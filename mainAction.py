from __future__ import annotations

import json
import os.path
from pathlib import Path
from faster_whisper import WhisperModel
from multimodal_summary import build_multimodal_summary
from PyQt6.QtCore import QRunnable
import subprocess, sys

import text_processing.audio as audio
from image_processing.OCR import run_ocr
from text_processing.LLMsummary import save_summary, summarize_with_llm
from image_processing.sharpness import select_keyframes
# from image_processing.model_keypoints import detect_model_keypoints_for_images, detect_model_slide_keypoints
from image_processing.keypoints_roi import detect_keypoints_for_images, detect_slide_keypoints
from utils import crop_frames_by_keypoints




class Main(QRunnable):
    def __init__(self, signal, link, set_videoname_signal, print_signal, llm_model, keypoints_output_mode = "auto"):
        super().__init__()
        self.signals = signal
        self.video = set_videoname_signal
        self.print_signal = print_signal
        self.link = link.strip('"')
        self.llm_model = (llm_model or "qwen").lower()
        self.OCR = False
        self.keypoints_model_path = Path("D:/Disk C/data/media/slide_dataset/output/best.weights.h5")
        self.keypoints_output_mode = keypoints_output_mode

    def run(self):
        try:
            work_root = Path(__file__).resolve().parent / "runs"
            work_root.mkdir(exist_ok=True)

            self.signals.result.emit("Статус: Подготовка входного видео")
            source_video = audio.resolve_input(self.link, work_root)
            run_dir = work_root / source_video.stem
            run_dir.mkdir(exist_ok=True)
            self.video.result.emit(source_video.stem)

            self.signals.result.emit("Статус: 1/7 Разделение аудио и видео")
            assets = audio.split_audio_video(source_video, run_dir)

            self.signals.result.emit("Статус: 2/7 Распознавание голоса")
            transcript_path = run_dir / "transcript.txt"
            if not (os.path.exists(transcript_path)):
                self._transcribe(assets.audio_wav, transcript_path)
            self.signals.result.emit("Статус: 3/7 Отбор ключевых кадров")
            frames_dir = run_dir / "keyframes"

            if detect_slide_keypoints is None:
                from image_processing.model_keypoints import detect_model_slide_keypoints
                roi_detector = lambda frame: detect_model_slide_keypoints(
                    frame_bgr=frame,
                    model_path=self.keypoints_model_path,
                    output_mode=self.keypoints_output_mode,
                )
            else:
                roi_detector = (
                    (lambda frame: detect_slide_keypoints)
                )   
            keyframes = select_keyframes(
                str(assets.local_video),
                str(frames_dir),
                sample_fps=1.0,
                roi_detector=roi_detector,
                detector_stride=5,
            )
            frame_paths = [Path(p) for _, p in keyframes]

            self.signals.result.emit("Статус: 4/7 Определение ключевых точек слайдов")
            if detect_keypoints_for_images is None:
                from image_processing.model_keypoints import detect_model_keypoints_for_images
                keypoints = detect_model_keypoints_for_images(
                    frame_paths,
                    model_path=self.keypoints_model_path,
                    output_mode=self.keypoints_output_mode,
                    image_size=320,
                    backbone="MobileNetV2",
                    dropout=0.2,
                    roi_pad_ratio=0.20,
                    max_side=1200,
                    min_line_len=80,
                    callback=self.print_signal.result.emit
                )
            else:
                keypoints = detect_keypoints_for_images(
                    frame_paths,
                    run_dir,
                )

            keypoints_path = run_dir / "keypoints.json"
            keypoints_path.write_text(json.dumps(keypoints, ensure_ascii=False, indent=2), encoding="utf-8")

            self.signals.result.emit("Статус: 5/7 Обрезка кадров по ключевым точкам")
            cropped = crop_frames_by_keypoints(
                frame_paths,
                keypoints,
                run_dir / "cropped",
            )
            self.signals.result.emit("Статус: 6/7 OCR")
            ocr_path = run_ocr(cropped, run_dir / "ocr")

            self.signals.result.emit("Статус: 7/7 Абстрактивная суммаризация")
            summary_input = transcript_path.read_text(encoding="utf-8")
            if self.OCR:
                summary_input += "\n\nТекст со слайдов:\n" + ocr_path.read_text(encoding="utf-8")
            # summary = summarize_with_llm(summary_input)
            # summary_path = save_summary(summary, run_dir / f"summary_{self.llm_model}.md")

            self.signals.result.emit("Статус: 7/7 Сборка мультимодального конспекта")
            summary_path, _ = build_multimodal_summary(
                transcript_path=transcript_path,
                out_path=run_dir / f"summary_{self.llm_model}.md",
                frame_paths=cropped,  # именно cropped, чтобы в конспект шли уже выделенные слайды
                keyframes=keyframes,  # если None, время возьмется из имени файла
                ocr_dir=run_dir / "ocr",
                model=self.llm_model,
                title=f"Конспект: {source_video.stem}",
                include_ocr=True,
                min_chars_for_llm=220,
                callback=self.print_signal.result.emit,
            )

            self.signals.result.emit("Статус: Работа завершена")
            self.print_signal.result.emit(f"Готово: {summary_path}")
        except Exception as exc:
            self.signals.result.emit("Статус: Ошибка")
            self.print_signal.result.emit(str(exc))

    def _prepare_keyframes(self, video_path: Path, frames_dir: Path) -> list[Path]:
        frames_dir.mkdir(parents=True, exist_ok=True)

        existing = sorted(
            p for p in frames_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        )

        if existing:
            self.print_signal.result.emit(
                f"[keyframes] найдено готовых кадров: {len(existing)}"
            )
            return existing

        self.print_signal.result.emit("[keyframes] запуск select_keyframes")
        keyframes = select_keyframes(
            str(video_path),
            str(frames_dir),
            sample_fps=1.0,
        )

        frame_paths = [Path(p) for _, p in keyframes if p]
        self.print_signal.result.emit(
            f"[keyframes] сохранено кадров: {len(frame_paths)}"
        )
        return frame_paths


    def _transcribe(self, audio_path: Path, out_path: Path):
        import subprocess
        import sys

        worker_script = Path(__file__).resolve().parent / "transcribe_worker.py"

        self.print_signal.result.emit("[transcribe] starting subprocess")

        process = subprocess.Popen(
            [sys.executable, str(worker_script), str(audio_path), str(out_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            bufsize=1
        )

        for line in process.stdout:
            line = line.rstrip()
            if not line:
                continue

            print(line, flush=True)
            self.print_signal.result.emit(line)

        return_code = process.wait()

        if return_code != 0:
            raise RuntimeError(f"Whisper subprocess failed with code {return_code}")

        self.print_signal.result.emit("[transcribe] subprocess finished")

    