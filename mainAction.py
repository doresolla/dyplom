from __future__ import annotations

import json
from pathlib import Path

import cv2
from PyQt6.QtCore import QRunnable
import subprocess, sys

import audio
from OCR import run_ocr
from LLMsummary import save_summary, summarize_with_llm
from keyPoints import detect_keypoints_for_images
from sharpness import select_keyframes


class Main(QRunnable):
    def __init__(self, signal, link, set_videoname_signal, print_signal):
        super().__init__()
        self.signals = signal
        self.video = set_videoname_signal
        self.print_signal = print_signal
        self.link = link.strip('"')

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
            self._transcribe(assets.audio_wav, transcript_path)

            self.signals.result.emit("Статус: 3/7 Отбор ключевых кадров")
            frames_dir = run_dir / "keyframes"
            keyframes = select_keyframes(str(assets.local_video), str(frames_dir), sample_fps=1.0)
            frame_paths = [Path(p) for _, p in keyframes]

            self.signals.result.emit("Статус: 4/7 Определение ключевых точек слайдов")
            keypoints = detect_keypoints_for_images(frame_paths)
            keypoints_path = run_dir / "keypoints.json"
            keypoints_path.write_text(json.dumps(keypoints, ensure_ascii=False, indent=2), encoding="utf-8")

            self.signals.result.emit("Статус: 5/7 Обрезка кадров по ключевым точкам")
            cropped = self._crop_frames_by_keypoints(frame_paths, keypoints, run_dir / "cropped")

            self.signals.result.emit("Статус: 6/7 OCR")
            ocr_path = run_ocr(cropped, run_dir / "ocr")

            self.signals.result.emit("Статус: 7/7 Абстрактивная суммаризация")
            summary = summarize_with_llm(
                transcript_path.read_text(encoding="utf-8") + "\n\n" + ocr_path.read_text(encoding="utf-8")
            )
            summary_path = save_summary(summary, run_dir / "summary.md")

            self.signals.result.emit("Статус: Работа завершена")
            self.print_signal.result.emit(f"Готово: {summary_path}")
        except Exception as exc:
            self.signals.result.emit("Статус: Ошибка")
            self.print_signal.result.emit(str(exc))

    def _transcribe(self, audio_path: Path, out_path: Path):
        # from faster_whisper import WhisperModel
        # model = WhisperModel("small", device="cpu", compute_type="int8")
        # segments, _ = model.transcribe(str(audio_path), beam_size=5, language="ru")
        # with out_path.open("w", encoding="utf-8") as f:
        #     for seg in segments:
        #         f.write(f"[{seg.start:.2f}-{seg.end:.2f}] {seg.text.strip()}\n")
        # def _transcribe(self, audio_path: Path, out_path: Path):
            script = Path(__file__).resolve().parent / "transcribe_worker.py"
            result = subprocess.run(
                [sys.executable, str(script), str(audio_path), str(out_path)],
                capture_output=True,
                text=True,
                encoding="utf-8"
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr or result.stdout or "Ошибка транскрибации")


    def _crop_frames_by_keypoints(self, frame_paths, keypoints_map, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        cropped_paths = []
        for frame_path in frame_paths:
            frame = cv2.imread(str(frame_path))
            if frame is None:
                continue
            meta = keypoints_map.get(str(frame_path))
            if not meta:
                continue
            x1, y1, x2, y2 = meta["bbox"]
            crop = frame[y1:y2, x1:x2]
            out_path = out_dir / frame_path.name
            cv2.imwrite(str(out_path), crop)
            cropped_paths.append(out_path)
        return cropped_paths
