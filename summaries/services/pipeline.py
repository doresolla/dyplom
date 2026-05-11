from __future__ import annotations

import csv
import inspect
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from django.conf import settings
from django.core.files import File
from django.core.mail import send_mail
from django.utils import timezone

from summaries.models import LectureNote, Notification, ProcessingLog
from summaries.services.markdown_pdf import generate_pdf_from_markdown
from summaries.services.filenames import make_safe_filename_stem


class PipelineImportError(RuntimeError):
    pass


@dataclass
class DbProgressLogger:
    note_id: int
    celery_task: Any | None = None

    def emit(self, message: str, progress: int | None = None) -> None:
        ProcessingLog.objects.create(note_id=self.note_id, message=message)
        update = {'status_message': message}
        if progress is not None:
            update['progress'] = max(0, min(100, progress))
        LectureNote.objects.filter(pk=self.note_id).update(**update)
        if self.celery_task is not None:
            self.celery_task.update_state(
                state='PROGRESS',
                meta={'note_id': self.note_id, 'message': message, 'progress': update.get('progress')},
            )


@contextmanager
def _pipeline_runtime_context() -> Iterator[Path]:
    """
    Запускает ML-код в тех же условиях, что и desktop/main.py:
    - добавляет корень ML-проекта в sys.path;
    - временно делает его текущей рабочей директорией;
    - прокидывает PYTHONPATH дочерним процессам.

    Это важно для LLM-модулей, если они читают локальные конфиги, относительные пути
    или запускают вспомогательные скрипты из корня проекта.
    """
    root = _prepare_python_path()
    old_cwd = Path.cwd()
    os.chdir(root)
    try:
        yield root
    finally:
        os.chdir(old_cwd)


def run_pipeline_for_note(note_id: int, celery_task: Any | None = None) -> dict[str, Any]:
    """
    Полный серверный пайплайн.

    Эта функция заменяет PyQt-класс Main(QRunnable): вместо сигналов GUI она пишет
    прогресс в БД, а запускается через Celery.
    """
    logger = DbProgressLogger(note_id=note_id, celery_task=celery_task)

    note = LectureNote.objects.select_related('owner').get(pk=note_id)
    LectureNote.objects.filter(pk=note_id).update(
        status=LectureNote.Status.PROCESSING,
        progress=1,
        status_message='Подготовка обработки',
        error_message='',
    )

    try:
        with _pipeline_runtime_context() as pipeline_root:
            modules = _import_pipeline_modules()

            source_video = Path(note.video_file.path).resolve()
            work_root = Path(settings.MEDIA_ROOT).resolve() / 'runs' / str(note.owner_id)
            run_dir = work_root / f'note_{note.id}'
            run_dir.mkdir(parents=True, exist_ok=True)

            llm_model = (note.llm_model or getattr(settings, 'DEFAULT_LLM_MODEL', 'qwen') or 'qwen').lower()
            summary_basename = make_safe_filename_stem(note.title, fallback=f'note_{note.id}')
            summary_md_name = f'{summary_basename}.md'
            summary_pdf_name = f'{summary_basename}.pdf'
            logger.emit(f'Статус: используется LLM-модель: {llm_model}')
            logger.emit(f'Статус: имя файла конспекта: {summary_basename}')
            logger.emit(f'Статус: корень ML-проекта: {pipeline_root}')

            logger.emit('Статус: 1/7 Разделение аудио и видео', 5)
            assets = modules['audio'].split_audio_video(source_video, run_dir)

            logger.emit('Статус: 2/7 Распознавание голоса', 15)
            transcript_path = run_dir / 'transcript.txt'
            if not transcript_path.exists():
                _transcribe(assets.audio_wav, transcript_path, logger, pipeline_root)
            _attach_file(note, 'transcript_file', transcript_path, f'transcripts/{note.id}/transcript.txt')

            logger.emit('Статус: 3/7 Отбор ключевых кадров', 35)
            keyframes, frame_paths = _prepare_content_keyframes(
                video_path=assets.local_video,
                work_dir=run_dir / 'content_selector',
                extract_content_keyframes=modules['extract_content_keyframes'],
                logger=logger,
                use_roi_images=False,
            )

            logger.emit('Статус: 4/7 Определение ROI/ключевых точек слайдов', 50)
            keypoints = _detect_keypoints(
                frame_paths=frame_paths,
                run_dir=run_dir,
                modules=modules,
                logger=logger,
            )
            keypoints_path = run_dir / 'keypoints.json'
            keypoints_path.write_text(json.dumps(keypoints, ensure_ascii=False, indent=2), encoding='utf-8')

            logger.emit('Статус: 5/7 Обрезка кадров по ключевым точкам', 63)
            cropped = modules['crop_frames_by_keypoints'](
                frame_paths,
                keypoints,
                run_dir / 'cropped',
            )

            logger.emit('Статус: 6/7 OCR', 75)
            ocr_path = modules['run_ocr'](cropped, run_dir / 'ocr')

            logger.emit('Статус: 7/7 Сборка мультимодального конспекта через LLM', 88)
            llm_callback_messages: list[str] = []
            summary_path, summary_meta = _build_summary_with_llm(
                build_multimodal_summary=modules['build_multimodal_summary'],
                transcript_path=transcript_path,
                out_path=run_dir / summary_md_name,
                frame_paths=cropped,
                keyframes=keyframes,
                ocr_dir=run_dir / 'ocr',
                model=llm_model,
                title=f'Конспект: {note.title}',
                include_ocr=True,
                min_chars_for_llm=220,
                logger=logger,
                callback_messages=llm_callback_messages,
            )

            summary_text = Path(summary_path).read_text(encoding='utf-8')
            _attach_file(note, 'summary_file', Path(summary_path), f'summaries/{note.id}/{summary_md_name}')

            logger.emit('Статус: Генерация PDF-конспекта', 94)
            pdf_path = generate_pdf_from_markdown(
                markdown_path=Path(summary_path),
                pdf_path=run_dir / summary_pdf_name,
                title=f'Конспект: {note.title}',
                log=logger.emit,
            )
            _attach_file(note, 'summary_pdf_file', pdf_path, f'summaries/{note.id}/{summary_pdf_name}')

            LectureNote.objects.filter(pk=note_id).update(
                status=LectureNote.Status.DONE,
                progress=100,
                status_message='Работа завершена',
                summary_text=summary_text,
                result_dir=str(run_dir),
                pipeline_meta={
                    'run_dir': str(run_dir),
                    'pipeline_root': str(pipeline_root),
                    'llm_model': llm_model,
                    'transcript_path': str(transcript_path),
                    'ocr_path': str(ocr_path),
                    'summary_path': str(summary_path),
                    'summary_pdf_path': str(pdf_path),
                    'summary_basename': summary_basename,
                    'summary_md_name': summary_md_name,
                    'summary_pdf_name': summary_pdf_name,
                    'n_keyframes': len(keyframes),
                    'llm_callback_messages': llm_callback_messages[-80:],
                    'summary_meta': _json_safe(summary_meta),
                },
                completed_at=timezone.now(),
            )

            _notify_success(note_id)
            return {'note_id': note_id, 'status': 'done'}

    except Exception as exc:
        message = str(exc)
        LectureNote.objects.filter(pk=note_id).update(
            status=LectureNote.Status.ERROR,
            status_message='Ошибка обработки',
            error_message=message,
        )
        logger.emit(f'Ошибка: {message}')
        _notify_error(note_id, message)
        raise


def _prepare_python_path() -> Path:
    pipeline_root = getattr(settings, 'PIPELINE_ROOT', '')
    if not pipeline_root:
        raise PipelineImportError('Не указан PIPELINE_ROOT в .env. Нужен путь к папке с ML-модулями проекта.')

    root = Path(pipeline_root).expanduser().resolve()
    if not root.exists():
        raise PipelineImportError(f'PIPELINE_ROOT не найден: {root}')

    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    current_pythonpath = os.environ.get('PYTHONPATH', '')
    paths = [p for p in current_pythonpath.split(os.pathsep) if p]
    if root_str not in paths:
        os.environ['PYTHONPATH'] = os.pathsep.join([root_str, *paths])
    os.environ.setdefault('PIPELINE_ROOT', root_str)
    return root


def _import_pipeline_modules() -> dict[str, Any]:
    try:
        import text_processing.audio as audio
        from image_processing.OCR import run_ocr
        from image_processing.detect_ROI_with_metrics.content_selector import extract_content_keyframes
        from image_processing.keypoints_roi import detect_keypoints_for_images
        from multimodal_summary import build_multimodal_summary
        from utils import crop_frames_by_keypoints
    except Exception as exc:
        raise PipelineImportError(
            'Не удалось импортировать модули пайплайна. Проверь PIPELINE_ROOT и зависимости ASR/OCR/LLM.'
        ) from exc

    return {
        'audio': audio,
        'run_ocr': run_ocr,
        'extract_content_keyframes': extract_content_keyframes,
        'detect_keypoints_for_images': detect_keypoints_for_images,
        'build_multimodal_summary': build_multimodal_summary,
        'crop_frames_by_keypoints': crop_frames_by_keypoints,
    }


def _transcribe(audio_path: Path, out_path: Path, logger: DbProgressLogger, pipeline_root: Path) -> None:
    worker_script = pipeline_root / 'text_processing' / 'transcribe_worker.py'
    if not worker_script.exists():
        raise RuntimeError(f'Не найден transcribe_worker.py: {worker_script}')

    logger.emit('[transcribe] starting subprocess')
    process = subprocess.Popen(
        [sys.executable, str(worker_script), str(audio_path), str(out_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        bufsize=1,
        cwd=str(pipeline_root),
        env=os.environ.copy(),
    )

    assert process.stdout is not None
    for line in process.stdout:
        line = line.rstrip()
        if line:
            logger.emit(line)

    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f'Whisper subprocess failed with code {return_code}')
    logger.emit('[transcribe] subprocess finished')


def _prepare_content_keyframes(
    video_path: Path,
    work_dir: Path,
    extract_content_keyframes,
    logger: DbProgressLogger,
    use_roi_images: bool = False,
) -> tuple[list[tuple[float, str]], list[Path]]:
    work_dir.mkdir(parents=True, exist_ok=True)
    manifest_csv = work_dir / 'keyframe_manifest.csv'

    if not manifest_csv.exists():
        logger.emit('[keyframes] запуск content_selector: метрики, пороги, сегменты')
        result = extract_content_keyframes(
            video_path=str(video_path),
            work_dir=str(work_dir),
            sample_fps=1.0,
            roi=None,
            auto_roi=True,
            min_stable_sec=1.0,
            guard_sec=0.5,
            min_segment_sec=2.5,
            drop_scroll_variants=True,
            primary_min_duration_sec=5.0,
            drop_contained_variants=True,
            containment_thr=0.97,
            max_containment_gap_sec=90.0,
            max_removed_ratio=0.03,
            min_contained_area_ratio=0.99,
            max_hamming=6,
        )
        manifest_csv = Path(result['manifest_csv'])
        logger.emit(
            f"[keyframes] samples={result.get('n_samples')}, "
            f"segments={result.get('n_segments')}, selected={result.get('n_keyframes')}"
        )
    else:
        logger.emit(f'[keyframes] найден готовый manifest: {manifest_csv}')

    if not manifest_csv.exists():
        raise RuntimeError(f'Не найден manifest ключевых кадров: {manifest_csv}')

    path_field = 'roi_path' if use_roi_images else 'image_path'
    keyframes: list[tuple[float, str]] = []

    with manifest_csv.open('r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_path = row.get(path_field) or row.get('image_path')
            raw_time = row.get('time_sec')
            if not raw_path or raw_time is None:
                continue
            frame_path = Path(raw_path)
            if not frame_path.exists():
                logger.emit(f'[keyframes] файл не найден: {frame_path}')
                continue
            keyframes.append((float(raw_time), str(frame_path)))

    keyframes.sort(key=lambda item: item[0])
    frame_paths = [Path(path) for _, path in keyframes]
    if not frame_paths:
        raise RuntimeError('content_selector не вернул ни одного ключевого кадра')

    logger.emit(f'[keyframes] сохранено кадров: {len(frame_paths)}')
    return keyframes, frame_paths


def _detect_keypoints(frame_paths: list[Path], run_dir: Path, modules: dict[str, Any], logger: DbProgressLogger):
    detect_keypoints_for_images = modules.get('detect_keypoints_for_images')
    if detect_keypoints_for_images is not None:
        return detect_keypoints_for_images(frame_paths, run_dir)

    try:
        from image_processing.model_keypoints import detect_model_keypoints_for_images
    except Exception as exc:
        raise RuntimeError('Не найден ни keypoints_roi, ни model_keypoints для определения ROI.') from exc

    model_path = getattr(settings, 'KEYPOINTS_MODEL_PATH', '')
    if not model_path:
        raise RuntimeError('Не указан KEYPOINTS_MODEL_PATH в .env')

    return detect_model_keypoints_for_images(
        frame_paths,
        model_path=Path(model_path),
        output_mode='auto',
        image_size=320,
        backbone='MobileNetV2',
        dropout=0.2,
        roi_pad_ratio=0.20,
        max_side=1200,
        min_line_len=80,
        callback=lambda text: logger.emit(str(text)),
    )


def _build_summary_with_llm(
    *,
    build_multimodal_summary,
    transcript_path: Path,
    out_path: Path,
    frame_paths: list[Path],
    keyframes: list[tuple[float, str]],
    ocr_dir: Path,
    model: str,
    title: str,
    include_ocr: bool,
    min_chars_for_llm: int,
    logger: DbProgressLogger,
    callback_messages: list[str],
):
    """
    Вызывает build_multimodal_summary как в desktop-пайплайне, но дополнительно
    включает строгие LLM-флаги, если такие параметры есть в твоей реализации.
    Это не ломает старую сигнатуру: неизвестные параметры не передаются.
    """
    def callback(text: Any) -> None:
        message = str(text)
        callback_messages.append(message)
        logger.emit(message)

    kwargs = {
        'transcript_path': transcript_path,
        'out_path': out_path,
        'frame_paths': frame_paths,
        'keyframes': keyframes,
        'ocr_dir': ocr_dir,
        'model': model,
        'title': title,
        'include_ocr': include_ocr,
        'min_chars_for_llm': min_chars_for_llm,
        'callback': callback,
    }

    try:
        params = inspect.signature(build_multimodal_summary).parameters
    except (TypeError, ValueError):
        params = {}

    optional_llm_flags = {
        'use_llm': True,
        'force_llm': True,
        'llm_required': True,
        'strict_llm': True,
    }
    for name, value in optional_llm_flags.items():
        if name in params:
            kwargs[name] = value
            logger.emit(f'[LLM] включен параметр {name}=True')

    if 'model_name' in params and 'model' not in params:
        kwargs['model_name'] = kwargs.pop('model')

    return build_multimodal_summary(**kwargs)


def _attach_file(note: LectureNote, field_name: str, src_path: Path, relative_name: str) -> None:
    """Копирует готовый артефакт пайплайна в Django FileField."""
    src_path = Path(src_path)
    if not src_path.exists():
        return

    with src_path.open('rb') as f:
        getattr(note, field_name).save(relative_name, File(f), save=True)


def _notify_success(note_id: int) -> None:
    note = LectureNote.objects.select_related('owner').get(pk=note_id)
    Notification.objects.create(
        user=note.owner,
        note=note,
        title='Конспект готов',
        message=f'Обработка видео «{note.title}» завершена. Конспект доступен в личном кабинете.',
    )
    if note.owner.email:
        send_mail(
            subject='Конспект видеолекции готов',
            message=f'Здравствуйте! Конспект «{note.title}» готов и доступен в личном кабинете.',
            from_email=settings.DEFAULT_FROM_EMAIL,
            recipient_list=[note.owner.email],
            fail_silently=True,
        )


def _notify_error(note_id: int, error_message: str) -> None:
    note = LectureNote.objects.select_related('owner').get(pk=note_id)
    Notification.objects.create(
        user=note.owner,
        note=note,
        title='Ошибка обработки',
        message=f'Не удалось обработать видео «{note.title}». Ошибка: {error_message}',
    )
    if note.owner.email:
        send_mail(
            subject='Ошибка обработки видеолекции',
            message=f'При обработке «{note.title}» возникла ошибка:\n\n{error_message}',
            from_email=settings.DEFAULT_FROM_EMAIL,
            recipient_list=[note.owner.email],
            fail_silently=True,
        )


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value, ensure_ascii=False)
        return value
    except TypeError:
        return str(value)
