"""
Сравнение трех способов определения области слайда на видео с ОДНИМ ручным ROI.

Эталон (ground truth) задается один раз на каждое видео через
select_manual_roi_from_video(...). Полученный BBox применяется ко всем
выбранным кадрам данного видео, поэтому покадровая CSV-разметка не нужна.

Методы:
1) classic_canny_contours -- существующий image_processing.keypoints_roi;
2) bbox_model             -- только bbox-модель из model_keypoints.py;
3) bbox_refined_quad      -- bbox-модель + уточнение quad по линиям.

Обработка bbox_model и bbox_refined_quad выполняется батчами; classic_canny_contours остаётся покадровым.

Результаты:
- detections.csv                -- IoU и время на каждом кадре;
- summary_overall.csv           -- итог по методам на всех видео;
- summary_by_video.csv          -- итог по каждому видео и методу;
- manual_roi/*.json             -- кэш выбранного вручную ROI;
- overlays/<video>/<method>/    -- изображения для визуального контроля.

Запуск выполняется из корня проекта, где доступен пакет image_processing.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

import cv2
import numpy as np

# В вашей структуре проекта функция ручного выбора ROI находится в модуле,
# где уже вызывается extract_content_keyframes. Если вы перенесли функцию,
# измените только эту строку импорта.
from image_processing.detect_ROI_with_metrics.content_selector import (
    select_manual_roi_from_video,
)

BBox = tuple[int, int, int, int]

METHOD_CLASSIC = "classic_canny_contours"
METHOD_BBOX = "bbox_model"
METHOD_REFINED = "bbox_refined_quad"
METHODS = (METHOD_CLASSIC, METHOD_BBOX, METHOD_REFINED)

DETAIL_FIELDS = [
    "video_id",
    "video_path",
    "frame_idx",
    "time_sec",
    "manual_roi_bbox",
    "manual_roi_points",
    "method",
    "inner_method",
    "pred_bbox",
    "pred_points",
    "iou",
    "iou_threshold",
    "is_success",
    "runtime_ms",
    "detection_error",
]

SUMMARY_FIELDS = [
    "method",
    "n_frames",
    "mean_iou",
    "median_iou",
    "successful_count",
    "success_rate",
    "mean_runtime_ms",
    "median_runtime_ms",
    "detector_errors",
]

SUMMARY_BY_VIDEO_FIELDS = ["video_id", "video_path"] + SUMMARY_FIELDS


@dataclass(frozen=True)
class VideoFrame:
    video_id: str
    video_path: Path
    frame_idx: int
    time_sec: float
    frame_bgr: np.ndarray


def stable_video_id(video_path: Path) -> str:
    """Устойчивое имя, чтобы видео с одинаковым stem не перезаписали ROI/overlays."""
    absolute = str(video_path.resolve()).encode("utf-8")
    suffix = hashlib.md5(absolute).hexdigest()[:8]
    return f"{video_path.stem}_{suffix}"


def write_csv_rows(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter=",")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def order_quad(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    return pts[np.argsort(angles)].astype(np.float32)


def rect_to_quad(bbox: BBox | list[int]) -> np.ndarray:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return np.asarray([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)


def clip_quad(quad: np.ndarray, width: int, height: int) -> np.ndarray:
    clipped = order_quad(quad).copy()
    clipped[:, 0] = np.clip(clipped[:, 0], 0, max(0, width - 1))
    clipped[:, 1] = np.clip(clipped[:, 1], 0, max(0, height - 1))
    return clipped


def bbox_from_quad(quad: np.ndarray) -> list[int]:
    pts = np.asarray(quad, dtype=np.float32).reshape(-1, 2)
    return [
        int(math.floor(float(pts[:, 0].min()))),
        int(math.floor(float(pts[:, 1].min()))),
        int(math.ceil(float(pts[:, 0].max()))),
        int(math.ceil(float(pts[:, 1].max()))),
    ]


def polygon_iou(true_quad: np.ndarray, pred_quad: np.ndarray) -> float:
    """IoU для ручного прямоугольного ROI и предсказанного bbox/quad."""
    true_poly = order_quad(true_quad)
    pred_poly = order_quad(pred_quad)
    true_area = abs(float(cv2.contourArea(true_poly)))
    pred_area = abs(float(cv2.contourArea(pred_poly)))
    if true_area <= 0.0 or pred_area <= 0.0:
        return 0.0
    intersection_area, _ = cv2.intersectConvexConvex(true_poly, pred_poly)
    union_area = true_area + pred_area - float(intersection_area)
    if union_area <= 0.0:
        return 0.0
    return max(0.0, min(1.0, float(intersection_area) / union_area))


def quad_json(quad: Optional[np.ndarray]) -> str:
    if quad is None:
        return ""
    return json.dumps(np.round(quad, 2).tolist(), ensure_ascii=False)


def normalize_detector_result(result: Any, width: int, height: int) -> tuple[np.ndarray, list[int], str]:
    """Приводит результат трех методов к единому четырехугольнику."""
    inner_method = ""
    if isinstance(result, dict):
        inner_method = str(result.get("method", ""))
        if result.get("points") is not None:
            quad = np.asarray(result["points"], dtype=np.float32).reshape(4, 2)
        elif result.get("bbox") is not None:
            quad = rect_to_quad([int(round(float(v))) for v in result["bbox"]])
        else:
            raise ValueError("Детектор вернул dict без points и bbox")
    else:
        values = np.asarray(result, dtype=np.float32)
        if values.shape == (4, 2):
            quad = values
        elif values.size == 4:
            quad = rect_to_quad([int(round(float(v))) for v in values.reshape(-1)])
        else:
            raise ValueError(f"Неожиданный формат результата детектора: {values.shape}")
    quad = clip_quad(quad, width, height)
    return quad, bbox_from_quad(quad), inner_method


def sample_video_frames(
    video_path: Path,
    sample_fps: float,
    start_sec: float = 0.0,
    end_sec: Optional[float] = None,
    max_frames: Optional[int] = None,
) -> Iterator[VideoFrame]:
    if sample_fps <= 0:
        raise ValueError("sample_fps должен быть больше 0")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Не удалось открыть видео: {video_path}")

    source_fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not np.isfinite(source_fps) or source_fps <= 0:
        cap.release()
        raise RuntimeError(f"Не удалось определить FPS видео: {video_path}")

    start_frame = max(0, int(round(start_sec * source_fps)))
    stop_frame = total_frames if end_sec is None else min(total_frames, int(round(end_sec * source_fps)))
    step = max(1, int(round(source_fps / sample_fps)))
    video_id = stable_video_id(video_path)

    produced = 0
    try:
        for frame_idx in range(start_frame, stop_frame, step):
            if max_frames is not None and produced >= max_frames:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            yield VideoFrame(
                video_id=video_id,
                video_path=video_path,
                frame_idx=frame_idx,
                time_sec=frame_idx / source_fps,
                frame_bgr=frame,
            )
            produced += 1
    finally:
        cap.release()


def iter_frame_batches(
    frames: Iterable[VideoFrame],
    batch_size: int,
) -> Iterator[list[VideoFrame]]:
    """Собирает кадры потоком, не сохраняя всё видео в оперативной памяти."""
    if batch_size <= 0:
        raise ValueError("batch_size должен быть больше 0")

    batch: list[VideoFrame] = []
    for frame in frames:
        batch.append(frame)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def detect_classic(frame_bgr: np.ndarray, debug_dir: Path) -> dict[str, Any]:
    from image_processing.keypoints_roi import detect_slide_keypoints

    try:
        return detect_slide_keypoints(frame_bgr)
    except TypeError as first_error:
        try:
            return detect_slide_keypoints(frame_bgr, debug_dir)
        except TypeError:
            raise TypeError(
                "Не совпала сигнатура detect_slide_keypoints в keypoints_roi.py. "
                "Измените адаптер detect_classic(...) под вашу версию функции."
            ) from first_error



def detect_model_batch(
    samples: list[VideoFrame],
    method: str,
    model_path: Optional[Path],
    image_size: int,
    backbone: str,
    dropout: float,
    roi_pad_ratio: float,
    max_side: int,
    min_line_len: int,
    fallback_to_full_image: bool,
) -> tuple[list[Optional[dict[str, Any]]], list[str], float]:
    """
    Запускает нейросетевой метод для одного батча кадров.

    Время возвращается в виде амортизированного времени на кадр:
    полное время выполнения метода для batch / число кадров в batch.
    Так сводные mean_runtime_ms сопоставимы с прежним форматом CSV.
    """
    from image_processing.model_keypoints_batched import detect_model_keypoints_for_frames

    if method not in (METHOD_BBOX, METHOD_REFINED):
        raise ValueError(f"Батчевый запуск не поддерживается для метода: {method}")
    if not samples:
        return [], [], 0.0

    frames_bgr = [sample.frame_bgr for sample in samples]
    started = time.perf_counter()
    try:
        results = detect_model_keypoints_for_frames(
            frames_bgr=frames_bgr,
            detector_mode=method,
            model_path=model_path,
            allow_fallback=False,  # ошибки не маскируются фиктивным default ROI
            image_size=image_size,
            backbone=backbone,
            dropout=dropout,
            roi_pad_ratio=roi_pad_ratio,
            max_side=max_side,
            min_line_len=min_line_len,
            fallback_to_full_image=fallback_to_full_image,
            batch_size=len(frames_bgr),  # ровно один model inference на текущий batch
        )
        if len(results) != len(samples):
            raise RuntimeError(
                f"Batch-детектор вернул {len(results)} результатов "
                f"для {len(samples)} кадров"
            )
        errors = [
            str(result.get("detector_error", "")).strip()
            if isinstance(result, dict)
            else ""
            for result in results
        ]
        outputs: list[Optional[dict[str, Any]]] = list(results)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        outputs = [None] * len(samples)
        errors = [error] * len(samples)

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    per_frame_runtime_ms = elapsed_ms / len(samples)
    return outputs, errors, per_frame_runtime_ms

def render_overlay(
    frame_bgr: np.ndarray,
    gt_quad: np.ndarray,
    pred_quad: Optional[np.ndarray],
    method: str,
    iou: float,
    success: bool,
    output_path: Path,
) -> None:
    canvas = frame_bgr.copy()
    cv2.polylines(
        canvas,
        [np.round(gt_quad).astype(np.int32).reshape((-1, 1, 2))],
        True,
        (0, 190, 0),
        3,
        cv2.LINE_AA,
    )
    if pred_quad is not None:
        cv2.polylines(
            canvas,
            [np.round(pred_quad).astype(np.int32).reshape((-1, 1, 2))],
            True,
            (0, 0, 230),
            3,
            cv2.LINE_AA,
        )
    label = f"{method} | IoU={iou:.3f} | success={int(success)} | GT=green PRED=red"
    cv2.rectangle(canvas, (0, 0), (min(canvas.shape[1] - 1, 1000), 38), (255, 255, 255), -1)
    cv2.putText(canvas, label, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 2, cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)


def append_detection_row(
    detail_rows: list[dict[str, Any]],
    sample: VideoFrame,
    manual_roi: BBox,
    method: str,
    raw_result: Optional[Any],
    runtime_ms: float,
    detection_error: str,
    args: argparse.Namespace,
    overlays_dir: Path,
) -> None:
    """Нормализует результат, рассчитывает IoU и при необходимости сохраняет overlay."""
    height, width = sample.frame_bgr.shape[:2]
    gt_quad = clip_quad(rect_to_quad(manual_roi), width, height)

    pred_quad: Optional[np.ndarray] = None
    pred_bbox: list[int] | str = ""
    inner_method = ""
    error = detection_error.strip()
    iou = 0.0

    if not error:
        try:
            if raw_result is None:
                raise RuntimeError("Детектор не вернул результат")
            pred_quad, pred_bbox, inner_method = normalize_detector_result(
                raw_result, width, height
            )
            iou = polygon_iou(gt_quad, pred_quad)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

    success = iou >= args.iou_threshold
    detail_rows.append(
        {
            "video_id": sample.video_id,
            "video_path": str(sample.video_path),
            "frame_idx": sample.frame_idx,
            "time_sec": f"{sample.time_sec:.4f}",
            "manual_roi_bbox": json.dumps(list(manual_roi), ensure_ascii=False),
            "manual_roi_points": quad_json(gt_quad),
            "method": method,
            "inner_method": inner_method,
            "pred_bbox": json.dumps(pred_bbox, ensure_ascii=False) if pred_bbox != "" else "",
            "pred_points": quad_json(pred_quad),
            "iou": f"{iou:.6f}",
            "iou_threshold": f"{args.iou_threshold:.2f}",
            "is_success": int(success),
            "runtime_ms": f"{runtime_ms:.4f}",
            "detection_error": error,
        }
    )

    save_overlay = args.save_overlays == "all" or (
        args.save_overlays == "failures" and not success
    )
    if save_overlay:
        overlay_name = f"frame_{sample.frame_idx:08d}_t_{sample.time_sec:010.3f}.jpg"
        render_overlay(
            sample.frame_bgr,
            gt_quad,
            pred_quad,
            method,
            iou,
            success,
            overlays_dir / sample.video_id / method / overlay_name,
        )


def summarize_rows(rows: list[dict[str, Any]], by_video: bool) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (str(row["video_id"]), str(row["video_path"]), str(row["method"])) if by_video else (str(row["method"]),)
        grouped[key].append(row)

    summary: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        method = key[-1]
        ious = [float(row["iou"]) for row in group]
        times = [float(row["runtime_ms"]) for row in group]
        successes = [int(row["is_success"]) for row in group]
        item: dict[str, Any] = {}
        if by_video:
            item["video_id"] = key[0]
            item["video_path"] = key[1]
        item.update(
            {
                "method": method,
                "n_frames": len(group),
                "mean_iou": round(statistics.fmean(ious), 6),
                "median_iou": round(statistics.median(ious), 6),
                "successful_count": sum(successes),
                "success_rate": round(sum(successes) / len(group), 6),
                "mean_runtime_ms": round(statistics.fmean(times), 4),
                "median_runtime_ms": round(statistics.median(times), 4),
                "detector_errors": sum(bool(str(row.get("detection_error", "")).strip()) for row in group),
            }
        )
        summary.append(item)
    return summary


def resolve_videos(video_args: list[str], videos_dir: Optional[str]) -> list[Path]:
    paths = [Path(raw) for raw in video_args]
    if videos_dir:
        root = Path(videos_dir)
        allowed = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
        paths.extend(sorted(path for path in root.rglob("*") if path.suffix.lower() in allowed))
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        if not resolved.exists():
            raise FileNotFoundError(f"Видео не найдено: {path}")
        seen.add(resolved)
        unique.append(resolved)
    if not unique:
        raise ValueError("Укажите хотя бы одно видео или каталог --videos-dir")
    return unique



def run_experiment(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch-size должен быть больше 0")

    videos = resolve_videos(args.videos, args.videos_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    roi_dir = output_dir / "manual_roi"
    overlays_dir = output_dir / "overlays"
    debug_dir = output_dir / "classic_debug"
    model_path = Path(args.model_path) if args.model_path else None

    detail_rows: list[dict[str, Any]] = []
    warmed_up = False

    for video_number, video_path in enumerate(videos, start=1):
        video_id = stable_video_id(video_path)
        cache_path = roi_dir / f"{video_id}.json"
        roi_dir.mkdir(parents=True, exist_ok=True)
        manual_roi: BBox = tuple(
            int(v)
            for v in select_manual_roi_from_video(
                video_path=str(video_path),
                time_sec=args.roi_time_sec,
                cache_path=cache_path,
                max_display_side=args.max_display_side,
                force_reselect=args.force_reselect,
            )
        )  # type: ignore[assignment]

        sampled_frames = sample_video_frames(
            video_path=video_path,
            sample_fps=args.sample_fps,
            start_sec=args.start_sec,
            end_sec=args.end_sec,
            max_frames=args.max_frames_per_video,
        )

        processed_for_video = 0
        received_any_batch = False

        for samples in iter_frame_batches(sampled_frames, args.batch_size):
            received_any_batch = True

            # Загрузка модели и построение TensorFlow-графа не включаются в замер.
            # Прогреваем на batch того же размера, что и первый рабочий batch.
            if not warmed_up:
                detect_classic(samples[0].frame_bgr, debug_dir)
                for neural_method in (METHOD_BBOX, METHOD_REFINED):
                    detect_model_batch(
                        samples=samples,
                        method=neural_method,
                        model_path=model_path,
                        image_size=args.image_size,
                        backbone=args.backbone,
                        dropout=args.dropout,
                        roi_pad_ratio=args.roi_pad_ratio,
                        max_side=args.max_side,
                        min_line_len=args.min_line_len,
                        fallback_to_full_image=not args.no_full_image_refine_fallback,
                    )
                warmed_up = True

            # Classic Canny не использует нейросеть и измеряется покадрово.
            for sample in samples:
                started = time.perf_counter()
                try:
                    classic_result: Optional[Any] = detect_classic(sample.frame_bgr, debug_dir)
                    classic_error = ""
                except Exception as exc:
                    classic_result = None
                    classic_error = f"{type(exc).__name__}: {exc}"
                classic_runtime_ms = (time.perf_counter() - started) * 1000.0

                append_detection_row(
                    detail_rows=detail_rows,
                    sample=sample,
                    manual_roi=manual_roi,
                    method=METHOD_CLASSIC,
                    raw_result=classic_result,
                    runtime_ms=classic_runtime_ms,
                    detection_error=classic_error,
                    args=args,
                    overlays_dir=overlays_dir,
                )

            # Методы запускаются независимо: время refined включает bbox inference и refinement.
            for neural_method in (METHOD_BBOX, METHOD_REFINED):
                batch_results, batch_errors, per_frame_runtime_ms = detect_model_batch(
                    samples=samples,
                    method=neural_method,
                    model_path=model_path,
                    image_size=args.image_size,
                    backbone=args.backbone,
                    dropout=args.dropout,
                    roi_pad_ratio=args.roi_pad_ratio,
                    max_side=args.max_side,
                    min_line_len=args.min_line_len,
                    fallback_to_full_image=not args.no_full_image_refine_fallback,
                )
                for sample, raw_result, error in zip(samples, batch_results, batch_errors):
                    append_detection_row(
                        detail_rows=detail_rows,
                        sample=sample,
                        manual_roi=manual_roi,
                        method=neural_method,
                        raw_result=raw_result,
                        runtime_ms=per_frame_runtime_ms,
                        detection_error=error,
                        args=args,
                        overlays_dir=overlays_dir,
                    )

            processed_for_video += len(samples)

        if not received_any_batch:
            raise RuntimeError(f"Из видео не получено ни одного исследуемого кадра: {video_path}")

        print(
            f"[{video_number}/{len(videos)}] {video_path.name}: ROI={manual_roi}, "
            f"кадров={processed_for_video}, batch_size={args.batch_size}, cache={cache_path}"
        )

    details_path = output_dir / "detections.csv"
    overall_path = output_dir / "summary_overall.csv"
    by_video_path = output_dir / "summary_by_video.csv"
    write_csv_rows(details_path, detail_rows, DETAIL_FIELDS)
    overall = summarize_rows(detail_rows, by_video=False)
    by_video = summarize_rows(detail_rows, by_video=True)
    write_csv_rows(overall_path, overall, SUMMARY_FIELDS)
    write_csv_rows(by_video_path, by_video, SUMMARY_BY_VIDEO_FIELDS)

    print(f"\nДетальные результаты: {details_path}")
    print(f"Сводка по всем видео: {overall_path}")
    print(f"Сводка по каждому видео: {by_video_path}")
    if args.save_overlays != "none":
        print(f"Контрольные изображения: {overlays_dir}")
    print(
        "\nВремя bbox_model и bbox_refined_quad: полное время batch, "
        "делённое на число кадров в этом batch."
    )
    print("\nИтог по методам:")
    for row in overall:
        print(
            f"  {row['method']}: mean IoU={row['mean_iou']:.4f}, "
            f"success={float(row['success_rate']):.2%}, "
            f"mean time={row['mean_runtime_ms']:.2f} ms"
        )

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Оценка выделения области слайда по одному ручному ROI на видео"
    )
    parser.add_argument(
        "videos",
        nargs="*",
        help="Пути к видео. Можно передать несколько файлов.",
    )
    parser.add_argument(
        "--videos-dir",
        default=None,
        help="Каталог видео; будут добавлены все поддерживаемые видеофайлы рекурсивно.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-path", default=r"D:\Disk C\data\media\slide_dataset\output\best.weights.h5", help="Путь к bbox-модели")

    parser.add_argument("--roi-time-sec", type=float, default=120.0, help="Кадр для ручного выбора ROI")
    parser.add_argument("--force-reselect", action="store_true", help="Заново выбрать ROI даже при наличии кэша")
    parser.add_argument("--max-display-side", type=int, default=1280)

    parser.add_argument("--sample-fps", type=float, default=1.0, help="Частота исследуемых кадров")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Размер batch для bbox_model и bbox_refined_quad; кадры читаются потоково",
    )
    parser.add_argument("--start-sec", type=float, default=0.0)
    parser.add_argument("--end-sec", type=float, default=None)
    parser.add_argument("--max-frames-per-video", type=int, default=None)
    parser.add_argument("--iou-threshold", type=float, default=0.70)
    parser.add_argument(
        "--save-overlays",
        choices=("failures", "all", "none"),
        default="failures",
        help="Какие кадры сохранять с наложенными GT и предсказанием",
    )

    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--backbone", default="MobileNetV2")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--roi-pad-ratio", type=float, default=0.20)
    parser.add_argument("--max-side", type=int, default=1200)
    parser.add_argument("--min-line-len", type=int, default=80)
    parser.add_argument(
        "--no-full-image-refine-fallback",
        action="store_true",
        help="Не искать линии во всем кадре, если уточнение внутри bbox не сработало",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()

# def main() -> None:
#     roi = select_manual_roi_from_video(
#         r"C:\Users\dondu\PycharmProjects\automatic_conspect33\runs\math\math.mp4", 
#         time_sec=120.0,
#         cache_path="selected_roi.json")
#     args = build_parser().parse_args()
#     args.func(args)


# if __name__ == "__main__":
#     main()


# def main() -> None:
#     roi = select_manual_roi_from_video(
#         r"C:\Users\dondu\PycharmProjects\automatic_conspect33\runs\math\math.mp4", 
#         time_sec=120.0,
#         cache_path="selected_roi.json")
#     args = build_parser().parse_args()
#     args.func(args)


# if __name__ == "__main__":
#     main()
