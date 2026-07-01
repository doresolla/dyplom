from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable, Literal, Sequence

import cv2
import numpy as np
import tensorflow as tf

from .detect_slide_with_model.slide_bbox_common import (
    clip_boxes_xywh_np,
    denormalize_xywh,
    load_trained_model,
)
from .detect_slide_with_model.refine_bbox import (
    bbox_xywh_to_quad,
    detect_presentation_surface,
    order_quad_pts,
)


BASE_DIR = Path(__file__).resolve().parent

# Можно переопределить через переменные окружения.
DEFAULT_MODEL_PATH = Path(
    os.getenv(
        "SLIDE_BBOX_MODEL_PATH",
        str(BASE_DIR / "models" / "slide_bbox" / "saved_model"),
    )
)
DEFAULT_IMAGE_SIZE = int(os.getenv("SLIDE_BBOX_IMAGE_SIZE", "320"))
DEFAULT_BACKBONE = os.getenv("SLIDE_BBOX_BACKBONE", "MobileNetV2")
DEFAULT_DROPOUT = float(os.getenv("SLIDE_BBOX_DROPOUT", "0.2"))
DEFAULT_ROI_PAD_RATIO = float(os.getenv("SLIDE_BBOX_ROI_PAD_RATIO", "0.20"))
DEFAULT_MAX_SIDE = int(os.getenv("SLIDE_BBOX_MAX_SIDE", "1200"))
DEFAULT_MIN_LINE_LEN = int(os.getenv("SLIDE_BBOX_MIN_LINE_LEN", "80"))
DEFAULT_BATCH_SIZE = int(os.getenv("SLIDE_BBOX_BATCH_SIZE", "32"))
DEFAULT_ALLOW_FALLBACK = os.getenv("KEYPOINTS_ALLOW_FALLBACK", "1") == "1"
DEFAULT_OUTPUT_MODE = "auto"  # оставлено для совместимости со старым вызовом

DetectorMode = Literal["bbox_model", "bbox_refined_quad"]
BBOX_MODEL: DetectorMode = "bbox_model"
BBOX_REFINED_QUAD: DetectorMode = "bbox_refined_quad"


def _resolve_model_path(model_path: str | Path | None) -> Path:
    return DEFAULT_MODEL_PATH if model_path is None else Path(model_path)


@lru_cache(maxsize=4)
def _load_bbox_model_cached(
    model_path_str: str,
    image_size: int,
    backbone: str,
    dropout: float,
):
    """Загружает модель только один раз для комбинации параметров."""
    return load_trained_model(
        model_path=model_path_str,
        image_size=image_size,
        backbone=backbone,
        dropout=dropout,
        compile_for_eval=False,
    )


def _notify(callback: Callable[[str], None] | None, text: str) -> None:
    if callback is not None:
        callback(text)


def _clip_quad_to_image(quad: np.ndarray, w: int, h: int) -> np.ndarray:
    quad = np.asarray(quad, dtype=np.float32).reshape(4, 2).copy()
    quad[:, 0] = np.clip(quad[:, 0], 0, max(0, w - 1))
    quad[:, 1] = np.clip(quad[:, 1], 0, max(0, h - 1))
    return order_quad_pts(quad)


def _bbox_xyxy_from_quad(quad: np.ndarray, w: int, h: int) -> list[int]:
    quad = np.asarray(quad, dtype=np.float32).reshape(-1, 2)
    x1 = int(np.floor(np.min(quad[:, 0])))
    y1 = int(np.floor(np.min(quad[:, 1])))
    x2 = int(np.ceil(np.max(quad[:, 0])))
    y2 = int(np.ceil(np.max(quad[:, 1])))

    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(1, min(x2, w))
    y2 = max(1, min(y2, h))
    if x2 <= x1:
        x2 = min(w, x1 + 1)
    if y2 <= y1:
        y2 = min(h, y1 + 1)
    return [x1, y1, x2, y2]


def _safe_default_quad(w: int, h: int, margin_ratio: float = 0.08) -> np.ndarray:
    mx = int(round(w * margin_ratio))
    my = int(round(h * margin_ratio))
    return np.array(
        [[mx, my], [w - mx, my], [w - mx, h - my], [mx, h - my]],
        dtype=np.float32,
    )


def _fallback_result(frame_bgr: np.ndarray, exc: Exception) -> dict:
    h, w = frame_bgr.shape[:2]
    quad = _clip_quad_to_image(_safe_default_quad(w, h), w, h)
    return {
        "points": np.round(quad).astype(int).tolist(),
        "bbox": _bbox_xyxy_from_quad(quad, w, h),
        "score": 0.0,
        "method": "safe_default_fallback",
        "detector_error": str(exc),
    }


def _prepare_batch_for_bbox_model(
    frames_bgr: Sequence[np.ndarray],
    image_size: int,
) -> tf.Tensor:
    """
    Подготавливает один tensor размера [batch, image_size, image_size, 3].

    Resize выполняется до stack, поэтому функция корректно работает даже
    для изображений разных исходных размеров.
    """
    prepared: list[tf.Tensor] = []
    for frame_bgr in frames_bgr:
        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("В batch передан пустой кадр")
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = tf.convert_to_tensor(frame_rgb, dtype=tf.uint8)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, (image_size, image_size), method="bilinear")
        prepared.append(image)
    return tf.stack(prepared, axis=0)


def _predict_bbox_from_batch(
    model,
    frames_bgr: Sequence[np.ndarray],
    image_size: int,
) -> list[tuple[list[float], list[float]]]:
    """
    Выполняет ОДИН inference для всего batch.

    Возвращает для каждого кадра пару:
      (bbox_abs_xywh, bbox_rel_xywh).
    """
    if not frames_bgr:
        return []

    inputs = _prepare_batch_for_bbox_model(frames_bgr, image_size=image_size)

    # model(..., training=False) не создаёт цикл Keras predict на каждый кадр.
    predictions = model(inputs, training=False)
    if isinstance(predictions, (tuple, list)):
        predictions = predictions[0]
    if hasattr(predictions, "numpy"):
        predictions = predictions.numpy()
    predictions = np.asarray(predictions, dtype=np.float32)

    if predictions.ndim != 2 or predictions.shape[0] != len(frames_bgr):
        raise RuntimeError(
            f"Неожиданная форма предсказаний bbox-модели: {predictions.shape}; "
            f"ожидалось ({len(frames_bgr)}, 4)"
        )

    result: list[tuple[list[float], list[float]]] = []
    for frame_bgr, pred_rel in zip(frames_bgr, predictions):
        h, w = frame_bgr.shape[:2]
        pred_rel = clip_boxes_xywh_np(pred_rel)
        pred_abs = denormalize_xywh(pred_rel, w, h)
        result.append(
            ([float(v) for v in pred_abs], [float(v) for v in pred_rel])
        )
    return result


def _result_from_bbox_prediction(
    frame_bgr: np.ndarray,
    model_xywh_abs: list[float],
    model_xywh_rel: list[float],
) -> dict:
    """Результат чистой bbox-модели без уточнения quad."""
    h, w = frame_bgr.shape[:2]
    quad = _clip_quad_to_image(bbox_xywh_to_quad(model_xywh_abs), w, h)
    return {
        "points": np.round(quad).astype(int).tolist(),
        "bbox": _bbox_xyxy_from_quad(quad, w, h),
        "score": 1.0,
        "method": "bbox_model",
        "model_bbox_xywh": [float(v) for v in model_xywh_abs],
        "model_bbox_rel_xywh": [float(v) for v in model_xywh_rel],
    }


def _result_from_refined_prediction(
    frame_bgr: np.ndarray,
    model_xywh_abs: list[float],
    model_xywh_rel: list[float],
    roi_pad_ratio: float,
    max_side: int,
    min_line_len: int,
    fallback_to_full_image: bool,
) -> dict:
    """
    Уточняет quad после пакетного нейросетевого предсказания bbox.

    Само уточнение остаётся покадровым: поиск линий выполняется для ROI
    конкретного кадра, но дорогой inference модели уже сделан batch-ом.
    """
    h, w = frame_bgr.shape[:2]
    quad, info = detect_presentation_surface(
        img=frame_bgr,
        model_roi_xywh=model_xywh_abs,
        roi_pad_ratio=roi_pad_ratio,
        max_side=max_side,
        min_line_len=min_line_len,
        fallback_to_full_image=fallback_to_full_image,
    )

    if quad is None:
        quad = bbox_xywh_to_quad(model_xywh_abs)
        method = "bbox_model_fallback"
        score = 0.5
    else:
        method = "bbox_model_refined_quad"
        score = 1.0

    quad = _clip_quad_to_image(quad, w, h)
    return {
        "points": np.round(quad).astype(int).tolist(),
        "bbox": _bbox_xyxy_from_quad(quad, w, h),
        "score": float(score),
        "method": method,
        "model_bbox_xywh": [float(v) for v in model_xywh_abs],
        "model_bbox_rel_xywh": [float(v) for v in model_xywh_rel],
        "search_bbox": info.get("search_xyxy") if isinstance(info, dict) else None,
        "refine_score": (
            float(info.get("best_score"))
            if isinstance(info, dict) and "best_score" in info
            else None
        ),
    }


def detect_model_keypoints_for_frames(
    frames_bgr: Sequence[np.ndarray],
    detector_mode: DetectorMode = BBOX_REFINED_QUAD,
    model_path: str | Path | None = None,
    allow_fallback: bool = DEFAULT_ALLOW_FALLBACK,
    image_size: int = DEFAULT_IMAGE_SIZE,
    backbone: str = DEFAULT_BACKBONE,
    dropout: float = DEFAULT_DROPOUT,
    roi_pad_ratio: float = DEFAULT_ROI_PAD_RATIO,
    max_side: int = DEFAULT_MAX_SIDE,
    min_line_len: int = DEFAULT_MIN_LINE_LEN,
    fallback_to_full_image: bool = True,
    batch_size: int = DEFAULT_BATCH_SIZE,
    callback: Callable[[str], None] | None = None,
) -> list[dict]:
    """
    Пакетный детектор для массива уже считанных кадров.

    detector_mode="bbox_model": только batch inference bbox-модели.
    detector_mode="bbox_refined_quad": batch inference + покадровое line refinement.

    Используйте эту функцию в эксперименте, если кадры уже извлекаются из видео.
    """
    if detector_mode not in (BBOX_MODEL, BBOX_REFINED_QUAD):
        raise ValueError(f"Неизвестный detector_mode: {detector_mode}")
    if batch_size <= 0:
        raise ValueError("batch_size должен быть положительным")
    if not frames_bgr:
        return []

    for frame in frames_bgr:
        if frame is None or frame.size == 0:
            raise ValueError("Список кадров содержит пустой кадр")

    resolved_model_path = _resolve_model_path(model_path)
    model = _load_bbox_model_cached(
        str(resolved_model_path), int(image_size), str(backbone), float(dropout)
    )

    results: list[dict] = []
    total = len(frames_bgr)
    for start in range(0, total, batch_size):
        batch = frames_bgr[start : start + batch_size]
        end = start + len(batch)
        try:
            predictions = _predict_bbox_from_batch(model, batch, image_size=int(image_size))
        except Exception as exc:
            if not allow_fallback:
                raise RuntimeError(f"Ошибка batch inference bbox-модели: {exc}") from exc
            results.extend(_fallback_result(frame, exc) for frame in batch)
            _notify(callback, f"[bbox] batch {start}:{end} завершён fallback: {exc}")
            continue

        for frame_bgr, (model_xywh_abs, model_xywh_rel) in zip(batch, predictions):
            try:
                if detector_mode == BBOX_MODEL:
                    item = _result_from_bbox_prediction(
                        frame_bgr, model_xywh_abs, model_xywh_rel
                    )
                else:
                    item = _result_from_refined_prediction(
                        frame_bgr=frame_bgr,
                        model_xywh_abs=model_xywh_abs,
                        model_xywh_rel=model_xywh_rel,
                        roi_pad_ratio=roi_pad_ratio,
                        max_side=max_side,
                        min_line_len=min_line_len,
                        fallback_to_full_image=fallback_to_full_image,
                    )
                results.append(item)
            except Exception as exc:
                if not allow_fallback:
                    raise RuntimeError(f"Ошибка refinement детектора: {exc}") from exc
                results.append(_fallback_result(frame_bgr, exc))

        _notify(
            callback,
            f"[bbox] mode={detector_mode}, обработано {end}/{total}, batch_size={len(batch)}",
        )

    return results


def detect_model_slide_keypoints(
    frame_bgr: np.ndarray,
    model_path: str | Path | None = None,
    output_mode: str = DEFAULT_OUTPUT_MODE,
    allow_fallback: bool = DEFAULT_ALLOW_FALLBACK,
    image_size: int = DEFAULT_IMAGE_SIZE,
    backbone: str = DEFAULT_BACKBONE,
    dropout: float = DEFAULT_DROPOUT,
    roi_pad_ratio: float = DEFAULT_ROI_PAD_RATIO,
    max_side: int = DEFAULT_MAX_SIDE,
    min_line_len: int = DEFAULT_MIN_LINE_LEN,
    fallback_to_full_image: bool = True,
    detector_mode: DetectorMode = BBOX_REFINED_QUAD,
) -> dict:
    """Совместимый однокадровый интерфейс. Для скорости используйте batch-функцию."""
    _ = output_mode
    return detect_model_keypoints_for_frames(
        frames_bgr=[frame_bgr],
        detector_mode=detector_mode,
        model_path=model_path,
        allow_fallback=allow_fallback,
        image_size=image_size,
        backbone=backbone,
        dropout=dropout,
        roi_pad_ratio=roi_pad_ratio,
        max_side=max_side,
        min_line_len=min_line_len,
        fallback_to_full_image=fallback_to_full_image,
        batch_size=1,
    )[0]


def detect_model_keypoints_for_images(
    image_paths: Iterable[Path],
    model_path: str | Path | None = None,
    output_mode: str = DEFAULT_OUTPUT_MODE,
    allow_fallback: bool = DEFAULT_ALLOW_FALLBACK,
    image_size: int = DEFAULT_IMAGE_SIZE,
    backbone: str = DEFAULT_BACKBONE,
    dropout: float = DEFAULT_DROPOUT,
    roi_pad_ratio: float = DEFAULT_ROI_PAD_RATIO,
    max_side: int = DEFAULT_MAX_SIDE,
    min_line_len: int = DEFAULT_MIN_LINE_LEN,
    callback: Callable[[str], None] | None = None,
    detector_mode: DetectorMode = BBOX_REFINED_QUAD,
    batch_size: int = DEFAULT_BATCH_SIZE,
    fallback_to_full_image: bool = True,
) -> dict[str, dict]:
    """
    Совместимый интерфейс для mainAction.py, теперь с batch inference.

    Кадры читаются частями по batch_size, поэтому все изображения не нужно
    одновременно хранить в памяти.
    """
    _ = output_mode
    if batch_size <= 0:
        raise ValueError("batch_size должен быть положительным")

    paths = [Path(path) for path in image_paths]
    result: dict[str, dict] = {}
    total = len(paths)

    for start in range(0, total, batch_size):
        batch_paths = paths[start : start + batch_size]
        valid_paths: list[Path] = []
        frames: list[np.ndarray] = []

        for image_path in batch_paths:
            frame = cv2.imread(str(image_path))
            if frame is None:
                _notify(callback, f"[bbox] файл не прочитан: {image_path}")
                continue
            valid_paths.append(image_path)
            frames.append(frame)

        if not frames:
            continue

        batch_results = detect_model_keypoints_for_frames(
            frames_bgr=frames,
            detector_mode=detector_mode,
            model_path=model_path,
            allow_fallback=allow_fallback,
            image_size=image_size,
            backbone=backbone,
            dropout=dropout,
            roi_pad_ratio=roi_pad_ratio,
            max_side=max_side,
            min_line_len=min_line_len,
            fallback_to_full_image=fallback_to_full_image,
            batch_size=len(frames),
            callback=None,
        )
        result.update({str(path): item for path, item in zip(valid_paths, batch_results)})
        _notify(
            callback,
            f"[bbox] mode={detector_mode}, обработано {min(start + batch_size, total)}/{total}",
        )

    return result
