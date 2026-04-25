from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable, Optional

import cv2
import numpy as np
import tensorflow as tf

from utils import _emit

from .detect_slide.slide_bbox_common import (
    load_trained_model,
    denormalize_xywh,
    clip_boxes_xywh_np,
)

from .detect_slide.refine_bbox import (
    detect_presentation_surface,
    bbox_xywh_to_quad,
    order_quad_pts,
)


BASE_DIR = Path(__file__).resolve().parent

# Можно переопределить через переменную окружения:
# set SLIDE_BBOX_MODEL_PATH=...
DEFAULT_MODEL_PATH = Path(
    os.getenv(
        "SLIDE_BBOX_MODEL_PATH",
        str(BASE_DIR / "models" / "slide_bbox" / "saved_model")
    )
)

DEFAULT_IMAGE_SIZE = int(os.getenv("SLIDE_BBOX_IMAGE_SIZE", "320"))
DEFAULT_BACKBONE = os.getenv("SLIDE_BBOX_BACKBONE", "MobileNetV2")
DEFAULT_DROPOUT = float(os.getenv("SLIDE_BBOX_DROPOUT", "0.2"))

DEFAULT_ROI_PAD_RATIO = float(os.getenv("SLIDE_BBOX_ROI_PAD_RATIO", "0.20"))
DEFAULT_MAX_SIDE = int(os.getenv("SLIDE_BBOX_MAX_SIDE", "1200"))
DEFAULT_MIN_LINE_LEN = int(os.getenv("SLIDE_BBOX_MIN_LINE_LEN", "80"))

DEFAULT_ALLOW_FALLBACK = os.getenv("KEYPOINTS_ALLOW_FALLBACK", "1") == "1"
DEFAULT_OUTPUT_MODE = "auto"  # оставлено только для совместимости со старым вызовом


def _resolve_model_path(model_path: str | Path | None) -> Path:
    if model_path is None:
        return DEFAULT_MODEL_PATH
    return Path(model_path)


@lru_cache(maxsize=4)
def _load_bbox_model_cached(
    model_path_str: str,
    image_size: int,
    backbone: str,
    dropout: float,
):
    return load_trained_model(
        model_path=model_path_str,
        image_size=image_size,
        backbone=backbone,
        dropout=dropout,
        compile_for_eval=False,
    )


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
        [
            [mx, my],
            [w - mx, my],
            [w - mx, h - my],
            [mx, h - my],
        ],
        dtype=np.float32,
    )


def _prepare_frame_for_bbox_model(frame_bgr: np.ndarray, image_size: int) -> tf.Tensor:
    """
    Препроцессинг повторяет смысл predict_single_image:
    RGB, float32 [0..1], resize до image_size.
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    image = tf.convert_to_tensor(frame_rgb, dtype=tf.uint8)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, (image_size, image_size), method="bilinear")

    return tf.expand_dims(image, axis=0)


def _predict_bbox_from_frame(
    model,
    frame_bgr: np.ndarray,
    image_size: int,
) -> tuple[list[float], list[float]]:
    """
    Возвращает:
    - model_xywh_abs: [x, y, w, h] в пикселях исходного кадра
    - model_xywh_rel: [x, y, w, h] в нормированных координатах
    """
    h, w = frame_bgr.shape[:2]

    inp = _prepare_frame_for_bbox_model(frame_bgr, image_size=image_size)
    pred_rel = model.predict(inp, verbose=0)[0]
    pred_rel = clip_boxes_xywh_np(pred_rel)

    pred_abs = denormalize_xywh(pred_rel, w, h)

    return [float(v) for v in pred_abs], [float(v) for v in pred_rel]


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
) -> dict:
    """
    Совместимый интерфейс со старым keyPoints.py.

    Возвращает:
    {
        "points": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
        "bbox": [x1, y1, x2, y2],
        "score": float,
        "method": str
    }

    Внутри:
    1) bbox-модель определяет грубый bbox слайда;
    2) detect_presentation_surface уточняет четырехугольник;
    3) если уточнить не удалось, используется bbox модели как quad.
    """
    _ = output_mode  # параметр оставлен для совместимости со старым вызовом

    if frame_bgr is None or frame_bgr.size == 0:
        raise ValueError("Пустой кадр передан в detect_slide_keypoints")

    frame_h, frame_w = frame_bgr.shape[:2]

    try:
        resolved_model_path = _resolve_model_path(model_path)

        model = _load_bbox_model_cached(
            str(resolved_model_path),
            int(image_size),
            str(backbone),
            float(dropout),
        )

        model_xywh_abs, model_xywh_rel = _predict_bbox_from_frame(
            model=model,
            frame_bgr=frame_bgr,
            image_size=int(image_size),
        )

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

        quad = _clip_quad_to_image(quad, frame_w, frame_h)
        bbox = _bbox_xyxy_from_quad(quad, frame_w, frame_h)

        result = {
            "points": np.round(quad).astype(int).tolist(),
            "bbox": bbox,
            "score": float(score),
            "method": method,

            # Дополнительные поля не ломают старый код,
            # но полезны для отладки.
            "model_bbox_xywh": [float(v) for v in model_xywh_abs],
            "model_bbox_rel_xywh": [float(v) for v in model_xywh_rel],
            "search_bbox": info.get("search_xyxy") if isinstance(info, dict) else None,
            "refine_score": float(info.get("best_score")) if isinstance(info, dict) and "best_score" in info else None,
        }

        return result

    except Exception as exc:
        if not allow_fallback:
            raise RuntimeError(f"Ошибка bbox/quad детектора: {exc}") from exc

        quad = _safe_default_quad(frame_w, frame_h)
        quad = _clip_quad_to_image(quad, frame_w, frame_h)
        bbox = _bbox_xyxy_from_quad(quad, frame_w, frame_h)

        return {
            "points": np.round(quad).astype(int).tolist(),
            "bbox": bbox,
            "score": 0.0,
            "method": "safe_default_fallback",
            "detector_error": str(exc),
        }


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
    callback: Optional[Callable[[str], None]] = None,
) -> dict[str, dict]:
    """
    Совместимо с текущим mainAction.py:
    принимает список путей к кадрам и возвращает словарь:
    {
        "path/to/image.jpg": {
            "points": ...,
            "bbox": ...,
            "score": ...,
            "method": ...
        }
    }
    """
    result: dict[str, dict] = {}

    for image_path in image_paths:
        image_path = Path(image_path)
        frame = cv2.imread(str(image_path))

        if frame is None:
            continue

        result[str(image_path)] = detect_model_slide_keypoints(
            frame_bgr=frame,
            model_path=model_path,
            output_mode=output_mode,
            allow_fallback=allow_fallback,
            image_size=image_size,
            backbone=backbone,
            dropout=dropout,
            roi_pad_ratio=roi_pad_ratio,
            max_side=max_side,
            min_line_len=min_line_len,
        )

    return result


