from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Any

import cv2
import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

try:
    import tensorflow as tf
    from tensorflow import keras
except Exception as exc:
    tf = None
    keras = None
    _TF_IMPORT_ERROR = exc
else:
    _TF_IMPORT_ERROR = None


BASE_DIR = Path(__file__).resolve().parent

# Путь к модели можно задать через переменную окружения KEYPOINTS_MODEL_PATH
# либо оставить файл по умолчанию в .\models\slide_keypoints.keras
DEFAULT_MODEL_PATH = Path(
    os.getenv(
        "KEYPOINTS_MODEL_PATH",
        str(BASE_DIR / "models" / "slide_keypoints.keras")
    )
)

# Режим интерпретации выхода модели:
# auto        - определить автоматически
# bbox_norm   - [x, y, w, h] в нормированных координатах [0..1]
# bbox_px     - [x, y, w, h] в пикселях относительно входа модели
# points_norm - [x1, y1, x2, y2, x3, y3, x4, y4] в [0..1]
# points_px   - те же 8 значений, но в пикселях относительно входа модели
DEFAULT_OUTPUT_MODE = os.getenv("KEYPOINTS_OUTPUT_MODE", "auto").strip().lower()

# Если модели нет или она упала на инференсе, использовать Canny fallback
DEFAULT_ALLOW_FALLBACK = os.getenv("KEYPOINTS_ALLOW_FALLBACK", "1") == "1"


def _safe_default_points(w: int, h: int, margin_ratio: float = 0.08) -> np.ndarray:
    mx = int(w * margin_ratio)
    my = int(h * margin_ratio)
    return np.array(
        [
            [mx, my],
            [w - mx, my],
            [w - mx, h - my],
            [mx, h - my],
        ],
        dtype=np.float32,
    )


def _clip_points(points: np.ndarray, w: int, h: int) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2).copy()
    pts[:, 0] = np.clip(pts[:, 0], 0, max(0, w - 1))
    pts[:, 1] = np.clip(pts[:, 1], 0, max(0, h - 1))
    return pts


def _order_points(points: np.ndarray) -> np.ndarray:
    """
    Упорядочивание: top-left, top-right, bottom-right, bottom-left.
    """
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    center = pts.mean(axis=0)

    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    pts = pts[np.argsort(angles)]

    sums = pts.sum(axis=1)
    start_idx = int(np.argmin(sums))
    pts = np.roll(pts, -start_idx, axis=0)

    # Проверка ориентации: если пошли против ожидаемого обхода — переворачиваем
    tl, p1, p2, p3 = pts
    cross = (p1[0] - tl[0]) * (p3[1] - tl[1]) - (p1[1] - tl[1]) * (p3[0] - tl[0])
    if cross < 0:
        pts = np.array([pts[0], pts[3], pts[2], pts[1]], dtype=np.float32)

    return pts


def _bbox_from_points(points: np.ndarray) -> list[int]:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    x1 = int(np.floor(np.min(pts[:, 0])))
    y1 = int(np.floor(np.min(pts[:, 1])))
    x2 = int(np.ceil(np.max(pts[:, 0])))
    y2 = int(np.ceil(np.max(pts[:, 1])))
    return [x1, y1, x2, y2]


def _canny_fallback(frame_bgr: np.ndarray) -> dict:
    h, w = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 140)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_area = 0.0

    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) != 4:
            continue
        area = cv2.contourArea(approx)
        if area < 0.1 * w * h:
            continue
        if area > best_area:
            best_area = area
            best = approx.reshape(4, 2).astype(np.float32)

    if best is None:
        best = _safe_default_points(w, h)
    else:
        best = _order_points(best)

    best = _clip_points(best, w, h)
    return {
        "points": best.astype(int).tolist(),
        "bbox": _bbox_from_points(best),
        "score": 0.0,
        "method": "canny_fallback",
    }


@lru_cache(maxsize=4)
def _load_model(model_path_str: str):
    if keras is None:
        raise ImportError(
            "TensorFlow/Keras не импортирован. "
            f"Исходная ошибка: {_TF_IMPORT_ERROR}"
        )

    model_path = Path(model_path_str)
    if not model_path.exists():
        raise FileNotFoundError(f"Модель не найдена: {model_path}")

    model = keras.models.load_model(str(model_path), compile=False)
    return model


def _resolve_model_path(model_path: str | Path | None) -> Path:
    if model_path is None:
        return DEFAULT_MODEL_PATH
    return Path(model_path)


def _get_model_input_spec(model) -> tuple[int, int, int]:
    """
    Возвращает (height, width, channels) для модели channels_last.
    """
    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    if input_shape is None or len(input_shape) != 4:
        raise ValueError(f"Неожиданный input_shape модели: {input_shape}")

    _, h, w, c = input_shape
    h = int(h) if h is not None else 224
    w = int(w) if w is not None else 224
    c = int(c) if c is not None else 3
    return h, w, c


def _prepare_image(frame_bgr: np.ndarray, target_h: int, target_w: int, channels: int) -> np.ndarray:
    if channels == 1:
        img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=-1)
        return img

    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    return img


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


def _extract_prediction_and_score(raw_pred: Any) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Поддерживает разные варианты выхода модели:
    - np.ndarray формы (B, 4) или (B, 8)
    - list/tuple из нескольких выходов
    - dict с ключами вроде points/bbox/score/confidence
    """
    if isinstance(raw_pred, dict):
        pred = None
        score = None

        for key in ("points", "keypoints", "coords", "coordinates", "bbox", "boxes", "output_1"):
            if key in raw_pred:
                pred = _to_numpy(raw_pred[key])
                break
        if pred is None:
            pred = _to_numpy(next(iter(raw_pred.values())))

        for key in ("score", "scores", "confidence", "conf", "output_2"):
            if key in raw_pred:
                score = _to_numpy(raw_pred[key])
                break

        return pred, score

    if isinstance(raw_pred, (list, tuple)):
        arrays = [_to_numpy(x) for x in raw_pred]
        pred = None
        score = None

        for arr in arrays:
            if arr.ndim >= 2 and arr.shape[-1] in (4, 8):
                pred = arr
                break

        if pred is None:
            pred = arrays[0]

        for arr in arrays:
            if arr is pred:
                continue
            if arr.ndim >= 1 and (arr.shape[-1] == 1 or arr.ndim == 1):
                score = arr
                break

        return pred, score

    pred = _to_numpy(raw_pred)
    return pred, None


def _sigmoid_if_needed(values: np.ndarray) -> np.ndarray:
    """
    Иногда линейный выход модели обучали на [0..1], но на инференсе он слегка выходит за пределы.
    Если значения небольшие по модулю, аккуратно переводим через sigmoid.
    """
    v = values.astype(np.float32)
    if np.min(v) >= 0.0 and np.max(v) <= 1.0:
        return v

    if np.max(np.abs(v)) <= 8.0 and (np.min(v) < 0.0 or np.max(v) > 1.0):
        return 1.0 / (1.0 + np.exp(-v))

    return v


def _decode_prediction(
    pred_row: np.ndarray,
    frame_w: int,
    frame_h: int,
    model_w: int,
    model_h: int,
    output_mode: str = "auto",
) -> np.ndarray:
    """
    Возвращает 4 точки слайда.
    """
    values = np.asarray(pred_row, dtype=np.float32).reshape(-1)

    if values.size not in (4, 8):
        raise ValueError(
            f"Ожидалось 4 или 8 значений на выходе модели, получено: {values.size}"
        )

    mode = (output_mode or "auto").strip().lower()

    if mode == "auto":
        if values.size == 8:
            v = _sigmoid_if_needed(values.copy())
            if np.min(v) >= -0.1 and np.max(v) <= 1.1:
                mode = "points_norm"
                values = v
            else:
                mode = "points_px"
        else:
            v = _sigmoid_if_needed(values.copy())
            if np.min(v) >= -0.1 and np.max(v) <= 1.1:
                mode = "bbox_norm"
                values = v
            else:
                mode = "bbox_px"

    if mode == "points_norm":
        pts = values.reshape(4, 2).copy()
        pts[:, 0] *= frame_w
        pts[:, 1] *= frame_h
        return pts

    if mode == "points_px":
        pts = values.reshape(4, 2).copy()
        pts[:, 0] *= frame_w / float(model_w)
        pts[:, 1] *= frame_h / float(model_h)
        return pts

    if mode == "bbox_norm":
        x, y, w, h = values.tolist()
        x1 = x * frame_w
        y1 = y * frame_h
        x2 = (x + w) * frame_w
        y2 = (y + h) * frame_h
        return np.array(
            [
                [x1, y1],
                [x2, y1],
                [x2, y2],
                [x1, y2],
            ],
            dtype=np.float32,
        )

    if mode == "bbox_px":
        x, y, w, h = values.tolist()
        sx = frame_w / float(model_w)
        sy = frame_h / float(model_h)
        x1 = x * sx
        y1 = y * sy
        x2 = (x + w) * sx
        y2 = (y + h) * sy
        return np.array(
            [
                [x1, y1],
                [x2, y1],
                [x2, y2],
                [x1, y2],
            ],
            dtype=np.float32,
        )

    raise ValueError(f"Неизвестный output_mode: {output_mode}")


def detect_slide_keypoints(
    frame_bgr: np.ndarray,
    model_path: str | Path | None = None,
    output_mode: str = DEFAULT_OUTPUT_MODE,
    allow_fallback: bool = DEFAULT_ALLOW_FALLBACK,
) -> dict:
    """
    Возвращает:
    {
        "points": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
        "bbox": [x1, y1, x2, y2],
        "score": float,
        "method": "keras_model" | "canny_fallback"
    }
    """
    if frame_bgr is None or frame_bgr.size == 0:
        raise ValueError("Пустой кадр передан в detect_slide_keypoints")

    frame_h, frame_w = frame_bgr.shape[:2]

    try:
        model_path = _resolve_model_path(model_path)
        model = _load_model(str(model_path))
        model_h, model_w, channels = _get_model_input_spec(model)

        inp = _prepare_image(frame_bgr, model_h, model_w, channels)
        inp = np.expand_dims(inp, axis=0)

        raw_pred = model.predict(inp, verbose=0)
        pred_arr, score_arr = _extract_prediction_and_score(raw_pred)

        pred_arr = np.asarray(pred_arr)
        if pred_arr.ndim == 1:
            pred_row = pred_arr
        else:
            pred_row = pred_arr[0]

        points = _decode_prediction(
            pred_row=pred_row,
            frame_w=frame_w,
            frame_h=frame_h,
            model_w=model_w,
            model_h=model_h,
            output_mode=output_mode,
        )

        points = _clip_points(points, frame_w, frame_h)
        points = _order_points(points)
        bbox = _bbox_from_points(points)

        score = 1.0
        if score_arr is not None:
            score_np = np.asarray(score_arr).reshape(-1)
            if score_np.size > 0:
                score = float(score_np[0])

        return {
            "points": points.astype(int).tolist(),
            "bbox": bbox,
            "score": score,
            "method": "keras_model",
        }

    except Exception as exc:
        if not allow_fallback:
            raise RuntimeError(f"Ошибка детекции ключевых точек Keras-моделью: {exc}") from exc

        result = _canny_fallback(frame_bgr)
        result["keras_error"] = str(exc)
        return result


def detect_keypoints_for_images(
    image_paths: Iterable[Path],
    model_path: str | Path | None = None,
    output_mode: str = DEFAULT_OUTPUT_MODE,
    allow_fallback: bool = DEFAULT_ALLOW_FALLBACK,
) -> dict[str, dict]:
    """
    Совместим с текущим mainAction.py:
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

        result[str(image_path)] = detect_slide_keypoints(
            frame_bgr=frame,
            model_path=model_path,
            output_mode=output_mode,
            allow_fallback=allow_fallback,
        )

    return result