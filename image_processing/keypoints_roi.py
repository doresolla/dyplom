from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Tuple

import cv2
import numpy as np
import json

def _fallback_quad(w: int, h: int) -> np.ndarray:
    margin_x = int(w * 0.08)
    margin_y = int(h * 0.08)
    return np.array(
        [
            [margin_x, margin_y],
            [w - margin_x, margin_y],
            [w - margin_x, h - margin_y],
            [margin_x, h - margin_y],
        ],
        dtype=np.int32,
    )


def _detect_quad_canny(frame_bgr: np.ndarray, min_area_ratio: float = 0.10) -> Optional[np.ndarray]:
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
        if area < min_area_ratio * w * h:
            continue

        if area > best_area:
            best_area = area
            best = approx.reshape(4, 2)

    return best


def _bbox_to_quad(bbox: list[int] | tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = map(int, bbox)
    return np.array(
        [
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2],
        ],
        dtype=np.int32,
    )


def _clip_bbox(
    bbox: list[int] | tuple[int, int, int, int] | None,
    w: int,
    h: int,
) -> Optional[list[int]]:
    if bbox is None:
        return None

    x1, y1, x2, y2 = map(int, bbox)
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(1, min(w, x2))
    y2 = max(1, min(h, y2))

    if x2 <= x1 + 5 or y2 <= y1 + 5:
        return None

    return [x1, y1, x2, y2]


def _refine_quad_in_bbox(frame_bgr: np.ndarray, bbox: list[int], pad: int = 8) -> Optional[np.ndarray]:
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = bbox

    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)

    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    quad = _detect_quad_canny(crop, min_area_ratio=0.35)
    if quad is None:
        return None

    quad[:, 0] += x1
    quad[:, 1] += y1
    return quad


def detect_slide_keypoints(
    frame_bgr: np.ndarray,
) -> dict:
    h, w = frame_bgr.shape[:2]
    best = None
    source = "canny"

    if best is None:
        best = _detect_quad_canny(frame_bgr)
        if best is None:
            best = _fallback_quad(w, h)
            source = "fallback"

    x, y, bw, bh = cv2.boundingRect(best.astype(np.int32))
    return {
        "points": best.astype(int).tolist(),
        "bbox": [int(x), int(y), int(x + bw), int(y + bh)],
        "source": source,
    }


def detect_keypoints_for_images(
    image_paths: Iterable[Path],
    run_dir: Path
) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for image_path in image_paths:
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue
        result[str(image_path)] = detect_slide_keypoints(frame)
    keypoints_path = run_dir / "keypoints.json"
    keypoints_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result