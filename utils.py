import cv2
from pathlib import Path
import numpy as np
from functools import lru_cache

from typing import Optional, Callable


def _emit(callback: Optional[Callable[[str], None]], text: str) -> None:
    if callback is not None:
        callback(text)

def crop_frames_by_keypoints(frame_paths,
                             keypoints_map, out_dir: Path, 
                             callback: Optional[Callable[[str], None]] = None
                             ) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cropped_paths = []

    for frame_path in frame_paths:
        frame = cv2.imread(str(frame_path))
        if frame is None:
            _emit(callback, f"[crop] не удалось открыть {frame_path}")
            continue

        meta = keypoints_map.get(str(frame_path))
        if not meta:
            _emit(callback, f"[crop] нет meta для {frame_path.name}")
            continue

        crop = None

        # Новый основной путь: перспективная обрезка по уточненному четырехугольнику.
        if "points" in meta and meta["points"]:
            try:
                crop = _warp_frame_by_quad(frame, meta["points"])
            except Exception as exc:
                _emit(callback, 
                      f"[crop] ошибка warp по quad для {frame_path.name}: {exc}"
                )
                crop = None

        # Старый fallback: обычная обрезка по bbox.
        if crop is None:
            if "bbox" not in meta:
                _emit(callback, f"[crop] нет bbox для {frame_path.name}")
                continue

            h, w = frame.shape[:2]
            x1, y1, x2, y2 = meta["bbox"]

            x1 = max(0, min(int(x1), w - 1))
            y1 = max(0, min(int(y1), h - 1))
            x2 = max(1, min(int(x2), w))
            y2 = max(1, min(int(y2), h))

            if x2 <= x1 or y2 <= y1:
                _emit(callback,
                    f"[crop] некорректный bbox для {frame_path.name}: {meta['bbox']}"
                )
                continue

            crop = frame[y1:y2, x1:x2]

        if crop is None or crop.size == 0:
            _emit(callback, f"[crop] пустой crop для {frame_path.name}")
            continue

        out_path = out_dir / frame_path.name
        cv2.imwrite(str(out_path), crop)
        cropped_paths.append(out_path)

    _emit(callback,f"[crop] обрезано кадров: {len(cropped_paths)}")
    return cropped_paths


def _order_quad_for_warp(points):
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    center = pts.mean(axis=0)

    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    pts = pts[np.argsort(angles)]

    start_idx = int(np.argmin(pts.sum(axis=1)))
    pts = np.roll(pts, -start_idx, axis=0)

    return pts



def _warp_frame_by_quad( frame, points):
    src = _order_quad_for_warp(points)

    tl, tr, br, bl = src

    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)

    out_w = int(round(max(width_top, width_bottom)))
    out_h = int(round(max(height_left, height_right)))

    if out_w < 10 or out_h < 10:
        return None

    dst = np.array(
        [
            [0, 0],
            [out_w - 1, 0],
            [out_w - 1, out_h - 1],
            [0, out_h - 1],
        ],
        dtype=np.float32,
    )

    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(frame, M, (out_w, out_h))


