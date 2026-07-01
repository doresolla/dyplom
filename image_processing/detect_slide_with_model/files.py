from pathlib import Path
from typing import  List, Dict
from quad import  quad_to_xyxy

from refine_bbox import xyxy_to_xywh, order_quad_pts

import numpy as np
import os
import csv

def collect_videos(input_path: str) -> List[str]:
    p = Path(input_path)
    if p.is_file():
        return [str(p)]

    exts = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}
    result = []
    for f in sorted(p.rglob("*")):
        if f.is_file() and f.suffix.lower() in exts:
            result.append(str(f))
    return result


def ensure_parent(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def make_frame_name(video_path: str, frame_idx: int, timestamp_sec: float) -> str:
    stem = Path(video_path).stem
    ts_ms = int(round(timestamp_sec * 1000.0))
    return f"{stem}_f{frame_idx:08d}_t{ts_ms:012d}.jpg"


def load_existing_keys(csv_path: str) -> set:
    if not os.path.exists(csv_path):
        return set()

    keys = set()
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["video_path"], int(row["frame_idx"]))
            keys.add(key)
    return keys




def build_annotation_row(
    image_save_path: str,
    video_path: str,
    frame_idx: int,
    timestamp_sec: float,
    frame_bgr: np.ndarray,
    source_mode: str,
    model_xywh,
    search_xyxy,
    quad: np.ndarray,
) -> Dict:
    h, w = frame_bgr.shape[:2]
    quad = order_quad_pts(quad)
    bbox_xyxy = quad_to_xyxy(quad, w, h)
    bbox_xywh = xyxy_to_xywh(bbox_xyxy)

    row = {
        "file_name": os.path.basename(image_save_path),
        "image_path": image_save_path,
        "video_path": video_path,
        "video_name": Path(video_path).name,
        "frame_idx": int(frame_idx),
        "timestamp_sec": float(timestamp_sec),
        "image_width": int(w),
        "image_height": int(h),
        "source_mode": source_mode,

        "model_x": float(model_xywh[0]),
        "model_y": float(model_xywh[1]),
        "model_w": float(model_xywh[2]),
        "model_h": float(model_xywh[3]),

        "search_x1": int(search_xyxy[0]),
        "search_y1": int(search_xyxy[1]),
        "search_x2": int(search_xyxy[2]),
        "search_y2": int(search_xyxy[3]),

        "bbox_x": int(bbox_xywh[0]),
        "bbox_y": int(bbox_xywh[1]),
        "bbox_w": int(bbox_xywh[2]),
        "bbox_h": int(bbox_xywh[3]),

        "quad_x1": float(quad[0, 0]),
        "quad_y1": float(quad[0, 1]),
        "quad_x2": float(quad[1, 0]),
        "quad_y2": float(quad[1, 1]),
        "quad_x3": float(quad[2, 0]),
        "quad_y3": float(quad[2, 1]),
        "quad_x4": float(quad[3, 0]),
        "quad_y4": float(quad[3, 1]),
    }
    return row

