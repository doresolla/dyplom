from __future__ import annotations

from refine_bbox import detect_presentation_surface

import csv
import math
import os
import random
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image, ImageDraw

from slide_bbox_common import denormalize_xywh, load_trained_model, predict_single_image

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(input_path: str) -> List[str]:
    p = Path(input_path)
    if p.is_file():
        return [str(p)]
    files = [str(x) for x in sorted(p.iterdir()) if x.is_file() and x.suffix.lower() in VALID_EXTS]
    if not files:
        raise RuntimeError(f"No images found in: {input_path}")
    return files


def draw_bbox(image_path: str, pred_abs_xywh: np.ndarray, save_path: str) -> None:
    img = Image.open(image_path).convert("RGB")
    img = draw_bbox_on_image(img, pred_abs_xywh)
    img.save(save_path)


def draw_bbox_on_image(
        image: Image.Image,
        pred_abs_xywh: np.ndarray,
        label: str = "pred",
        color=(255, 0, 0),
        width: int = 3,
) -> Image.Image:
    img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    x, y, w, h = [float(v) for v in pred_abs_xywh.tolist()]
    x2 = x + w
    y2 = y + h
    draw.rectangle([x, y, x2, y2], outline=color, width=width)
    draw.text((x + 4, max(0, y - 14)), label, fill=color)
    return img


def run_prediction(
        input_path: str,
        model_path: str,
        image_size: int = 320,
        backbone: str = "MobileNetV2",
        dropout: float = 0.2,
        output_dir: str = "./predict_output",
        save_vis: bool = False,
) -> List[dict]:
    os.makedirs(output_dir, exist_ok=True)
    vis_dir = os.path.join(output_dir, "visualizations")
    if save_vis:
        os.makedirs(vis_dir, exist_ok=True)

    model = load_trained_model(
        model_path=model_path,
        image_size=image_size,
        backbone=backbone,
        dropout=dropout,
        compile_for_eval=False,
    )

    image_paths = collect_images(input_path)
    rows: List[dict] = []

    for image_path in image_paths:
        pred_rel = predict_single_image(model, image_path=image_path, image_size=image_size)

        with Image.open(image_path) as img:
            width, height = img.size
        pred_abs = denormalize_xywh(pred_rel, width, height)

        quad, info = detect_presentation_surface(
            img,
            max_side=1200,
            min_line_len=80,
            model_roi_xywh=pred_abs,
            roi_pad_ratio=0.20,
            fallback_to_full_image=False,
        )
        if quad is None:
            x, y, w, h = pred_abs
            quad = np.array(
                [
                    [x, y],
                    [x + w, y],
                    [x + w, y + h],
                    [x, y + h],
                ],
                dtype=np.float32,
            )

        row = {
            "file_name": os.path.basename(image_path),
            "image_path": image_path,
            "image_width": width,
            "image_height": height,
            "pred_x": float(pred_abs[0]),
            "pred_y": float(pred_abs[1]),
            "pred_w": float(pred_abs[2]),
            "pred_h": float(pred_abs[3]),
            "pred_x_rel": float(pred_rel[0]),
            "pred_y_rel": float(pred_rel[1]),
            "pred_w_rel": float(pred_rel[2]),
            "pred_h_rel": float(pred_rel[3]),
        }
        rows.append(row)

        if save_vis:
            save_path = os.path.join(vis_dir, os.path.basename(image_path))
            draw_bbox(image_path, pred_abs, save_path)

    csv_path = os.path.join(output_dir, "predictions.csv")
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print(f"Predicted {len(rows)} images")
    print(f"Saved predictions to: {csv_path}")
    if save_vis:
        print(f"Saved visualizations to: {vis_dir}")

    return rows

