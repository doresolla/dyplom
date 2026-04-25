from __future__ import annotations

import csv
import json
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

from slide_bbox_common import (
    build_dataset,
    clip_boxes_xywh_np,
    denormalize_xywh,
    iou_xywh_np,
    load_coco_single_box_records,
    load_trained_model,
)


def run_evaluation(
    test_dir: str,
    model_path: str,
    image_size: int = 320,
    batch_size: int = 16,
    backbone: str = "MobileNetV2",
    dropout: float = 0.2,
    category_ids: Sequence[int] = (1,),
    output_dir: str = "./eval_output",
) -> Tuple[Dict[str, float], List[dict]]:
    os.makedirs(output_dir, exist_ok=True)

    records, _ = load_coco_single_box_records(test_dir, category_ids=category_ids)
    ds = build_dataset(records, image_size=image_size, batch_size=batch_size, training=False)

    model = load_trained_model(
        model_path=model_path,
        image_size=image_size,
        backbone=backbone,
        dropout=dropout,
        compile_for_eval=True,
    )

    metrics = model.evaluate(ds, return_dict=True, verbose=1)
    preds = model.predict(ds, verbose=1)
    preds = clip_boxes_xywh_np(preds)

    per_image_rows: List[dict] = []
    ious: List[float] = []
    mae_xywh: List[float] = []

    for rec, pred in zip(records, preds):
        gt = np.asarray(rec["bbox"], dtype=np.float32)
        pred = np.asarray(pred, dtype=np.float32)
        iou = iou_xywh_np(gt, pred)
        ious.append(iou)
        mae_xywh.append(float(np.mean(np.abs(gt - pred))))

        gt_abs = denormalize_xywh(gt, rec["width"], rec["height"])
        pred_abs = denormalize_xywh(pred, rec["width"], rec["height"])
        per_image_rows.append(
            {
                "image_id": rec["image_id"],
                "file_name": rec["file_name"],
                "image_path": rec["image_path"],
                "image_width": rec["width"],
                "image_height": rec["height"],
                "iou": float(iou),
                "gt_x": float(gt_abs[0]),
                "gt_y": float(gt_abs[1]),
                "gt_w": float(gt_abs[2]),
                "gt_h": float(gt_abs[3]),
                "pred_x": float(pred_abs[0]),
                "pred_y": float(pred_abs[1]),
                "pred_w": float(pred_abs[2]),
                "pred_h": float(pred_abs[3]),
            }
        )

    thresholds = [0.5, 0.75, 0.9]
    summary = {
        "loss": float(metrics.get("loss", 0.0)),
        "mean_iou_metric": float(metrics.get("mean_iou", 0.0)),
        "mae_metric": float(metrics.get("mae", 0.0)),
        "num_images": len(records),
        "mean_iou_manual": float(np.mean(ious)),
        "median_iou_manual": float(np.median(ious)),
        "mean_abs_error_xywh": float(np.mean(mae_xywh)),
    }
    for t in thresholds:
        summary[f"acc_iou_ge_{str(t).replace('.', '_')}"] = float(np.mean(np.array(ious) >= t))

    with open(os.path.join(output_dir, "evaluation_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    csv_path = os.path.join(output_dir, "per_image_predictions.csv")
    if per_image_rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(per_image_rows[0].keys()))
            writer.writeheader()
            writer.writerows(per_image_rows)

    print("Evaluation summary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved summary to: {os.path.join(output_dir, 'evaluation_summary.json')}")
    print(f"Saved per-image predictions to: {csv_path}")

    return summary, per_image_rows


