from __future__ import annotations

import csv
import json
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np
import tensorflow as tf

from slide_bbox_common import build_dataset, build_model, load_coco_single_box_records, set_seed


def summarize_box_areas(records: Sequence[dict], split_name: str) -> Dict[str, float]:
    areas = [float(r["bbox"][2] * r["bbox"][3]) for r in records]
    stats = {
        "num_images": len(records),
        "bbox_area_min": float(np.min(areas)),
        "bbox_area_mean": float(np.mean(areas)),
        "bbox_area_max": float(np.max(areas)),
    }
    print(f"{split_name}: {stats['num_images']} images")
    print(
        f"{split_name} bbox area stats (normalized): "
        f"min={stats['bbox_area_min']:.4f}, "
        f"mean={stats['bbox_area_mean']:.4f}, "
        f"max={stats['bbox_area_max']:.4f}"
    )
    return stats


def save_training_artifacts(
    history: tf.keras.callbacks.History,
    output_dir: str,
    train_stats: Dict[str, float],
    val_stats: Dict[str, float],
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    history_csv = os.path.join(output_dir, "history.csv")
    history_dict = history.history
    with open(history_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        keys = list(history_dict.keys())
        writer.writerow(keys)
        num_rows = len(history_dict[keys[0]]) if keys else 0
        for i in range(num_rows):
            writer.writerow([history_dict[k][i] for k in keys])

    summary = {
        "train": train_stats,
        "valid": val_stats,
        "best_val_loss": float(np.min(history_dict.get("val_loss", [np.nan]))),
        "best_val_mean_iou": float(np.max(history_dict.get("val_mean_iou", [np.nan]))),
        "final_train_loss": float(history_dict.get("loss", [np.nan])[-1]),
        "final_val_loss": float(history_dict.get("val_loss", [np.nan])[-1]),
    }
    with open(os.path.join(output_dir, "training_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)



def run_training(
    train_dir: str,
    val_dir: str,
    output_dir: str,
    image_size: int = 320,
    batch_size: int = 16,
    epochs: int = 30,
    learning_rate: float = 1e-3,
    backbone: str = "MobileNetV2",
    dropout: float = 0.2,
    category_ids: Sequence[int] = (1,),
    seed: int = 42,
    pretrained: bool = True,
    train_backbone: bool = True,
) -> Tuple[tf.keras.Model, tf.keras.callbacks.History, Dict[str, Dict[str, float]]]:
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    train_records, _ = load_coco_single_box_records(train_dir, category_ids=category_ids)
    val_records, _ = load_coco_single_box_records(val_dir, category_ids=category_ids)

    train_stats = summarize_box_areas(train_records, "Train")
    val_stats = summarize_box_areas(val_records, "Valid")

    train_ds = build_dataset(train_records, image_size=image_size, batch_size=batch_size, training=True)
    val_ds = build_dataset(val_records, image_size=image_size, batch_size=batch_size, training=False)

    model = build_model(
        image_size=image_size,
        backbone=backbone,
        learning_rate=learning_rate,
        dropout=dropout,
        pretrained=pretrained,
        train_backbone=train_backbone,
        compile_model=True,
    )

    callbacks: List[tf.keras.callbacks.Callback] = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(output_dir, "best.weights.h5"),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(output_dir, "last.weights.h5"),
            save_best_only=False,
            save_weights_only=True,
            verbose=0,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(os.path.join(output_dir, "fit_log.csv"), append=False),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    saved_model_dir = os.path.join(output_dir, "saved_model")
    model.save(saved_model_dir)
    save_training_artifacts(history, output_dir, train_stats, val_stats)

    stats = {"train": train_stats, "valid": val_stats}
    return model, history, stats

