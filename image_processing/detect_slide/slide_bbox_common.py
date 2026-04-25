from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
from PIL import Image, ImageDraw
import math
import numpy as np
import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# -----------------------------
# COCO loading
# -----------------------------

def find_annotation_file(dataset_dir: str) -> str:
    path = os.path.join(dataset_dir, "_annotations.coco.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"COCO annotation file not found: {path}")
    return path


def _choose_single_box(anns: Sequence[dict]) -> dict:
    return max(anns, key=lambda a: float(a.get("area", a["bbox"][2] * a["bbox"][3])))


def load_coco_single_box_records(dataset_dir: str, category_ids: Sequence[int] = (1,)) -> Tuple[List[dict], Dict[int, str]]:
    ann_path = find_annotation_file(dataset_dir)
    with open(ann_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    image_by_id = {int(img["id"]): img for img in coco["images"]}
    category_name = {int(cat["id"]): str(cat["name"]) for cat in coco.get("categories", [])}

    ann_by_image: Dict[int, List[dict]] = {}
    for ann in coco["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        cid = int(ann["category_id"])
        if cid not in category_ids:
            continue
        ann_by_image.setdefault(int(ann["image_id"]), []).append(ann)

    records: List[dict] = []
    missing_images = 0

    for image_id, anns in ann_by_image.items():
        img = image_by_id.get(image_id)
        if img is None:
            continue

        width = int(img["width"])
        height = int(img["height"])
        file_name = str(img["file_name"])
        image_path = os.path.join(dataset_dir, file_name)
        if not os.path.exists(image_path):
            missing_images += 1
            continue

        ann = _choose_single_box(anns)
        x, y, w, h = [float(v) for v in ann["bbox"]]
        x = np.clip(x, 0.0, float(width - 1))
        y = np.clip(y, 0.0, float(height - 1))
        w = np.clip(w, 1.0, float(width) - x)
        h = np.clip(h, 1.0, float(height) - y)

        bbox_rel_xywh = np.array([x / width, y / height, w / width, h / height], dtype=np.float32)
        bbox_rel_xywh = np.clip(bbox_rel_xywh, 0.0, 1.0)

        records.append(
            {
                "image_path": image_path,
                "bbox": bbox_rel_xywh,
                "width": width,
                "height": height,
                "image_id": image_id,
                "category_id": int(ann["category_id"]),
                "file_name": file_name,
            }
        )

    if not records:
        raise RuntimeError(
            f"No records found in {dataset_dir}. Check category_ids={tuple(category_ids)} and image paths."
        )

    print(f"Loaded {len(records)} records from {dataset_dir}")
    if missing_images:
        print(f"Skipped {missing_images} records with missing image files")

    return records, category_name


# -----------------------------
# BBox math
# -----------------------------

def xywh_to_xyxy_tf(boxes: tf.Tensor) -> tf.Tensor:
    x, y, w, h = tf.split(boxes, 4, axis=-1)
    return tf.concat([x, y, x + w, y + h], axis=-1)


def xyxy_to_xywh_tf(boxes: tf.Tensor) -> tf.Tensor:
    x1, y1, x2, y2 = tf.split(boxes, 4, axis=-1)
    return tf.concat([x1, y1, x2 - x1, y2 - y1], axis=-1)


def clip_boxes_xywh_tf(boxes: tf.Tensor) -> tf.Tensor:
    xyxy = xywh_to_xyxy_tf(boxes)
    x1, y1, x2, y2 = tf.split(xyxy, 4, axis=-1)
    x1 = tf.clip_by_value(x1, 0.0, 1.0)
    y1 = tf.clip_by_value(y1, 0.0, 1.0)
    x2 = tf.clip_by_value(x2, 0.0, 1.0)
    y2 = tf.clip_by_value(y2, 0.0, 1.0)
    x2 = tf.maximum(x2, x1 + 1e-5)
    y2 = tf.maximum(y2, y1 + 1e-5)
    return xyxy_to_xywh_tf(tf.concat([x1, y1, x2, y2], axis=-1))


def pairwise_iou_xywh_tf(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    true_xyxy = xywh_to_xyxy_tf(clip_boxes_xywh_tf(y_true))
    pred_xyxy = xywh_to_xyxy_tf(clip_boxes_xywh_tf(y_pred))

    tx1, ty1, tx2, ty2 = tf.split(true_xyxy, 4, axis=-1)
    px1, py1, px2, py2 = tf.split(pred_xyxy, 4, axis=-1)

    ix1 = tf.maximum(tx1, px1)
    iy1 = tf.maximum(ty1, py1)
    ix2 = tf.minimum(tx2, px2)
    iy2 = tf.minimum(ty2, py2)

    inter_w = tf.maximum(ix2 - ix1, 0.0)
    inter_h = tf.maximum(iy2 - iy1, 0.0)
    inter = inter_w * inter_h

    true_area = tf.maximum(tx2 - tx1, 0.0) * tf.maximum(ty2 - ty1, 0.0)
    pred_area = tf.maximum(px2 - px1, 0.0) * tf.maximum(py2 - py1, 0.0)
    union = true_area + pred_area - inter
    return inter / (union + 1e-7)


class SlideBBoxLoss(tf.keras.losses.Loss):
    def __init__(
        self,
        huber_delta: float = 0.05,
        huber_weight: float = 1.0,
        iou_weight: float = 2.0,
        reduction=tf.keras.losses.Reduction.AUTO,
        name: str = "slide_bbox_loss",
        **kwargs,
    ):
        super().__init__(reduction=reduction, name=name, **kwargs)
        self.huber = tf.keras.losses.Huber(delta=huber_delta, reduction=tf.keras.losses.Reduction.NONE)
        self.huber_delta = float(huber_delta)
        self.huber_weight = float(huber_weight)
        self.iou_weight = float(iou_weight)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(y_true, tf.float32)
        y_pred = clip_boxes_xywh_tf(tf.cast(y_pred, tf.float32))
        huber = tf.reduce_mean(self.huber(y_true, y_pred), axis=-1)
        iou = tf.squeeze(pairwise_iou_xywh_tf(y_true, y_pred), axis=-1)
        return self.huber_weight * huber + self.iou_weight * (1.0 - iou)

    def get_config(self):
        config = super().get_config()
        config.update({
            "huber_delta": self.huber_delta,
            "huber_weight": self.huber_weight,
            "iou_weight": self.iou_weight,
        })
        return config


class MeanIoUBox(tf.keras.metrics.Metric):
    def __init__(self, name: str = "mean_iou", **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = clip_boxes_xywh_tf(tf.cast(y_pred, tf.float32))
        iou = tf.squeeze(pairwise_iou_xywh_tf(y_true, y_pred), axis=-1)
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, tf.float32)
            iou = iou * sample_weight
            num = tf.reduce_sum(sample_weight)
        else:
            num = tf.cast(tf.size(iou), tf.float32)
        self.total.assign_add(tf.reduce_sum(iou))
        self.count.assign_add(num)

    def result(self):
        return self.total / (self.count + 1e-7)

    def reset_states(self):
        self.total.assign(0.0)
        self.count.assign(0.0)


# -----------------------------
# NumPy helpers for evaluation / drawing
# -----------------------------

def clip_boxes_xywh_np(boxes: np.ndarray) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float32)
    out = boxes.copy()
    out[..., 0] = np.clip(out[..., 0], 0.0, 1.0)
    out[..., 1] = np.clip(out[..., 1], 0.0, 1.0)
    out[..., 2] = np.clip(out[..., 2], 1e-6, 1.0)
    out[..., 3] = np.clip(out[..., 3], 1e-6, 1.0)
    out[..., 2] = np.minimum(out[..., 2], 1.0 - out[..., 0])
    out[..., 3] = np.minimum(out[..., 3], 1.0 - out[..., 1])
    return out


def xywh_to_xyxy_np(boxes: np.ndarray) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float32)
    x1 = boxes[..., 0]
    y1 = boxes[..., 1]
    x2 = boxes[..., 0] + boxes[..., 2]
    y2 = boxes[..., 1] + boxes[..., 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def iou_xywh_np(gt_box: np.ndarray, pred_box: np.ndarray) -> float:
    gt = xywh_to_xyxy_np(clip_boxes_xywh_np(gt_box))
    pr = xywh_to_xyxy_np(clip_boxes_xywh_np(pred_box))

    ix1 = max(float(gt[0]), float(pr[0]))
    iy1 = max(float(gt[1]), float(pr[1]))
    ix2 = min(float(gt[2]), float(pr[2]))
    iy2 = min(float(gt[3]), float(pr[3]))

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    gt_area = max(0.0, float(gt[2] - gt[0])) * max(0.0, float(gt[3] - gt[1]))
    pr_area = max(0.0, float(pr[2] - pr[0])) * max(0.0, float(pr[3] - pr[1]))
    union = gt_area + pr_area - inter
    return inter / (union + 1e-7)


def denormalize_xywh(box: np.ndarray, width: int, height: int) -> np.ndarray:
    x, y, w, h = np.asarray(box, dtype=np.float32)
    return np.array([x * width, y * height, w * width, h * height], dtype=np.float32)


# -----------------------------
# Data pipeline
# -----------------------------

def load_image_and_bbox(image_path: tf.Tensor, bbox: tf.Tensor, image_size: int):
    image_bytes = tf.io.read_file(image_path)
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, (image_size, image_size), method="bilinear")
    bbox = tf.cast(bbox, tf.float32)
    bbox = clip_boxes_xywh_tf(bbox)
    return image, bbox


def augment_train(image: tf.Tensor, bbox: tf.Tensor):
    do_flip = tf.less(tf.random.uniform([]), 0.5)

    def _flip():
        flipped_image = tf.image.flip_left_right(image)
        x, y, w, h = tf.unstack(bbox)
        flipped_bbox = tf.stack([1.0 - x - w, y, w, h])
        return flipped_image, flipped_bbox

    image, bbox = tf.cond(do_flip, _flip, lambda: (image, bbox))

    image = tf.image.random_brightness(image, max_delta=0.12)
    image = tf.image.random_contrast(image, lower=0.85, upper=1.15)
    image = tf.image.random_saturation(image, lower=0.85, upper=1.15)
    image = tf.image.random_hue(image, max_delta=0.02)

    do_gray = tf.less(tf.random.uniform([]), 0.10)
    def _gray():
        gray = tf.image.rgb_to_grayscale(image)
        return tf.image.grayscale_to_rgb(gray)
    image = tf.cond(do_gray, _gray, lambda: image)

    image = tf.clip_by_value(image, 0.0, 1.0)
    bbox = clip_boxes_xywh_tf(bbox)
    return image, bbox


def build_dataset(records: List[dict], image_size: int, batch_size: int, training: bool = False) -> tf.data.Dataset:
    image_paths = [r["image_path"] for r in records]
    boxes = np.stack([r["bbox"] for r in records]).astype(np.float32)

    ds = tf.data.Dataset.from_tensor_slices((image_paths, boxes))
    if training:
        ds = ds.shuffle(min(len(records), 2048), reshuffle_each_iteration=True)
    ds = ds.map(lambda p, b: load_image_and_bbox(p, b, image_size), num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.map(augment_train, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds


# -----------------------------
# Model
# -----------------------------

def _make_backbone(backbone: str, image_size: int, pretrained: bool, train_backbone: bool):
    weights = "imagenet" if pretrained else None
    input_shape = (image_size, image_size, 3)

    if backbone == "MobileNetV2":
        base = tf.keras.applications.MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights=weights,
        )
        # Dataset images are already float32 in [0, 1].
        # MobileNetV2 expects [-1, 1], so use a serializable built-in layer.
        preprocess_layer = tf.keras.layers.Rescaling(scale=2.0, offset=-1.0, name="preprocess")
    elif backbone == "EfficientNetB0":
        base = tf.keras.applications.EfficientNetB0(
            input_shape=input_shape,
            include_top=False,
            weights=weights,
        )
        # EfficientNet in tf.keras.applications expects raw [0, 255] pixels.
        # Use a built-in layer instead of Lambda so SavedModel loads cleanly.
        preprocess_layer = tf.keras.layers.Rescaling(scale=255.0, offset=0.0, name="preprocess")
    else:
        raise ValueError("backbone must be 'MobileNetV2' or 'EfficientNetB0'")

    base.trainable = train_backbone
    return base, preprocess_layer


def build_model(
    image_size: int = 320,
    backbone: str = "MobileNetV2",
    learning_rate: float = 1e-3,
    dropout: float = 0.2,
    pretrained: bool = True,
    train_backbone: bool = True,
    compile_model: bool = True,
) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(image_size, image_size, 3), name="image")
    base, preprocess_layer = _make_backbone(backbone, image_size, pretrained, train_backbone)

    x = preprocess_layer(inputs)
    x = base(x, training=train_backbone)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(4, activation="sigmoid", name="bbox")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="slide_bbox_regressor")

    if compile_model:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=SlideBBoxLoss(),
            metrics=[MeanIoUBox(), tf.keras.metrics.MeanAbsoluteError(name="mae")],
        )
    return model


def load_trained_model(
    model_path: str,
    image_size: int = 320,

    backbone: str = "MobileNetV2",
    dropout: float = 0.2,
    compile_for_eval: bool = True,
) -> tf.keras.Model:
    custom_objects = {
        "SlideBBoxLoss": SlideBBoxLoss,
        "MeanIoUBox": MeanIoUBox,
        "slide_bbox_loss": SlideBBoxLoss(),
    }

    def _compile_if_needed(model: tf.keras.Model) -> tf.keras.Model:
        if compile_for_eval:
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                loss=SlideBBoxLoss(),
                metrics=[MeanIoUBox(), tf.keras.metrics.MeanAbsoluteError(name="mae")],
            )
        return model

    if os.path.isdir(model_path):
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
        return _compile_if_needed(model)

    if model_path.endswith(".h5") or model_path.endswith(".weights.h5"):
        if model_path.endswith(".weights.h5"):
            model = build_model(
                image_size=image_size,
                backbone=backbone,
                dropout=dropout,
                pretrained=False,
                train_backbone=True,
                compile_model=compile_for_eval,
            )
            model.load_weights(model_path)
            return model
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
        return _compile_if_needed(model)

    raise ValueError(f"Unsupported model format: {model_path}")


def predict_single_image(model: tf.keras.Model, image_path: str, image_size: int) -> np.ndarray:
    image_bytes = tf.io.read_file(image_path)
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, (image_size, image_size), method="bilinear")
    pred = model.predict(tf.expand_dims(image, axis=0), verbose=0)[0]
    return clip_boxes_xywh_np(pred)

def draw_boxes_on_image(
    image: Image.Image,
    gt_abs_xywh: np.ndarray,
    pred_abs_xywh: np.ndarray,
    gt_color=(0, 255, 0),
    pred_color=(255, 0, 0),
    width: int = 3,
    mode: str = 'pred'
) -> Image.Image:
    img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    if mode == 'eval':
        gx, gy, gw, gh = [float(v) for v in gt_abs_xywh.tolist()]
        draw.rectangle([gx, gy, gx + gw, gy + gh], outline=gt_color, width=width)
        draw.text((gx + 4, max(0, gy - 14)), "gt", fill=gt_color)
    px, py, pw, ph = [float(v) for v in pred_abs_xywh.tolist()]
    draw.rectangle([px, py, px + pw, py + ph], outline=pred_color, width=width)
    draw.text((px + 4, max(0, py - 28)), "pred", fill=pred_color)
    return img



def show_samples_inline(
    rows: List[dict],
    max_images: int = 6,
    cols: int = 3,
    mode: str = 'pred',
    order: str = "worst",
    seed: int = 42,
    figsize_per_image: float = 5.0,
):
    if not rows:
        raise RuntimeError("rows is empty. Run run_evaluation(...) first.")

    import matplotlib.pyplot as plt

    selected = list(rows)
    if mode == 'eval':
        if order == "worst":
            selected.sort(key=lambda x: float(x["iou"]))
        elif order == "best":
            selected.sort(key=lambda x: float(x["iou"]), reverse=True)
        elif order == "random":
            rnd = random.Random(seed)
            rnd.shuffle(selected)
        else:
            raise ValueError("order must be 'worst', 'best' or 'random'")
    else:
        if order == "random":
            rnd = random.Random(seed)
            rnd.shuffle(selected)
        selected = selected[:max_images]
    selected = selected[:max_images]
    cols = max(1, int(cols))
    n = len(selected)
    rows_count = int(math.ceil(n / cols))

    fig, axes = plt.subplots(rows_count, cols, figsize=(figsize_per_image * cols, figsize_per_image * rows_count))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.reshape(rows_count, cols)

    for ax in axes.flat:
        ax.axis("off")

    for ax, row in zip(axes.flat, selected):
        image = Image.open(row["image_path"]).convert("RGB")
        gt_abs = np.array([row["gt_x"], row["gt_y"], row["gt_w"], row["gt_h"]], dtype=np.float32)
        pred_abs = np.array([row["pred_x"], row["pred_y"], row["pred_w"], row["pred_h"]], dtype=np.float32)
        vis = draw_boxes_on_image(image, gt_abs, pred_abs)
        ax.imshow(vis)
        ax.set_title(f"{row['file_name']}\nIoU={row['iou']:.3f}", fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    plt.show()
