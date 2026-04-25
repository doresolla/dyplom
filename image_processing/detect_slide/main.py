import os
import csv
import math
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import cv2
import numpy as np
from slide_bbox_common import load_trained_model, predict_single_image, denormalize_xywh

from quad import (draw_quad,
                  QuadEditor)
from refine_bbox import (detect_presentation_surface,
                         clamp_xyxy, xywh_to_xyxy,
                         draw_box, xyxy_to_xywh, bbox_xywh_to_quad)
from files import (collect_videos, ensure_parent,
                   load_existing_keys,
                   make_frame_name,
                   build_annotation_row)

def predict_rough_xywh_from_frame(
    model,
    frame_bgr: np.ndarray,
    image_size: int,
    temp_image_path: str,
):
    """
    Здесь используется ваш уже существующий predict_single_image(...),
    чтобы не менять вашу схему препроцессинга.
    """
    ensure_parent(temp_image_path)
    ok = cv2.imwrite(temp_image_path, frame_bgr)
    if not ok:
        raise RuntimeError(f"Не удалось сохранить временный кадр: {temp_image_path}")

    pred_rel = predict_single_image(
        model,
        image_path=temp_image_path,
        image_size=image_size,
    )

    h, w = frame_bgr.shape[:2]

    pred_abs = denormalize_xywh(pred_rel, w, h)
    pred_abs = [float(v) for v in pred_abs]
    pred_rel = [float(v) for v in pred_rel]
    return pred_abs, pred_rel


def expand_model_roi_xywh(
    model_roi_xywh,
    image_shape,
    pad_ratio: float = 0.10,
) -> List[int]:
    """
    Расширение от размеров ВСЕГО изображения, а не от bbox.
    """
    h_img, w_img = image_shape[:2]
    x1, y1, x2, y2 = xywh_to_xyxy(model_roi_xywh)

    pad_x = int(round(pad_ratio * w_img))
    pad_y = int(round(pad_ratio * h_img))

    search_xyxy = [x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y]
    return clamp_xyxy(search_xyxy, w_img, h_img)



def save_preview(
    save_path: str,
    frame_bgr: np.ndarray,
    rough_xywh,
    search_xyxy,
    quad: np.ndarray,
):
    vis = frame_bgr.copy()
    vis = draw_box(vis, xywh_to_xyxy(rough_xywh), color=(0, 255, 255), thickness=2, label="model")
    vis = draw_box(vis, search_xyxy, color=(255, 255, 0), thickness=2, label="search")
    vis = draw_quad(vis, quad, color=(0, 0, 255), thickness=2)

    ensure_parent(save_path)
    cv2.imwrite(save_path, vis)



ANNOTATION_FIELDS = [
    "file_name",
    "image_path",
    "video_path",
    "video_name",
    "frame_idx",
    "timestamp_sec",
    "image_width",
    "image_height",
    "source_mode",          # auto / manual / bbox_fallback

    "model_x",
    "model_y",
    "model_w",
    "model_h",

    "search_x1",
    "search_y1",
    "search_x2",
    "search_y2",

    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",

    "quad_x1", "quad_y1",
    "quad_x2", "quad_y2",
    "quad_x3", "quad_y3",
    "quad_x4", "quad_y4",
]

def iter_sampled_frames(video_path: str, sample_every_sec: float):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Не удалось открыть видео: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1e-6:
        fps = 25.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(round(sample_every_sec * fps)))

    try:
        for frame_idx in range(0, total_frames, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            timestamp_sec = frame_idx / fps
            yield frame_idx, timestamp_sec, frame
    finally:
        cap.release()


def append_annotation(csv_path: str, row: Dict):
    file_exists = os.path.exists(csv_path)
    ensure_parent(csv_path)

    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ANNOTATION_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

def process_videos_for_dataset(
    input_path: str,
    model_path: str,
    dataset_dir: str,
    image_size: int = 224,
    backbone: str = "MobileNetV2",
    dropout: float = 0.2,
    sample_every_sec: float = 3.0,
    roi_pad_ratio: float = 0.10,
    max_side: int = 1200,
    min_line_len: int = 80,
):
    videos = collect_videos(input_path)
    if not videos:
        raise FileNotFoundError(f"Видео не найдены: {input_path}")

    images_dir = os.path.join(dataset_dir, "images")
    previews_dir = os.path.join(dataset_dir, "previews")
    temp_dir = os.path.join(dataset_dir, "_temp_frames")
    csv_path = os.path.join(dataset_dir, "annotations.csv")

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(previews_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    done_keys = load_existing_keys(csv_path)

    model = load_trained_model(
        model_path=model_path,
        image_size=image_size,
        backbone=backbone,
        dropout=dropout,
        compile_for_eval=False,
    )

    for video_path in videos:
        video_name = Path(video_path).name
        print(f"\n=== Видео: {video_name} ===")

        for frame_idx, timestamp_sec, frame_bgr in iter_sampled_frames(video_path, sample_every_sec):
            key = (video_path, frame_idx)
            if key in done_keys:
                continue

            temp_frame_path = os.path.join(
                temp_dir,
                f"{Path(video_path).stem}_f{frame_idx:08d}.png",
            )

            try:
                model_xywh, pred_rel = predict_rough_xywh_from_frame(
                    model=model,
                    frame_bgr=frame_bgr,
                    image_size=image_size,
                    temp_image_path=temp_frame_path,
                )
            except Exception as e:
                print(f"[WARN] Ошибка модели на кадре {frame_idx}: {e}")
                continue

            quad_auto, refine_info = detect_presentation_surface(
                img=frame_bgr,
                model_roi_xywh=model_xywh,
                roi_pad_ratio=roi_pad_ratio,
                max_side=max_side,
                min_line_len=min_line_len,
                fallback_to_full_image=True,
            )

            search_xyxy = refine_info["search_xyxy"]

            if quad_auto is None:
                quad_auto = bbox_xywh_to_quad(model_xywh)
                initial_mode = "bbox_fallback"
            else:
                initial_mode = "auto"
            editor = QuadEditor(
                frame_bgr=frame_bgr,
                video_name=video_name,
                frame_idx=frame_idx,
                timestamp_sec=timestamp_sec,
                rough_xywh=model_xywh,
                search_xyxy=search_xyxy,
                auto_quad=quad_auto,
            )

            final_quad, user_mode = editor.edit()

            if user_mode == "quit":
                print("Остановка по команде пользователя.")
                return

            if user_mode == "skip":
                print(f"[SKIP] {video_name} frame={frame_idx}")
                continue

            source_mode = user_mode
            if user_mode == "auto" and initial_mode == "bbox_fallback":
                source_mode = "bbox_fallback"

            image_name = make_frame_name(video_path, frame_idx, timestamp_sec)
            image_save_path = os.path.join(images_dir, image_name)
            preview_save_path = os.path.join(previews_dir, image_name)

            cv2.imwrite(image_save_path, frame_bgr)
            save_preview(
                save_path=preview_save_path,
                frame_bgr=frame_bgr,
                rough_xywh=model_xywh,
                search_xyxy=search_xyxy,
                quad=final_quad,
            )

            row = build_annotation_row(
                image_save_path=image_save_path,
                video_path=video_path,
                frame_idx=frame_idx,
                timestamp_sec=timestamp_sec,
                frame_bgr=frame_bgr,
                source_mode=source_mode,
                model_xywh=model_xywh,
                search_xyxy=search_xyxy,
                quad=final_quad,
            )
            append_annotation(csv_path, row)
            done_keys.add(key)

            print(f"[SAVE] {video_name} frame={frame_idx} mode={source_mode}")

    print("\nГотово.")

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Полуавтоматическая разметка ROI/quad из видео"
    )
    parser.add_argument("--input-path", required=True, help="Видео или папка с видео")
    parser.add_argument("--model-path", required=True, help="Путь к обученной модели")
    parser.add_argument("--dataset-dir", required=True, help="Куда сохранять датасет")

    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--backbone", type=str, default="MobileNetV2")
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--sample-every-sec", type=float, default=1.0)
    parser.add_argument("--roi-pad-ratio", type=float, default=0.10)
    parser.add_argument("--max-side", type=int, default=1200)
    parser.add_argument("--min-line-len", type=int, default=80)

    args = parser.parse_args()

    process_videos_for_dataset(
        input_path=args.input_path,
        model_path=args.model_path,
        dataset_dir=args.dataset_dir,
        image_size=args.image_size,
        backbone=args.backbone,
        dropout=args.dropout,
        sample_every_sec=args.sample_every_sec,
        roi_pad_ratio=args.roi_pad_ratio,
        max_side=args.max_side,
        min_line_len=args.min_line_len,
    )


if __name__ == "__main__":
    main()