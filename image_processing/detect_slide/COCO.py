import json
from datetime import datetime


COCO_INFO = {
    "year": str(datetime.now().year),
    "version": "1",
    "description": "Semi-automatic ROI annotations",
    "contributor": "",
    "url": "",
    "date_created": datetime.now().astimezone().isoformat(),
}

COCO_LICENSES = [
    {
        "id": 1,
        "url": "https://creativecommons.org/licenses/by/4.0/",
        "name": "CC BY 4.0",
    }
]

# Делаем совместимо с вашим примером
COCO_CATEGORIES = [
    {"id": 0, "name": "slideregion", "supercategory": "none"},
    {"id": 1, "name": "projection screen", "supercategory": "slideregion"},
]


def load_or_init_coco(coco_json_path: str) -> dict:
    if os.path.exists(coco_json_path):
        with open(coco_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    return {
        "info": COCO_INFO,
        "licenses": COCO_LICENSES,
        "categories": COCO_CATEGORIES,
        "images": [],
        "annotations": [],
    }


def save_coco(coco_json_path: str, coco_data: dict):
    ensure_parent(coco_json_path)
    with open(coco_json_path, "w", encoding="utf-8") as f:
        json.dump(coco_data, f, ensure_ascii=False)


def build_existing_coco_keys(coco_data: dict) -> set:
    """
    Чтобы не дублировать записи при повторном запуске.
    Ключ делаем по file_name.
    """
    keys = set()
    for img in coco_data.get("images", []):
        keys.add(img["file_name"])
    return keys


def next_image_id(coco_data: dict) -> int:
    if not coco_data["images"]:
        return 0
    return max(img["id"] for img in coco_data["images"]) + 1


def next_annotation_id(coco_data: dict) -> int:
    if not coco_data["annotations"]:
        return 0
    return max(ann["id"] for ann in coco_data["annotations"]) + 1


def append_coco_image_and_annotation(
    coco_data: dict,
    file_name: str,
    width: int,
    height: int,
    bbox_xywh,
    category_id: int = 1,
):
    image_id = next_image_id(coco_data)
    ann_id = next_annotation_id(coco_data)

    x, y, w, h = [float(v) for v in bbox_xywh]

    image_item = {
        "id": image_id,
        "license": 1,
        "file_name": file_name,
        "height": int(height),
        "width": int(width),
        "date_captured": datetime.now().astimezone().isoformat(),
    }

    ann_item = {
        "id": ann_id,
        "image_id": image_id,
        "category_id": int(category_id),
        "bbox": [x, y, w, h],
        "area": float(w * h),
        "segmentation": [],
        "iscrowd": 0,
    }

    coco_data["images"].append(image_item)
    coco_data["annotations"].append(ann_item)