from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import cv2
import pytesseract


_CLEAN_RE = re.compile(r"\s+")

TESSERACT_EXE = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
pytesseract.pytesseract.tesseract_cmd = str(TESSERACT_EXE)

def process_image(image_path: Path, lang: str = "rus+eng") -> str:
    img = cv2.imread(str(image_path))
    if img is None:
        return ""
    text = pytesseract.image_to_string(img, lang=lang)
    return _CLEAN_RE.sub(" ", text).strip()


def run_ocr(images: Iterable[Path], out_dir: Path, lang: str = "rus+eng") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    merged_path = out_dir / "ocr_merged.txt"
    with merged_path.open("w", encoding="utf-8") as merged:
        for image_path in images:
            txt = process_image(image_path, lang=lang)
            single_out = out_dir / f"{image_path.stem}.txt"
            single_out.write_text(txt, encoding="utf-8")
            if txt:
                merged.write(f"[{image_path.name}]\n{txt}\n\n")
    return merged_path
