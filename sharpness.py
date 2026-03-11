import os
import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable
from skimage.metrics import structural_similarity as ssim
import imagehash
from PIL import Image

MIN_EDGE_DENSITY = 60
MAX_EDGE_DENSITY = 160

KSIZE_X = 15
KSIZE_Y = 5

@dataclass
class FrameInfo:
    t: float
    frame_bgr: np.ndarray
    roi_bgr: np.ndarray
    score: float
    metrics: dict



def normalize_roi(frame_bgr: np.ndarray, roi: Optional[Tuple[int, int, int, int]]) -> Optional[Tuple[int, int, int, int]]:
    if roi is None:
        return None

    x1, y1, x2, y2 = map(int, roi)
    h, w = frame_bgr.shape[:2]

    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(1, min(w, x2))
    y2 = max(1, min(h, y2))

    if x2 <= x1 + 5 or y2 <= y1 + 5:
        return None

    return (x1, y1, x2, y2)

# Crop image by keypoints (ROI - region of interest)
def crop_roi(frame_bgr: np.ndarray, roi: Optional[Tuple[int,int,int,int]] = None) -> np.ndarray:
    """
    roi = (x1, y1, x2, y2). If None -> use whole frame.
    """
    if roi is None:
        return frame_bgr
    x1, y1, x2, y2 = roi
    h, w = frame_bgr.shape[:2]
    x1 = max(0, min(w-1, x1)); x2 = max(0, min(w, x2))
    y1 = max(0, min(h-1, y1)); y2 = max(0, min(h, y2))
    if x2 <= x1+5 or y2 <= y1+5:
        return frame_bgr
    return frame_bgr[y1:y2, x1:x2]

def normalize_roi(
    frame_bgr: np.ndarray,
    roi: Optional[Tuple[int, int, int, int]],
) -> Optional[Tuple[int, int, int, int]]:
    if roi is None:
        return None

    x1, y1, x2, y2 = map(int, roi)
    h, w = frame_bgr.shape[:2]

    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(1, min(w, x2))
    y2 = max(1, min(h, y2))

    if x2 <= x1 + 5 or y2 <= y1 + 5:
        return None

    return (x1, y1, x2, y2)




def laplacian_var(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def edge_density(gray: np.ndarray) -> float:
    edges = cv2.Canny(gray, MIN_EDGE_DENSITY, MAX_EDGE_DENSITY)
    return float((edges > 0).mean())

def gray_entropy(gray: np.ndarray) -> float:
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    p = hist / (hist.sum() + 1e-9)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())

def textness_fast(gray: np.ndarray) -> float:
    """
    Быстрый индикатор "текстовости" без OCR:
    blackhat + градиент + морфология -> сколько "штрихов" найдено.
    """
    # Подходит для белого/светлого фона со слайдом, но работает и в среднем случае
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (KSIZE_X, KSIZE_Y))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    grad = cv2.Sobel(blackhat, cv2.CV_32F, 1, 0, ksize=3)
    grad = np.absolute(grad)
    grad = (255 * (grad / (grad.max() + 1e-9))).astype(np.uint8)

    thr = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=1)
    thr = cv2.erode(thr, None, iterations=1)
    thr = cv2.dilate(thr, None, iterations=1)

    return float((thr > 0).mean())

def compute_content_metrics(roi_bgr: np.ndarray) -> dict:
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    # лёгкое подавление шума, чтобы метрики были стабильнее
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)

    m = {
        "sharp": laplacian_var(gray_blur),
        "edge": edge_density(gray_blur),
        "text": textness_fast(gray_blur),
        "entropy": gray_entropy(gray_blur),
    }
    return m

def robust_score(metrics: dict) -> float:
    """
    Простейшая нормализация "на месте":
    - sharp логарифмируем (у него большой разброс)
    - остальные оставляем как есть
    """
    sharp = np.log1p(metrics["sharp"])
    edge = metrics["edge"]
    text = metrics["text"]
    ent  = metrics["entropy"] / 8.0  # грубо приводим к ~[0..1]
    # веса можно подкрутить под твой датасет
    return float(0.40 * sharp + 0.25 * edge + 0.25 * text + 0.10 * ent)

# ---------- Slide/content change ----------
def ssim_change(a_gray: np.ndarray, b_gray: np.ndarray) -> float:
    # приводим к одному размеру для стабильности
    a = cv2.resize(a_gray, (320, 180))
    b = cv2.resize(b_gray, (320, 180))
    return float(ssim(a, b))

def is_change(prev_roi_bgr: np.ndarray, roi_bgr: np.ndarray, thr: float = 0.72) -> bool:
    pa = cv2.cvtColor(prev_roi_bgr, cv2.COLOR_BGR2GRAY)
    pb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    sim = ssim_change(pa, pb)
    return sim < thr

def extract_candidates(
    video_path: str,
    sample_fps: float = 1.0,
    roi: Optional[Tuple[int, int, int, int]] = None,
    change_thr: float = 0.72,
    roi_detector: Optional[Callable[[np.ndarray], dict]] = None,
    detector_stride: int = 5,
) -> List[FrameInfo]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(fps / sample_fps)))

    candidates: List[FrameInfo] = []
    prev_roi = None
    last_dynamic_roi = roi
    sampled_idx = 0

    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if i % step != 0:
            i += 1
            continue

        t = float(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)

        current_roi = roi

        if roi_detector is not None:
            need_refresh = (sampled_idx % detector_stride == 0) or (last_dynamic_roi is None)
            if need_refresh:
                try:
                    meta = roi_detector(frame)
                    bbox = meta.get("bbox") if meta else None
                    bbox = normalize_roi(frame, tuple(bbox)) if bbox else None
                    if bbox is not None:
                        last_dynamic_roi = bbox
                except Exception:
                    pass
            current_roi = last_dynamic_roi

        roi_bgr = crop_roi(frame, current_roi)

        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        if gray.mean() < 8:
            i += 1
            sampled_idx += 1
            continue

        m = compute_content_metrics(roi_bgr)
        sc = robust_score(m)

        changed = False
        if prev_roi is not None and is_change(prev_roi, roi_bgr, thr=change_thr):
            changed = True
            sc *= 1.15

        candidates.append(
            FrameInfo(
                t=t,
                frame_bgr=frame,
                roi_bgr=roi_bgr,
                score=sc,
                metrics={**m, "changed": changed},
            )
        )

        prev_roi = roi_bgr
        i += 1
        sampled_idx += 1

    cap.release()
    return candidates

# ---------- Segment + selection ----------
def segment_by_change(cands: List[FrameInfo]) -> List[List[FrameInfo]]:
    if not cands:
        return []
    segs = [[cands[0]]]
    for c in cands[1:]:
        if c.metrics.get("changed", False):
            segs.append([c])
        else:
            segs[-1].append(c)
    return segs

def best_per_segment(segs: List[List[FrameInfo]]) -> List[FrameInfo]:
    out = []
    for seg in segs:
        best = max(seg, key=lambda x: x.score)
        out.append(best)
    return out

def dedup_phash(keyframes: List[FrameInfo], max_hamming: int = 6) -> List[FrameInfo]:
    """
    Дедуп по perceptual hash на ROI.
    max_hamming: чем меньше, тем агрессивнее удаление похожих.
    """
    kept: List[FrameInfo] = []
    hashes = []
    for k in keyframes:
        rgb = cv2.cvtColor(k.roi_bgr, cv2.COLOR_BGR2RGB)
        h = imagehash.phash(Image.fromarray(rgb))
        is_dup = any((h - hh) <= max_hamming for hh in hashes)
        if not is_dup:
            kept.append(k)
            hashes.append(h)
    return kept
def select_keyframes(
    video_path: str,
    out_dir: str,
    sample_fps: float = 1.0,
    roi: Optional[Tuple[int, int, int, int]] = None,
    change_thr: float = 0.72,
    max_hamming: int = 6,
    roi_detector: Optional[Callable[[np.ndarray], dict]] = None,
    detector_stride: int = 5,
) -> List[Tuple[float, str]]:
    os.makedirs(out_dir, exist_ok=True)

    cands = extract_candidates(
        video_path,
        sample_fps=sample_fps,
        roi=roi,
        change_thr=change_thr,
        roi_detector=roi_detector,
        detector_stride=detector_stride,
    )
    segs = segment_by_change(cands)
    keys = best_per_segment(segs)
    keys = dedup_phash(keys, max_hamming=max_hamming)

    saved = []
    for idx, k in enumerate(keys):
        fname = f"key_{idx:04d}_{k.t:.2f}s.jpg"
        path = os.path.join(out_dir, fname)
        cv2.imwrite(path, k.frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        saved.append((k.t, path))
    return saved