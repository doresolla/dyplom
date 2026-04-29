from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from skimage.metrics import structural_similarity as ssim

try:
    import imagehash  # type: ignore
except Exception:
    class _FallbackImageHash:
        def __init__(self, bits: np.ndarray):
            self.hash = np.asarray(bits, dtype=bool).reshape(-1)

        def __sub__(self, other: "_FallbackImageHash") -> int:
            return int(np.count_nonzero(self.hash != other.hash))

        def __str__(self) -> str:
            bit_str = "".join("1" if v else "0" for v in self.hash.astype(np.uint8))
            width = max(1, len(bit_str) // 4)
            return f"{int(bit_str, 2):0{width}x}"

        __repr__ = __str__

    class _FallbackImageHashModule:
        ImageHash = _FallbackImageHash

        @staticmethod
        def phash(image: Image.Image, hash_size: int = 8, highfreq_factor: int = 4) -> _FallbackImageHash:
            img = np.asarray(image.convert("L"))
            size = hash_size * highfreq_factor
            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
            dct = cv2.dct(np.float32(img))
            lowfreq = dct[:hash_size, :hash_size]
            med = float(np.median(lowfreq[1:, :]))
            bits = lowfreq > med
            return _FallbackImageHash(bits)

        @staticmethod
        def hex_to_hash(hexstr: str, hash_size: int = 8) -> _FallbackImageHash:
            nbits = hash_size * hash_size
            as_int = int(str(hexstr), 16)
            bits = np.array(list(np.binary_repr(as_int, width=nbits)), dtype="U1") == "1"
            return _FallbackImageHash(bits)

    imagehash = _FallbackImageHashModule()  # type: ignore

try:
    from .debug_contours import find_page_rect  # type: ignore
except Exception:
    find_page_rect = None

try:
    from .debug_contours import detect_slide_roi_from_video  # type: ignore
except Exception:
    detect_slide_roi_from_video = None

BBox = Tuple[int, int, int, int]
MIN_EDGE_DENSITY = 60
MAX_EDGE_DENSITY = 160
KSIZE_X = 15
KSIZE_Y = 5
THUMB_SIZE = (320, 180)


@dataclass
class Thresholds:
    ssim_drop_thr: float
    phash_thr: float
    change_score_thr: float
    stable_ssim_thr: float
    stable_phash_thr: float
    stable_change_thr: float
    return_ssim_thr: float
    return_phash_thr: float
    simple_change_ssim_thr: float
    simple_change_phash_thr: float
    simple_change_score_thr: float


@dataclass
class KeyframeRecord:
    segment_id: int
    frame_idx: int
    time_sec: float
    content_score: float
    image_path: str
    roi_path: str


# -----------------------------------------------------------------------------
# ROI utils
# -----------------------------------------------------------------------------
def normalize_roi(frame_bgr: np.ndarray, roi: Optional[BBox]) -> Optional[BBox]:
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

    return x1, y1, x2, y2


def crop_roi(frame_bgr: np.ndarray, roi: Optional[BBox] = None) -> np.ndarray:
    if roi is None:
        return frame_bgr
    roi = normalize_roi(frame_bgr, roi)
    if roi is None:
        return frame_bgr
    x1, y1, x2, y2 = roi
    return frame_bgr[y1:y2, x1:x2]


def detect_constant_roi(video_path: str) -> Optional[BBox]:
    if find_page_rect is None:
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)

    probe_frame_idx = int(min(max(0, fps * 60.0), max(0, total_frames - 1)))
    if total_frames > 0 and probe_frame_idx <= 0:
        probe_frame_idx = total_frames // 2

    cap.set(cv2.CAP_PROP_POS_FRAMES, probe_frame_idx)
    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        return None

    try:
        rect, _ = find_page_rect(frame, debug=False)
        return normalize_roi(frame, rect)
    except Exception:
        return None


# -----------------------------------------------------------------------------
# Content metrics
# -----------------------------------------------------------------------------
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
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (KSIZE_X, KSIZE_Y))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    grad = cv2.Sobel(blackhat, cv2.CV_32F, 1, 0, ksize=3)
    grad = np.abs(grad)
    grad = (255 * (grad / (grad.max() + 1e-9))).astype(np.uint8)

    thr = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=1)
    thr = cv2.erode(thr, None, iterations=1)
    thr = cv2.dilate(thr, None, iterations=1)
    return float((thr > 0).mean())


def compute_content_metrics(roi_bgr: np.ndarray) -> Dict[str, float]:
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)
    return {
        "sharp": laplacian_var(gray_blur),
        "edge": edge_density(gray_blur),
        "text": textness_fast(gray_blur),
        "entropy": gray_entropy(gray_blur),
    }


def robust_content_score(metrics: Dict[str, float]) -> float:
    sharp = np.log1p(metrics["sharp"])
    edge = metrics["edge"]
    text = metrics["text"]
    ent = metrics["entropy"] / 8.0
    return float(0.40 * sharp + 0.25 * edge + 0.25 * text + 0.10 * ent)


# -----------------------------------------------------------------------------
# Visual similarity
# -----------------------------------------------------------------------------
def roi_to_thumb(roi_bgr: np.ndarray, size: Tuple[int, int] = THUMB_SIZE) -> np.ndarray:
    return cv2.resize(roi_bgr, size, interpolation=cv2.INTER_AREA)


def gray_for_similarity(thumb_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(thumb_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (3, 3), 0)


def ssim_same(a_gray: np.ndarray, b_gray: np.ndarray) -> float:
    return float(ssim(a_gray, b_gray))


def phash_from_bgr(image_bgr: np.ndarray):
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return imagehash.phash(Image.fromarray(rgb))


def phash_distance(hash_a, hash_b) -> int:
    return int(hash_a - hash_b)


# -----------------------------------------------------------------------------
# Pass 1: sample video and dump metrics + thumbnails
# -----------------------------------------------------------------------------
def sample_video_metrics(
    video_path: str,
    out_csv: str,
    sample_fps: float = 1.0,
    roi: Optional[BBox] = None,
    auto_roi: bool = True,
    thumbs_dir: Optional[str] = None,
) -> pd.DataFrame:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(float(fps) / float(sample_fps))))

    if roi is None and auto_roi:
        roi, all_rects, infos = detect_slide_roi_from_video(
            video_path,
            sample_fps=0.5,
            max_frames=20,
            debug=False
        )
        print("FINAL ROI:", roi)

    thumbs_root = Path(thumbs_dir) if thumbs_dir else Path(out_csv).with_suffix("").parent / "thumbs"
    thumbs_root.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    prev_gray: Optional[np.ndarray] = None
    prev_hash = None
    prev_metrics: Optional[Dict[str, float]] = None

    frame_idx = 0
    sampled_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % step != 0:
            frame_idx += 1
            continue

        time_sec = float(frame_idx / fps)
        roi_bgr = crop_roi(frame, roi)
        if roi_bgr.size == 0:
            frame_idx += 1
            continue

        thumb_bgr = roi_to_thumb(roi_bgr)
        gray = gray_for_similarity(thumb_bgr)
        if float(gray.mean()) < 8.0:
            frame_idx += 1
            sampled_idx += 1
            continue

        curr_hash = phash_from_bgr(thumb_bgr)
        metrics = compute_content_metrics(roi_bgr)
        content_score = robust_content_score(metrics)

        thumb_path = thumbs_root / f"thumb_{sampled_idx:06d}_f{frame_idx:08d}.jpg"
        cv2.imwrite(str(thumb_path), thumb_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

        if prev_gray is None or prev_hash is None or prev_metrics is None:
            ssim_prev = 1.0
            ssim_drop = 0.0
            phash_prev = 0
            d_sharp = 0.0
            d_edge = 0.0
            d_text = 0.0
            d_entropy = 0.0
        else:
            ssim_prev = ssim_same(prev_gray, gray)
            ssim_drop = 1.0 - ssim_prev
            phash_prev = phash_distance(prev_hash, curr_hash)
            d_sharp = abs(metrics["sharp"] - prev_metrics["sharp"])
            d_edge = abs(metrics["edge"] - prev_metrics["edge"])
            d_text = abs(metrics["text"] - prev_metrics["text"])
            d_entropy = abs(metrics["entropy"] - prev_metrics["entropy"])

        change_score = (
            0.60 * ssim_drop
            + 0.25 * (phash_prev / 64.0)
            + 0.10 * min(1.0, d_text / 0.08)
            + 0.05 * min(1.0, d_edge / 0.08)
        )

        rows.append(
            {
                "sample_id": sampled_idx,
                "frame_idx": frame_idx,
                "time_sec": time_sec,
                "sharp": metrics["sharp"],
                "edge": metrics["edge"],
                "text": metrics["text"],
                "entropy": metrics["entropy"],
                "content_score": content_score,
                "ssim_prev": ssim_prev,
                "ssim_drop": ssim_drop,
                "phash_prev": phash_prev,
                "d_sharp": d_sharp,
                "d_edge": d_edge,
                "d_text": d_text,
                "d_entropy": d_entropy,
                "change_score": change_score,
                "thumb_path": str(thumb_path),
                "phash_hex": str(curr_hash),
                "roi_x1": roi[0] if roi else -1,
                "roi_y1": roi[1] if roi else -1,
                "roi_x2": roi[2] if roi else -1,
                "roi_y2": roi[3] if roi else -1,
            }
        )

        prev_gray = gray
        prev_hash = curr_hash
        prev_metrics = metrics
        frame_idx += 1
        sampled_idx += 1

    cap.release()

    df = pd.DataFrame(rows)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8")
    return df


# -----------------------------------------------------------------------------
# Thresholds
# -----------------------------------------------------------------------------
def _robust_threshold(series: pd.Series, k: float = 3.0) -> float:
    x = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=np.float64)
    if x.size == 0:
        return 0.0
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-9
    return float(med + k * 1.4826 * mad)


def estimate_thresholds(metrics_df: pd.DataFrame) -> Thresholds:
    if metrics_df.empty:
        return Thresholds(
            ssim_drop_thr=0.08,
            phash_thr=6.0,
            change_score_thr=0.18,
            stable_ssim_thr=0.92,
            stable_phash_thr=6.0,
            stable_change_thr=0.10,
            return_ssim_thr=0.95,
            return_phash_thr=6.0,
            simple_change_ssim_thr=0.72,
            simple_change_phash_thr=6.0,
            simple_change_score_thr=0.18,
        )

    ssim_drop_thr = max(0.08, _robust_threshold(metrics_df["ssim_drop"], k=3.0))
    phash_thr = max(6.0, _robust_threshold(metrics_df["phash_prev"], k=2.5))
    change_score_thr = max(0.18, _robust_threshold(metrics_df["change_score"], k=3.0))

    ssim_drop_thr = float(min(ssim_drop_thr, 0.60))
    phash_thr = float(min(phash_thr, 30.0))
    change_score_thr = float(min(change_score_thr, 0.95))

    stable_ssim_thr = float(np.clip(1.0 - max(0.03, ssim_drop_thr * 0.55), 0.88, 0.98))
    stable_phash_thr = float(np.clip(max(4.0, phash_thr * 0.40), 4.0, 10.0))
    stable_change_thr = float(np.clip(max(0.04, change_score_thr * 0.45), 0.04, 0.18))

    return_ssim_thr = float(np.clip(max(0.93, stable_ssim_thr + 0.02), 0.93, 0.99))
    return_phash_thr = float(np.clip(max(4.0, stable_phash_thr * 1.15), 4.0, 12.0))

    simple_change_ssim_thr = float(np.clip(1.0 - max(0.18, ssim_drop_thr * 1.35), 0.72, 0.88))
    simple_change_phash_thr = float(np.clip(max(6.0, phash_thr * 0.90), 6.0, 20.0))
    simple_change_score_thr = float(np.clip(max(0.18, change_score_thr * 0.85), 0.18, 0.60))

    return Thresholds(
        ssim_drop_thr=ssim_drop_thr,
        phash_thr=phash_thr,
        change_score_thr=change_score_thr,
        stable_ssim_thr=stable_ssim_thr,
        stable_phash_thr=stable_phash_thr,
        stable_change_thr=stable_change_thr,
        return_ssim_thr=return_ssim_thr,
        return_phash_thr=return_phash_thr,
        simple_change_ssim_thr=simple_change_ssim_thr,
        simple_change_phash_thr=simple_change_phash_thr,
        simple_change_score_thr=simple_change_score_thr,
    )


# -----------------------------------------------------------------------------
# Cache + visual helpers used by post-filters
# -----------------------------------------------------------------------------
def _load_visual_cache(df: pd.DataFrame) -> Tuple[List[np.ndarray], List[object]]:
    grays: List[np.ndarray] = []
    hashes: List[object] = []

    for row in df.itertuples(index=False):
        thumb = cv2.imread(str(row.thumb_path), cv2.IMREAD_COLOR)
        if thumb is None:
            raise RuntimeError(f"Cannot read thumb: {row.thumb_path}")
        grays.append(gray_for_similarity(thumb))
        hashes.append(imagehash.hex_to_hash(str(row.phash_hex)))

    return grays, hashes


def _visual_similarity(idx_a: int, idx_b: int, grays: List[np.ndarray], hashes: List[object]) -> Tuple[float, int]:
    return ssim_same(grays[idx_a], grays[idx_b]), phash_distance(hashes[idx_a], hashes[idx_b])


def _overlap_slices(length: int, shift: int) -> Optional[Tuple[slice, slice]]:
    if abs(shift) >= length:
        return None
    if shift >= 0:
        return slice(0, length - shift), slice(shift, length)
    shift_abs = -shift
    return slice(shift_abs, length), slice(0, length - shift_abs)


def _aligned_overlap_similarity(
    a_gray: np.ndarray,
    b_gray: np.ndarray,
    max_dx_ratio: float = 0.18,
    max_dy_ratio: float = 0.40,
) -> Tuple[float, float, float, float, float]:
    h, w = a_gray.shape[:2]
    a32 = np.float32(a_gray) / 255.0
    b32 = np.float32(b_gray) / 255.0

    (dx, dy), response = cv2.phaseCorrelate(a32, b32)
    dx_i = int(round(dx))
    dy_i = int(round(dy))

    max_dx = int(round(w * max_dx_ratio))
    max_dy = int(round(h * max_dy_ratio))
    if abs(dx_i) > max_dx or abs(dy_i) > max_dy:
        return float(dx), float(dy), float(response), 0.0, 0.0

    x_slices = _overlap_slices(w, dx_i)
    y_slices = _overlap_slices(h, dy_i)
    if x_slices is None or y_slices is None:
        return float(dx), float(dy), float(response), 0.0, 0.0

    ax, bx = x_slices
    ay, by = y_slices
    a_crop = a_gray[ay, ax]
    b_crop = b_gray[by, bx]
    if a_crop.size == 0 or b_crop.size == 0:
        return float(dx), float(dy), float(response), 0.0, 0.0

    overlap_ratio = float(a_crop.shape[0] * a_crop.shape[1]) / float(h * w)
    if a_crop.shape[0] < 32 or a_crop.shape[1] < 32:
        return float(dx), float(dy), float(response), overlap_ratio, 0.0

    aligned_ssim = ssim_same(a_crop, b_crop)
    return float(dx), float(dy), float(response), overlap_ratio, float(aligned_ssim)


def _segments_are_scroll_related(
    idx_a: int,
    idx_b: int,
    grays: List[np.ndarray],
    min_scroll_px: int = 10,
    aligned_ssim_thr: float = 0.90,
    overlap_ratio_thr: float = 0.65,
    response_thr: float = 0.05,
    min_ssim_gain: float = 0.05,
) -> bool:
    a_gray = grays[idx_a]
    b_gray = grays[idx_b]
    raw_ssim = ssim_same(a_gray, b_gray)
    dx, dy, response, overlap_ratio, aligned_ssim = _aligned_overlap_similarity(a_gray, b_gray)

    shift_ok = max(abs(dx), abs(dy)) >= float(min_scroll_px)
    gain_ok = aligned_ssim >= raw_ssim + min_ssim_gain

    return bool(
        shift_ok
        and response >= response_thr
        and overlap_ratio >= overlap_ratio_thr
        and aligned_ssim >= aligned_ssim_thr
        and gain_ok
    )


def _ink_mask_from_bgr(
    image_bgr: np.ndarray,
    gray_thr: int = 235,
    sat_thr: int = 35,
    max_val_thr: int = 250,
) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    sat = hsv[..., 1]
    val = hsv[..., 2]

    mask = ((gray < gray_thr) | ((sat > sat_thr) & (val < max_val_thr))).astype(np.uint8) * 255
    mask = cv2.medianBlur(mask, 3)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def _aligned_overlap_arrays(a: np.ndarray, b: np.ndarray, dx_i: int, dy_i: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
    h, w = a.shape[:2]
    x_slices = _overlap_slices(w, dx_i)
    y_slices = _overlap_slices(h, dy_i)
    if x_slices is None or y_slices is None:
        return None, None, 0.0

    ax, bx = x_slices
    ay, by = y_slices
    a_crop = a[ay, ax]
    b_crop = b[by, bx]
    if a_crop.size == 0 or b_crop.size == 0:
        return None, None, 0.0

    overlap_ratio = float(a_crop.shape[0] * a_crop.shape[1]) / float(h * w)
    return a_crop, b_crop, overlap_ratio


def _containment_metrics(prev_bgr: np.ndarray, curr_bgr: np.ndarray) -> Dict[str, float]:
    prev_gray = gray_for_similarity(prev_bgr)
    curr_gray = gray_for_similarity(curr_bgr)

    dx, dy, response, _, _ = _aligned_overlap_similarity(
        prev_gray,
        curr_gray,
        max_dx_ratio=0.06,
        max_dy_ratio=0.06,
    )
    dx_i = int(round(dx))
    dy_i = int(round(dy))

    prev_mask = _ink_mask_from_bgr(prev_bgr)
    curr_mask = _ink_mask_from_bgr(curr_bgr)
    prev_crop, curr_crop, overlap_ratio = _aligned_overlap_arrays(prev_mask, curr_mask, dx_i, dy_i)

    if prev_crop is None or curr_crop is None:
        return {
            "recall_prev_in_curr": 0.0,
            "precision_prev_in_curr": 0.0,
            "area_ratio": 0.0,
            "added_ratio": 0.0,
            "removed_ratio": 1.0,
            "overlap_ratio": 0.0,
            "shift_px": float(max(abs(dx), abs(dy))),
            "response": float(response),
        }

    prev_bin = prev_crop > 0
    curr_bin = curr_crop > 0
    prev_area = float(prev_bin.sum())
    curr_area = float(curr_bin.sum())
    if prev_area < 1.0 or curr_area < 1.0:
        return {
            "recall_prev_in_curr": 0.0,
            "precision_prev_in_curr": 0.0,
            "area_ratio": 0.0,
            "added_ratio": 0.0,
            "removed_ratio": 1.0,
            "overlap_ratio": overlap_ratio,
            "shift_px": float(max(abs(dx), abs(dy))),
            "response": float(response),
        }

    inter = float((prev_bin & curr_bin).sum())
    recall_prev_in_curr = inter / (prev_area + 1e-9)
    precision_prev_in_curr = inter / (curr_area + 1e-9)
    area_ratio = curr_area / (prev_area + 1e-9)
    added_ratio = max(0.0, curr_area - inter) / (prev_area + 1e-9)
    removed_ratio = max(0.0, prev_area - inter) / (prev_area + 1e-9)

    return {
        "recall_prev_in_curr": float(recall_prev_in_curr),
        "precision_prev_in_curr": float(precision_prev_in_curr),
        "area_ratio": float(area_ratio),
        "added_ratio": float(added_ratio),
        "removed_ratio": float(removed_ratio),
        "overlap_ratio": float(overlap_ratio),
        "shift_px": float(max(abs(dx), abs(dy))),
        "response": float(response),
    }


def _frame_contains_previous(
    prev_bgr: np.ndarray,
    curr_bgr: np.ndarray,
    containment_thr: float = 0.98,
    max_removed_ratio: float = 0.02,
    min_area_ratio: float = 0.995,
    max_shift_px: float = 10.0,
) -> Tuple[bool, Dict[str, float]]:
    metrics = _containment_metrics(prev_bgr, curr_bgr)
    is_contained = bool(
        metrics["recall_prev_in_curr"] >= containment_thr
        and metrics["removed_ratio"] <= max_removed_ratio
        and metrics["area_ratio"] >= min_area_ratio
        and metrics["shift_px"] <= max_shift_px
    )
    return is_contained, metrics


# -----------------------------------------------------------------------------
# Segment logic: simplified, close to sharpness.py
# -----------------------------------------------------------------------------
def _estimate_sample_step_sec(df: pd.DataFrame, sample_fps: float = 1.0) -> float:
    if len(df) >= 2:
        diffs = np.diff(df["time_sec"].astype(float).to_numpy())
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if diffs.size > 0:
            return float(np.median(diffs))
    return 1.0 / max(float(sample_fps), 1e-6)


def _choose_segment_anchor(df: pd.DataFrame, start_idx: int, end_idx: int) -> int:
    seg_rows = df.iloc[start_idx:end_idx + 1]
    ranked = seg_rows.sort_values(
        by=["sharp", "content_score", "text", "edge"],
        ascending=[False, False, False, False],
    )
    return int(ranked.index[0])


def mark_stable_segments(
    metrics_df: pd.DataFrame,
    thresholds: Optional[Thresholds] = None,
    sample_fps: float = 1.0,
    min_stable_sec: float = 1.0,
    min_stable_frames: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    del min_stable_sec, min_stable_frames

    if metrics_df.empty:
        empty = metrics_df.copy()
        empty["row_stable"] = []
        empty["state"] = []
        empty["segment_id"] = []
        empty["is_change"] = []
        return empty, empty.iloc[0:0].copy()

    df = metrics_df.reset_index(drop=True).copy()
    thr = thresholds or estimate_thresholds(df)
    step_sec = _estimate_sample_step_sec(df, sample_fps=sample_fps)

    is_change = np.zeros(len(df), dtype=bool)
    is_change[0] = True

    for i in range(1, len(df)):
        row = df.iloc[i]
        big_visual_jump = float(row["ssim_prev"]) <= thr.simple_change_ssim_thr
        force_jump = bool(
            float(row["phash_prev"]) >= max(thr.simple_change_phash_thr, thr.phash_thr * 1.10)
            or float(row["change_score"]) >= max(thr.simple_change_score_thr, thr.change_score_thr * 1.05)
        )
        is_change[i] = bool(big_visual_jump or force_jump)

    segment_id = np.cumsum(is_change.astype(np.int32)) - 1
    df["is_change"] = is_change.astype(int)
    df["row_stable"] = 1
    df["state"] = "stable"
    df["segment_id"] = segment_id.astype(int)
    df["is_stable_candidate"] = 1

    segments: List[Dict[str, object]] = []
    start = 0
    current_seg = int(segment_id[0])

    for i in range(1, len(df) + 1):
        is_new = i == len(df) or int(segment_id[i]) != current_seg
        if not is_new:
            continue

        end = i - 1
        anchor_idx = _choose_segment_anchor(df, start, end)
        t_start = float(df.iloc[start]["time_sec"])
        t_end = float(df.iloc[end]["time_sec"]) + step_sec

        segments.append(
            {
                "segment_id": current_seg,
                "start_idx": start,
                "end_idx": end,
                "t_start": t_start,
                "t_end": t_end,
                "leave_old_idx": end,
                "enter_new_idx": start,
                "t_leave_old": float(df.iloc[end]["time_sec"]),
                "t_enter_new": float(df.iloc[start]["time_sec"]),
                "t_boundary": t_start,
                "anchor_idx": anchor_idx,
            }
        )

        if i < len(df):
            start = i
            current_seg = int(segment_id[i])

    seg_df = pd.DataFrame(segments)
    return df, seg_df


# -----------------------------------------------------------------------------
# Post-filters: scroll families, contained variants, phash dedup
# -----------------------------------------------------------------------------
def _choose_family_representative(family: List[pd.Series], primary_min_duration_sec: float = 5.0) -> int:
    ordered = sorted(
        family,
        key=lambda row: (float(row["duration_sec"]), -float(row["t_start"])),
        reverse=True,
    )
    substantive = [row for row in family if float(row["duration_sec"]) >= float(primary_min_duration_sec)]
    if substantive:
        return int(sorted(substantive, key=lambda row: (float(row["t_start"]), -float(row["duration_sec"])))[0]["segment_id"])
    return int(ordered[0]["segment_id"])


def _collapse_scroll_segment_ids(marked_df: pd.DataFrame, segments_df: pd.DataFrame, primary_min_duration_sec: float = 5.0) -> List[int]:
    if segments_df.empty:
        return []

    grays, _ = _load_visual_cache(marked_df)
    seg_rows = [row for _, row in segments_df.sort_values("segment_id").iterrows()]

    families: List[List[pd.Series]] = []
    current_family: List[pd.Series] = [seg_rows[0]]

    for row in seg_rows[1:]:
        prev = current_family[-1]
        related = _segments_are_scroll_related(int(prev["anchor_idx"]), int(row["anchor_idx"]), grays)
        if related:
            current_family.append(row)
        else:
            families.append(current_family)
            current_family = [row]
    families.append(current_family)

    return [_choose_family_representative(family, primary_min_duration_sec=primary_min_duration_sec) for family in families]


def _collapse_contained_keyframes(
    selected_df: pd.DataFrame,
    max_gap_sec: float = 90.0,
    containment_thr: float = 0.98,
    max_removed_ratio: float = 0.02,
    min_area_ratio: float = 0.995,
) -> pd.DataFrame:
    if len(selected_df) <= 1:
        return selected_df.copy()

    work_df = selected_df.sort_values(["time_sec", "segment_id"]).reset_index(drop=True).copy()
    thumbs: List[Optional[np.ndarray]] = []
    for row in work_df.itertuples(index=False):
        thumbs.append(cv2.imread(str(row.thumb_path), cv2.IMREAD_COLOR))

    keep_indices: List[int] = [0]
    family_last = 0

    for i in range(1, len(work_df)):
        prev_thumb = thumbs[family_last]
        curr_thumb = thumbs[i]
        gap_sec = float(work_df.iloc[i]["time_sec"]) - float(work_df.iloc[family_last]["time_sec"])

        contained = False
        if prev_thumb is not None and curr_thumb is not None and gap_sec <= float(max_gap_sec):
            contained, _ = _frame_contains_previous(
                prev_thumb,
                curr_thumb,
                containment_thr=containment_thr,
                max_removed_ratio=max_removed_ratio,
                min_area_ratio=min_area_ratio,
            )

        if contained:
            family_last = i
            keep_indices[-1] = i
        else:
            keep_indices.append(i)
            family_last = i

    return work_df.iloc[sorted(set(keep_indices))].reset_index(drop=True)


def _dedup_selected_by_phash(selected_df: pd.DataFrame, max_hamming: int = 6) -> pd.DataFrame:
    if selected_df.empty or len(selected_df) <= 1:
        return selected_df.copy()

    work_df = selected_df.sort_values(["time_sec", "segment_id"]).reset_index(drop=True).copy()
    kept_rows: List[pd.Series] = []
    hashes: List[object] = []

    for _, row in work_df.iterrows():
        h = imagehash.hex_to_hash(str(row["phash_hex"]))
        is_dup = any(phash_distance(h, prev_h) <= int(max_hamming) for prev_h in hashes)
        if not is_dup:
            kept_rows.append(row)
            hashes.append(h)

    if not kept_rows:
        return work_df.iloc[0:0].copy()
    return pd.DataFrame(kept_rows).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Keyframe selection
# -----------------------------------------------------------------------------
def select_keyframes_from_segments(
    marked_df: pd.DataFrame,
    segments_df: pd.DataFrame,
    guard_sec: float = 0.5,
    min_segment_sec: float = 2.5,
    drop_scroll_variants: bool = True,
    primary_min_duration_sec: float = 5.0,
    drop_contained_variants: bool = True,
    containment_thr: float = 0.98,
    max_containment_gap_sec: float = 90.0,
    max_removed_ratio: float = 0.02,
    min_contained_area_ratio: float = 0.995,
    max_hamming: int = 6,
) -> pd.DataFrame:
    if marked_df.empty or segments_df.empty:
        return marked_df.iloc[0:0].copy()

    work_segments = segments_df.copy()
    work_segments["duration_sec"] = work_segments["t_end"].astype(float) - work_segments["t_start"].astype(float)

    if min_segment_sec > 0:
        work_segments = work_segments[work_segments["duration_sec"] >= float(min_segment_sec)].copy()

    if work_segments.empty:
        return marked_df.iloc[0:0].copy()

    if drop_scroll_variants and len(work_segments) > 1:
        keep_segment_ids = _collapse_scroll_segment_ids(
            marked_df=marked_df,
            segments_df=work_segments,
            primary_min_duration_sec=primary_min_duration_sec,
        )
        work_segments = work_segments[work_segments["segment_id"].isin(keep_segment_ids)].copy()

    chosen_rows = []
    for seg in work_segments.sort_values("segment_id").itertuples(index=False):
        seg_rows = marked_df[marked_df["segment_id"] == int(seg.segment_id)].copy()
        if seg_rows.empty:
            continue

        t0 = float(seg.t_start) + float(guard_sec)
        t1 = float(seg.t_end) - float(guard_sec)
        inner = seg_rows[(seg_rows["time_sec"] >= t0) & (seg_rows["time_sec"] <= t1)]
        if not inner.empty:
            seg_rows = inner

        ranked = seg_rows.sort_values(
            by=["sharp", "content_score", "text", "edge"],
            ascending=[False, False, False, False],
        )
        best = ranked.iloc[0].copy()
        best["segment_t_start"] = float(seg.t_start)
        best["segment_t_end"] = float(seg.t_end)
        best["segment_t_boundary"] = float(seg.t_boundary)
        best["segment_duration_sec"] = float(seg.duration_sec)
        chosen_rows.append(best)

    if not chosen_rows:
        return marked_df.iloc[0:0].copy()

    selected_df = pd.DataFrame(chosen_rows).reset_index(drop=True)

    if drop_contained_variants and len(selected_df) > 1:
        selected_df = _collapse_contained_keyframes(
            selected_df,
            max_gap_sec=max_containment_gap_sec,
            containment_thr=containment_thr,
            max_removed_ratio=max_removed_ratio,
            min_area_ratio=min_contained_area_ratio,
        )

    selected_df = _dedup_selected_by_phash(selected_df, max_hamming=max_hamming)
    return selected_df.reset_index(drop=True)


# -----------------------------------------------------------------------------
# Save selected frames
# -----------------------------------------------------------------------------
def save_keyframes(
    video_path: str,
    selected_df: pd.DataFrame,
    out_dir: str,
    roi: Optional[BBox] = None,
) -> List[KeyframeRecord]:
    out_path = Path(out_dir)
    frames_dir = out_path / "frames"
    roi_dir = out_path / "roi"
    frames_dir.mkdir(parents=True, exist_ok=True)
    roi_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    selected_map = {int(row.frame_idx): row for row in selected_df.itertuples(index=False)}

    records: List[KeyframeRecord] = []
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx not in selected_map:
            frame_idx += 1
            continue

        row = selected_map[frame_idx]
        if roi is None and int(getattr(row, "roi_x1", -1)) >= 0:
            roi_local = (
                int(row.roi_x1),
                int(row.roi_y1),
                int(row.roi_x2),
                int(row.roi_y2),
            )
        else:
            roi_local = roi

        roi_bgr = crop_roi(frame, roi_local)
        base = f"seg_{int(row.segment_id):04d}_f_{frame_idx:07d}_{float(row.time_sec):08.2f}s"
        frame_file = frames_dir / f"{base}.jpg"
        roi_file = roi_dir / f"{base}_roi.jpg"

        cv2.imwrite(str(frame_file), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        cv2.imwrite(str(roi_file), roi_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

        records.append(
            KeyframeRecord(
                segment_id=int(row.segment_id),
                frame_idx=frame_idx,
                time_sec=float(row.time_sec),
                content_score=float(row.content_score),
                image_path=str(frame_file),
                roi_path=str(roi_file),
            )
        )
        frame_idx += 1

    cap.release()
    return records


# -----------------------------------------------------------------------------
# Full pipeline
# -----------------------------------------------------------------------------
def extract_content_keyframes(
    video_path: str,
    work_dir: str,
    sample_fps: float = 1.0,
    roi: Optional[BBox] = None,
    auto_roi: bool = True,
    min_stable_sec: float = 1.0,
    min_stable_frames: Optional[int] = None,
    guard_sec: float = 0.5,
    min_segment_sec: float = 2.5,
    drop_scroll_variants: bool = True,
    primary_min_duration_sec: float = 5.0,
    drop_contained_variants: bool = True,
    containment_thr: float = 0.98,
    max_containment_gap_sec: float = 90.0,
    max_removed_ratio: float = 0.02,
    min_contained_area_ratio: float = 0.995,
    max_hamming: int = 6,
) -> Dict[str, object]:
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)

    metrics_csv = work / "frame_metrics.csv"
    thumbs_dir = work / "thumbs"
    print("1. Computing metrics")
    metrics_df = sample_video_metrics(
        video_path=video_path,
        out_csv=str(metrics_csv),
        sample_fps=sample_fps,
        roi=roi,
        auto_roi=auto_roi,
        thumbs_dir=str(thumbs_dir),
    )

    print("2. Estimating thresholds")
    thresholds = estimate_thresholds(metrics_df)

    print("3. Segmenting by visual change")
    marked_df, segments_df = mark_stable_segments(
        metrics_df,
        thresholds=thresholds,
        sample_fps=sample_fps,
        min_stable_sec=min_stable_sec,
        min_stable_frames=min_stable_frames,
    )

    marked_csv = work / "frame_metrics_marked.csv"
    segments_csv = work / "segments_manifest.csv"
    marked_df.to_csv(marked_csv, index=False, encoding="utf-8")
    segments_df.to_csv(segments_csv, index=False, encoding="utf-8")

    print("4. Selecting representative keyframes")
    selected_df = select_keyframes_from_segments(
        marked_df=marked_df,
        segments_df=segments_df,
        guard_sec=guard_sec,
        min_segment_sec=min_segment_sec,
        drop_scroll_variants=drop_scroll_variants,
        primary_min_duration_sec=primary_min_duration_sec,
        drop_contained_variants=drop_contained_variants,
        containment_thr=containment_thr,
        max_containment_gap_sec=max_containment_gap_sec,
        max_removed_ratio=max_removed_ratio,
        min_contained_area_ratio=min_contained_area_ratio,
        max_hamming=max_hamming,
    )
    selected_csv = work / "selected_keyframes.csv"
    selected_df.to_csv(selected_csv, index=False, encoding="utf-8")

    if roi is None and not metrics_df.empty and int(metrics_df.iloc[0]["roi_x1"]) >= 0:
        roi = (
            int(metrics_df.iloc[0]["roi_x1"]),
            int(metrics_df.iloc[0]["roi_y1"]),
            int(metrics_df.iloc[0]["roi_x2"]),
            int(metrics_df.iloc[0]["roi_y2"]),
        )

    print("5. Saving keyframes")
    saved = save_keyframes(
        video_path=video_path,
        selected_df=selected_df,
        out_dir=str(work / "keyframes"),
        roi=roi,
    )

    manifest = pd.DataFrame(
        [
            {
                "segment_id": r.segment_id,
                "frame_idx": r.frame_idx,
                "time_sec": r.time_sec,
                "content_score": r.content_score,
                "image_path": r.image_path,
                "roi_path": r.roi_path,
            }
            for r in saved
        ]
    )
    manifest_csv = work / "keyframe_manifest.csv"
    manifest.to_csv(manifest_csv, index=False, encoding="utf-8")

    return {
        "roi": roi,
        "metrics_csv": str(metrics_csv),
        "marked_csv": str(marked_csv),
        "segments_csv": str(segments_csv),
        "selected_csv": str(selected_csv),
        "manifest_csv": str(manifest_csv),
        "keyframes_dir": str(work / "keyframes"),
        "n_samples": int(len(metrics_df)),
        "n_segments": int(len(segments_df)),
        "n_keyframes": int(len(saved)),
        "thresholds": thresholds.__dict__,
    }


def _prepare_content_keyframes(
        self,
        video_path: Path,
        work_dir: Path,
        use_roi_images: bool = False,
    ) -> tuple[list[tuple[float, str]], list[Path]]:
        """
        Адаптер для нового content_selector.extract_content_keyframes(...).

        Возвращает данные в старом формате mainAction.py:
            keyframes  = [(time_sec, image_path), ...]
            frame_paths = [Path(image_path), ...]

        use_roi_images=False:
            сохраняем текущую логику mainAction.py:
            keypoints -> crop -> OCR -> multimodal summary.

        use_roi_images=True:
            берем уже готовые ROI-кропы из content_selector.
            В этом режиме этап keypoints/crop лучше отключать отдельно.
        """
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

        manifest_csv = work_dir / "keyframe_manifest.csv"

        if not manifest_csv.exists():
            self.print_signal.result.emit(
                "[keyframes] запуск content_selector.extract_content_keyframes"
            )

            result = extract_content_keyframes(
                video_path=str(video_path),
                work_dir=str(work_dir),
                sample_fps=1.0,
                min_stable_sec=1.0,
                guard_sec=0.5,
                min_segment_sec=2.5,
                drop_scroll_variants=True,
                primary_min_duration_sec=5.0,
                drop_contained_variants=True,
                containment_thr=0.97,
                max_containment_gap_sec=90.0,
                max_removed_ratio=0.03,
                min_contained_area_ratio=0.99,
            )
            manifest_csv = Path(result["manifest_csv"])
        else:
            self.print_signal.result.emit(
                f"[keyframes] найден готовый manifest: {manifest_csv}"
            )

        if not manifest_csv.exists():
            raise RuntimeError(f"Не найден manifest ключевых кадров: {manifest_csv}")

        path_field = "roi_path" if use_roi_images else "image_path"
        keyframes: list[tuple[float, str]] = []

        with manifest_csv.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row:
                    continue

                raw_path = row.get(path_field) or row.get("image_path")
                raw_time = row.get("time_sec")
                if not raw_path or raw_time is None:
                    continue

                frame_path = Path(raw_path)
                if not frame_path.exists():
                    self.print_signal.result.emit(
                        f"[keyframes] файл не найден: {frame_path}"
                    )
                    continue

                keyframes.append((float(raw_time), str(frame_path)))

        keyframes.sort(key=lambda item: item[0])
        frame_paths = [Path(path) for _, path in keyframes]

        if not frame_paths:
            raise RuntimeError("content_selector не вернул ни одного ключевого кадра")

        self.print_signal.result.emit(f"[keyframes] отобрано кадров: {len(frame_paths)}")
        return keyframes, frame_paths


def _prepare_keyframes(self, video_path: Path, frames_dir: Path) -> list[Path]:
    frames_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(
        p for p in frames_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )

    if existing:
        self.print_signal.result.emit(
            f"[keyframes] найдено готовых кадров: {len(existing)}"
        )
        return existing

    self.print_signal.result.emit("[keyframes] запуск select_keyframes")
    keyframes = select_keyframes(
        str(video_path),
        str(frames_dir),
        sample_fps=1.0,
    )

    frame_paths = [Path(p) for _, p in keyframes if p]
    self.print_signal.result.emit(
        f"[keyframes] сохранено кадров: {len(frame_paths)}"
    )
    return frame_paths



__all__ = [
    "Thresholds",
    "KeyframeRecord",
    "normalize_roi",
    "crop_roi",
    "detect_constant_roi",
    "compute_content_metrics",
    "sample_video_metrics",
    "estimate_thresholds",
    "mark_stable_segments",
    "select_keyframes_from_segments",
    "save_keyframes",
    "extract_content_keyframes",
]
