from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any, Dict, List, Optional, Tuple

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
    from debug_contours import find_page_rect  # type: ignore
except Exception:
    find_page_rect = None

try:
    from debug_contours import detect_slide_roi_from_video  # type: ignore
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


def _read_video_frame_at_sec(video_path: str, time_sec: float = 120.0) -> tuple[np.ndarray, float, int]:
    """
    Возвращает кадр видео около заданного времени.

    Если видео короче time_sec, берется середина видео.
    Возвращает:
        frame_bgr, actual_time_sec, frame_idx
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if total_frames > 0:
        duration_sec = total_frames / max(fps, 1e-9)
        if duration_sec <= 0:
            target_sec = 0.0
        elif duration_sec < float(time_sec):
            target_sec = duration_sec * 0.5
        else:
            target_sec = float(time_sec)

        frame_idx = int(round(target_sec * fps))
        frame_idx = max(0, min(total_frames - 1, frame_idx))
    else:
        frame_idx = max(0, int(round(float(time_sec) * fps)))

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()

    if not ok or frame is None:
        # запасной вариант: середина видео или первый кадр
        fallback_idx = total_frames // 2 if total_frames > 0 else 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, fallback_idx)
        ok, frame = cap.read()
        frame_idx = fallback_idx

    cap.release()

    if not ok or frame is None:
        raise RuntimeError(f"Cannot read frame from video: {video_path}")

    actual_time_sec = float(frame_idx / max(fps, 1e-9))
    return frame, actual_time_sec, frame_idx


def _resize_for_roi_window(frame_bgr: np.ndarray, max_side: int = 1280) -> tuple[np.ndarray, float]:
    """
    Уменьшает кадр только для удобного отображения.
    scale нужен, чтобы потом пересчитать ROI обратно в координаты исходного кадра.
    """
    h, w = frame_bgr.shape[:2]
    if max(h, w) <= max_side:
        return frame_bgr.copy(), 1.0

    scale = float(max_side) / float(max(h, w))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    preview = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return preview, scale


def load_roi_json(cache_path: str | Path, frame_bgr: Optional[np.ndarray] = None) -> Optional[BBox]:
    """
    Загружает ранее выбранный ROI из json.
    """
    path = Path(cache_path)
    if not path.exists():
        return None

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        roi_raw = data.get("roi")
        if not isinstance(roi_raw, (list, tuple)) or len(roi_raw) != 4:
            return None
        roi = tuple(map(int, roi_raw))  # type: ignore[assignment]
        if frame_bgr is not None:
            return normalize_roi(frame_bgr, roi)
        return roi  # type: ignore[return-value]
    except Exception:
        return None


def save_roi_json(
    cache_path: str | Path,
    roi: BBox,
    frame_bgr: np.ndarray,
    video_path: str,
    selected_time_sec: float,
    selected_frame_idx: int,
) -> None:
    """
    Сохраняет выбранный ROI, чтобы не выбирать его повторно для того же запуска.
    """
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    h, w = frame_bgr.shape[:2]
    data = {
        "video_path": str(video_path),
        "frame_width": int(w),
        "frame_height": int(h),
        "selected_time_sec": float(selected_time_sec),
        "selected_frame_idx": int(selected_frame_idx),
        "roi": [int(v) for v in roi],
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def select_manual_roi_from_video(
    video_path: str,
    time_sec: float = 120.0,
    cache_path: Optional[str | Path] = None,
    window_name: str = "Select slide ROI",
    max_display_side: int = 1280,
    force_reselect: bool = False,
) -> BBox:
    """
    Ручной выбор ROI один раз на видео.

    Управление в окне OpenCV:
        1. Выделить область мышью.
        2. Enter или Space — подтвердить.
        3. C или Esc — отменить.

    Возвращает ROI в координатах исходного кадра:
        (x1, y1, x2, y2), где x2/y2 — правая/нижняя граница для среза frame[y1:y2, x1:x2].
    """
    frame, actual_time_sec, frame_idx = _read_video_frame_at_sec(video_path, time_sec=time_sec)

    if cache_path is not None and not force_reselect:
        cached_roi = load_roi_json(cache_path, frame_bgr=frame)
        if cached_roi is not None:
            return cached_roi

    preview, scale = _resize_for_roi_window(frame, max_side=max_display_side)

    shown = preview.copy()
    cv2.putText(
        shown,
        "Select ROI: Enter/Space=OK, C/Esc=Cancel",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, shown)

    x, y, w, h = cv2.selectROI(
        window_name,
        shown,
        fromCenter=False,
        showCrosshair=True,
    )
    cv2.destroyWindow(window_name)

    if int(w) <= 5 or int(h) <= 5:
        raise RuntimeError("ROI не выбран: выделите область слайда и подтвердите Enter/Space")

    x1 = int(round(x / scale))
    y1 = int(round(y / scale))
    x2 = int(round((x + w) / scale))
    y2 = int(round((y + h) / scale))

    roi = normalize_roi(frame, (x1, y1, x2, y2))
    if roi is None:
        raise RuntimeError("Выбран некорректный ROI: область слишком маленькая или вне кадра")

    if cache_path is not None:
        save_roi_json(
            cache_path=cache_path,
            roi=roi,
            frame_bgr=frame,
            video_path=video_path,
            selected_time_sec=actual_time_sec,
            selected_frame_idx=frame_idx,
        )

    return roi


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


def prepare_metrics_for_variant(metrics_df: pd.DataFrame, use_phash: bool = True) -> pd.DataFrame:
    """
    Подготавливает копию таблицы метрик для конкретного варианта эксперимента.

    use_phash=False нужен для ablation study "без pHash":
    - phash_prev принудительно обнуляется;
    - change_score пересчитывается только по SSIM и изменениям простых метрик;
    - далее сегментация, дедупликация и выбор кадров выполняются без pHash.
    """
    df = metrics_df.copy()

    if use_phash:
        return df

    if "phash_prev_original" not in df.columns and "phash_prev" in df.columns:
        df["phash_prev_original"] = df["phash_prev"]

    df["phash_prev"] = 0.0

    ssim_drop = pd.to_numeric(df.get("ssim_drop", 0.0), errors="coerce").fillna(0.0)
    d_text = pd.to_numeric(df.get("d_text", 0.0), errors="coerce").fillna(0.0)
    d_edge = pd.to_numeric(df.get("d_edge", 0.0), errors="coerce").fillna(0.0)

    df["change_score"] = (
        0.75 * ssim_drop
        + 0.15 * np.minimum(1.0, d_text / 0.08)
        + 0.10 * np.minimum(1.0, d_edge / 0.08)
    ).astype(float)

    return df


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
    use_phash: bool = True,
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
        phash_jump = bool(
            use_phash
            and float(row["phash_prev"]) >= max(thr.simple_change_phash_thr, thr.phash_thr * 1.10)
        )
        score_jump = bool(
            float(row["change_score"]) >= max(thr.simple_change_score_thr, thr.change_score_thr * 1.05)
        )
        is_change[i] = bool(big_visual_jump or phash_jump or score_jump)

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
    use_phash: bool = True,
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

    if use_phash:
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

def merge_short_segments(
    marked_df: pd.DataFrame,
    segments_df: pd.DataFrame,
    min_semantic_segment_sec: float = 10.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Склеивает слишком короткие визуальные сегменты.

    Смысл:
    раздел лекции короче 10-30 секунд обычно является шумом:
    мигание, переход, движение мыши, прокрутка, смена яркости и т.п.
    """

    if marked_df.empty or segments_df.empty:
        return marked_df, segments_df

    df = marked_df.copy().reset_index(drop=True)
    segs = segments_df.copy().reset_index(drop=True)

    segs["duration_sec"] = (
        segs["t_end"].astype(float) - segs["t_start"].astype(float)
    )

    # Карта старый segment_id -> новый semantic segment_id
    old_to_new = {}
    current_new_id = 0

    for i, seg in segs.iterrows():
        seg_id = int(seg["segment_id"])
        duration = float(seg["duration_sec"])

        if i == 0:
            old_to_new[seg_id] = current_new_id
            continue

        if duration < min_semantic_segment_sec:
            # короткий сегмент считаем шумом и клеим к предыдущему
            old_to_new[seg_id] = current_new_id
        else:
            current_new_id += 1
            old_to_new[seg_id] = current_new_id

    df["segment_id"] = df["segment_id"].map(old_to_new).astype(int)

    # Пересобираем segments_df уже по новым segment_id
    new_segments = []

    step_sec = _estimate_sample_step_sec(df)

    for new_id, group in df.groupby("segment_id", sort=True):
        start_idx = int(group.index.min())
        end_idx = int(group.index.max())

        t_start = float(group["time_sec"].iloc[0])
        t_end = float(group["time_sec"].iloc[-1]) + step_sec

        anchor_idx = int(
            group.sort_values(
                by=["sharp", "content_score", "text", "edge"],
                ascending=[False, False, False, False],
            ).index[0]
        )

        new_segments.append(
            {
                "segment_id": int(new_id),
                "start_idx": start_idx,
                "end_idx": end_idx,
                "t_start": t_start,
                "t_end": t_end,
                "t_boundary": t_start,
                "anchor_idx": anchor_idx,
                "duration_sec": t_end - t_start,
            }
        )

    new_segments_df = pd.DataFrame(new_segments)

    return df, new_segments_df
# -----------------------------------------------------------------------------
# Full pipeline
# -----------------------------------------------------------------------------
def _infer_roi_from_metrics(metrics_df: pd.DataFrame) -> Optional[BBox]:
    if metrics_df.empty:
        return None

    first = metrics_df.iloc[0]
    try:
        if int(first.get("roi_x1", -1)) < 0:
            return None
        return (
            int(first["roi_x1"]),
            int(first["roi_y1"]),
            int(first["roi_x2"]),
            int(first["roi_y2"]),
        )
    except Exception:
        return None


def _safe_variant_name(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(name).strip())
    safe = "_".join(part for part in safe.split("_") if part)
    return safe or "variant"


def _write_variant_config(path: Path, config: Dict[str, Any]) -> None:
    cleaned: Dict[str, Any] = {}
    for key, value in config.items():
        if isinstance(value, Path):
            cleaned[key] = str(value)
        elif isinstance(value, tuple):
            cleaned[key] = list(value)
        else:
            cleaned[key] = value
    path.write_text(json.dumps(cleaned, ensure_ascii=False, indent=2), encoding="utf-8")


def run_keyframe_variant_from_metrics(
    video_path: str,
    metrics_df: pd.DataFrame,
    work_dir: str,
    sample_fps: float = 1.0,
    roi: Optional[BBox] = None,
    min_stable_sec: float = 1.0,
    min_stable_frames: Optional[int] = None,
    guard_sec: float = 0.5,
    min_segment_sec: float = 2.5,
    min_semantic_segment_sec: float = 20.0,
    drop_scroll_variants: bool = True,
    primary_min_duration_sec: float = 5.0,
    drop_contained_variants: bool = True,
    containment_thr: float = 0.98,
    max_containment_gap_sec: float = 90.0,
    max_removed_ratio: float = 0.02,
    min_contained_area_ratio: float = 0.995,
    max_hamming: int = 6,
    use_phash: bool = True,
    variant_name: str = "variant",
) -> Dict[str, object]:
    """
    Запускает этапы сегментации и выбора ключевых кадров на уже посчитанных метриках.

    Это удобно для исследования: метрики считаются один раз, а затем можно менять
    фильтры и пороги без повторного прохода по видео.
    """
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)

    config = {
        "variant_name": variant_name,
        "sample_fps": sample_fps,
        "roi": roi,
        "min_stable_sec": min_stable_sec,
        "min_stable_frames": min_stable_frames,
        "guard_sec": guard_sec,
        "min_segment_sec": min_segment_sec,
        "min_semantic_segment_sec": min_semantic_segment_sec,
        "drop_scroll_variants": drop_scroll_variants,
        "primary_min_duration_sec": primary_min_duration_sec,
        "drop_contained_variants": drop_contained_variants,
        "containment_thr": containment_thr,
        "max_containment_gap_sec": max_containment_gap_sec,
        "max_removed_ratio": max_removed_ratio,
        "min_contained_area_ratio": min_contained_area_ratio,
        "max_hamming": max_hamming,
        "use_phash": use_phash,
    }
    _write_variant_config(work / "variant_config.json", config)

    variant_metrics_df = prepare_metrics_for_variant(metrics_df, use_phash=use_phash)

    thresholds = estimate_thresholds(variant_metrics_df)

    marked_df, visual_segments_df = mark_stable_segments(
        variant_metrics_df,
        thresholds=thresholds,
        sample_fps=sample_fps,
        min_stable_sec=min_stable_sec,
        min_stable_frames=min_stable_frames,
        use_phash=use_phash,
    )

    n_visual_segments = int(len(visual_segments_df))

    if float(min_semantic_segment_sec) > 0:
        marked_df, segments_df = merge_short_segments(
            marked_df,
            visual_segments_df,
            min_semantic_segment_sec=float(min_semantic_segment_sec),
        )
    else:
        segments_df = visual_segments_df.copy()
        if not segments_df.empty and "duration_sec" not in segments_df.columns:
            segments_df["duration_sec"] = (
                segments_df["t_end"].astype(float) - segments_df["t_start"].astype(float)
            )

    marked_csv = work / "frame_metrics_marked.csv"
    segments_csv = work / "segments_manifest.csv"
    marked_df.to_csv(marked_csv, index=False, encoding="utf-8")
    segments_df.to_csv(segments_csv, index=False, encoding="utf-8")

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
        use_phash=use_phash,
    )

    selected_csv = work / "selected_keyframes.csv"
    selected_df.to_csv(selected_csv, index=False, encoding="utf-8")

    if roi is None:
        roi = _infer_roi_from_metrics(metrics_df)

    saved = save_keyframes(
        video_path=video_path,
        selected_df=selected_df,
        out_dir=str(work / "keyframes"),
        roi=roi,
    )

    manifest = pd.DataFrame(
        [
            {
                "variant_name": variant_name,
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

    result: Dict[str, object] = {
        "variant_name": variant_name,
        "work_dir": str(work),
        "roi": roi,
        "marked_csv": str(marked_csv),
        "segments_csv": str(segments_csv),
        "selected_csv": str(selected_csv),
        "manifest_csv": str(manifest_csv),
        "keyframes_dir": str(work / "keyframes"),
        "n_samples": int(len(metrics_df)),
        "n_visual_segments_before_semantic": n_visual_segments,
        "n_segments": int(len(segments_df)),
        "n_keyframes": int(len(saved)),
        "min_semantic_segment_sec": float(min_semantic_segment_sec),
        "min_segment_sec": float(min_segment_sec),
        "drop_scroll_variants": bool(drop_scroll_variants),
        "drop_contained_variants": bool(drop_contained_variants),
        "use_phash": bool(use_phash),
        "thresholds": thresholds.__dict__,
    }

    (work / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return result


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
    min_semantic_segment_sec: float = 20.0,
    drop_scroll_variants: bool = True,
    primary_min_duration_sec: float = 5.0,
    drop_contained_variants: bool = True,
    containment_thr: float = 0.98,
    max_containment_gap_sec: float = 90.0,
    max_removed_ratio: float = 0.02,
    min_contained_area_ratio: float = 0.995,
    max_hamming: int = 6,
    use_phash: bool = True,
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

    if roi is None:
        roi = _infer_roi_from_metrics(metrics_df)

    print("2. Segmenting and selecting keyframes")
    result = run_keyframe_variant_from_metrics(
        video_path=video_path,
        metrics_df=metrics_df,
        work_dir=str(work),
        sample_fps=sample_fps,
        roi=roi,
        min_stable_sec=min_stable_sec,
        min_stable_frames=min_stable_frames,
        guard_sec=guard_sec,
        min_segment_sec=min_segment_sec,
        min_semantic_segment_sec=min_semantic_segment_sec,
        drop_scroll_variants=drop_scroll_variants,
        primary_min_duration_sec=primary_min_duration_sec,
        drop_contained_variants=drop_contained_variants,
        containment_thr=containment_thr,
        max_containment_gap_sec=max_containment_gap_sec,
        max_removed_ratio=max_removed_ratio,
        min_contained_area_ratio=min_contained_area_ratio,
        max_hamming=max_hamming,
        use_phash=use_phash,
        variant_name="single_run",
    )

    result["metrics_csv"] = str(metrics_csv)
    return result


def build_default_ablation_plan(
    baseline_min_semantic_segment_sec: float = 20.0,
    semantic_values: Tuple[float, ...] = (5.0, 10.0, 20.0, 30.0),
) -> List[Dict[str, Any]]:
    """
    План экспериментов для статьи/презентации.

    Варианты:
    - baseline: полный алгоритм;
    - no_scroll_family_filter: без объединения сегментов, связанных прокруткой;
    - no_contained_filter: без удаления вложенных/дописанных вариантов;
    - no_scroll_filters: без обеих фильтраций прокрутки;
    - no_phash: pHash не участвует в change_score, порогах и дедупликации;
    - no_semantic_filter: короткие смысловые сегменты не склеиваются;
    - no_roi: анализируется весь кадр, а не область слайда/доски;
    - semantic_Xs: перебор min_semantic_segment_sec.
    """
    plan: List[Dict[str, Any]] = [
        {
            "variant_name": "baseline",
            "metrics_source": "roi",
            "drop_scroll_variants": True,
            "drop_contained_variants": True,
            "use_phash": True,
            "min_semantic_segment_sec": float(baseline_min_semantic_segment_sec),
        },
        {
            "variant_name": "no_scroll_family_filter",
            "metrics_source": "roi",
            "drop_scroll_variants": False,
            "drop_contained_variants": True,
            "use_phash": True,
            "min_semantic_segment_sec": float(baseline_min_semantic_segment_sec),
        },
        {
            "variant_name": "no_contained_filter",
            "metrics_source": "roi",
            "drop_scroll_variants": True,
            "drop_contained_variants": False,
            "use_phash": True,
            "min_semantic_segment_sec": float(baseline_min_semantic_segment_sec),
        },
        {
            "variant_name": "no_scroll_filters",
            "metrics_source": "roi",
            "drop_scroll_variants": False,
            "drop_contained_variants": False,
            "use_phash": True,
            "min_semantic_segment_sec": float(baseline_min_semantic_segment_sec),
        },
        {
            "variant_name": "no_phash",
            "metrics_source": "roi",
            "drop_scroll_variants": True,
            "drop_contained_variants": True,
            "use_phash": False,
            "min_semantic_segment_sec": float(baseline_min_semantic_segment_sec),
        },
        {
            "variant_name": "no_semantic_filter",
            "metrics_source": "roi",
            "drop_scroll_variants": True,
            "drop_contained_variants": True,
            "use_phash": True,
            "min_semantic_segment_sec": 0.0,
        },
        {
            "variant_name": "no_roi",
            "metrics_source": "full_frame",
            "drop_scroll_variants": True,
            "drop_contained_variants": True,
            "use_phash": True,
            "min_semantic_segment_sec": float(baseline_min_semantic_segment_sec),
        },
    ]

    existing_names = {item["variant_name"] for item in plan}
    for value in semantic_values:
        name = f"semantic_{int(value) if float(value).is_integer() else value}s"
        if name in existing_names:
            continue
        plan.append(
            {
                "variant_name": name,
                "metrics_source": "roi",
                "drop_scroll_variants": True,
                "drop_contained_variants": True,
                "use_phash": True,
                "min_semantic_segment_sec": float(value),
            }
        )
    return plan


def _load_or_compute_metrics(
    video_path: str,
    metrics_dir: Path,
    sample_fps: float,
    roi: Optional[BBox],
    auto_roi: bool,
) -> pd.DataFrame:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = metrics_dir / "frame_metrics.csv"
    thumbs_dir = metrics_dir / "thumbs"

    if metrics_csv.exists():
        return pd.read_csv(metrics_csv)

    return sample_video_metrics(
        video_path=video_path,
        out_csv=str(metrics_csv),
        sample_fps=sample_fps,
        roi=roi,
        auto_roi=auto_roi,
        thumbs_dir=str(thumbs_dir),
    )


def _make_manual_evaluation_template(
    results: List[Dict[str, object]],
    out_csv: Path,
) -> None:
    """
    Создает CSV для ручной оценки.

    Важно:
    - строки row_type="selected_frame" размечаются как TP/FP через is_true_positive;
    - строка row_type="variant_summary" добавляется один раз на каждый вариант,
      даже если вариант не выбрал ни одного кадра;
    - в variant_summary нужно вручную заполнить fn_count — сколько нужных
      ключевых кадров/состояний алгоритм пропустил.
    """
    rows: List[Dict[str, object]] = []

    for result in results:
        variant_name = str(result.get("variant_name", ""))

        # Служебная строка для FN по варианту. Благодаря ей FN можно указать
        # даже тогда, когда алгоритм не выбрал ни одного кадра.
        rows.append(
            {
                "variant_name": variant_name,
                "row_type": "variant_summary",
                "time_sec": "",
                "frame_idx": "",
                "segment_id": "",
                "image_path": "",
                "roi_path": "",
                "is_true_positive": "",
                "fn_count": "",
                "error_type": "",
                "comment": "Заполните fn_count: сколько нужных ключевых кадров/состояний пропущено этим вариантом",
            }
        )

        manifest_path = Path(str(result.get("manifest_csv", "")))
        if not manifest_path.exists():
            continue

        manifest = pd.read_csv(manifest_path)
        for row in manifest.itertuples(index=False):
            rows.append(
                {
                    "variant_name": getattr(row, "variant_name", variant_name),
                    "row_type": "selected_frame",
                    "time_sec": float(getattr(row, "time_sec")),
                    "frame_idx": int(getattr(row, "frame_idx")),
                    "segment_id": int(getattr(row, "segment_id")),
                    "image_path": getattr(row, "image_path"),
                    "roi_path": getattr(row, "roi_path"),
                    "is_true_positive": "",
                    "fn_count": "",
                    "error_type": "",
                    "comment": "",
                }
            )

    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8")



def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_Нет данных_"

    columns = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]

    for row in df.itertuples(index=False):
        values = []
        for value in row:
            text = str(value)
            text = text.replace("|", "\\|").replace("\n", " ")
            values.append(text)
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def _write_ablation_report(summary_df: pd.DataFrame, out_md: Path) -> None:
    if summary_df.empty:
        out_md.write_text("# Ablation study\n\nНет результатов.\n", encoding="utf-8")
        return

    report_df = summary_df.copy()
    baseline = report_df[report_df["variant_name"] == "baseline"]
    if not baseline.empty:
        base_keyframes = int(baseline.iloc[0]["n_keyframes"])
        base_segments = int(baseline.iloc[0]["n_segments"])
        report_df["delta_keyframes_vs_baseline"] = report_df["n_keyframes"].astype(int) - base_keyframes
        report_df["delta_segments_vs_baseline"] = report_df["n_segments"].astype(int) - base_segments

    cols = [
        "variant_name",
        "metrics_source",
        "min_semantic_segment_sec",
        "use_phash",
        "drop_scroll_variants",
        "drop_contained_variants",
        "n_visual_segments_before_semantic",
        "n_segments",
        "n_keyframes",
    ]
    cols = [c for c in cols if c in report_df.columns]

    md = [
        "# Ablation study отбора ключевых кадров",
        "",
        "## Что сравнивается",
        "",
        "- `baseline` — полный вариант алгоритма.",
        "- `no_scroll_family_filter` — отключено объединение сегментов, связанных прокруткой.",
        "- `no_contained_filter` — отключено удаление вложенных/дописанных вариантов.",
        "- `no_scroll_filters` — отключены обе фильтрации прокрутки.",
        "- `no_phash` — pHash не используется в сегментации и дедупликации.",
        "- `no_semantic_filter` — отключено склеивание слишком коротких смысловых сегментов.",
        "- `no_roi` — анализируется весь кадр.",
        "- `semantic_Xs` — перебор `min_semantic_segment_sec`.",
        "",
        "## Сводная таблица",
        "",
        _dataframe_to_markdown(report_df[cols]),
        "",
        "## Как оценивать качество вручную",
        "",
        "Откройте `manual_evaluation_template.csv` и заполните:",
        "",
        "- в строках `row_type = selected_frame`: `is_true_positive = 1`, если кадр действительно является полезным ключевым кадром;",
        "- в строках `row_type = selected_frame`: `is_true_positive = 0`, если это ложное срабатывание;",
        "- в строке `row_type = variant_summary`: `fn_count` — сколько нужных ключевых кадров/состояний этот вариант пропустил;",
        "- `error_type` — например: `scroll`, `cursor`, `teacher`, `transition`, `duplicate`, `blur`.",
        "",
        "После разметки можно посчитать `precision = TP / (TP + FP)`, `recall = TP / (TP + FN)` и `F1` по каждому варианту.",
    ]

    out_md.write_text("\n".join(md), encoding="utf-8")


def run_keyframe_ablation_study(
    video_path: str,
    work_dir: str,
    sample_fps: float = 1.0,
    roi: Optional[BBox] = None,
    auto_roi: bool = True,
    min_stable_sec: float = 1.0,
    min_stable_frames: Optional[int] = None,
    guard_sec: float = 0.5,
    min_segment_sec: float = 2.5,
    baseline_min_semantic_segment_sec: float = 20.0,
    semantic_values: Tuple[float, ...] = (0.0, 5.0, 10.0, 20.0, 30.0),
    primary_min_duration_sec: float = 5.0,
    containment_thr: float = 0.98,
    max_containment_gap_sec: float = 90.0,
    max_removed_ratio: float = 0.02,
    min_contained_area_ratio: float = 0.995,
    max_hamming: int = 6,
    variants: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, object]:
    """
    Запускает серию экспериментов для исследования влияния отдельных компонентов.

    На выходе:
    - ablation_summary.csv — сводка по числу сегментов/кадров;
    - ablation_report.md — краткий отчет для статьи/презентации;
    - manual_evaluation_template.csv — таблица для ручной разметки TP/FP;
    - отдельная папка на каждый вариант с кадрами и manifest.
    """
    root = Path(work_dir)
    root.mkdir(parents=True, exist_ok=True)

    plan = variants or build_default_ablation_plan(
        baseline_min_semantic_segment_sec=baseline_min_semantic_segment_sec,
        semantic_values=semantic_values,
    )

    metrics_roi = _load_or_compute_metrics(
        video_path=video_path,
        metrics_dir=root / "_metrics_roi",
        sample_fps=sample_fps,
        roi=roi,
        auto_roi=auto_roi,
    )

    roi_for_saving = roi or _infer_roi_from_metrics(metrics_roi)

    metrics_full_frame: Optional[pd.DataFrame] = None

    results: List[Dict[str, object]] = []

    for item in plan:
        variant_name = _safe_variant_name(str(item.get("variant_name", "variant")))
        metrics_source = str(item.get("metrics_source", "roi"))

        if metrics_source == "full_frame":
            if metrics_full_frame is None:
                metrics_full_frame = _load_or_compute_metrics(
                    video_path=video_path,
                    metrics_dir=root / "_metrics_full_frame",
                    sample_fps=sample_fps,
                    roi=None,
                    auto_roi=False,
                )
            metrics_df = metrics_full_frame
            variant_roi = None
        else:
            metrics_df = metrics_roi
            variant_roi = roi_for_saving

        print(f"[ablation] {variant_name}")

        result = run_keyframe_variant_from_metrics(
            video_path=video_path,
            metrics_df=metrics_df,
            work_dir=str(root / variant_name),
            sample_fps=sample_fps,
            roi=variant_roi,
            min_stable_sec=min_stable_sec,
            min_stable_frames=min_stable_frames,
            guard_sec=guard_sec,
            min_segment_sec=float(item.get("min_segment_sec", min_segment_sec)),
            min_semantic_segment_sec=float(
                item.get("min_semantic_segment_sec", baseline_min_semantic_segment_sec)
            ),
            drop_scroll_variants=bool(item.get("drop_scroll_variants", True)),
            primary_min_duration_sec=float(
                item.get("primary_min_duration_sec", primary_min_duration_sec)
            ),
            drop_contained_variants=bool(item.get("drop_contained_variants", True)),
            containment_thr=float(item.get("containment_thr", containment_thr)),
            max_containment_gap_sec=float(
                item.get("max_containment_gap_sec", max_containment_gap_sec)
            ),
            max_removed_ratio=float(item.get("max_removed_ratio", max_removed_ratio)),
            min_contained_area_ratio=float(
                item.get("min_contained_area_ratio", min_contained_area_ratio)
            ),
            max_hamming=int(item.get("max_hamming", max_hamming)),
            use_phash=bool(item.get("use_phash", True)),
            variant_name=variant_name,
        )
        result["metrics_source"] = metrics_source
        results.append(result)

    summary_rows = []
    for result in results:
        row = {
            "variant_name": result["variant_name"],
            "metrics_source": result.get("metrics_source", "roi"),
            "min_semantic_segment_sec": result["min_semantic_segment_sec"],
            "min_segment_sec": result["min_segment_sec"],
            "use_phash": result["use_phash"],
            "drop_scroll_variants": result["drop_scroll_variants"],
            "drop_contained_variants": result["drop_contained_variants"],
            "n_samples": result["n_samples"],
            "n_visual_segments_before_semantic": result["n_visual_segments_before_semantic"],
            "n_segments": result["n_segments"],
            "n_keyframes": result["n_keyframes"],
            "work_dir": result["work_dir"],
            "manifest_csv": result["manifest_csv"],
        }
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    if not summary_df.empty and (summary_df["variant_name"] == "baseline").any():
        baseline = summary_df[summary_df["variant_name"] == "baseline"].iloc[0]
        summary_df["delta_segments_vs_baseline"] = (
            summary_df["n_segments"].astype(int) - int(baseline["n_segments"])
        )
        summary_df["delta_keyframes_vs_baseline"] = (
            summary_df["n_keyframes"].astype(int) - int(baseline["n_keyframes"])
        )

    summary_csv = root / "ablation_summary.csv"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8")

    manual_eval_csv = root / "manual_evaluation_template.csv"
    _make_manual_evaluation_template(results, manual_eval_csv)

    report_md = root / "ablation_report.md"
    _write_ablation_report(summary_df, report_md)

    return {
        "work_dir": str(root),
        "summary_csv": str(summary_csv),
        "manual_evaluation_csv": str(manual_eval_csv),
        "report_md": str(report_md),
        "results": results,
    }


def _read_manual_evaluation_csv(evaluation_csv: str) -> pd.DataFrame:
    """
    Читает CSV после Excel максимально устойчиво:
    - UTF-8 with BOM;
    - запятая / точка с запятой / tab;
    - лишние пробелы в названиях колонок;
    - BOM в первом заголовке.
    """
    path = Path(evaluation_csv)
    if not path.exists():
        raise FileNotFoundError(f"Не найден файл разметки: {evaluation_csv}")

    last_error: Optional[Exception] = None
    candidates = [
        {"encoding": "utf-8-sig", "sep": None, "engine": "python"},
        {"encoding": "utf-8-sig", "sep": ","},
        {"encoding": "utf-8-sig", "sep": ";"},
        {"encoding": "utf-8-sig", "sep": "\t"},
        {"encoding": "utf-8", "sep": None, "engine": "python"},
        {"encoding": "cp1251", "sep": None, "engine": "python"},
    ]

    required = {"variant_name", "is_true_positive"}

    for kwargs in candidates:
        try:
            df = pd.read_csv(path, **kwargs)
        except Exception as exc:
            last_error = exc
            continue

        df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]

        # Иногда Excel/редактор добавляет полностью пустые служебные колонки.
        unnamed_cols = [c for c in df.columns if c.lower().startswith("unnamed")]
        if unnamed_cols:
            df = df.drop(columns=unnamed_cols)

        if required.issubset(set(df.columns)):
            return df

    # Диагностика: читаем хотя бы как текст и показываем первую строку.
    try:
        first_line = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()[0]
    except Exception:
        first_line = "<не удалось прочитать первую строку>"

    try:
        debug_df = pd.read_csv(path, encoding="utf-8-sig", sep=None, engine="python", nrows=2)
        debug_columns = [str(c).replace("\ufeff", "").strip() for c in debug_df.columns]
    except Exception:
        debug_columns = []

    raise ValueError(
        "В evaluation_csv должны быть колонки variant_name и is_true_positive.\n"
        f"Файл: {evaluation_csv}\n"
        f"Первая строка файла: {first_line}\n"
        f"Распознанные колонки: {debug_columns}\n"
        "Проверьте, что это именно manual_evaluation_template.csv, "
        "а не ablation_summary.csv / precision_summary.csv, и что первая строка содержит заголовки."
    ) from last_error


def _parse_binary_label(value: object) -> Optional[int]:
    """
    Приводит ручную метку к 0/1.
    Поддерживает значения, которые часто появляются после Excel:
    1, 0, 1.0, 0.0, TRUE/FALSE, да/нет, tp/fp.
    """
    if value is None or pd.isna(value):
        return None

    s = str(value).strip().lower()
    if s == "":
        return None

    positive = {"1", "1.0", "true", "yes", "y", "да", "истина", "tp", "тп"}
    negative = {"0", "0.0", "false", "no", "n", "нет", "ложь", "fp", "фп"}

    if s in positive:
        return 1
    if s in negative:
        return 0

    # На случай, если Excel сохранил десятичную запятую.
    s = s.replace(",", ".")
    try:
        x = float(s)
    except ValueError:
        return None

    if x == 1.0:
        return 1
    if x == 0.0:
        return 0
    return None


def summarize_manual_evaluation(
    evaluation_csv: str,
    out_csv: Optional[str] = None,
) -> pd.DataFrame:
    """
    Считает качество вариантов после ручной разметки manual_evaluation_template.csv.

    В строках row_type="selected_frame" нужно поставить:
        is_true_positive = 1 — корректный ключевой кадр;
        is_true_positive = 0 — ложное срабатывание.

    В строке row_type="variant_summary" для каждого варианта нужно поставить:
        fn_count — сколько нужных ключевых кадров/состояний алгоритм пропустил.

    Возвращает таблицу:
        variant_name, selected, labeled, TP, FP, FN,
        precision_percent, recall_percent, f1_percent.
    """
    df = _read_manual_evaluation_csv(evaluation_csv)
    work = df.copy()

    # Совместимость со старыми CSV, где еще не было row_type/fn_count.
    if "row_type" not in work.columns:
        work["row_type"] = "selected_frame"
    if "fn_count" not in work.columns:
        work["fn_count"] = 0

    # Нормализуем тип строки после Excel.
    work["row_type"] = work["row_type"].fillna("selected_frame").astype(str).str.strip()
    work.loc[work["row_type"] == "", "row_type"] = "selected_frame"

    # Метки TP/FP устойчиво парсим из 1/0, TRUE/FALSE, да/нет и т.п.
    work["label"] = work["is_true_positive"].map(_parse_binary_label)

    # FN — число уровня варианта. Excel может оставить пустые ячейки или записать числа текстом.
    work["fn_count_numeric"] = pd.to_numeric(
        work["fn_count"].astype(str).str.replace(",", ".", regex=False),
        errors="coerce",
    )

    rows: List[Dict[str, object]] = []
    for variant_name, group in work.groupby("variant_name", sort=True, dropna=False):
        variant_name_str = str(variant_name).strip()
        if not variant_name_str or variant_name_str.lower() == "nan":
            continue

        selected_rows = group[group["row_type"] != "variant_summary"]
        labeled = selected_rows[selected_rows["label"].isin([0, 1])]

        tp = int((labeled["label"] == 1).sum())
        fp = int((labeled["label"] == 0).sum())

        summary_rows = group[group["row_type"] == "variant_summary"]
        if summary_rows.empty:
            fn_values = group["fn_count_numeric"].dropna()
        else:
            fn_values = summary_rows["fn_count_numeric"].dropna()

        # FN — показатель уровня варианта, а не отдельного кадра.
        # Берем максимум, чтобы случайное копирование FN в несколько строк не завышало результат.
        fn = int(round(float(fn_values.max()))) if not fn_values.empty else 0
        fn = max(0, fn)

        selected = int(len(selected_rows))
        labeled_count = int(len(labeled))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        rows.append(
            {
                "variant_name": variant_name_str,
                "selected": selected,
                "labeled": labeled_count,
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "precision_percent": round(100.0 * precision, 2),
                "recall_percent": round(100.0 * recall, 2),
                "f1_percent": round(100.0 * f1, 2),
            }
        )

    summary_df = pd.DataFrame(rows)

    if out_csv is not None:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    return summary_df


__all__ = [
    "Thresholds",
    "KeyframeRecord",
    "normalize_roi",
    "crop_roi",
    "load_roi_json",
    "save_roi_json",
    "select_manual_roi_from_video",
    "detect_constant_roi",
    "compute_content_metrics",
    "sample_video_metrics",
    "estimate_thresholds",
    "mark_stable_segments",
    "select_keyframes_from_segments",
    "save_keyframes",
    "prepare_metrics_for_variant",
    "run_keyframe_variant_from_metrics",
    "build_default_ablation_plan",
    "run_keyframe_ablation_study",
    "summarize_manual_evaluation",
    "extract_content_keyframes",
]
