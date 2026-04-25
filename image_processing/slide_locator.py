from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

import cv2
import numpy as np


PointArray = np.ndarray  # shape (4, 2), order: tl, tr, br, bl


@dataclass
class SlideDetection:
    found: bool
    mode_used: str
    keypoints: Optional[PointArray]
    bbox: Optional[Tuple[int, int, int, int]]
    confidence: float
    metrics: Dict[str, float] = field(default_factory=dict)
    debug: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VideoComplexityReport:
    video_path: str
    sampled_frames: int
    reliable_frames: int
    success_rate: float
    median_confidence: float
    mean_confidence: float
    temporal_jitter: float
    recommended_mode: str  # classic | hybrid | neural
    frame_reports: List[SlideDetection] = field(default_factory=list)
    thresholds: Dict[str, float] = field(default_factory=dict)


class NeuralSlideDetector(Protocol):
    """
    Интерфейс для внешнего нейросетевого детектора.

    predict(frame_bgr) должен вернуть словарь одного из видов:
    1) {"keypoints": np.ndarray shape (4,2), "confidence": float}
    2) {"bbox": (x1, y1, x2, y2), "confidence": float}
    """

    def predict(self, frame_bgr: np.ndarray) -> Dict[str, Any]:
        ...


# =========================================================
# Геометрия
# =========================================================

def order_points_clockwise(pts: np.ndarray) -> PointArray:
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] != 4:
        raise ValueError("Expected exactly 4 points")

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    return np.array([tl, tr, br, bl], dtype=np.float32)


def clip_quad_to_frame(pts: PointArray, frame_shape: Tuple[int, int, int]) -> PointArray:
    h, w = frame_shape[:2]
    out = pts.copy().astype(np.float32)
    out[:, 0] = np.clip(out[:, 0], 0, w - 1)
    out[:, 1] = np.clip(out[:, 1], 0, h - 1)
    return out


def quad_bbox(pts: PointArray, frame_shape: Tuple[int, int, int]) -> Tuple[int, int, int, int]:
    h, w = frame_shape[:2]
    xs = np.clip(pts[:, 0], 0, w - 1)
    ys = np.clip(pts[:, 1], 0, h - 1)
    x1, y1 = int(xs.min()), int(ys.min())
    x2, y2 = int(xs.max()), int(ys.max())
    return x1, y1, x2, y2


def polygon_area(pts: PointArray) -> float:
    pts = np.asarray(pts, dtype=np.float32)
    x = pts[:, 0]
    y = pts[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def rectified_size(pts: PointArray) -> Tuple[int, int]:
    tl, tr, br, bl = pts
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)

    width = max(int(round(max(width_a, width_b))), 1)
    height = max(int(round(max(height_a, height_b))), 1)
    return width, height


def warp_slide(frame_bgr: np.ndarray, pts: PointArray) -> np.ndarray:
    pts = order_points_clockwise(pts)
    width, height = rectified_size(pts)

    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(pts.astype(np.float32), dst)
    return cv2.warpPerspective(frame_bgr, M, (width, height))


def _safe_angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    v1 = a - b
    v2 = c - b
    n1 = np.linalg.norm(v1) + 1e-6
    n2 = np.linalg.norm(v2) + 1e-6
    cosang = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return math.degrees(math.acos(cosang))


def quad_angle_score(pts: PointArray) -> float:
    pts = order_points_clockwise(pts)
    angles = []

    for i in range(4):
        a = pts[(i - 1) % 4]
        b = pts[i]
        c = pts[(i + 1) % 4]
        angles.append(_safe_angle_deg(a, b, c))

    mean_dev = float(np.mean([abs(a - 90.0) for a in angles]))
    score = max(0.0, 1.0 - mean_dev / 35.0)
    return float(np.clip(score, 0.0, 1.0))


def aspect_score(
    pts: PointArray,
    targets: Sequence[float] = (4 / 3, 16 / 10, 16 / 9),
) -> float:
    w, h = rectified_size(pts)
    ratio = max(w, 1) / max(h, 1)
    vals = [math.exp(-abs(math.log((ratio + 1e-6) / t)) / 0.35) for t in targets]
    return float(np.clip(max(vals), 0.0, 1.0))


def area_score(pts: PointArray, frame_shape: Tuple[int, int, int]) -> float:
    area = polygon_area(pts)
    frame_area = frame_shape[0] * frame_shape[1]
    r = area / max(frame_area, 1)

    if r <= 0.03:
        return 0.0
    if 0.10 <= r <= 0.85:
        return 1.0
    if r < 0.10:
        return float(np.clip((r - 0.03) / 0.07, 0.0, 1.0))
    return float(np.clip(1.0 - (r - 0.85) / 0.15, 0.0, 1.0))


def center_score(pts: PointArray, frame_shape: Tuple[int, int, int]) -> float:
    h, w = frame_shape[:2]
    c_frame = np.array([w / 2.0, h / 2.0], dtype=np.float32)
    c_quad = np.mean(pts, axis=0)
    dist = np.linalg.norm(c_quad - c_frame)
    diag = math.hypot(w, h) + 1e-6
    return float(np.clip(1.0 - dist / (0.6 * diag), 0.0, 1.0))


def border_margin_score(
    pts: PointArray,
    frame_shape: Tuple[int, int, int],
    margin_ratio: float = 0.02,
) -> float:
    h, w = frame_shape[:2]
    mx = max(2.0, w * margin_ratio)
    my = max(2.0, h * margin_ratio)

    near = 0
    for x, y in pts:
        if x <= mx or x >= (w - 1 - mx) or y <= my or y >= (h - 1 - my):
            near += 1

    lut = {0: 1.0, 1: 0.75, 2: 0.45, 3: 0.15, 4: 0.0}
    return float(lut.get(int(near), 0.0))


def rectangularity_score(contour: np.ndarray, pts: PointArray) -> float:
    contour_area = float(cv2.contourArea(contour))
    q_area = max(polygon_area(pts), 1.0)
    fill = contour_area / q_area
    return float(np.clip(fill, 0.0, 1.0))


# =========================================================
# Препроцессинг изображения
# =========================================================

def adaptive_canny(gray: np.ndarray) -> np.ndarray:
    med = float(np.median(gray))
    lo = int(max(0, 0.66 * med))
    hi = int(min(255, max(lo + 20, 1.33 * med)))

    if hi - lo < 25:
        lo = max(0, lo - 10)
        hi = min(255, hi + 20)

    return cv2.Canny(gray, lo, hi, L2gradient=True)


def build_candidate_masks(gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(blur)

    _, otsu = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    p75 = np.percentile(clahe, 75)
    bright = np.where(clahe >= p75, 255, 0).astype(np.uint8)
    bright = cv2.bitwise_or(otsu, bright)

    edges = adaptive_canny(clahe)

    k1 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, k1, iterations=2)
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, k2, iterations=1)

    edge_fill = cv2.dilate(edges, None, iterations=1)
    edge_fill = cv2.morphologyEx(edge_fill, cv2.MORPH_CLOSE, k1, iterations=2)

    return bright, cv2.bitwise_or(edges, edge_fill)


def line_support(edge_map: np.ndarray, p1: np.ndarray, p2: np.ndarray, thickness: int = 3) -> float:
    mask = np.zeros(edge_map.shape, dtype=np.uint8)
    cv2.line(
        mask,
        (int(round(p1[0])), int(round(p1[1]))),
        (int(round(p2[0])), int(round(p2[1]))),
        255,
        thickness=thickness,
    )

    total = int((mask > 0).sum())
    if total == 0:
        return 0.0

    support = int(((edge_map > 0) & (mask > 0)).sum())
    return support / total


def edge_support_score(edge_map: np.ndarray, pts: PointArray) -> float:
    pts = order_points_clockwise(pts)
    vals = []
    for i in range(4):
        vals.append(line_support(edge_map, pts[i], pts[(i + 1) % 4], thickness=3))
    return float(np.clip(np.mean(vals) * 2.0, 0.0, 1.0))


def contrast_score(gray: np.ndarray, pts: PointArray) -> float:
    h, w = gray.shape[:2]
    poly_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(poly_mask, pts.astype(np.int32), 255)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    dil = cv2.dilate(poly_mask, kernel, iterations=1)
    ring = cv2.subtract(dil, poly_mask)

    inside = gray[poly_mask > 0]
    outside = gray[ring > 0]

    if inside.size < 50 or outside.size < 50:
        return 0.0

    diff = abs(float(inside.mean()) - float(outside.mean())) / 255.0
    return float(np.clip(diff * 2.0, 0.0, 1.0))


def blur_score(frame_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    val = cv2.Laplacian(gray, cv2.CV_64F).var()

    if val <= 40:
        return 0.0
    if val >= 250:
        return 1.0
    return float((val - 40) / (250 - 40))


# =========================================================
# Оценка кандидата
# =========================================================

def _score_candidate(
    contour: np.ndarray,
    pts: PointArray,
    gray: np.ndarray,
    edge_map: np.ndarray,
    frame_shape: Tuple[int, int, int],
    frame_bgr: np.ndarray,
) -> Tuple[float, Dict[str, float]]:
    pts = clip_quad_to_frame(order_points_clockwise(pts), frame_shape)

    metrics = {
        "area": area_score(pts, frame_shape),
        "aspect": aspect_score(pts),
        "angles": quad_angle_score(pts),
        "rectangularity": rectangularity_score(contour, pts),
        "edge_support": edge_support_score(edge_map, pts),
        "contrast": contrast_score(gray, pts),
        "center": center_score(pts, frame_shape),
        "border_margin": border_margin_score(pts, frame_shape),
        "blur": blur_score(frame_bgr),
    }

    confidence = (
        0.17 * metrics["area"]
        + 0.10 * metrics["aspect"]
        + 0.14 * metrics["angles"]
        + 0.18 * metrics["rectangularity"]
        + 0.22 * metrics["edge_support"]
        + 0.08 * metrics["contrast"]
        + 0.04 * metrics["center"]
        + 0.03 * metrics["border_margin"]
        + 0.04 * metrics["blur"]
    )

    if metrics["rectangularity"] < 0.72:
        confidence *= 0.80
    if metrics["edge_support"] < 0.40:
        confidence *= 0.85
    if metrics["border_margin"] < 0.30 and metrics["edge_support"] < 0.60:
        confidence *= 0.75

    return float(np.clip(confidence, 0.0, 1.0)), metrics


def is_reliable_classic_detection(
    det: SlideDetection,
    conf_threshold: float = 0.58,
) -> bool:
    if not det.found or det.keypoints is None:
        return False

    m = det.metrics
    return (
        det.confidence >= conf_threshold
        and m.get("rectangularity", 0.0) >= 0.72
        and m.get("edge_support", 0.0) >= 0.58
        and m.get("angles", 0.0) >= 0.70
        and m.get("area", 0.0) >= 0.45
        and m.get("border_margin", 0.0) >= 0.45
    )


# =========================================================
# Классический детектор
# =========================================================

def _contours_from_mask(mask: np.ndarray) -> List[np.ndarray]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def _contour_to_quad(contour: np.ndarray) -> Optional[PointArray]:
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

    if len(approx) == 4 and cv2.isContourConvex(approx):
        return order_points_clockwise(approx.reshape(4, 2).astype(np.float32))

    hull = cv2.convexHull(contour)
    peri_h = cv2.arcLength(hull, True)
    approx_h = cv2.approxPolyDP(hull, 0.02 * peri_h, True)

    if len(approx_h) == 4 and cv2.isContourConvex(approx_h):
        return order_points_clockwise(approx_h.reshape(4, 2).astype(np.float32))

    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect).astype(np.float32)
    return order_points_clockwise(box)


def detect_slide_keypoints_classic(
    frame_bgr: np.ndarray,
    min_area_ratio: float = 0.04,
    conf_threshold: float = 0.45,
) -> SlideDetection:
    h, w = frame_bgr.shape[:2]
    frame_area = h * w

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    bright_mask, edge_mask = build_candidate_masks(gray)

    all_contours = _contours_from_mask(bright_mask) + _contours_from_mask(edge_mask)
    seen = set()
    best: Optional[SlideDetection] = None

    for contour in all_contours:
        area = float(cv2.contourArea(contour))
        if area < min_area_ratio * frame_area:
            continue

        x, y, bw, bh = cv2.boundingRect(contour)
        signature = (int(x / 10), int(y / 10), int(bw / 10), int(bh / 10))
        if signature in seen:
            continue
        seen.add(signature)

        quad = _contour_to_quad(contour)
        if quad is None:
            continue

        conf, metrics = _score_candidate(
            contour=contour,
            pts=quad,
            gray=gray,
            edge_map=edge_mask,
            frame_shape=frame_bgr.shape,
            frame_bgr=frame_bgr,
        )

        bbox = quad_bbox(quad, frame_bgr.shape)
        det = SlideDetection(
            found=conf >= conf_threshold,
            mode_used="classic",
            keypoints=quad,
            bbox=bbox,
            confidence=conf,
            metrics=metrics,
            debug={"candidate_bbox": bbox},
        )

        if best is None or det.confidence > best.confidence:
            best = det

    if best is None:
        return SlideDetection(
            found=False,
            mode_used="classic",
            keypoints=None,
            bbox=None,
            confidence=0.0,
            metrics={},
            debug={},
        )

    if best.confidence < conf_threshold:
        best.found = False

    return best


# =========================================================
# Нейросетевой детектор и refinement
# =========================================================

def _expand_bbox(
    bbox: Tuple[int, int, int, int],
    frame_shape: Tuple[int, int, int],
    scale: float = 0.06,
) -> Tuple[int, int, int, int]:
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1
    dx = int(round(scale * bw))
    dy = int(round(scale * bh))

    return (
        max(0, x1 - dx),
        max(0, y1 - dy),
        min(w - 1, x2 + dx),
        min(h - 1, y2 + dy),
    )


def refine_quad_in_bbox(
    frame_bgr: np.ndarray,
    bbox: Tuple[int, int, int, int],
    conf_threshold: float = 0.35,
) -> SlideDetection:
    x1, y1, x2, y2 = _expand_bbox(bbox, frame_bgr.shape, scale=0.08)
    roi = frame_bgr[y1:y2, x1:x2]

    if roi.size == 0:
        return SlideDetection(False, "classic_refine", None, None, 0.0)

    det = detect_slide_keypoints_classic(roi, min_area_ratio=0.08, conf_threshold=conf_threshold)

    if det.keypoints is not None:
        quad = det.keypoints.copy()
        quad[:, 0] += x1
        quad[:, 1] += y1
        quad = clip_quad_to_frame(quad, frame_bgr.shape)

        return SlideDetection(
            found=det.found,
            mode_used="classic_refine",
            keypoints=quad,
            bbox=quad_bbox(quad, frame_bgr.shape),
            confidence=det.confidence,
            metrics=det.metrics,
            debug={"refined_from_bbox": bbox},
        )

    quad = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
    return SlideDetection(
        found=True,
        mode_used="bbox_fallback",
        keypoints=quad,
        bbox=(x1, y1, x2, y2),
        confidence=0.30,
        metrics={"from_bbox_only": 1.0},
        debug={"refined_from_bbox": bbox},
    )


def detect_slide_keypoints_neural(
    frame_bgr: np.ndarray,
    neural_detector: NeuralSlideDetector,
) -> SlideDetection:
    pred = neural_detector.predict(frame_bgr) or {}
    conf = float(pred.get("confidence", 0.0))

    if pred.get("keypoints") is not None:
        quad = order_points_clockwise(np.asarray(pred["keypoints"], dtype=np.float32))
        quad = clip_quad_to_frame(quad, frame_bgr.shape)

        return SlideDetection(
            found=True,
            mode_used="neural_keypoints",
            keypoints=quad,
            bbox=quad_bbox(quad, frame_bgr.shape),
            confidence=conf,
            metrics={"neural_conf": conf},
            debug={},
        )

    if pred.get("bbox") is not None:
        bbox = tuple(map(int, pred["bbox"]))
        refined = refine_quad_in_bbox(frame_bgr, bbox)
        refined.mode_used = "neural_bbox+" + refined.mode_used
        refined.confidence = max(refined.confidence, conf)
        refined.metrics["neural_conf"] = conf
        return refined

    return SlideDetection(
        found=False,
        mode_used="neural",
        keypoints=None,
        bbox=None,
        confidence=0.0,
        metrics={},
        debug={},
    )


def detect_slide_keypoints(
    frame_bgr: np.ndarray,
    mode: str = "auto",
    neural_detector: Optional[NeuralSlideDetector] = None,
    classic_conf_threshold: float = 0.45,
) -> SlideDetection:
    """
    mode:
      - classic
      - neural
      - hybrid
      - auto
    """
    mode = mode.lower().strip()

    if mode == "classic":
        return detect_slide_keypoints_classic(frame_bgr, conf_threshold=classic_conf_threshold)

    if mode == "neural":
        if neural_detector is None:
            raise ValueError("mode='neural', but neural_detector is None")
        return detect_slide_keypoints_neural(frame_bgr, neural_detector)

    if mode in {"hybrid", "auto"}:
        classic = detect_slide_keypoints_classic(frame_bgr, conf_threshold=classic_conf_threshold)
        if is_reliable_classic_detection(classic):
            return classic

        if neural_detector is not None:
            neural = detect_slide_keypoints_neural(frame_bgr, neural_detector)
            if neural.found:
                return neural

        return classic

    raise ValueError(f"Unsupported mode: {mode}")


# =========================================================
# Анализ видео
# =========================================================

def _iter_sampled_frames(
    video_path: str,
    sample_every_sec: float = 2.0,
    max_samples: int = 80,
) -> Iterable[Tuple[float, np.ndarray]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps if frame_count > 0 else 0.0

    if duration > 0 and max_samples > 0:
        adaptive_step = max(sample_every_sec, duration / max_samples)
    else:
        adaptive_step = sample_every_sec

    step_frames = max(1, int(round(adaptive_step * fps)))

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if idx % step_frames == 0:
            t = float(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
            yield t, frame

        idx += 1

    cap.release()


def _mean_corner_jitter(
    dets: List[SlideDetection],
    frame_shape: Optional[Tuple[int, int]] = None,
) -> float:
    good = [d for d in dets if d.keypoints is not None]
    if len(good) < 2:
        return 0.0

    if frame_shape is not None:
        h, w = frame_shape
        diag = math.hypot(w, h) + 1e-6
    else:
        all_pts = np.vstack([g.keypoints for g in good])
        max_xy = all_pts.max(axis=0)
        diag = math.hypot(float(max_xy[0]), float(max_xy[1])) + 1e-6

    diffs = []
    for a, b in zip(good[:-1], good[1:]):
        diffs.append(float(np.linalg.norm(a.keypoints - b.keypoints, axis=1).mean()) / diag)

    return float(np.mean(diffs)) if diffs else 0.0


def analyze_video_complexity(
    video_path: str,
    sample_every_sec: float = 2.0,
    max_samples: int = 80,
    frame_conf_threshold: float = 0.45,
    reliable_frame_conf: float = 0.58,
) -> VideoComplexityReport:
    reports: List[SlideDetection] = []
    sampled = 0
    frame_shape: Optional[Tuple[int, int]] = None

    for _, frame in _iter_sampled_frames(
        video_path,
        sample_every_sec=sample_every_sec,
        max_samples=max_samples,
    ):
        sampled += 1
        frame_shape = frame.shape[:2]
        det = detect_slide_keypoints_classic(frame, conf_threshold=frame_conf_threshold)
        reports.append(det)

    if sampled == 0:
        return VideoComplexityReport(
            video_path=video_path,
            sampled_frames=0,
            reliable_frames=0,
            success_rate=0.0,
            median_confidence=0.0,
            mean_confidence=0.0,
            temporal_jitter=1.0,
            recommended_mode="neural",
            frame_reports=[],
        )

    confidences = [r.confidence for r in reports]
    reliable = [
        r for r in reports
        if is_reliable_classic_detection(r, conf_threshold=reliable_frame_conf)
    ]

    success_rate = len(reliable) / sampled
    median_conf = float(np.median(confidences))
    mean_conf = float(np.mean(confidences))
    jitter = _mean_corner_jitter(reliable, frame_shape=frame_shape)

    if success_rate >= 0.72 and median_conf >= 0.60 and jitter <= 0.05:
        mode = "classic"
    elif success_rate < 0.45 or median_conf < 0.48 or jitter > 0.10:
        mode = "neural"
    else:
        mode = "hybrid"

    return VideoComplexityReport(
        video_path=video_path,
        sampled_frames=sampled,
        reliable_frames=len(reliable),
        success_rate=success_rate,
        median_confidence=median_conf,
        mean_confidence=mean_conf,
        temporal_jitter=jitter,
        recommended_mode=mode,
        frame_reports=reports,
        thresholds={
            "frame_conf_threshold": frame_conf_threshold,
            "reliable_frame_conf": reliable_frame_conf,
            "simple_success_rate": 0.72,
            "simple_median_conf": 0.60,
            "simple_jitter": 0.05,
            "complex_success_rate": 0.45,
            "complex_median_conf": 0.48,
            "complex_jitter": 0.10,
        },
    )


# =========================================================
# Извлечение ключевых точек на видео
# =========================================================

def _smooth_quad(
    prev_pts: Optional[PointArray],
    cur_pts: Optional[PointArray],
    alpha: float = 0.75,
) -> Optional[PointArray]:
    if cur_pts is None:
        return prev_pts
    if prev_pts is None:
        return cur_pts
    return (alpha * prev_pts + (1.0 - alpha) * cur_pts).astype(np.float32)


def extract_video_keypoints(
    video_path: str,
    mode: str = "auto",
    neural_detector: Optional[NeuralSlideDetector] = None,
    sample_every_sec: float = 1.0,
    max_samples: int = 300,
    smooth: bool = True,
) -> Tuple[VideoComplexityReport, List[Tuple[float, SlideDetection]]]:
    report = analyze_video_complexity(video_path)
    selected_mode = mode.lower().strip()

    if selected_mode == "auto":
        selected_mode = report.recommended_mode

    results: List[Tuple[float, SlideDetection]] = []
    prev_pts: Optional[PointArray] = None

    for t, frame in _iter_sampled_frames(
        video_path,
        sample_every_sec=sample_every_sec,
        max_samples=max_samples,
    ):
        det = detect_slide_keypoints(
            frame,
            mode=selected_mode,
            neural_detector=neural_detector,
        )

        if selected_mode in {"classic", "hybrid"}:
            if (not is_reliable_classic_detection(det)) and neural_detector is not None:
                nn_det = detect_slide_keypoints_neural(frame, neural_detector)
                if nn_det.found and nn_det.confidence >= det.confidence:
                    det = nn_det

        if smooth and det.keypoints is not None:
            smoothed = _smooth_quad(prev_pts, det.keypoints, alpha=0.75)
            det.keypoints = smoothed
            det.bbox = quad_bbox(smoothed, frame.shape)
            prev_pts = smoothed
        elif det.keypoints is not None:
            prev_pts = det.keypoints

        results.append((t, det))

    return report, results


# =========================================================
# Обертка над YOLO
# =========================================================

class YoloSlideDetector:
    """
    Обертка над Ultralytics YOLO.

    Поддерживает:
    1) модель детекции -> bbox
    2) pose/keypoints модель -> 4 точки

    Установка:
        pip install ultralytics
    """

    def __init__(self, model_path: str, conf: float = 0.25):
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        self.conf = conf

    def predict(self, frame_bgr: np.ndarray) -> Dict[str, Any]:
        res = self.model.predict(frame_bgr, verbose=False, conf=self.conf)
        if not res:
            return {}

        r = res[0]
        out: Dict[str, Any] = {}

        kps = getattr(r, "keypoints", None)
        if kps is not None and getattr(kps, "xy", None) is not None and len(kps.xy) > 0:
            pts = np.asarray(kps.xy[0].cpu().numpy(), dtype=np.float32)
            if pts.shape[0] >= 4:
                out["keypoints"] = pts[:4]
                if getattr(r, "boxes", None) is not None and len(r.boxes):
                    out["confidence"] = float(r.boxes.conf[0].cpu().numpy())
                else:
                    out["confidence"] = 0.5
                return out

        if getattr(r, "boxes", None) is not None and len(r.boxes) > 0:
            box = r.boxes.xyxy[0].cpu().numpy().astype(float)
            conf = float(r.boxes.conf[0].cpu().numpy())
            out["bbox"] = tuple(map(int, box))
            out["confidence"] = conf
            return out

        return {}


# =========================================================
# Визуализация
# =========================================================

def draw_detection(frame_bgr: np.ndarray, det: SlideDetection) -> np.ndarray:
    out = frame_bgr.copy()

    if det.keypoints is not None:
        pts = det.keypoints.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(out, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        colors = [(255, 0, 0), (0, 255, 255), (0, 0, 255), (255, 255, 0)]
        for i, p in enumerate(det.keypoints.astype(np.int32)):
            cv2.circle(out, tuple(p), 6, colors[i], -1)
            cv2.putText(
                out,
                f"P{i+1}",
                tuple(p + 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                colors[i],
                2,
                cv2.LINE_AA,
            )

    label = f"{det.mode_used} | conf={det.confidence:.2f} | found={int(det.found)}"
    cv2.putText(out, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
    return out


__all__ = [
    "SlideDetection",
    "VideoComplexityReport",
    "NeuralSlideDetector",
    "YoloSlideDetector",
    "analyze_video_complexity",
    "detect_slide_keypoints",
    "detect_slide_keypoints_classic",
    "detect_slide_keypoints_neural",
    "extract_video_keypoints",
    "is_reliable_classic_detection",
    "warp_slide",
    "draw_detection",
]