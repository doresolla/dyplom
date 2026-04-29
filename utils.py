import cv2
from pathlib import Path
import numpy as np
from functools import lru_cache

from typing import Optional, Callable, Any, List
import re
from collections import Counter
import math
import os


TIME_FROM_NAME_RE = re.compile(r"_(?P<time>\d+(?:\.\d+)?)s(?:\.[^.]+)?$")
TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9_+#-]{2,}")

RUS_STOPWORDS = {
    "это", "как", "что", "для", "или", "при", "если", "есть", "так", "его", "ее", "их",
    "она", "они", "мы", "вы", "ты", "я", "но", "а", "и", "в", "во", "на", "по", "из",
    "к", "ко", "у", "о", "об", "от", "до", "же", "ли", "не", "да", "ну", "то", "же",
    "бы", "быть", "вот", "тут", "там", "уже", "еще", "ещё", "только", "когда", "где",
    "который", "которая", "которые", "такой", "такая", "такие", "этот", "эта", "эти",
    "the", "and", "for", "with", "this", "that", "from", "into", "then", "than", "are",
    "was", "were", "have", "has", "had", "not", "you", "your", "our", "but", "can",
}

TRANSITION_PREFIXES = (
    "итак", "теперь", "далее", "следующий", "следующая", "перейдем", "перейдём",
    "рассмотрим", "обсудим", "подведем", "подведём", "сначала", "во-первых", "во вторых",
    "во-вторых", "наконец", "important", "next", "now", "let us", "moving on",
)


def _emit(callback: Optional[Callable[[str], None]], text: str) -> None:
    if callback is not None:
        callback(text)

def crop_frames_by_keypoints(frame_paths,
                             keypoints_map, out_dir: Path, 
                             callback: Optional[Callable[[str], None]] = None
                             ) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cropped_paths = []

    for frame_path in frame_paths:
        frame = cv2.imread(str(frame_path))
        if frame is None:
            _emit(callback, f"[crop] не удалось открыть {frame_path}")
            continue

        meta = keypoints_map.get(str(frame_path))
        if not meta:
            _emit(callback, f"[crop] нет meta для {frame_path.name}")
            continue

        crop = None

        # Новый основной путь: перспективная обрезка по уточненному четырехугольнику.
        if "points" in meta and meta["points"]:
            try:
                crop = _warp_frame_by_quad(frame, meta["points"])
            except Exception as exc:
                _emit(callback, 
                      f"[crop] ошибка warp по quad для {frame_path.name}: {exc}"
                )
                crop = None

        # Старый fallback: обычная обрезка по bbox.
        if crop is None:
            if "bbox" not in meta:
                _emit(callback, f"[crop] нет bbox для {frame_path.name}")
                continue

            h, w = frame.shape[:2]
            x1, y1, x2, y2 = meta["bbox"]

            x1 = max(0, min(int(x1), w - 1))
            y1 = max(0, min(int(y1), h - 1))
            x2 = max(1, min(int(x2), w))
            y2 = max(1, min(int(y2), h))

            if x2 <= x1 or y2 <= y1:
                _emit(callback,
                    f"[crop] некорректный bbox для {frame_path.name}: {meta['bbox']}"
                )
                continue

            crop = frame[y1:y2, x1:x2]

        if crop is None or crop.size == 0:
            _emit(callback, f"[crop] пустой crop для {frame_path.name}")
            continue

        out_path = out_dir / frame_path.name
        cv2.imwrite(str(out_path), crop)
        cropped_paths.append(out_path)

    _emit(callback,f"[crop] обрезано кадров: {len(cropped_paths)}")
    return cropped_paths


def _order_quad_for_warp(points):
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    center = pts.mean(axis=0)

    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    pts = pts[np.argsort(angles)]

    start_idx = int(np.argmin(pts.sum(axis=1)))
    pts = np.roll(pts, -start_idx, axis=0)

    return pts

def _warp_frame_by_quad( frame, points):
    src = _order_quad_for_warp(points)

    tl, tr, br, bl = src

    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)

    out_w = int(round(max(width_top, width_bottom)))
    out_h = int(round(max(height_left, height_right)))

    if out_w < 10 or out_h < 10:
        return None

    dst = np.array(
        [
            [0, 0],
            [out_w - 1, 0],
            [out_w - 1, out_h - 1],
            [0, out_h - 1],
        ],
        dtype=np.float32,
    )

    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(frame, M, (out_w, out_h))

def _clean_text(text: str | None) -> str:
    if text is None:
        return ""
    text = text.replace("\xa0", " ")
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _format_ts(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _relative_posix(target: Path, base_dir: Path) -> str:
    rel = os.path.relpath(str(target), str(base_dir))
    return rel.replace("\\", "/")


def _extract_time_from_path(path: Path) -> Optional[float]:
    match = TIME_FROM_NAME_RE.search(path.name)
    if match:
        try:
            return float(match.group("time"))
        except Exception:
            return None
    return None


def _escape_md(text: str) -> str:
    return text.replace("\\", "\\\\").replace("*", r"\*").replace("_", r"\_")


def _truncate(text: str, max_chars: int = 450) -> str:
    text = _clean_text(text)
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars].rsplit(" ", 1)[0].strip()
    return cut + " …"


def _duration(start: float, end: float) -> float:
    return max(0.0, float(end) - float(start))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _split_sentences(text: str) -> List[str]:
    text = _clean_text(text)
    if not text:
        return []
    parts = re.split(r"(?<=[.!?…])\s+", text)
    return [p.strip(" -•\n\t") for p in parts if p.strip()]


def _tokenize(text: str) -> List[str]:
    text = _clean_text(text).lower().replace("ё", "е")
    tokens = TOKEN_RE.findall(text)
    return [t for t in tokens if t not in RUS_STOPWORDS and len(t) > 1]


def _vectorize(text: str) -> Counter:
    return Counter(_tokenize(text))


def _cosine_sim(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0

    common = set(a) & set(b)
    dot = sum(a[t] * b[t] for t in common)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a <= 1e-12 or norm_b <= 1e-12:
        return 0.0
    return float(dot / (norm_a * norm_b))


def _transition_cue_score(text: str) -> float:
    t = _clean_text(text).lower()
    if not t:
        return 0.0
    score = 0.0
    for prefix in TRANSITION_PREFIXES:
        if t.startswith(prefix):
            score += 0.5
    if re.search(r"\b(итак|теперь|далее|перейд[её]м|следующ|рассмотрим|обсудим)\b", t):
        score += 0.35
    if re.search(r":\s*$", t[:50]):
        score += 0.15
    return min(score, 1.0)


def _top_keywords(text: str, limit: int = 6) -> List[str]:
    freq = _vectorize(text)
    if not freq:
        return []
    return [token for token, _ in freq.most_common(limit)]


