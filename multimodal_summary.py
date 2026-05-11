from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from text_processing.LLMsummary import summarize_with_llm, generate_with_llm
from utils import _emit


TRANSCRIPT_LINE_RE = re.compile(
    r"^\[(?P<start>\d+(?:\.\d+)?)\s*-\s*(?P<end>\d+(?:\.\d+)?)\]\s*(?P<text>.*)$"
)

TIME_FROM_NAME_RE = re.compile(r"_(?P<time>\d+(?:\.\d+)?)s(?:\.[^.]+)?$")
TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9_+#-]{2,}")

STOPWORDS = {
    "это", "как", "что", "для", "или", "при", "если", "есть", "так", "его", "ее", "её",
    "их", "она", "они", "мы", "вы", "ты", "я", "но", "а", "и", "в", "во", "на", "по",
    "из", "к", "ко", "у", "о", "об", "от", "до", "же", "ли", "не", "да", "ну", "то",
    "бы", "быть", "вот", "тут", "там", "уже", "еще", "ещё", "только", "когда", "где",
    "который", "которая", "которые", "такой", "такая", "такие", "этот", "эта", "эти",
    "the", "and", "for", "with", "this", "that", "from", "into", "then", "than", "are",
    "was", "were", "have", "has", "had", "not", "you", "your", "our", "but", "can",
}

TRANSITION_WORDS = (
    "итак", "теперь", "далее", "следующий", "следующая", "перейдем", "перейдём",
    "рассмотрим", "обсудим", "подведем", "подведём", "сначала", "во-первых",
    "во-вторых", "наконец", "важно", "следовательно",
    "important", "next", "now", "let us", "moving on",
)


# =========================================================
# БАЗОВЫЕ УТИЛИТЫ
# =========================================================

def _clean_text(text: str | None) -> str:
    if text is None:
        return ""
    text = text.replace("\xa0", " ")
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _escape_md(text: str) -> str:
    return text.replace("\\", "\\\\").replace("*", r"\*").replace("_", r"\_")


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
    if not match:
        return None

    try:
        return float(match.group("time"))
    except Exception:
        return None


def _duration(start: float, end: float) -> float:
    return max(0.0, float(end) - float(start))


def _truncate(text: str, max_chars: int = 700) -> str:
    text = _clean_text(text)
    if len(text) <= max_chars:
        return text

    cut = text[:max_chars].rsplit(" ", 1)[0].strip()
    return cut + " …"


def _split_sentences(text: str) -> List[str]:
    text = _clean_text(text)
    if not text:
        return []

    parts = re.split(r"(?<=[.!?…])\s+", text)
    return [p.strip(" -•\n\t") for p in parts if p.strip()]


def _tokenize(text: str) -> List[str]:
    text = _clean_text(text).lower().replace("ё", "е")
    tokens = TOKEN_RE.findall(text)
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


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


def _top_keywords(text: str, limit: int = 6) -> List[str]:
    freq = _vectorize(text)
    if not freq:
        return []

    return [token for token, _ in freq.most_common(limit)]


def _transition_score(text: str) -> float:
    text = _clean_text(text).lower()
    if not text:
        return 0.0

    score = 0.0

    for word in TRANSITION_WORDS:
        if text.startswith(word):
            score += 0.55

    if re.search(r"\b(итак|теперь|далее|перейд[её]м|следующ|рассмотрим|обсудим)\b", text):
        score += 0.35

    return min(score, 1.0)


def _parse_json_object(raw: str) -> Optional[dict]:
    raw = _clean_text(raw)

    try:
        value = json.loads(raw)
        if isinstance(value, dict):
            return value
    except Exception:
        pass

    match = re.search(r"\{.*\}", raw, flags=re.S)
    if not match:
        return None

    try:
        value = json.loads(match.group(0))
        if isinstance(value, dict):
            return value
    except Exception:
        return None

    return None


def _parse_json_list(raw: str) -> Optional[list]:
    raw = _clean_text(raw)

    try:
        value = json.loads(raw)
        if isinstance(value, list):
            return value
    except Exception:
        pass

    match = re.search(r"\[.*\]", raw, flags=re.S)
    if not match:
        return None

    try:
        value = json.loads(match.group(0))
        if isinstance(value, list):
            return value
    except Exception:
        return None

    return None


# =========================================================
# ЧТЕНИЕ ТРАНСКРИПТА, OCR, КАДРОВ
# =========================================================

def parse_transcript(transcript_path: str | Path) -> List[Dict[str, Any]]:
    transcript_path = Path(transcript_path)
    items: List[Dict[str, Any]] = []

    if not transcript_path.exists():
        return items

    for raw_line in transcript_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        match = TRANSCRIPT_LINE_RE.match(line)
        if not match:
            continue

        start = float(match.group("start"))
        end = float(match.group("end"))
        text = _clean_text(match.group("text"))

        if not text:
            continue

        items.append(
            {
                "start": start,
                "end": end,
                "center": (start + end) / 2.0,
                "text": text,
            }
        )

    return items


def load_ocr_map(ocr_dir: str | Path | None) -> Dict[str, str]:
    if ocr_dir is None:
        return {}

    ocr_dir = Path(ocr_dir)
    if not ocr_dir.exists():
        return {}

    result: Dict[str, str] = {}

    for txt_file in sorted(ocr_dir.glob("*.txt")):
        if txt_file.name.lower() == "ocr_merged.txt":
            continue

        result[txt_file.stem] = _clean_text(txt_file.read_text(encoding="utf-8"))

    return result


def normalize_frames(
    frame_paths: Optional[Iterable[str | Path]] = None,
    keyframes: Optional[Iterable[Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Важный момент.

    keyframes обычно содержит время и путь к исходному полному кадру.
    frame_paths после crop_frames_by_keypoints содержит уже обрезанные кадры.

    Поэтому время берём из keyframes, а путь стараемся заменить на cropped-версию
    с тем же stem, чтобы в markdown попадал именно обрезанный слайд/доска.
    """
    cropped_by_stem: Dict[str, Path] = {}

    if frame_paths:
        for p in frame_paths:
            path_obj = Path(p)
            cropped_by_stem[path_obj.stem] = path_obj

    records: List[Dict[str, Any]] = []

    def add_record(frame_time: Optional[float], frame_path: str | Path) -> None:
        original_path = Path(frame_path)
        final_path = cropped_by_stem.get(original_path.stem, original_path)

        if frame_time is None:
            frame_time = _extract_time_from_path(original_path)
        if frame_time is None:
            frame_time = _extract_time_from_path(final_path)

        records.append(
            {
                "time": float(frame_time) if frame_time is not None else 0.0,
                "path": final_path,
                "stem": final_path.stem,
            }
        )

    if keyframes:
        for item in keyframes:
            if isinstance(item, dict):
                path = item.get("path") or item.get("image_path")
                if path:
                    add_record(item.get("time") or item.get("time_sec"), path)

            elif isinstance(item, (tuple, list)) and len(item) >= 2:
                add_record(item[0], item[1])

            elif isinstance(item, (str, Path)):
                add_record(None, item)

    if not records and frame_paths:
        for frame_path in frame_paths:
            add_record(None, frame_path)

    records.sort(key=lambda x: (x["time"], x["path"].name if x["path"] else ""))

    deduped: List[Dict[str, Any]] = []
    seen: set[Tuple[float, str]] = set()

    for rec in records:
        key = (round(float(rec["time"]), 2), str(rec["path"]))
        if key in seen:
            continue

        seen.add(key)
        deduped.append(rec)

    return deduped


def _softmax(scores: List[float], temperature: float = 0.35) -> List[float]:
    """
    Преобразует набор оценок похожести в вероятности.
    Чем меньше temperature, тем увереннее распределение.
    """
    if not scores:
        return []

    temperature = max(1e-6, float(temperature))

    scaled = [s / temperature for s in scores]
    max_s = max(scaled)

    exps = [math.exp(s - max_s) for s in scaled]
    total = sum(exps)

    if total <= 1e-12:
        return [1.0 / len(scores)] * len(scores)

    return [x / total for x in exps]


def _normalize_plan_sections(plan: List[Any]) -> List[Dict[str, Any]]:
    """
    Поддерживает оба варианта:
    1. ["Раздел 1", "Раздел 2"]
    2. [{"title": "...", "description": "..."}]
    """
    sections: List[Dict[str, Any]] = []

    for item in plan:
        if isinstance(item, str):
            title = _clean_text(item)
            description = ""

        elif isinstance(item, dict):
            title = _clean_text(
                item.get("title")
                or item.get("name")
                or item.get("section")
                or ""
            )
            description = _clean_text(
                item.get("description")
                or item.get("summary")
                or ""
            )

        else:
            title = _clean_text(str(item))
            description = ""

        if not title:
            continue

        sections.append(
            {
                "index": len(sections),
                "title": title,
                "description": description,
            }
        )

    return sections


def build_section_profiles(plan: List[Any]) -> List[Dict[str, Any]]:
    """
    Профиль раздела нужен для сравнения текста окна с пунктом плана.
    """
    sections = _normalize_plan_sections(plan)

    profiles: List[Dict[str, Any]] = []

    for section in sections:
        title = section["title"]
        description = section.get("description", "")

        profile_text = _clean_text(f"{title}. {description}")

        profiles.append(
            {
                "index": section["index"],
                "title": title,
                "description": description,
                "profile_text": profile_text,
                "vector": _vectorize(profile_text),
            }
        )

    return profiles


def _make_window_from_items(
    transcript_items: List[Dict[str, Any]],
    item_start: int,
    item_end: int,
    index: int = 1,
) -> Dict[str, Any]:
    """
    Собирает окно текста из ASR-фрагментов.
    """
    item_start = max(0, int(item_start))
    item_end = min(len(transcript_items) - 1, int(item_end))

    block = transcript_items[item_start:item_end + 1]

    text = _clean_text(" ".join(item["text"] for item in block))

    return {
        "index": index,
        "item_start": item_start,
        "item_end": item_end,
        "start": float(block[0]["start"]),
        "end": float(block[-1]["end"]),
        "center": (float(block[0]["start"]) + float(block[-1]["end"])) / 2.0,
        "text": text,
        "vector": _vectorize(text),
        "cue": _transition_score(block[0]["text"]),
    }


def estimate_section_membership(
    text: str,
    section_profiles: List[Dict[str, Any]],
    position_ratio: float | None = None,
    semantic_weight: float = 0.85,
    position_weight: float = 0.15,
    temperature: float = 0.35,
) -> Dict[str, Any]:
    """
    Оценивает вероятность принадлежности текста к каждому разделу плана.

    position_ratio — положение окна внутри лекции от 0 до 1.
    Это слабый prior, чтобы ранние окна чуть больше тяготели к ранним разделам,
    а поздние — к поздним.
    """
    if not section_profiles:
        return {
            "section_probs": [],
            "section_scores": [],
            "best_section_idx": None,
            "best_section_title": "",
            "confidence": 0.0,
            "margin": 0.0,
            "ambiguous": True,
        }

    text_vec = _vectorize(text)
    k = len(section_profiles)

    scores: List[float] = []

    for j, profile in enumerate(section_profiles):
        semantic_score = _cosine_sim(text_vec, profile["vector"])

        if position_ratio is not None and k > 1:
            expected_pos = j / (k - 1)
            position_score = max(0.0, 1.0 - abs(position_ratio - expected_pos))
        else:
            position_score = 0.0

        score = semantic_weight * semantic_score + position_weight * position_score
        scores.append(score)

    probs = _softmax(scores, temperature=temperature)

    best_idx = max(range(k), key=lambda i: probs[i])
    sorted_probs = sorted(probs, reverse=True)

    confidence = sorted_probs[0]
    second = sorted_probs[1] if len(sorted_probs) > 1 else 0.0
    margin = confidence - second

    ambiguous = confidence < 0.45 or margin < 0.15

    return {
        "section_probs": probs,
        "section_scores": scores,
        "best_section_idx": best_idx,
        "best_section_title": section_profiles[best_idx]["title"],
        "confidence": confidence,
        "margin": margin,
        "ambiguous": ambiguous,
    }


def _window_position_ratio(
    window: Dict[str, Any],
    lecture_start: float,
    lecture_end: float,
) -> float:
    duration = max(1e-6, lecture_end - lecture_start)
    center = float(window["center"])
    return max(0.0, min(1.0, (center - lecture_start) / duration))


def _find_middle_split_item(
    transcript_items: List[Dict[str, Any]],
    item_start: int,
    item_end: int,
) -> int:
    """
    Ищет ASR-фрагмент около середины окна по времени.
    """
    start_time = float(transcript_items[item_start]["start"])
    end_time = float(transcript_items[item_end]["end"])
    mid_time = (start_time + end_time) / 2.0

    best_i = item_start
    best_gap = 1e18

    for i in range(item_start, item_end):
        candidate_time = float(transcript_items[i]["end"])
        gap = abs(candidate_time - mid_time)

        if gap < best_gap:
            best_gap = gap
            best_i = i

    return best_i


def classify_window_with_recursive_split(
    transcript_items: List[Dict[str, Any]],
    item_start: int,
    item_end: int,
    section_profiles: List[Dict[str, Any]],
    lecture_start: float,
    lecture_end: float,
    min_split_sec: float = 10.0,
    min_items: int = 2,
    max_depth: int = 4,
    depth: int = 0,
) -> List[Dict[str, Any]]:
    """
    Классифицирует окно.
    Если принадлежность к разделу неоднозначна — делит окно и классифицирует части.
    """
    window = _make_window_from_items(
        transcript_items=transcript_items,
        item_start=item_start,
        item_end=item_end,
    )

    position_ratio = _window_position_ratio(
        window=window,
        lecture_start=lecture_start,
        lecture_end=lecture_end,
    )

    result = estimate_section_membership(
        text=window["text"],
        section_profiles=section_profiles,
        position_ratio=position_ratio,
    )

    window.update(result)

    duration = _duration(window["start"], window["end"])
    item_count = item_end - item_start + 1

    can_split = (
        window["ambiguous"]
        and depth < max_depth
        and duration >= min_split_sec * 2
        and item_count > min_items
    )

    if not can_split:
        return [window]

    split_item = _find_middle_split_item(
        transcript_items=transcript_items,
        item_start=item_start,
        item_end=item_end,
    )

    if split_item <= item_start or split_item >= item_end:
        return [window]

    left = classify_window_with_recursive_split(
        transcript_items=transcript_items,
        item_start=item_start,
        item_end=split_item,
        section_profiles=section_profiles,
        lecture_start=lecture_start,
        lecture_end=lecture_end,
        min_split_sec=min_split_sec,
        min_items=min_items,
        max_depth=max_depth,
        depth=depth + 1,
    )

    right = classify_window_with_recursive_split(
        transcript_items=transcript_items,
        item_start=split_item + 1,
        item_end=item_end,
        section_profiles=section_profiles,
        lecture_start=lecture_start,
        lecture_end=lecture_end,
        min_split_sec=min_split_sec,
        min_items=min_items,
        max_depth=max_depth,
        depth=depth + 1,
    )

    return left + right


def classify_text_windows_by_plan(
    transcript_items: List[Dict[str, Any]],
    windows: List[Dict[str, Any]],
    plan: List[Any],
    min_split_sec: float = 10.0,
    callback: Optional[Callable[[str], None]] = None,
) -> List[Dict[str, Any]]:
    """
    Второй шаг пайплайна:
    - берёт окна текста;
    - оценивает вероятность принадлежности каждого окна к каждому разделу;
    - неоднозначные окна делит;
    - возвращает классифицированные окна.
    """
    if not transcript_items:
        return []

    if not windows:
        return []

    section_profiles = build_section_profiles(plan)

    if not section_profiles:
        return windows

    lecture_start = float(transcript_items[0]["start"])
    lecture_end = float(transcript_items[-1]["end"])

    classified: List[Dict[str, Any]] = []

    _emit(callback, "[summary] классификация текстовых окон по разделам плана")

    for window in windows:
        parts = classify_window_with_recursive_split(
            transcript_items=transcript_items,
            item_start=int(window["item_start"]),
            item_end=int(window["item_end"]),
            section_profiles=section_profiles,
            lecture_start=lecture_start,
            lecture_end=lecture_end,
            min_split_sec=min_split_sec,
        )

        classified.extend(parts)

    for idx, window in enumerate(classified, start=1):
        window["index"] = idx

    ambiguous_count = sum(1 for w in classified if w.get("ambiguous"))

    _emit(
        callback,
        f"[summary] классифицировано окон: {len(classified)}, "
        f"неоднозначных после деления: {ambiguous_count}"
    )

    return classified

# =========================================================
# 1. ГРУБЫЙ ПЛАН ЛЕКЦИИ ПО ТРАНСКРИПТУ
# =========================================================

def _desired_section_count(total_duration_sec: float) -> int:
    if total_duration_sec <= 0:
        return 1

    # примерно один крупный раздел на 3 минуты, но не больше 12
    return max(1, min(12, int(round(total_duration_sec / 180.0)) + 1))


def _compact_transcript_for_plan(
    transcript_items: List[Dict[str, Any]],
    max_chars_per_block: int = 900,
    target_blocks: int = 10,
) -> str:
    """
    Делает компактное представление транскрипта для построения плана.

    Не пытаемся передать LLM весь транскрипт.
    Передаём последовательные блоки:
    [00:00–01:20] текст...
    [01:20–02:40] текст...
    """
    if not transcript_items:
        return ""

    n = len(transcript_items)
    target_blocks = max(1, min(target_blocks, n))
    chunk_size = max(1, math.ceil(n / target_blocks))

    blocks = []

    for i in range(0, n, chunk_size):
        chunk = transcript_items[i:i + chunk_size]
        if not chunk:
            continue

        start = float(chunk[0]["start"])
        end = float(chunk[-1]["end"])
        text = _clean_text(" ".join(item["text"] for item in chunk))
        text = _truncate(text, max_chars_per_block)

        blocks.append(
            f"[{_format_ts(start)}–{_format_ts(end)}]\n{text}"
        )

    return "\n\n".join(blocks)


def _normalize_plan_items(raw_items: list) -> List[Dict[str, Any]]:
    """
    Приводит ответ LLM к единому виду:
    [
        {"index": 1, "title": "...", "description": "..."},
        ...
    ]
    """
    result: List[Dict[str, Any]] = []
    seen_titles = set()

    for item in raw_items:
        title = ""
        description = ""

        if isinstance(item, str):
            title = item
            description = ""

        elif isinstance(item, dict):
            title = (
                item.get("title")
                or item.get("name")
                or item.get("section")
                or ""
            )
            description = item.get("description") or item.get("summary") or ""

        title = _clean_text(str(title))
        description = _clean_text(str(description))

        if not title:
            continue

        title = title.strip(" -—:;.0123456789")
        title = title[:90]

        key = title.lower().replace("ё", "е")

        if key in seen_titles:
            continue

        seen_titles.add(key)

        result.append(
            {
                "index": len(result) + 1,
                "title": title,
                "description": description[:350],
            }
        )

    return result


def _fallback_lecture_plan(
    transcript_items: List[Dict[str, Any]],
    target_sections: int,
) -> List[Dict[str, Any]]:
    """
    Запасной вариант без LLM.
    Даёт не идеальные названия, но пайплайн не падает.
    """
    if not transcript_items:
        return [
            {
                "index": 1,
                "title": "Раздел 1",
                "description": "",
            }
        ]

    n = len(transcript_items)
    chunk_size = max(1, math.ceil(n / target_sections))

    plan: List[Dict[str, Any]] = []

    for i in range(0, n, chunk_size):
        chunk = transcript_items[i:i + chunk_size]
        chunk_text = _clean_text(" ".join(item["text"] for item in chunk))

        keywords = _top_keywords(chunk_text, limit=4)

        if keywords:
            title = " / ".join(keywords)
        else:
            title = f"Раздел {len(plan) + 1}"

        plan.append(
            {
                "index": len(plan) + 1,
                "title": title,
                "description": _truncate(chunk_text, 250),
            }
        )

    return plan[:12]


def build_lecture_plan(
    transcript_items: List[Dict[str, Any]],
    model: str | None = None,
    callback: Optional[Callable[[str], None]] = None,
    use_llm: bool = True,
) -> List[Dict[str, Any]]:
    """
    Строит тематический план лекции.

    Возвращает не просто список строк, а список объектов:
    [
        {
            "index": 1,
            "title": "...",
            "description": "..."
        }
    ]

    title нужен для вывода в конспект.
    description потом пригодится для оценки принадлежности текста к разделу.
    """
    if not transcript_items:
        return [
            {
                "index": 1,
                "title": "Раздел 1",
                "description": "",
            }
        ]

    full_text = _clean_text(" ".join(item["text"] for item in transcript_items))

    if not full_text:
        return [
            {
                "index": 1,
                "title": "Раздел 1",
                "description": "",
            }
        ]

    total_duration_sec = max(float(item["end"]) for item in transcript_items)
    target_sections = _desired_section_count(total_duration_sec)

    if not use_llm:
        _emit(callback, "[summary] построение плана без LLM")
        return _fallback_lecture_plan(
            transcript_items=transcript_items,
            target_sections=target_sections,
        )

    compact_transcript = _compact_transcript_for_plan(
        transcript_items=transcript_items,
        max_chars_per_block=900,
        target_blocks=max(6, target_sections * 2),
    )

    prompt = f"""
Построй тематический план видеолекции по транскрипту.

Требования:
- разделы должны идти в порядке лекции;
- желательное число разделов: {target_sections};
- допустимое число разделов: от {max(1, target_sections - 2)} до {min(12, target_sections + 3)};
- названия разделов должны быть короткими: 3–8 слов;
- не добавляй темы, которых нет в транскрипте;
- не делай отдельный раздел для короткой фразы-перехода;
- каждый раздел должен отражать крупный смысловой блок лекции;
- description — 1 короткое предложение о содержании раздела.

Верни строго JSON-массив объектов без пояснений:

[
  {{
    "title": "Название раздела",
    "description": "Краткое описание раздела"
  }}
]

Транскрипт:
{compact_transcript}
""".strip()

    try:
        _emit(callback, "[summary] построение плана лекции с помощью LLM")

        raw = generate_with_llm(
            prompt=prompt,
            model=model,
            system=(
                "Ты строишь учебный план видеолекции. "
                "Пиши по-русски. "
                "Не добавляй факты, которых нет в транскрипте. "
                "Возвращай только валидный JSON."
            ),
            max_new_tokens=800,
        )

        parsed = _parse_json_list(raw)

        if parsed:
            plan = _normalize_plan_items(parsed)

            if plan:
                return plan[:12]

        _emit(callback, "[summary] LLM вернула некорректный план, fallback")

    except Exception as exc:
        _emit(callback, f"[summary] план через LLM не построен, fallback: {exc}")

    return _fallback_lecture_plan(
        transcript_items=transcript_items,
        target_sections=target_sections,
    )


# =========================================================
# 2. ОКНА ТЕКСТА, А НЕ ОТДЕЛЬНЫЕ ПРЕДЛОЖЕНИЯ
# =========================================================

def build_text_windows(
    transcript_items: List[Dict[str, Any]],
    min_window_sec: float = 20.0,
    max_window_sec: float = 55.0,
    min_chars: int = 350,
    max_chars: int = 1400,
) -> List[Dict[str, Any]]:
    """
    Окно = несколько соседних ASR-фрагментов.
    Это устойчивее, чем делить лекцию по отдельным предложениям.
    """
    windows: List[Dict[str, Any]] = []

    if not transcript_items:
        return windows

    i = 0
    idx = 1
    n = len(transcript_items)

    while i < n:
        block: List[Dict[str, Any]] = []
        j = i

        while j < n:
            block.append(transcript_items[j])

            start = float(block[0]["start"])
            end = float(block[-1]["end"])
            text = _clean_text(" ".join(item["text"] for item in block))

            dur = _duration(start, end)

            enough = dur >= min_window_sec and len(text) >= min_chars
            too_big = dur >= max_window_sec or len(text) >= max_chars

            j += 1

            if enough or too_big:
                break

        if not block:
            break

        text = _clean_text(" ".join(item["text"] for item in block))

        windows.append(
            {
                "index": idx,
                "item_start": i,
                "item_end": i + len(block) - 1,
                "start": float(block[0]["start"]),
                "end": float(block[-1]["end"]),
                "text": text,
                "vector": _vectorize(text),
                "cue": _transition_score(block[0]["text"]),
            }
        )

        idx += 1
        i += len(block)

    return windows

def _normalize_probs(probs: List[float], n: int) -> List[float]:
    if not probs or len(probs) != n:
        return [1.0 / n] * n

    cleaned = [max(1e-9, float(p)) for p in probs]
    total = sum(cleaned)

    if total <= 1e-12:
        return [1.0 / n] * n

    return [p / total for p in cleaned]


def _window_section_probs(
    window: Dict[str, Any],
    n_sections: int,
) -> List[float]:
    """
    Достаёт вероятности принадлежности окна к разделам.

    Если section_probs нет, строит грубое распределение по best_section_idx.
    """
    probs = window.get("section_probs")

    if probs and len(probs) == n_sections:
        return _normalize_probs(probs, n_sections)

    best_idx = window.get("best_section_idx")

    result = [0.05 / max(1, n_sections - 1)] * n_sections

    if best_idx is None:
        return [1.0 / n_sections] * n_sections

    best_idx = int(best_idx)

    if 0 <= best_idx < n_sections:
        result[best_idx] = 0.95

    return _normalize_probs(result, n_sections)


def decode_ordered_section_labels(
    classified_windows: List[Dict[str, Any]],
    n_sections: int,
    force_all_sections: bool = True,
) -> List[int]:
    """
    Сглаживает последовательность разделов.

    Разрешённые переходы:
    - остаться в текущем разделе;
    - перейти к следующему разделу.

    Это защищает от скачков вида:
    0 -> 1 -> 3 -> 2.
    """
    if not classified_windows:
        return []

    if n_sections <= 1:
        return [0] * len(classified_windows)

    n_windows = len(classified_windows)

    if n_windows < n_sections:
        force_all_sections = False

    neg_inf = -1e18

    dp = [[neg_inf] * n_sections for _ in range(n_windows)]
    parent = [[-1] * n_sections for _ in range(n_windows)]

    first_probs = _window_section_probs(classified_windows[0], n_sections)

    if force_all_sections:
        # Считаем, что лекция начинается с первого пункта плана.
        dp[0][0] = math.log(max(first_probs[0], 1e-9))
    else:
        for j in range(n_sections):
            # Небольшой штраф за старт не с первого раздела.
            dp[0][j] = math.log(max(first_probs[j], 1e-9)) - 0.4 * j

    for i in range(1, n_windows):
        probs = _window_section_probs(classified_windows[i], n_sections)

        for j in range(n_sections):
            emission = math.log(max(probs[j], 1e-9))

            candidates = []

            # Остаться в текущем разделе.
            candidates.append((dp[i - 1][j] + 0.08, j))

            # Перейти из предыдущего раздела в следующий.
            if j > 0:
                candidates.append((dp[i - 1][j - 1] - 0.02, j - 1))

            best_score, best_prev = max(candidates, key=lambda x: x[0])

            dp[i][j] = best_score + emission
            parent[i][j] = best_prev

    if force_all_sections:
        last_state = n_sections - 1
    else:
        last_state = max(range(n_sections), key=lambda j: dp[-1][j])

    labels = [last_state]

    for i in range(n_windows - 1, 0, -1):
        last_state = parent[i][last_state]

        if last_state < 0:
            last_state = labels[-1]

        labels.append(last_state)

    labels.reverse()
    return labels

def _dominant_label_for_segment(
    segment: Dict[str, Any],
    windows: List[Dict[str, Any]],
    labels: List[int],
) -> int:
    """
    Определяет главный label внутри сегмента.
    Используется после merge_short_segments, когда сегмент мог объединить
    несколько маленьких кусков.
    """
    weights = Counter()

    seg_start = float(segment["start"])
    seg_end = float(segment["end"])

    for window, label in zip(windows, labels):
        w_start = float(window["start"])
        w_end = float(window["end"])

        overlap = max(0.0, min(seg_end, w_end) - max(seg_start, w_start))

        if overlap > 0:
            weights[int(label)] += overlap

    if not weights:
        return 0

    return weights.most_common(1)[0][0]


def build_segments_from_classified_windows(
    transcript_items: List[Dict[str, Any]],
    classified_windows: List[Dict[str, Any]],
    plan: List[Any],
    min_section_sec: float = 30.0,
    force_all_sections: bool = True,
    callback: Optional[Callable[[str], None]] = None,
) -> List[Dict[str, Any]]:
    """
    Третий шаг:
    - сглаживает метки окон;
    - ищет границы разделов;
    - собирает текст каждого раздела;
    - назначает title/start/end.
    """
    if not transcript_items:
        return []

    if not classified_windows:
        return []

    sections = _normalize_plan_sections(plan)

    if not sections:
        return []

    n_sections = len(sections)

    _emit(callback, "[summary] сглаживание последовательности разделов")

    labels = decode_ordered_section_labels(
        classified_windows=classified_windows,
        n_sections=n_sections,
        force_all_sections=force_all_sections,
    )

    boundary_items: List[int] = []

    for i in range(len(classified_windows) - 1):
        current_label = labels[i]
        next_label = labels[i + 1]

        if current_label != next_label:
            boundary_items.append(int(classified_windows[i]["item_end"]))

    _emit(callback, f"[summary] найдено границ разделов: {len(boundary_items)}")

    segments = _segments_from_boundaries(
        transcript_items=transcript_items,
        boundary_items=boundary_items,
    )

    segments = merge_short_segments(
        segments=segments,
        min_section_sec=min_section_sec,
    )

    for segment in segments:
        label = _dominant_label_for_segment(
            segment=segment,
            windows=classified_windows,
            labels=labels,
        )

        label = max(0, min(label, n_sections - 1))

        segment["section_idx"] = label
        segment["title"] = sections[label]["title"]

    for idx, segment in enumerate(segments, start=1):
        segment["index"] = idx

    _emit(callback, f"[summary] итоговых разделов: {len(segments)}")

    return segments

# =========================================================
# 3. ГРУБАЯ СЕГМЕНТАЦИЯ ПО ОКНАМ
# =========================================================

def _score_window_boundaries(windows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    scores: List[Dict[str, Any]] = []

    for i in range(len(windows) - 1):
        left = windows[i]
        right = windows[i + 1]

        sim = _cosine_sim(left["vector"], right["vector"])
        cue = max(left.get("cue", 0.0) * 0.3, right.get("cue", 0.0))

        split_score = (1.0 - sim) + cue

        scores.append(
            {
                "after_window": left["index"],
                "boundary_item": left["item_end"],
                "boundary_time": float(left["end"]),
                "similarity": sim,
                "cue": cue,
                "split_score": split_score,
            }
        )

    return scores


def _segments_from_boundaries(
    transcript_items: List[Dict[str, Any]],
    boundary_items: List[int],
) -> List[Dict[str, Any]]:
    if not transcript_items:
        return []

    boundaries = [0]
    for b in sorted(set(boundary_items)):
        next_start = b + 1
        if 0 < next_start < len(transcript_items):
            boundaries.append(next_start)

    boundaries.append(len(transcript_items))
    boundaries = sorted(set(boundaries))

    segments: List[Dict[str, Any]] = []

    for i in range(len(boundaries) - 1):
        s = boundaries[i]
        e = boundaries[i + 1] - 1
        block = transcript_items[s:e + 1]

        if not block:
            continue

        segments.append(
            {
                "index": len(segments) + 1,
                "item_start": s,
                "item_end": e,
                "start": float(block[0]["start"]),
                "end": float(block[-1]["end"]),
                "text": _clean_text(" ".join(item["text"] for item in block)),
            }
        )

    return segments


def merge_short_segments(
    segments: List[Dict[str, Any]],
    min_section_sec: float = 30.0,
) -> List[Dict[str, Any]]:
    if not segments:
        return []

    merged = [dict(seg) for seg in segments]

    changed = True
    while changed and len(merged) > 1:
        changed = False

        for i, seg in enumerate(list(merged)):
            dur = _duration(seg["start"], seg["end"])

            if dur >= min_section_sec:
                continue

            if i == 0:
                target = 1
            elif i == len(merged) - 1:
                target = i - 1
            else:
                left_sim = _cosine_sim(_vectorize(merged[i - 1]["text"]), _vectorize(seg["text"]))
                right_sim = _cosine_sim(_vectorize(seg["text"]), _vectorize(merged[i + 1]["text"]))
                target = i - 1 if left_sim >= right_sim else i + 1

            a = min(i, target)
            b = max(i, target)

            merged[a] = {
                "index": merged[a]["index"],
                "item_start": min(merged[a]["item_start"], merged[b]["item_start"]),
                "item_end": max(merged[a]["item_end"], merged[b]["item_end"]),
                "start": min(merged[a]["start"], merged[b]["start"]),
                "end": max(merged[a]["end"], merged[b]["end"]),
                "text": _clean_text(merged[a]["text"] + " " + merged[b]["text"]),
            }

            merged.pop(b)
            changed = True
            break

    for idx, seg in enumerate(merged, start=1):
        seg["index"] = idx

    return merged


def rough_segment_transcript(
    transcript_items: List[Dict[str, Any]],
    windows: List[Dict[str, Any]],
    plan_titles: List[str],
    min_section_sec: float = 30.0,
    max_section_sec: float = 300.0,
) -> List[Dict[str, Any]]:
    if not transcript_items:
        return []

    if not windows:
        return _segments_from_boundaries(transcript_items, [])

    total_duration = _duration(transcript_items[0]["start"], transcript_items[-1]["end"])
    target_count = len(plan_titles) if plan_titles else _desired_section_count(total_duration)
    target_count = max(1, min(target_count, 12))

    boundary_scores = _score_window_boundaries(windows)
    sorted_candidates = sorted(boundary_scores, key=lambda x: x["split_score"], reverse=True)

    selected_items: List[int] = []

    for cand in sorted_candidates:
        if len(selected_items) >= target_count - 1:
            break

        b = int(cand["boundary_item"])

        left_start = transcript_items[0]["start"]
        for prev_b in sorted(selected_items):
            if prev_b < b:
                left_start = transcript_items[prev_b + 1]["start"]

        right_end = transcript_items[-1]["end"]
        for next_b in sorted(selected_items):
            if next_b > b:
                right_end = transcript_items[next_b]["end"]
                break

        left_dur = _duration(left_start, transcript_items[b]["end"])
        right_dur = _duration(transcript_items[b + 1]["start"], right_end)

        if left_dur < min_section_sec or right_dur < min_section_sec:
            continue

        selected_items.append(b)

    # Принудительно режем слишком длинные разделы.
    changed = True
    while changed:
        changed = False
        segments = _segments_from_boundaries(transcript_items, selected_items)

        for seg in segments:
            dur = _duration(seg["start"], seg["end"])
            if dur <= max_section_sec:
                continue

            best_item = None
            best_gap = 1e18

            for k in range(seg["item_start"], seg["item_end"]):
                left_dur = _duration(seg["start"], transcript_items[k]["end"])
                right_dur = _duration(transcript_items[k + 1]["start"], seg["end"])

                if left_dur < min_section_sec or right_dur < min_section_sec:
                    continue

                gap = abs(left_dur - max_section_sec)
                if gap < best_gap:
                    best_gap = gap
                    best_item = k

            if best_item is not None and best_item not in selected_items:
                selected_items.append(best_item)
                changed = True
                break

    segments = _segments_from_boundaries(transcript_items, selected_items)
    return merge_short_segments(segments, min_section_sec=min_section_sec)


# =========================================================
# 4. УТОЧНЕНИЕ ГРАНИЦ С ПОМОЩЬЮ LLM
# =========================================================

def _nearest_boundary_item_by_time(
    transcript_items: List[Dict[str, Any]],
    boundary_time: float,
) -> Optional[int]:
    if len(transcript_items) < 2:
        return None

    best_i = None
    best_gap = 1e18

    for i in range(len(transcript_items) - 1):
        t = float(transcript_items[i]["end"])
        gap = abs(t - boundary_time)

        if gap < best_gap:
            best_gap = gap
            best_i = i

    return best_i


def refine_boundaries_with_llm(
    transcript_items: List[Dict[str, Any]],
    segments: List[Dict[str, Any]],
    windows: List[Dict[str, Any]],
    plan_titles: List[str],
    model: str | None = None,
    callback: Optional[Callable[[str], None]] = None,
    min_section_sec: float = 30.0,
    search_radius_sec: float = 70.0,
) -> List[Dict[str, Any]]:
    """
    LLM не получает право произвольно нарезать всю лекцию.
    Она только выбирает лучшую границу из допустимых временных кандидатов рядом
    с уже найденной грубой границей.
    """
    if len(segments) <= 1:
        return segments

    boundary_items = [int(seg["item_end"]) for seg in segments[:-1]]

    for b_idx, old_item in enumerate(list(boundary_items)):
        old_time = float(transcript_items[old_item]["end"])

        candidates = []
        for w in windows:
            candidate_time = float(w["end"])
            if abs(candidate_time - old_time) <= search_radius_sec:
                item = _nearest_boundary_item_by_time(transcript_items, candidate_time)
                if item is None:
                    continue

                candidates.append(
                    {
                        "time": float(transcript_items[item]["end"]),
                        "item": int(item),
                        "text_before": _truncate(transcript_items[item]["text"], 220),
                        "text_after": _truncate(transcript_items[item + 1]["text"], 220)
                        if item + 1 < len(transcript_items)
                        else "",
                    }
                )

        if not candidates:
            continue

        # Убираем дубли по item.
        unique_candidates = []
        seen_items = set()

        for cand in candidates:
            if cand["item"] in seen_items:
                continue
            seen_items.add(cand["item"])
            unique_candidates.append(cand)

        left_seg = segments[b_idx]
        right_seg = segments[b_idx + 1]

        plan_text = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(plan_titles))

        candidates_text = "\n".join(
            (
                f"- time={round(c['time'], 2)}, item={c['item']}\n"
                f"  before: {c['text_before']}\n"
                f"  after: {c['text_after']}"
            )
            for c in unique_candidates
        )

        prompt = (
            "Нужно уточнить границу между двумя тематическими разделами видеолекции.\n"
            "Выбери одну границу только из списка candidate boundaries.\n"
            "Нельзя выбирать время, которого нет в списке.\n\n"
            "Грубый план лекции:\n"
            f"{plan_text}\n\n"
            "Левый раздел, конец которого уточняется:\n"
            f"{_truncate(left_seg['text'], 1200)}\n\n"
            "Правый раздел, начало которого уточняется:\n"
            f"{_truncate(right_seg['text'], 1200)}\n\n"
            "Candidate boundaries:\n"
            f"{candidates_text}\n\n"
            "Верни строго JSON такого вида:\n"
            '{"boundary_time": 123.45, "reason": "кратко"}'
        )

        try:
            _emit(
                callback,
                f"[summary] LLM-уточнение границы {b_idx + 1}/{len(boundary_items)} около {_format_ts(old_time)}",
            )

            raw = summarize_with_llm(prompt, model=model)
            parsed = _parse_json_object(raw)

            if not parsed or "boundary_time" not in parsed:
                continue

            chosen_time = float(parsed["boundary_time"])
            best = min(unique_candidates, key=lambda c: abs(c["time"] - chosen_time))
            new_item = int(best["item"])

            test_items = list(boundary_items)
            test_items[b_idx] = new_item
            test_segments = _segments_from_boundaries(transcript_items, test_items)

            valid = True
            for seg in test_segments:
                if _duration(seg["start"], seg["end"]) < min_section_sec:
                    valid = False
                    break

            if valid:
                boundary_items[b_idx] = new_item

        except Exception as exc:
            _emit(callback, f"[summary] граница не уточнена LLM, fallback: {exc}")

    refined = _segments_from_boundaries(transcript_items, boundary_items)
    refined = merge_short_segments(refined, min_section_sec=min_section_sec)

    return refined


def assign_titles_to_segments(
    segments: List[Dict[str, Any]],
    plan_titles: List[str],
) -> List[Dict[str, Any]]:
    if not segments:
        return []

    if not plan_titles:
        for i, seg in enumerate(segments, start=1):
            keywords = _top_keywords(seg["text"], limit=4)
            seg["title"] = " / ".join(keywords) if keywords else f"Раздел {i}"
        return segments

    plan_vecs = [_vectorize(title) for title in plan_titles]
    last_plan_idx = 0

    for i, seg in enumerate(segments):
        seg_vec = _vectorize(seg["text"])

        best_idx = min(last_plan_idx, len(plan_titles) - 1)
        best_score = -1.0

        for j in range(last_plan_idx, len(plan_titles)):
            score = _cosine_sim(seg_vec, plan_vecs[j])

            # лёгкий штраф за слишком далёкий пункт плана
            score -= 0.03 * abs(j - i)

            if score > best_score:
                best_score = score
                best_idx = j

        seg["title"] = plan_titles[best_idx]

        if best_idx < len(plan_titles) - 1:
            last_plan_idx = best_idx

    for idx, seg in enumerate(segments, start=1):
        seg["index"] = idx
        seg.setdefault("title", f"Раздел {idx}")

    return segments


# =========================================================
# 5. ВЫБОР КАДРОВ ПО ТАЙМИНГУ И СМЫСЛОВОЙ БЛИЗОСТИ
# =========================================================

def _frame_time_score(frame_time: float, start: float, end: float) -> float:
    center = (start + end) / 2.0
    half = max(12.0, (end - start) / 2.0)
    return max(0.0, 1.0 - abs(frame_time - center) / (half + 12.0))


def _frame_information_score(ocr_text: str) -> float:
    ocr_text = _clean_text(ocr_text)

    if not ocr_text:
        return 0.10

    token_count = len(_tokenize(ocr_text))
    line_count = len([x for x in ocr_text.splitlines() if x.strip()])

    score = 0.2 + token_count / 45.0 + line_count / 30.0
    return max(0.0, min(1.0, score))


def _select_frame_limit_for_segment(
    segment: Dict[str, Any],
    candidates: List[Dict[str, Any]],
) -> int:
    if not candidates:
        return 0

    dur = _duration(segment["start"], segment["end"])

    if dur < 90:
        return 1

    if dur < 240:
        return 1 if len(candidates) <= 2 else 2

    return 2 if len(candidates) <= 4 else 3


def attach_frames_to_segments(
    segments: List[Dict[str, Any]],
    frames: List[Dict[str, Any]],
    ocr_map: Dict[str, str],
    time_margin_sec: float = 18.0,
) -> List[Dict[str, Any]]:
    frame_records: List[Dict[str, Any]] = []

    for frame in frames:
        stem = frame.get("stem")
        ocr_text = _clean_text(ocr_map.get(stem, ""))

        frame_records.append(
            {
                "time": float(frame["time"]),
                "path": frame["path"],
                "stem": stem,
                "ocr_text": ocr_text,
                "vector": _vectorize(ocr_text),
            }
        )

    for seg in segments:
        start = float(seg["start"])
        end = float(seg["end"])

        seg_text = _clean_text(seg.get("title", "") + " " + seg.get("text", ""))
        seg_vec = _vectorize(seg_text)

        candidates: List[Dict[str, Any]] = []

        for frame in frame_records:
            time_val = float(frame["time"])

            if time_val < start - time_margin_sec or time_val > end + time_margin_sec:
                continue

            time_score = _frame_time_score(time_val, start, end)
            semantic_score = _cosine_sim(seg_vec, frame["vector"]) if frame["ocr_text"] else 0.0
            info_score = _frame_information_score(frame["ocr_text"])

            # Тайминг важен, но OCR-смысл тоже влияет.
            total_score = 0.45 * time_score + 0.40 * semantic_score + 0.15 * info_score

            candidates.append(
                {
                    **frame,
                    "time_score": time_score,
                    "semantic_score": semantic_score,
                    "info_score": info_score,
                    "score": total_score,
                }
            )

        candidates.sort(
            key=lambda x: (
                x["score"],
                x["semantic_score"],
                x["info_score"],
                -abs(x["time"] - (start + end) / 2.0),
            ),
            reverse=True,
        )

        limit = _select_frame_limit_for_segment(seg, candidates)
        selected: List[Dict[str, Any]] = []

        for cand in candidates:
            # Если уже есть хотя бы один хороший кадр, совсем слабые не добавляем.
            if selected and cand["score"] < 0.22:
                continue

            duplicate = False

            for prev in selected:
                same_text = _cosine_sim(cand["vector"], prev["vector"])
                close_time = abs(cand["time"] - prev["time"]) <= 25.0

                if same_text >= 0.92 and close_time:
                    duplicate = True
                    break

            if duplicate:
                continue

            selected.append(cand)

            if len(selected) >= limit:
                break

        seg["frames"] = selected

        if not selected:
            seg["mode"] = "text_only"
        elif len(selected) == 1:
            seg["mode"] = "mixed"
        else:
            seg["mode"] = "rich"

    return segments


def attach_all_keyframes_by_time(
    segments: List[Dict[str, Any]],
    frames: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Строго распределяет все ключевые кадры по разделам.
    Каждый кадр попадает в тот раздел, внутри временного интервала которого
    находится frame["time"].

    Результат кладётся в segment["all_keyframes"].
    """
    if not segments:
        return segments

    for segment in segments:
        segment["all_keyframes"] = []

    for frame in frames:
        frame_time = float(frame["time"])

        best_segment = None

        for i, segment in enumerate(segments):
            start = float(segment["start"])
            end = float(segment["end"])

            is_last = i == len(segments) - 1

            if start <= frame_time < end or (is_last and start <= frame_time <= end):
                best_segment = segment
                break

        # Если кадр немного выпал из-за неточной границы,
        # прикрепляем к ближайшему разделу.
        if best_segment is None:
            best_segment = min(
                segments,
                key=lambda seg: min(
                    abs(frame_time - float(seg["start"])),
                    abs(frame_time - float(seg["end"])),
                )
            )

        best_segment["all_keyframes"].append(frame)

    for segment in segments:
        segment["all_keyframes"].sort(key=lambda x: float(x["time"]))

    return segments

# =========================================================
# 6. СУММАРИЗАЦИЯ РАЗДЕЛОВ
# =========================================================

def _fallback_summary(source_text: str, max_sentences: int = 4) -> str:
    text = _clean_text(source_text)

    if not text:
        return "- Нет данных для конспекта."

    parts = _split_sentences(text)
    parts = parts[:max_sentences]

    if not parts:
        return f"- {text[:300]}"

    return "\n".join(f"- {p}" for p in parts)


def _compose_segment_source_text(segment: Dict[str, Any]) -> str:
    parts: List[str] = []

    title = _clean_text(segment.get("title", ""))
    speech = _clean_text(segment.get("text", ""))

    if title:
        parts.append("НАЗВАНИЕ РАЗДЕЛА:\n" + title)

    if speech:
        parts.append("ТЕКСТ ЛЕКТОРА:\n" + speech)

    frames = segment.get("frames") or []

    ocr_parts = []
    for idx, frame in enumerate(frames, start=1):
        ocr_text = _clean_text(frame.get("ocr_text", ""))
        if not ocr_text:
            continue

        ocr_parts.append(
            f"КАДР {idx}, время {_format_ts(frame['time'])}:\n{ocr_text}"
        )

    if ocr_parts:
        parts.append("ТЕКСТ СО СЛАЙДОВ / ДОСКИ:\n" + "\n\n".join(ocr_parts))

    return "\n\n".join(parts).strip()

def summarize_segment(
    segment: Dict[str, Any],
    model: str | None = None,
    min_chars_for_llm: int = 220,
    max_chars_for_llm: int = 3500,
) -> str:
    source_text = _compose_segment_source_text(segment)

    if not source_text:
        return "- Для этого раздела не удалось получить текст."

    if len(source_text) < min_chars_for_llm:
        return _fallback_summary(source_text)

    # Жёстко ограничиваем размер входа для LLM.
    # Это важно для RTX 3060 и обычной ОЗУ.
    source_text_for_llm = _truncate(source_text, max_chars_for_llm)

    prompt = (
        "Сделай конспект для одного тематического раздела видеолекции.\n\n"
        "Требования:\n"
        "- 3-5 коротких пунктов;\n"
        "- по-русски;\n"
        "- не добавляй факты от себя;\n"
        "- сохрани термины, определения, числа и обозначения;\n"
        "- если OCR шумный, игнорируй мусор;\n"
        "- если на слайде есть важный текст, включи его в конспект;\n"
        "- не пиши вводные фразы вроде 'в этом фрагменте рассказывается'.\n\n"
        "Формат ответа: markdown-список.\n\n"
        f"{source_text_for_llm}"
    )

    try:
        summary = summarize_with_llm(prompt, model=model).strip()
        if summary:
            return summary
    except Exception as exc:
        print(f"[summary] LLM не смог обработать раздел, fallback: {exc}")

    return _fallback_summary(source_text)


def summarize_segments(
    segments: List[Dict[str, Any]],
    model: str | None = None,
    min_chars_for_llm: int = 220,
    callback: Optional[Callable[[str], None]] = None,
) -> None:
    total = len(segments)

    for i, segment in enumerate(segments, start=1):
        _emit(
            callback,
            f"[summary] раздел {i}/{total}: "
            f"{_format_ts(segment['start'])}–{_format_ts(segment['end'])} | "
            f"frames={len(segment.get('frames') or [])} | "
            f"mode={segment.get('mode', 'text_only')}",
        )

        # segment["summary"] = summarize_segment(
        #     segment=segment,
        #     model=model,
        #     min_chars_for_llm=min_chars_for_llm,
        # )
        segment["summary"] = summarize_segment(
            segment=segment,
            model=model,
            min_chars_for_llm=min_chars_for_llm,
            max_chars_for_llm=3500,
        )


# =========================================================
# 7. АДАПТИВНАЯ СБОРКА MARKDOWN
# =========================================================

def _overall_visual_mode(segments: List[Dict[str, Any]]) -> str:
    if not segments:
        return "text_only"

    total_images = sum(len(seg.get("frames") or []) for seg in segments)

    if total_images == 0:
        return "text_only"

    if total_images <= len(segments):
        return "mixed"

    return "rich"


def _mode_ru(mode: str) -> str:
    if mode == "text_only":
        return "текстовый"
    if mode == "mixed":
        return "смешанный"
    if mode == "rich":
        return "насыщенный"
    return mode


def render_markdown(
    segments: List[Dict[str, Any]],
    out_path: str | Path,
    title: str = "Конспект лекции",
    include_ocr: bool = True,
    plan_titles: Optional[List[str]] = None,
) -> str:
    out_path = Path(out_path)
    out_dir = out_path.parent

    overall_mode = _overall_visual_mode(segments)

    lines: List[str] = [
        f"# {title}",
        "",
        f"**Тип конспекта:** {_mode_ru(overall_mode)}",
        "",
    ]

    if plan_titles:
        lines.append("## Грубый план лекции")
        lines.append("")

        for idx, item in enumerate(plan_titles, start=1):
            lines.append(f"{idx}. {item}")

        lines.append("")

    for segment in segments:
        start_ts = _format_ts(float(segment["start"]))
        end_ts = _format_ts(float(segment["end"]))
        segment_title = segment.get("title") or f"Раздел {segment['index']}"
        mode = segment.get("mode", "text_only")

        lines.append(f"## Раздел {segment['index']}: {segment_title}")
        lines.append("")
        lines.append(f"**Время:** {start_ts}–{end_ts}")
        lines.append("")
        lines.append(f"**Формат раздела:** {_mode_ru(mode)}")
        lines.append("")

        frames = segment.get("frames") or []

        for frame in frames:
            img_path = Path(frame["path"])
            if not img_path.exists():
                continue

            rel_img = _relative_posix(img_path, out_dir)
            img_ts = _format_ts(float(frame["time"]))

            lines.append(f"![{_escape_md(segment_title)} | {img_ts}]({rel_img})")
            lines.append("")

        summary = _clean_text(segment.get("summary", ""))

        if summary:
            lines.append("**Краткий конспект:**")
            lines.append(summary)
            lines.append("")

        if include_ocr and frames:
            ocr_blocks = []

            for frame in frames:
                ocr_text = _clean_text(frame.get("ocr_text", ""))
                if not ocr_text:
                    continue

                ocr_blocks.append(
                    f"- {_format_ts(float(frame['time']))}: {_escape_md(_truncate(ocr_text, 500))}"
                )

            if ocr_blocks:
                lines.append("**Фрагменты текста на слайдах / доске:**")
                lines.extend(ocr_blocks)
                lines.append("")

        lines.append("---")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


# =========================================================
# 8. ГЛАВНАЯ ФУНКЦИЯ. ИНТЕРФЕЙС ОСТАЁТСЯ ПРЕЖНИМ
# =========================================================

def build_multimodal_summary(
    transcript_path: str | Path,
    out_path: str | Path,
    frame_paths: Optional[Iterable[str | Path]] = None,
    keyframes: Optional[Iterable[Any]] = None,
    ocr_dir: str | Path | None = None,
    model: str | None = None,
    title: str = "Конспект лекции",
    include_ocr: bool = True,
    min_chars_for_llm: int = 220,
    USE_OCR: bool = False,
    callback: Optional[Callable[[str], None]] = None,
) -> Tuple[Path, List[Dict[str, Any]]]:
    transcript_path = Path(transcript_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    _emit(callback, "[summary] чтение транскрипта")
    transcript_items = parse_transcript(transcript_path)
    if USE_OCR:
        _emit(callback, "[summary] чтение OCR")
        ocr_map = load_ocr_map(ocr_dir)
    else:
        ocr_map = {}

    _emit(callback, "[summary] подготовка кадров")
    frames = normalize_frames(frame_paths=frame_paths, keyframes=keyframes)

    if not transcript_items:
        segments = [
            {
                "index": 1,
                "start": 0.0,
                "end": 0.0,
                "title": "Конспект",
                "text": "",
                "frames": [],
                "mode": "text_only",
                "summary": "- Транскрипт пустой или не найден.",
            }
        ]

        markdown = render_markdown(
            segments=segments,
            out_path=out_path,
            title=title,
            include_ocr=include_ocr,
            plan_titles=[],
        )
        out_path.write_text(markdown, encoding="utf-8")
        return out_path, segments

    total_duration_sec = max(float(item["end"]) for item in transcript_items)
    _emit(callback, f"[summary] длительность лекции по транскрипту: {_format_ts(total_duration_sec)}")

    plan = build_lecture_plan(
        transcript_items=transcript_items,
        model=model,
        callback=callback,
        use_llm=True,
    )
    
    plan_titles = [item["title"] for item in plan]

    _emit(callback, "[summary] разбиение транскрипта на текстовые окна")
    windows = build_text_windows(
        transcript_items=transcript_items,
        min_window_sec=20.0,
        max_window_sec=55.0,
        min_chars=350,
        max_chars=1400,
    )

        
    classified_windows = classify_text_windows_by_plan(
        transcript_items=transcript_items,
        windows=windows,
        plan=plan,
        min_split_sec=10.0,
        callback=callback,
    )
    _emit(callback, f"[summary] окон текста: {len(windows)}")

    segments = build_segments_from_classified_windows(
        transcript_items=transcript_items,
        classified_windows=classified_windows,
        plan=plan,
        min_section_sec=30.0,
        force_all_sections=True,
        callback=callback,
    )
    
    # Все ключевые кадры по строгому таймингу.
    segments = attach_all_keyframes_by_time(
        segments=segments,
        frames=frames,
    )
   
    _emit(callback, "[summary] подбор кадров по таймингу и смысловой близости")
    # Лучшие кадры для отображения в конспекте.
    segments = attach_frames_to_segments(
        segments=segments,
        frames=frames,
        ocr_map=ocr_map,
        time_margin_sec=18.0,
    )
    
    for segment in segments:
        print(
            f"Раздел {segment['index']}: {segment['title']} | "
            f"{_format_ts(segment['start'])}–{_format_ts(segment['end'])} | "
            f"keyframes={len(segment.get('all_keyframes') or [])} | "
            f"selected={len(segment.get('frames') or [])}"
        )

    _emit(callback, "[summary] суммаризация разделов")
    summarize_segments(
        segments=segments,
        model=model,
        min_chars_for_llm=min_chars_for_llm,
        callback=callback,
    )

    _emit(callback, "[summary] сборка markdown")
    markdown = render_markdown(
        segments=segments,
        out_path=out_path,
        title=title,
        include_ocr=include_ocr,
        plan_titles=plan_titles,
    )

    out_path.write_text(markdown, encoding="utf-8")

    return out_path, segments

