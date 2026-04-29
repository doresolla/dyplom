from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
import math
from collections import Counter

from text_processing.LLMsummary import summarize_with_llm

from utils import _emit

TRANSCRIPT_LINE_RE = re.compile(
    r"^\[(?P<start>\d+(?:\.\d+)?)\s*-\s*(?P<end>\d+(?:\.\d+)?)\]\s*(?P<text>.*)$"
)
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


def _clean_text(text: str) -> str:
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
    if match:
        try:
            return float(match.group("time"))
        except Exception:
            return None
    return None

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


def build_text_windows(
    transcript_items: List[Dict[str, Any]],
    max_items: int = 6,
    stride: int = 3,
    min_window_sec: float = 20.0,
) -> List[Dict[str, Any]]:
    windows: List[Dict[str, Any]] = []
    if not transcript_items:
        return windows

    n = len(transcript_items)
    i = 0
    idx = 1

    while i < n:
        j = min(n, i + max_items)

        # немного расширяем окно, если оно слишком короткое по времени
        while j < n and _duration(transcript_items[i]["start"], transcript_items[j - 1]["end"]) < min_window_sec:
            j += 1

        block = transcript_items[i:j]
        text = _clean_text(" ".join(item["text"] for item in block))
        windows.append(
            {
                "index": idx,
                "item_start": i,
                "item_end": j - 1,
                "start": float(block[0]["start"]),
                "end": float(block[-1]["end"]),
                "text": text,
                "vector": _vectorize(text),
                "cue": _transition_cue_score(block[0]["text"]),
            }
        )

        idx += 1
        if j >= n:
            break
        i += max(1, stride)

    return windows


def _score_adjacent_windows(windows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    scores: List[Dict[str, Any]] = []
    for i in range(len(windows) - 1):
        left = windows[i]
        right = windows[i + 1]
        sim = _cosine_sim(left["vector"], right["vector"])
        cue = max(left.get("cue", 0.0) * 0.4, right.get("cue", 0.0))
        split_score = max(0.0, 1.0 - sim) + cue
        scores.append(
            {
                "after_window": left["index"],
                "left_window": left,
                "right_window": right,
                "similarity": sim,
                "cue": cue,
                "split_score": split_score,
            }
        )
    return scores


def _desired_section_count(total_duration_sec: float) -> int:
    if total_duration_sec <= 0:
        return 1
    return max(1, min(12, int(round(total_duration_sec / 180.0)) + 1))


def build_lecture_plan(
    transcript_items: List[Dict[str, Any]],
    model: str | None = None,
    callback: Optional[Callable[[str], None]] = None,
) -> List[str]:
    if not transcript_items:
        return ["Раздел 1"]

    full_text = _clean_text(" ".join(item["text"] for item in transcript_items))
    if not full_text:
        return ["Раздел 1"]

    total_duration_sec = max(float(item["end"]) for item in transcript_items)
    target_sections = _desired_section_count(total_duration_sec)

    excerpt = full_text[:14000]
    prompt = (
        "Нужно построить грубый план видеолекции только по транскрипту.\n\n"
        f"Ожидаемое число разделов: {target_sections}.\n"
        "Верни только нумерованный список названий разделов, без пояснений и без markdown-таблиц.\n"
        "Названия должны быть короткими: 3-8 слов.\n"
        "Если тем меньше, не выдумывай лишние.\n\n"
        f"Транскрипт:\n{excerpt}"
    )

    try:
        _emit(callback, "[summary] построение грубого плана лекции")
        raw = summarize_with_llm(prompt, model=model).strip()
        titles: List[str] = []
        for line in raw.splitlines():
            line = _clean_text(line)
            line = re.sub(r"^[-*•]+\s*", "", line)
            line = re.sub(r"^\d+[).:-]?\s*", "", line)
            if not line:
                continue
            if len(line) > 90:
                line = _truncate(line, 90)
            titles.append(line)
        titles = [t for i, t in enumerate(titles) if t and t not in titles[:i]]
        if titles:
            return titles[:12]
    except Exception as exc:
        _emit(callback, f"[summary] план LLM не построен, fallback: {exc}")

    keywords = _top_keywords(full_text, limit=target_sections)
    if not keywords:
        return [f"Раздел {i + 1}" for i in range(target_sections)]
    return [f"Тема: {kw}" for kw in keywords]


def rough_segment_transcript(
    transcript_items: List[Dict[str, Any]],
    windows: List[Dict[str, Any]],
    min_section_sec: float = 25.0,
    max_section_sec: float = 240.0,
) -> List[Dict[str, Any]]:
    if not transcript_items:
        return []

    if not windows:
        text = _clean_text(" ".join(item["text"] for item in transcript_items))
        return [
            {
                "index": 1,
                "start": float(transcript_items[0]["start"]),
                "end": float(transcript_items[-1]["end"]),
                "item_start": 0,
                "item_end": len(transcript_items) - 1,
                "text": text,
            }
        ]

    scores = _score_adjacent_windows(windows)
    split_points: List[int] = []

    if scores:
        split_values = [s["split_score"] for s in scores]
        mean_val = sum(split_values) / len(split_values)
        std_val = math.sqrt(sum((v - mean_val) ** 2 for v in split_values) / len(split_values))
        threshold = mean_val + max(0.12, 0.35 * std_val)

        last_split_item = 0
        for score in scores:
            boundary_item = int(score["left_window"]["item_end"])
            next_item = boundary_item + 1
            if next_item >= len(transcript_items):
                continue

            dur = _duration(transcript_items[last_split_item]["start"], transcript_items[boundary_item]["end"])
            if score["split_score"] >= threshold and dur >= min_section_sec:
                split_points.append(boundary_item)
                last_split_item = next_item

    # добавляем сплиты по длине, если блок стал слишком большим
    adjusted_splits: List[int] = []
    prev = 0
    all_candidates = sorted(set(split_points))
    candidate_idx = 0

    while prev < len(transcript_items):
        next_boundary = all_candidates[candidate_idx] if candidate_idx < len(all_candidates) else len(transcript_items) - 1
        current_dur = _duration(transcript_items[prev]["start"], transcript_items[next_boundary]["end"])

        if current_dur <= max_section_sec or prev == next_boundary:
            if next_boundary < len(transcript_items) - 1:
                adjusted_splits.append(next_boundary)
            prev = next_boundary + 1
            candidate_idx += 1 if candidate_idx < len(all_candidates) else 0
            continue

        # ищем внутреннюю точку разрыва ближе к max_section_sec
        best_k = None
        best_gap = 1e18
        for k in range(prev, next_boundary):
            dur = _duration(transcript_items[prev]["start"], transcript_items[k]["end"])
            gap = abs(dur - max_section_sec)
            if dur >= min_section_sec and gap < best_gap:
                best_gap = gap
                best_k = k

        if best_k is None:
            best_k = min(next_boundary - 1, prev + 1)

        adjusted_splits.append(best_k)
        prev = best_k + 1

    boundaries = [0] + [b + 1 for b in sorted(set(adjusted_splits)) if b + 1 < len(transcript_items)] + [len(transcript_items)]

    segments: List[Dict[str, Any]] = []
    for i in range(len(boundaries) - 1):
        s = boundaries[i]
        e = boundaries[i + 1] - 1
        block = transcript_items[s:e + 1]
        segments.append(
            {
                "index": i + 1,
                "item_start": s,
                "item_end": e,
                "start": float(block[0]["start"]),
                "end": float(block[-1]["end"]),
                "text": _clean_text(" ".join(item["text"] for item in block)),
            }
        )

    return merge_short_segments(segments, min_section_sec=min_section_sec)


def merge_short_segments(
    segments: List[Dict[str, Any]],
    min_section_sec: float = 25.0,
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

    for i, seg in enumerate(merged, start=1):
        seg["index"] = i
    return merged


def refine_segments_with_plan(
    segments: List[Dict[str, Any]],
    plan_titles: List[str],
) -> List[Dict[str, Any]]:
    if not segments:
        return []

    refined = [dict(seg) for seg in segments]

    # Сначала слабое уточнение границ: сливаем соседние сегменты,
    # если они обе очень похожи на один и тот же пункт плана.
    if plan_titles and len(refined) > 1:
        plan_vecs = [_vectorize(t) for t in plan_titles]
        changed = True
        while changed and len(refined) > 1:
            changed = False
            for i in range(len(refined) - 1):
                left = refined[i]
                right = refined[i + 1]
                left_v = _vectorize(left["text"])
                right_v = _vectorize(right["text"])

                left_best = max(range(len(plan_vecs)), key=lambda j: _cosine_sim(left_v, plan_vecs[j]))
                right_best = max(range(len(plan_vecs)), key=lambda j: _cosine_sim(right_v, plan_vecs[j]))
                bridge_sim = _cosine_sim(left_v, right_v)
                total_dur = _duration(left["start"], right["end"])

                if left_best == right_best and bridge_sim >= 0.18 and total_dur <= 360.0:
                    refined[i] = {
                        "index": left["index"],
                        "item_start": min(left["item_start"], right["item_start"]),
                        "item_end": max(left["item_end"], right["item_end"]),
                        "start": min(left["start"], right["start"]),
                        "end": max(left["end"], right["end"]),
                        "text": _clean_text(left["text"] + " " + right["text"]),
                    }
                    refined.pop(i + 1)
                    changed = True
                    break

    # Затем присваиваем заголовки разделам
    if plan_titles:
        plan_vecs = [_vectorize(t) for t in plan_titles]
        next_min_idx = 0
        for seg in refined:
            seg_v = _vectorize(seg["text"])
            best_idx = next_min_idx
            best_score = -1.0
            for j in range(next_min_idx, len(plan_titles)):
                score = _cosine_sim(seg_v, plan_vecs[j])
                if score > best_score:
                    best_score = score
                    best_idx = j
            seg["title"] = plan_titles[best_idx]
            if best_idx < len(plan_titles) - 1:
                next_min_idx = best_idx
    else:
        for i, seg in enumerate(refined, start=1):
            keywords = _top_keywords(seg["text"], limit=4)
            seg["title"] = " / ".join(keywords) if keywords else f"Раздел {i}"

    for i, seg in enumerate(refined, start=1):
        seg["index"] = i
        seg.setdefault("title", f"Раздел {i}")

    return refined


def _frame_time_score(frame_time: float, start: float, end: float) -> float:
    center = (start + end) / 2.0
    span = max(10.0, (end - start) / 2.0 + 8.0)
    return max(0.0, 1.0 - abs(frame_time - center) / span)


def _frame_information_score(ocr_text: str) -> float:
    ocr_text = _clean_text(ocr_text)
    if not ocr_text:
        return 0.15
    token_count = len(_tokenize(ocr_text))
    return min(1.0, 0.2 + token_count / 40.0)


def _select_count_for_segment(segment: Dict[str, Any], candidates: List[Dict[str, Any]]) -> int:
    dur = _duration(segment["start"], segment["end"])
    if not candidates:
        return 0
    if dur < 90:
        return 1
    if dur < 240:
        return 1 if len(candidates) < 3 else 2
    return 2 if len(candidates) < 5 else 3


def attach_frames_to_segments(
    segments: List[Dict[str, Any]],
    frames: List[Dict[str, Any]],
    ocr_map: Dict[str, str],
) -> List[Dict[str, Any]]:
    if not segments:
        return []

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
        seg["frames"] = []
        seg_vec = _vectorize(seg.get("text", "") + " " + seg.get("title", ""))
        start = float(seg["start"])
        end = float(seg["end"])
        candidates: List[Dict[str, Any]] = []

        for frame in frame_records:
            time_val = frame["time"]
            if time_val < start - 10.0 or time_val > end + 10.0:
                continue

            time_score = _frame_time_score(time_val, start, end)
            text_score = _cosine_sim(seg_vec, frame["vector"]) if frame["ocr_text"] else 0.0
            info_score = _frame_information_score(frame["ocr_text"])
            total_score = 0.5 * time_score + 0.35 * text_score + 0.15 * info_score

            candidates.append(
                {
                    **frame,
                    "time_score": time_score,
                    "text_score": text_score,
                    "info_score": info_score,
                    "score": total_score,
                }
            )

        candidates.sort(key=lambda x: (x["score"], x["info_score"], -abs(x["time"] - (start + end) / 2.0)), reverse=True)

        limit = _select_count_for_segment(seg, candidates)
        selected: List[Dict[str, Any]] = []
        for cand in candidates:
            if cand["score"] < 0.18 and selected:
                continue

            is_duplicate = False
            for prev in selected:
                text_dup = _cosine_sim(cand["vector"], prev["vector"])
                if text_dup >= 0.92 and abs(cand["time"] - prev["time"]) <= 20.0:
                    is_duplicate = True
                    break
            if is_duplicate:
                continue

            selected.append(cand)
            if len(selected) >= limit:
                break

        seg["frames"] = selected
        seg["mode"] = "text_only" if not selected else ("mixed" if len(selected) == 1 else "rich")

    return segments


# =========================================================
# SUMMARY GENERATION
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
    speech = _clean_text(segment.get("text", ""))
    parts: List[str] = []

    if segment.get("title"):
        parts.append("ПРЕДПОЛАГАЕМОЕ НАЗВАНИЕ РАЗДЕЛА:\n" + _clean_text(segment["title"]))
    if speech:
        parts.append("ТЕКСТ ЛЕКТОРА:\n" + speech)

    frames = segment.get("frames") or []
    if frames:
        ocr_chunks = []
        for idx, frame in enumerate(frames, start=1):
            ocr_text = _clean_text(frame.get("ocr_text", ""))
            if not ocr_text:
                continue
            ocr_chunks.append(
                f"КАДР {idx} ({_format_ts(frame['time'])}):\n{ocr_text}"
            )
        if ocr_chunks:
            parts.append("ТЕКСТ СО СЛАЙДОВ / ДОСКИ:\n" + "\n\n".join(ocr_chunks))

    return "\n\n".join(parts).strip()


def summarize_segment(
    segment: Dict[str, Any],
    model: str | None = None,
    min_chars_for_llm: int = 220,
) -> str:
    source_text = _compose_segment_source_text(segment)
    if not source_text:
        return "- Для этого раздела не удалось получить текст."

    if len(source_text) < min_chars_for_llm:
        return _fallback_summary(source_text)

    prompt = (
        "Сделай конспект для одного тематического раздела видеолекции.\n\n"
        "Требования:\n"
        "- 3-6 коротких пунктов;\n"
        "- по-русски;\n"
        "- не добавляй факты от себя;\n"
        "- сохрани термины, определения, числа и обозначения;\n"
        "- если OCR шумный, игнорируй мусор;\n"
        "- если на слайде есть важный текст, включи его в конспект;\n"
        "- не пиши вводные фразы вроде 'в этом фрагменте рассказывается'.\n\n"
        "Формат ответа: markdown-список.\n\n"
        f"{source_text}"
    )

    try:
        summary = summarize_with_llm(prompt, model=model).strip()
        if summary:
            return summary
    except Exception:
        pass
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
            f"frames={len(segment.get('frames') or [])} | mode={segment.get('mode', 'text_only')}",
        )
        segment["summary"] = summarize_segment(
            segment=segment,
            model=model,
            min_chars_for_llm=min_chars_for_llm,
        )


def _overall_visual_mode(segments: List[Dict[str, Any]]) -> str:
    if not segments:
        return "text_only"
    with_images = sum(1 for s in segments if s.get("frames"))
    total_images = sum(len(s.get("frames") or []) for s in segments)

    if with_images == 0:
        return "text_only"
    if total_images <= len(segments):
        return "mixed"
    return "rich"


# =========================================================
# MARKDOWN RENDER
# =========================================================

def render_markdown(
    segments: List[Dict[str, Any]],
    out_path: str | Path,
    title: str = "Конспект лекции",
    include_ocr: bool = True,
    plan_titles: Optional[List[str]] = None,
) -> str:
    out_path = Path(out_path)
    out_dir = out_path.parent

    lines: List[str] = [f"# {title}", ""]

    overall_mode = _overall_visual_mode(segments)
    lines.append(f"**Режим конспекта:** {overall_mode}")
    lines.append("")

    if plan_titles:
        lines.append("## Грубый план лекции")
        lines.append("")
        for idx, item in enumerate(plan_titles, start=1):
            lines.append(f"{idx}. {item}")
        lines.append("")

    for segment in segments:
        start_ts = _format_ts(float(segment["start"]))
        end_ts = _format_ts(float(segment["end"]))
        segment_title = segment.get("title", f"Раздел {segment['index']}")
        lines.append(f"## Раздел {segment['index']}: {segment_title}")
        lines.append("")
        lines.append(f"**Время:** {start_ts}–{end_ts}")
        lines.append("")
        lines.append(f"**Формат:** {segment.get('mode', 'text_only')}")
        lines.append("")

        frames = segment.get("frames") or []
        for frame in frames:
            rel_img = _relative_posix(Path(frame["path"]), out_dir)
            lines.append(f"![{segment.get('title', 'Раздел')} | {_format_ts(frame['time'])}]({rel_img})")
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
                ocr_blocks.append(f"- {_format_ts(frame['time'])}: {_escape_md(_truncate(ocr_text, 500))}")
            if ocr_blocks:
                lines.append("**Фрагменты текста на слайдах / доске:**")
                lines.extend(ocr_blocks)
                lines.append("")

        lines.append("---")
        lines.append("")

    return "\n".join(lines).strip() + "\n"

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
    records: List[Dict[str, Any]] = []

    def add_record(frame_time: Optional[float], frame_path: str | Path) -> None:
        path_obj = Path(frame_path)
        time_value = frame_time if frame_time is not None else _extract_time_from_path(path_obj)
        records.append(
            {
                "time": float(time_value) if time_value is not None else 0.0,
                "path": path_obj,
                "stem": path_obj.stem,
            }
        )

    if keyframes:
        for item in keyframes:
            if isinstance(item, dict):
                add_record(item.get("time"), item.get("path"))
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




# =========================================================
# FULL MULTIMODAL PIPELINE
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
    callback: Optional[Callable[[str], None]] = None,
) -> Tuple[Path, List[Dict[str, Any]]]:
    transcript_path = Path(transcript_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    _emit(callback, "[summary] чтение транскрипта")
    transcript_items = parse_transcript(transcript_path)

    _emit(callback, "[summary] чтение OCR")
    ocr_map = load_ocr_map(ocr_dir)

    _emit(callback, "[summary] подготовка кадров")
    frames = normalize_frames(frame_paths=frame_paths, keyframes=keyframes)

    total_duration_sec = max((float(item["end"]) for item in transcript_items), default=0.0)
    _emit(callback, f"[summary] длительность распознанной лекции: {_format_ts(total_duration_sec)}")

    plan_titles = build_lecture_plan(
        transcript_items=transcript_items,
        model=model,
        callback=callback,
    )

    _emit(callback, "[summary] сегментация по окнам текста")
    windows = build_text_windows(
        transcript_items=transcript_items,
        max_items=6,
        stride=3,
        min_window_sec=20.0,
    )

    rough_segments = rough_segment_transcript(
        transcript_items=transcript_items,
        windows=windows,
        min_section_sec=25.0,
        max_section_sec=240.0,
    )

    _emit(callback, f"[summary] грубых разделов: {len(rough_segments)}")
    segments = refine_segments_with_plan(rough_segments, plan_titles)
    _emit(callback, f"[summary] после уточнения: {len(segments)}")

    segments = attach_frames_to_segments(
        segments=segments,
        frames=frames,
        ocr_map=ocr_map,
    )

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
