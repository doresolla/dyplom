from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from text_processing.LLMsummary import summarize_with_llm


TRANSCRIPT_LINE_RE = re.compile(
    r"^\[(?P<start>\d+(?:\.\d+)?)\s*-\s*(?P<end>\d+(?:\.\d+)?)\]\s*(?P<text>.*)$"
)
TIME_FROM_NAME_RE = re.compile(r"_(?P<time>\d+(?:\.\d+)?)s(?:\.[^.]+)?$")


def _emit(callback: Optional[Callable[[str], None]], text: str) -> None:
    if callback is not None:
        callback(text)


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
    records: List[Dict[str, Any]] = []

    def add_record(frame_time: Optional[float], frame_path: str | Path) -> None:
        path_obj = Path(frame_path)
        if frame_time is None:
            frame_time = _extract_time_from_path(path_obj)

        records.append(
            {
                "time": float(frame_time) if frame_time is not None else 0.0,
                "path": path_obj,
                "stem": path_obj.stem,
            }
        )

    if keyframes:
        for item in keyframes:
            # вариант: {"time": ..., "path": ...}
            if isinstance(item, dict):
                add_record(item.get("time"), item.get("path"))
                continue

            # вариант: (time, path)
            if isinstance(item, (tuple, list)) and len(item) >= 2:
                add_record(item[0], item[1])
                continue

            # вариант: просто строка / Path
            if isinstance(item, (str, Path)):
                add_record(None, item)
                continue

    # fallback: если keyframes передали, но не удалось распарсить ни одного кадра
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

def build_segments(
    transcript_items: List[Dict[str, Any]],
    frames: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    transcript_end = 0.0
    if transcript_items:
        transcript_end = max(float(item["end"]) for item in transcript_items)

    if not frames:
        return [
            {
                "index": 1,
                "start": 0.0,
                "end": transcript_end,
                "image_path": None,
                "image_stem": None,
                "speech_text": "",
                "ocr_text": "",
                "summary": "",
            }
        ]

    segments: List[Dict[str, Any]] = []
    for i, frame in enumerate(frames):
        start = 0.0 if i == 0 else float(frames[i]["time"])
        end = (
            float(frames[i + 1]["time"])
            if i + 1 < len(frames)
            else transcript_end
        )

        if end < start:
            end = start

        segments.append(
            {
                "index": i + 1,
                "start": start,
                "end": end,
                "image_path": frame["path"],
                "image_stem": frame["stem"],
                "speech_text": "",
                "ocr_text": "",
                "summary": "",
            }
        )

    return segments


def assign_transcript_to_segments(
    segments: List[Dict[str, Any]],
    transcript_items: List[Dict[str, Any]],
) -> None:
    if not segments:
        return

    for item in transcript_items:
        center = float(item["center"])
        chosen = None

        for segment in segments:
            if segment["start"] <= center < segment["end"]:
                chosen = segment
                break

        if chosen is None:
            chosen = segments[-1]

        if chosen["speech_text"]:
            chosen["speech_text"] += " " + item["text"]
        else:
            chosen["speech_text"] = item["text"]

    for segment in segments:
        segment["speech_text"] = _clean_text(segment["speech_text"])


def assign_ocr_to_segments(
    segments: List[Dict[str, Any]],
    ocr_map: Dict[str, str],
) -> None:
    for segment in segments:
        stem = segment.get("image_stem")
        segment["ocr_text"] = _clean_text(ocr_map.get(stem, ""))


def _fallback_summary(source_text: str, max_sentences: int = 4) -> str:
    text = _clean_text(source_text)
    if not text:
        return "- Нет данных для конспекта."

    parts = re.split(r"(?<=[.!?])\s+", text)
    parts = [p.strip(" -•\n\t") for p in parts if p.strip()]
    parts = parts[:max_sentences]

    if not parts:
        return f"- {text[:300]}"

    return "\n".join(f"- {p}" for p in parts)


def _compose_source_text(segment: Dict[str, Any]) -> str:
    speech = _clean_text(segment.get("speech_text", ""))
    ocr = _clean_text(segment.get("ocr_text", ""))

    parts: List[str] = []

    if speech:
        parts.append("ТЕКСТ ЛЕКТОРА:\n" + speech)
    if ocr:
        parts.append("ТЕКСТ СО СЛАЙДА:\n" + ocr)

    return "\n\n".join(parts).strip()


def summarize_segments(
    segments: List[Dict[str, Any]],
    model: str | None = None,
    min_chars_for_llm: int = 220,
    callback: Optional[Callable[[str], None]] = None,
) -> None:
    total = len(segments)

    for i, segment in enumerate(segments, start=1):
        source_text = _compose_source_text(segment)

        _emit(
            callback,
            f"[multimodal] блок {i}/{total}: "
            f"{_format_ts(segment['start'])}–{_format_ts(segment['end'])}",
        )

        if not source_text:
            segment["summary"] = "- Для этого блока не удалось получить текст."
            continue

        if len(source_text) < min_chars_for_llm:
            segment["summary"] = _fallback_summary(source_text)
            continue

        prompt = (
            "Сделай краткий конспект только для этого фрагмента лекции.\n\n"
            "Требования:\n"
            "- 3-6 пунктов;\n"
            "- по-русски;\n"
            "- сохрани важные термины, определения, числа и обозначения;\n"
            "- не добавляй факты от себя;\n"
            "- если есть мусор OCR, игнорируй его.\n\n"
            "Формат ответа: Markdown-список.\n\n"
            f"{source_text}"
        )

        try:
            segment["summary"] = summarize_with_llm(prompt, model=model).strip()
            if not segment["summary"]:
                segment["summary"] = _fallback_summary(source_text)
        except Exception:
            segment["summary"] = _fallback_summary(source_text)


def render_markdown(
    segments: List[Dict[str, Any]],
    out_path: str | Path,
    title: str = "Конспект лекции",
    include_ocr: bool = True,
) -> str:
    out_path = Path(out_path)
    out_dir = out_path.parent

    lines: List[str] = [f"# {title}", ""]

    for segment in segments:
        start_ts = _format_ts(float(segment["start"]))
        end_ts = _format_ts(float(segment["end"]))
        lines.append(f"## Блок {segment['index']}")
        lines.append("")
        lines.append(f"**Время:** {start_ts}–{end_ts}")
        lines.append("")

        image_path = segment.get("image_path")
        if image_path:
            rel_img = _relative_posix(Path(image_path), out_dir)
            lines.append(f"![Блок {segment['index']}]({rel_img})")
            lines.append("")

        summary = _clean_text(segment.get("summary", ""))
        if summary:
            lines.append("**Краткий конспект:**")
            lines.append(summary)
            lines.append("")

        if include_ocr:
            ocr_text = _clean_text(segment.get("ocr_text", ""))
            if ocr_text:
                lines.append("**Текст на слайде (OCR):**")
                lines.append("")
                lines.append(f"> {_escape_md(ocr_text)}")
                lines.append("")

        lines.append("---")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


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

    _emit(callback, "[multimodal] чтение транскрипта")
    transcript_items = parse_transcript(transcript_path)

    _emit(callback, "[multimodal] чтение OCR")
    ocr_map = load_ocr_map(ocr_dir)

    _emit(callback, "[multimodal] подготовка сегментов")
    frames = normalize_frames(frame_paths=frame_paths, keyframes=keyframes)
    segments = build_segments(transcript_items, frames)

    assign_transcript_to_segments(segments, transcript_items)
    assign_ocr_to_segments(segments, ocr_map)

    _emit(callback, "[multimodal] суммаризация по блокам")
    summarize_segments(
        segments=segments,
        model=model,
        min_chars_for_llm=min_chars_for_llm,
        callback=callback,
    )

    _emit(callback, "[multimodal] сборка markdown")
    markdown = render_markdown(
        segments=segments,
        out_path=out_path,
        title=title,
        include_ocr=include_ocr,
    )
    out_path.write_text(markdown, encoding="utf-8")

    return out_path, segments