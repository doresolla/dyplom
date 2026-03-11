from __future__ import annotations

import re, os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from yt_dlp import YoutubeDL


@dataclass
class MediaAssets:
    source_video: Path
    local_video: Path
    audio_wav: Path


def _safe_name(name: str) -> str:
    name = name.replace(" ", "_")
    name = re.sub(r"[^\w.-]", "_", name)
    name = re.sub(r"_{2,}", "_", name)
    return name.strip("_.") or "video"


def download_video(url: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "merge_output_format": "mp4",
        "outtmpl": str(out_dir / "%(title)s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
    }
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
    candidate = Path(filename)
    if candidate.suffix.lower() != ".mp4":
        candidate = candidate.with_suffix(".mp4")
    return candidate
def _find_ffmpeg() -> str:
    candidates = [
        os.getenv("FFMPEG_EXE"),
        shutil.which("ffmpeg"),
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
    ]

    for candidate in candidates:
        if not candidate:
            continue
        p = Path(candidate)
        if p.exists():
            return str(p)

    raise FileNotFoundError(
        "ffmpeg.exe не найден. Укажи путь через переменную окружения FFMPEG_EXE "
        "или добавь ffmpeg в PATH."
    )

def split_audio_video(video_path: Path, out_dir: Path) -> MediaAssets:
    out_dir.mkdir(parents=True, exist_ok=True)

    video_copy = out_dir / f"{_safe_name(video_path.stem)}.mp4"
    if video_path.resolve() != video_copy.resolve():
        shutil.copy2(video_path, video_copy)

    audio_path = out_dir / f"{video_copy.stem}.wav"
    ffmpeg_exe = _find_ffmpeg()

    cmd = [
        ffmpeg_exe,
        "-y",
        "-hide_banner",
        "-loglevel", "error",
        "-i", str(video_copy),
        "-map", "a:0?",
        "-vn",
        "-acodec", "pcm_s16le",
        "-ac", "1",
        "-ar", "16000",
        str(audio_path),
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    if result.returncode != 0:
        err = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(
            "Ошибка ffmpeg при извлечении аудио.\n"
            f"Код возврата: {result.returncode}\n"
            f"Команда: {' '.join(cmd)}\n"
            f"{err}"
        )

    if not audio_path.exists() or audio_path.stat().st_size == 0:
        raise RuntimeError("ffmpeg завершился без ошибки, но WAV-файл не был создан.")

    return MediaAssets(source_video=video_path, local_video=video_copy, audio_wav=audio_path)

def resolve_input(source: str, out_dir: Path) -> Path:
    maybe_path = Path(source.strip('"'))
    if maybe_path.exists():
        return maybe_path.resolve()
    return download_video(source, out_dir)
