from pathlib import Path
from faster_whisper import WhisperModel
import sys

audio_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])

model = WhisperModel("small", device="cpu", compute_type="int8", cpu_threads=1)
segments, _ = model.transcribe(str(audio_path), beam_size=5, language="ru")

with out_path.open("w", encoding="utf-8") as f:
    for seg in segments:
        f.write(f"[{seg.start:.2f}-{seg.end:.2f}] {seg.text.strip()}\n")