from pathlib import Path
import sys
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["CT2_VERBOSE"] = "1"

from faster_whisper import WhisperModel


def log(msg: str, log_path: Path | None = None):
    print(msg, flush=True)

    if log_path is not None:
        try:
            with log_path.open("a", encoding="utf-8") as f:
                f.write(msg + "\n")
                f.flush()
        except Exception:
            pass


def main():
    audio_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])

    log("[worker] creating model")
    model = WhisperModel("small", device="cuda", compute_type="float16", cpu_threads=1)

    log("[worker] model created")
    segments, info = model.transcribe(
        str(audio_path),
        beam_size=5,
        language="ru"
    )

    log(f"[worker] language={info.language}, p={info.language_probability:.3f}")

    with out_path.open("w", encoding="utf-8") as out:
        for i, seg in enumerate(segments, 1):
            line = f"[{seg.start:.2f}-{seg.end:.2f}] {seg.text.strip()}"
            out.write(line + "\n")
            out.flush()
            log(f"[seg {i}] {line}")

    log("[worker] done")
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    main()