import os

os.environ["CT2_VERBOSE"] = "1"
os.environ["CT2_USE_MKL"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"

from faster_whisper import WhisperModel

audio_path = r"C:\Users\dondu\PycharmProjects\automatic_conspect33\runs\теория вероятности\теория_вероятности.wav"

model = WhisperModel(
    "tiny",
    device="cuda",
    compute_type="float16",
    cpu_threads=1,
)

segments, info = model.transcribe(
    audio_path,
    beam_size=1,
    language="ru",
)

print(info.language, info.language_probability)
for seg in segments:
    print(f"[{seg.start:.2f}-{seg.end:.2f}] {seg.text}")