#!/usr/bin/env python3
"""Разовый ручной замер large-v3 beam=5 на A5000. Не часть автоматизации
(SPEC.md §9, Этап 0) — запускается вручную на etu-pc-1, не на плате."""
import json
import sys
import time

from faster_whisper import WhisperModel, BatchedInferencePipeline

AUDIO = "/root/asr-bench-gpu/reference.wav"

model = WhisperModel("large-v3", device="cuda", compute_type="float16")
batched = BatchedInferencePipeline(model=model)

start = time.monotonic()
segments, info = batched.transcribe(
    AUDIO, language="ru", beam_size=5, vad_filter=True, batch_size=16,
    word_timestamps=True,
)
segments = list(segments)
wall_clock_s = time.monotonic() - start

words = [
    {"start": w.start, "end": w.end, "word": w.word}
    for s in segments for w in (s.words or [])
]

out = {
    "audio_duration_s": info.duration,
    "wall_clock_s": wall_clock_s,
    "rtf": wall_clock_s / info.duration,
    "segments": [{"start": s.start, "end": s.end, "text": s.text} for s in segments],
    "words": words,
}
json.dump(out, sys.stdout, ensure_ascii=False)
