#!/usr/bin/env python3
"""Полный замер на A5000: матрица моделей × beam_size.

Запускается вручную на etu-pc-1 (SPEC.md §9, Этап 0 — разовый замер, не часть
автоматизации). Отдаёт JSON со словами и таймингами; WER считается на плате,
чтобы нормализация текста была одной и той же реализацией для всех цифр.

Единица работы — дорожка целиком, без нашей нарезки: у faster-whisper свой VAD
и батчинг (SPEC.md §9, этап 1.5).
"""
import json
import sys
import time

from faster_whisper import BatchedInferencePipeline, WhisperModel

AUDIO = "/root/asr-bench-gpu/reference.wav"
MODELS = ["large-v3", "large-v3-turbo", "medium", "small"]
BEAM_SIZES = [5, 1]

results = []
for model_name in MODELS:
    print(f"=== {model_name} ===", file=sys.stderr, flush=True)
    load_start = time.monotonic()
    model = WhisperModel(model_name, device="cuda", compute_type="float16")
    batched = BatchedInferencePipeline(model=model)
    load_s = time.monotonic() - load_start

    for beam_size in BEAM_SIZES:
        start = time.monotonic()
        segments, info = batched.transcribe(
            AUDIO, language="ru", beam_size=beam_size, vad_filter=True,
            batch_size=16, word_timestamps=True,
        )
        segments = list(segments)
        wall_clock_s = time.monotonic() - start

        results.append({
            "model": model_name,
            "beam_size": beam_size,
            "load_s": load_s,
            "wall_clock_s": wall_clock_s,
            "audio_duration_s": info.duration,
            "rtf": wall_clock_s / info.duration,
            "n_segments": len(segments),
            "words": [
                {"start": w.start, "end": w.end, "word": w.word}
                for s in segments for w in (s.words or [])
            ],
        })
        print(f"  beam={beam_size}: RTF={wall_clock_s / info.duration:.4f}",
              file=sys.stderr, flush=True)

    del batched, model

json.dump(results, sys.stdout, ensure_ascii=False)
