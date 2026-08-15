#!/usr/bin/env python3
"""Собрать gold.tsv из расшифровки Толка, сохранив ручную разметку владельца.

Внутри `wer_window` строки не трогаются вообще — это выверенный вручную эталон.
Всё, что после окна, берётся из расшифровки Толка: время начала — её, конец
подрезается по VAD (иначе короткое «угу» растягивается на полминуты тишины),
`speaker_id` приводится к латинице строчными по SPEC.md §7.3.

Результат — черновик для правки в Audacity, а не эталон: расшифровка Толка
сама по себе с ошибками. Для WER по-прежнему берётся только `wer_window`.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bench

# Имена в Толке -> speaker_id (SPEC.md §7.3: латиницей строчными, как в манифесте)
SPEAKER_IDS = {
    "Андрей Павлов": "andrey",
    "Павел Шубин": "pavel",
    "Данила Дубровин": "danila",
    "Мельник Артем": "artem",
    "Ширинкин Александр": "alexander",
}

# Как владелец подписал говорящих в ручной части (кириллица) -> тот же speaker_id
MANUAL_SPEAKER_IDS = {
    "Андрей": "andrey",
    "Павел": "pavel",
    "Данила": "danila",
    "Артём": "artem",
    "Артем": "artem",
    "Александр": "alexander",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", required=True, help="существующий gold.tsv с ручной разметкой")
    parser.add_argument("--tolk", required=True, help="расшифровка Толка (.md)")
    parser.add_argument("--audio", required=True, help="запись, по ней считается VAD")
    parser.add_argument("--vad-model", default="/srv/asr/models/silero_vad.onnx")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with open(args.gold, encoding="utf-8") as f:
        gold = bench.parse_gold_tsv(f.read())
    if gold.wer_window is None:
        print("в gold.tsv нет wer_window — не понять, где ручная часть", file=sys.stderr)
        return 1
    w0, w1 = gold.wer_window

    with open(args.tolk, encoding="utf-8") as f:
        tolk_rows = bench.parse_tolk_transcript(f.read())

    unknown = {r.speaker for r in tolk_rows} - set(SPEAKER_IDS)
    if unknown:
        print(f"неизвестные говорящие в расшифровке Толка: {sorted(unknown)}", file=sys.stderr)
        return 1

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        wav = os.path.join(tmp, "reference.wav")
        print("нормализация аудио...", file=sys.stderr)
        bench.normalize_audio(args.audio, wav)
        duration_s = bench.probe_duration_s(wav)
        print("VAD...", file=sys.stderr)
        vad = bench.run_vad(wav, args.vad_model)
    print(f"VAD: {len(vad)} сегментов, запись {duration_s:.1f}с", file=sys.stderr)

    lines = [
        f"# call_id: {gold.call_id}\n",
        f"# wer_window: {w0:.3f} {w1:.3f}\n",
        "#\n",
        "# Внутри wer_window — выверенный вручную эталон, его и берёт bench.py для WER.\n",
        "# После окна — черновая расшифровка Контур.Толка с ошибками распознавания:\n",
        "# правится в Audacity, эталоном не является, пока не вычитана.\n",
        "# Границы реплик после окна: начало от Толка, конец подрезан по VAD.\n",
    ]

    manual = [s for s in gold.segments if s.t0 < w1]
    for seg in manual:
        speaker_id = MANUAL_SPEAKER_IDS.get(seg.speaker, seg.speaker)
        lines.append(f"{seg.t0:.3f}\t{seg.t1:.3f}\t{speaker_id}: {seg.text}\n")

    tail = [r for r in tolk_rows if r.t0 >= w1]
    for i, row in enumerate(tail):
        t_next = tail[i + 1].t0 if i + 1 < len(tail) else duration_s
        t1 = bench.clip_end_by_vad(row.t0, t_next, vad)
        lines.append(f"{row.t0:.3f}\t{t1:.3f}\t{SPEAKER_IDS[row.speaker]}: {row.text}\n")

    tmp_out = args.out + ".tmp"
    with open(tmp_out, "w", encoding="utf-8") as f:
        f.writelines(lines)
    os.replace(tmp_out, args.out)

    print(f"записано: {len(manual)} ручных строк + {len(tail)} из Толка -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
