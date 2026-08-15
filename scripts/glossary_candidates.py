#!/usr/bin/env python3
"""Собрать кандидатов в глоссарий из нескольких расшифровок одного созвона.

SPEC.md §7.2: «кандидаты собираются автоматически (латиница плюс слова, которые
модель пишет по-разному в одинаковых контекстах) и отдаются владельцу на
вычитку». Здесь ровно это: на вход идут расшифровка Толка и JSON с пословными
таймкодами от разных моделей, на выход — список кандидатов с вариантами
написания и временем, где их слушать.

Решение, что из этого термин и как он пишется канонически, принимает владелец:
скрипт не догадывается о предметной области.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bench


def tolk_words(path: str) -> list[tuple[float, str]]:
    """Слова расшифровки Толка. Своих таймкодов у слов нет, поэтому они
    раскладываются равномерно внутри реплики — для попадания в нужное окно
    этого достаточно."""
    with open(path, encoding="utf-8") as f:
        rows = bench.parse_tolk_transcript(f.read())
    out = []
    for i, row in enumerate(rows):
        t_end = rows[i + 1].t0 if i + 1 < len(rows) else row.t0 + 10.0
        words = row.text.split()
        if not words:
            continue
        step = max((t_end - row.t0) / len(words), 0.0)
        for j, word in enumerate(words):
            out.append((row.t0 + j * step, word))
    return out


def gpu_sources(path: str) -> list[list[tuple[float, str]]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    runs = data if isinstance(data, list) else [data]
    return [[(w["start"], w["word"]) for w in run["words"]] for run in runs]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tolk", help="расшифровка Толка (.md)")
    parser.add_argument("--gpu-json", action="append", default=[],
                        help="JSON с пословными таймкодами (можно повторять)")
    parser.add_argument("--window-s", type=float, default=30.0)
    parser.add_argument("--cutoff", type=float, default=0.65)
    parser.add_argument("--min-total", type=int, default=3,
                        help="не показывать кандидатов реже этого числа вхождений")
    parser.add_argument("--limit", type=int, default=60)
    args = parser.parse_args()

    sources = []
    if args.tolk:
        sources.append(tolk_words(args.tolk))
    for path in args.gpu_json:
        sources.extend(gpu_sources(path))
    if len(sources) < 2:
        print("нужно минимум два источника, иначе разнобой не с чем сравнивать", file=sys.stderr)
        return 1
    print(f"источников: {len(sources)}", file=sys.stderr)

    candidates = bench.collect_glossary_candidates(
        sources, window_s=args.window_s, cutoff=args.cutoff,
    )
    shown = [c for c in candidates if c.total >= args.min_total][:args.limit]

    print(f"{'вхожд':>6}  {'лат':>3}  {'первое вхождение':>16}  варианты написания")
    for c in shown:
        forms = sorted(c.forms.items(), key=lambda kv: -kv[1])
        forms_str = ", ".join(f"{w}×{n}" if n > 1 else w for w, n in forms[:8])
        t = min(c.windows)
        print(f"{c.total:>6}  {'да' if c.has_latin else '  ':>3}  "
              f"{int(t) // 60:>13}:{int(t) % 60:02d}  {forms_str}")
    print(f"\nвсего кандидатов: {len(candidates)}, показано: {len(shown)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
