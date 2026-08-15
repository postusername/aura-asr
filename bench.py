#!/usr/bin/env python3
"""Замер этапа 0: RTF / WER / F1 по терминам глоссария для whisper.cpp на плате.

См. SPEC.md §9 «Этап 0 — замер». Вход: реальная запись созвона, gold.tsv
(эталон в формате §7.3), glossary.json (формат §7.2). Модель whisper.cpp
поднимается один раз в режиме whisper-server, нарезка идёт по VAD
(sherpa-onnx-vad), сегменты уходят к серверу по HTTP — так же, как будет
работать этап 1, иначе цифры не переносятся.

Внешняя зависимость сверх numpy — только jiwer, и только здесь.
"""
from __future__ import annotations

import argparse
import dataclasses
import difflib
import glob
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import uuid
from dataclasses import dataclass, field


class Terminated(Exception):
    """SIGTERM во время замера. Нужно перехватывать, чтобы вернуть governor
    и убить whisper-server — по умолчанию Python на SIGTERM завершается сразу,
    минуя finally (сами так и оставили governor на 'performance' при аварийной
    остановке из-за перегрева)."""


def _raise_terminated(signum, frame):
    raise Terminated()

try:
    import jiwer
except ImportError:
    jiwer = None  # нужен только для итогового счёта WER, чистая логика работает и без него


# ---------------------------------------------------------------------------
# gold.tsv (§7.3)
# ---------------------------------------------------------------------------

@dataclass
class GoldSegment:
    t0: float
    t1: float
    speaker: str
    text: str


@dataclass
class GoldTranscript:
    call_id: str
    wer_window: tuple[float, float] | None
    segments: list[GoldSegment]


def parse_gold_tsv(text: str) -> GoldTranscript:
    """Разбирает gold.tsv. Формат см. SPEC.md §7.3."""
    call_id = None
    wer_window = None
    segments: list[GoldSegment] = []

    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip("\n")
        if not line.strip():
            continue
        if line.startswith("#"):
            body = line[1:].strip()
            if body.startswith("call_id:"):
                call_id = body[len("call_id:"):].strip()
            elif body.startswith("wer_window:"):
                parts = body[len("wer_window:"):].split()
                if len(parts) != 2:
                    raise ValueError(f"gold.tsv:{lineno}: неверный формат wer_window: {line!r}")
                wer_window = (float(parts[0]), float(parts[1]))
            continue

        cols = line.split("\t")
        if len(cols) != 3:
            raise ValueError(f"gold.tsv:{lineno}: ожидалось 3 колонки через TAB, получено {len(cols)}: {line!r}")
        t0_s, t1_s, rest = cols
        try:
            t0, t1 = float(t0_s), float(t1_s)
        except ValueError as exc:
            raise ValueError(f"gold.tsv:{lineno}: t0/t1 не числа: {line!r}") from exc
        if ":" not in rest:
            raise ValueError(f"gold.tsv:{lineno}: нет 'speaker_id:' перед текстом: {line!r}")
        speaker, _, seg_text = rest.partition(":")
        segments.append(GoldSegment(t0=t0, t1=t1, speaker=speaker.strip(), text=seg_text.strip()))

    if call_id is None:
        raise ValueError("gold.tsv: отсутствует обязательный заголовок '# call_id: ...'")

    return GoldTranscript(call_id=call_id, wer_window=wer_window, segments=segments)


def gold_reference_text(gold: GoldTranscript) -> str:
    """Эталонный текст внутри wer_window, отсортированный по t0."""
    if gold.wer_window is None:
        return ""
    w0, w1 = gold.wer_window
    in_window = [s for s in gold.segments if s.text and s.t0 < w1 and s.t1 > w0]
    in_window.sort(key=lambda s: s.t0)
    return " ".join(s.text for s in in_window)


# ---------------------------------------------------------------------------
# расшифровка Контур.Толка (вход для gold.tsv и для кандидатов глоссария)
# ---------------------------------------------------------------------------

@dataclass
class TolkRow:
    t0: float
    speaker: str
    text: str


_TOLK_TIME_RE = re.compile(r"^(\d{2}):(\d{2}):(\d{2})$")


def parse_tolk_transcript(text: str) -> list[TolkRow]:
    """Экспорт расшифровки Толка: `HH:MM:SS<TAB>Имя Фамилия<TAB>текст`.
    Первая строка — заголовок, битые строки пропускаются молча: файл приходит
    из внешней системы и мусор в нём не наша забота."""
    rows = []
    for line in text.splitlines():
        cols = line.rstrip("\n").split("\t")
        if len(cols) != 3:
            continue
        m = _TOLK_TIME_RE.match(cols[0].strip())
        if not m:
            continue
        h, mi, s = (int(x) for x in m.groups())
        rows.append(TolkRow(t0=h * 3600 + mi * 60 + s, speaker=cols[1].strip(), text=cols[2].strip()))
    return rows


def clip_end_by_vad(
    t0: float,
    t_next: float,
    vad_segments: list[tuple[float, float]],
    fallback_s: float = 2.0,
) -> float:
    """Конец реплики по VAD, а не «до следующей реплики».

    Толк даёт только время начала, поэтому наивное t1 = t0 следующей реплики
    растягивает короткое «угу» на полминуты тишины. Берём конец последнего
    речевого куска внутри интервала; если речи там нет — короткий fallback.
    Результат всегда строго больше t0.

    Если t_next не позже t0 (у Толка две реплики попали на одну секунду —
    перебивание), ограничение сверху не применяется: по SPEC.md §7.3
    перекрывающиеся интервалы допустимы, а сегмент нулевой длины — нет.
    """
    limit = t_next if t_next > t0 else t0 + fallback_s
    ends = [min(e, limit) for s, e in vad_segments if s < limit and e > t0]
    end = max(ends) if ends else t0 + fallback_s
    return max(min(end, limit), t0 + 0.001)


# ---------------------------------------------------------------------------
# кандидаты в глоссарий (§7.2: латиница + разнобой написания в одном контексте)
# ---------------------------------------------------------------------------

_LATIN_RE = re.compile(r"[A-Za-z]")


def has_latin(token: str) -> bool:
    return bool(_LATIN_RE.search(token))


# ponytail: фонетическая транслитерация «на глазок», нужна только чтобы сравнить
# кириллическое искажение с латинским оригиналом («лок» ~ «LOC»). Потолок —
# грубые пары вроде «щ»→«sch» и потеря мягкости; апгрейд при надобности —
# готовая таблица ГОСТ/ISO 9, но для поиска кандидатов на вычитку хватает.
_TRANSLIT = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e",
    "ж": "zh", "з": "z", "и": "i", "й": "i", "к": "k", "л": "l", "м": "m",
    "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "х": "h", "ц": "c", "ч": "ch", "ш": "sh", "щ": "sch",
    "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "u", "я": "a",
}


def translit_key(token: str) -> str:
    """Ключ сравнения: нижний регистр без пунктуации, кириллица транслитерирована."""
    clean = _normalize_for_glossary_match(token)
    return "".join(_TRANSLIT.get(ch, ch) for ch in clean)


def cluster_variants(word_lists: list[list[str]], cutoff: float = 0.65) -> list[list[str]]:
    """Группирует похожие написания из разных расшифровок одного окна.
    Сравнение идёт по транслитерированному ключу, иначе «лок» и «LOC» никогда
    не сойдутся. Возвращает кластеры (списки написаний как они встретились)."""
    clusters: list[list[str]] = []
    for words in word_lists:
        for word in words:
            key = translit_key(word)
            placed = False
            for cluster in clusters:
                keys = [translit_key(w) for w in cluster]
                if difflib.get_close_matches(key, keys, n=1, cutoff=cutoff):
                    cluster.append(word)
                    placed = True
                    break
            if not placed:
                clusters.append([word])
    return clusters


@dataclass
class GlossaryCandidate:
    forms: dict[str, int]          # написание -> сколько раз встретилось
    windows: list[float]           # начала окон, где встретился
    has_latin: bool

    @property
    def total(self) -> int:
        return sum(self.forms.values())


def collect_glossary_candidates(
    sources: list[list[tuple[float, str]]],
    window_s: float = 30.0,
    cutoff: float = 0.65,
    min_len: int = 5,
) -> list[GlossaryCandidate]:
    """Кандидаты в глоссарий из нескольких расшифровок одного аудио.

    Сигнал ровно из SPEC.md §7.2: латиница плюс слова, которые пишутся
    по-разному в одинаковых контекстах. Контекст задаётся окном window_s —
    расшифровки выровнены по времени, поэтому одно и то же слово у разных
    моделей попадает в одно окно.

    Короткие кириллические слова отбрасываются (min_len): на них difflib
    слишком лоялен — «как», «так» и «там» отличаются одной буквой из трёх и
    слипаются в один кластер, давая сплошной шум из служебных слов. Латиница
    берётся от двух символов: аббревиатуры («XR», «LOC», «SSH») — как раз то,
    ради чего глоссарий и заводится.
    """
    by_window: dict[int, list[list[str]]] = {}
    for source in sources:
        per_window: dict[int, list[str]] = {}
        for t, word in source:
            clean = _normalize_for_glossary_match(word)
            floor = 2 if has_latin(clean) else min_len
            if len(clean) < floor:
                continue
            per_window.setdefault(int(t // window_s), []).append(word.strip())
        for idx, words in per_window.items():
            by_window.setdefault(idx, []).append(words)

    merged: dict[str, GlossaryCandidate] = {}
    for idx, word_lists in sorted(by_window.items()):
        for cluster in cluster_variants(word_lists, cutoff):
            spellings = {_normalize_for_glossary_match(w) for w in cluster}
            latin = any(has_latin(w) for w in cluster)
            # интересны только несогласие моделей либо латиница
            if len(spellings) < 2 and not latin:
                continue
            key = min(spellings)
            cand = merged.get(key)
            if cand is None:
                cand = GlossaryCandidate(forms={}, windows=[], has_latin=latin)
                merged[key] = cand
            cand.has_latin = cand.has_latin or latin
            cand.windows.append(idx * window_s)
            for w in cluster:
                cand.forms[w] = cand.forms.get(w, 0) + 1
    return sorted(merged.values(), key=lambda c: -c.total)


# ---------------------------------------------------------------------------
# glossary.json (§7.2)
# ---------------------------------------------------------------------------

@dataclass
class GlossaryTerm:
    canon: str
    variants: list[str]
    prompt: bool


def load_glossary(path: str) -> list[GlossaryTerm]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [
        GlossaryTerm(canon=t["canon"], variants=t.get("variants", []), prompt=t.get("prompt", False))
        for t in data["terms"]
    ]


def glossary_prompt_text(terms: list[GlossaryTerm]) -> str:
    """Термины с prompt=true через запятую — то, что уходит в --prompt whisper.cpp."""
    return ", ".join(t.canon for t in terms if t.prompt)


# ---------------------------------------------------------------------------
# нормализация текста для WER (§7.4)
# ---------------------------------------------------------------------------

_UNITS = {
    "ноль": 0, "один": 1, "одна": 1, "одно": 1, "два": 2, "две": 2, "три": 3,
    "четыре": 4, "пять": 5, "шесть": 6, "семь": 7, "восемь": 8, "девять": 9,
}
_TEENS = {
    "десять": 10, "одиннадцать": 11, "двенадцать": 12, "тринадцать": 13,
    "четырнадцать": 14, "пятнадцать": 15, "шестнадцать": 16, "семнадцать": 17,
    "восемнадцать": 18, "девятнадцать": 19,
}
_TENS = {
    "двадцать": 20, "тридцать": 30, "сорок": 40, "пятьдесят": 50,
    "шестьдесят": 60, "семьдесят": 70, "восемьдесят": 80, "девяносто": 90,
}
_HUNDREDS = {
    "сто": 100, "двести": 200, "триста": 300, "четыреста": 400, "пятьсот": 500,
    "шестьсот": 600, "семьсот": 700, "восемьсот": 800, "девятьсот": 900,
}


def _consume_number(tokens: list[str], i: int) -> tuple[int | None, int]:
    """Пытается разобрать числительное с позиции i. ponytail: только 0-999,
    больше в технических созвонах почти не звучит словами (версии читают по цифрам).
    Апгрейд — добавить tysyachi/million при первой реальной надобности."""
    n = len(tokens)
    j = i
    value = 0
    matched = False

    if j < n and tokens[j].lower() in _HUNDREDS:
        value += _HUNDREDS[tokens[j].lower()]
        matched = True
        j += 1

    if j < n and tokens[j].lower() in _TEENS:
        value += _TEENS[tokens[j].lower()]
        matched = True
        j += 1
    elif j < n and tokens[j].lower() in _TENS:
        value += _TENS[tokens[j].lower()]
        matched = True
        j += 1
        if j < n and tokens[j].lower() in _UNITS:
            value += _UNITS[tokens[j].lower()]
            j += 1
    elif j < n and tokens[j].lower() in _UNITS:
        value += _UNITS[tokens[j].lower()]
        matched = True
        j += 1

    if not matched:
        return None, i
    return value, j


def ru_number_words_to_digits(text: str) -> str:
    """Числительные словами → цифрами. Остальной текст не трогается."""
    tokens = text.split(" ")
    out = []
    i = 0
    while i < len(tokens):
        value, j = _consume_number(tokens, i)
        if value is not None:
            out.append(str(value))
            i = j
        else:
            out.append(tokens[i])
            i += 1
    return " ".join(out)


def normalize_for_wer(text: str) -> str:
    """Нижний регистр, числа цифрами, пунктуация снимается, дефисы значимы."""
    text = ru_number_words_to_digits(text)
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _normalize_for_glossary_match(text: str) -> str:
    """Регистронезависимо и без пунктуации (включая дефис) — см. §7.2."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# постобработка глоссария (два эшелона защиты терминологии, §9.1)
# ---------------------------------------------------------------------------

def apply_glossary(
    text: str,
    terms: list[GlossaryTerm],
    cutoff: float = 0.85,
    fuzzy_min_len: int = 4,
) -> str:
    """Подстановка канонической формы термина вместо распознанного варианта.

    Точное совпадение с вариантом срабатывает всегда, нечёткое — только на
    словах от fuzzy_min_len букв. Иначе на коротких словах difflib съедает
    обычную речь: «ло» подтягивается к «лок» (LOC), «саш» к «ссаш» (SSH) —
    имя человека превращается в протокол. Порог 0.85 подобран на записи
    2026-08-12: при 0.8 «надо» девять раз становилось «NetBird».

    Склонения латинских акронимов снимаются точным префиксным правилом.
    """
    tokens = text.split(" ")
    norm_tokens = [_normalize_for_glossary_match(t) for t in tokens]

    for term in sorted(terms, key=lambda t: -len(t.canon.split())):
        candidates_norm = [_normalize_for_glossary_match(c) for c in [term.canon] + term.variants]
        # варианты могут быть длиннее канона по числу слов ("вда 5050" при VDA5050) —
        # окно сравнения перебирается по всем встречающимся длинам, от длинных к коротким
        lengths = sorted({len(c.split()) for c in [term.canon] + term.variants}, reverse=True)
        is_latin_acronym = len(term.canon.split()) == 1 and term.canon.isascii() and term.canon.isupper()
        declension_re = None
        if is_latin_acronym:
            declension_re = re.compile(re.escape(term.canon.lower()) + r"[а-яё]{0,5}$")

        for n in lengths:
            i = 0
            while i <= len(tokens) - n:
                window_norm = " ".join(norm_tokens[i:i + n])
                matched = False
                if n == 1 and declension_re is not None and declension_re.match(window_norm):
                    matched = True
                elif window_norm in candidates_norm:
                    matched = True
                elif len(window_norm) >= fuzzy_min_len and difflib.get_close_matches(
                    window_norm, candidates_norm, n=1, cutoff=cutoff
                ):
                    matched = True
                if matched:
                    tokens[i] = term.canon
                    norm_tokens[i] = _normalize_for_glossary_match(term.canon)
                    del tokens[i + 1:i + n]
                    del norm_tokens[i + 1:i + n]
                i += 1

    return " ".join(tokens)


def count_canon_occurrences(text: str, canon: str) -> int:
    """Число вхождений канонической формы термина, с учётом регистра, по границе слова."""
    pattern = r"(?<!\w)" + re.escape(canon) + r"(?!\w)"
    return len(re.findall(pattern, text))


@dataclass
class TermF1:
    tp: int
    fp: int
    fn: int

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


def term_f1(terms: list[GlossaryTerm], gold_text: str, hyp_text: str) -> TermF1:
    """F1 по вхождениям терминов глоссария в пределах wer_window (§7.4).
    Считается по мультимножеству: для каждого термина сравнивается число
    вхождений канонической формы в эталоне и в гипотезе."""
    tp = fp = fn = 0
    for term in terms:
        actual = count_canon_occurrences(gold_text, term.canon)
        predicted = count_canon_occurrences(hyp_text, term.canon)
        term_tp = min(actual, predicted)
        tp += term_tp
        fp += predicted - term_tp
        fn += actual - term_tp
    return TermF1(tp=tp, fp=fp, fn=fn)


# ---------------------------------------------------------------------------
# RTF
# ---------------------------------------------------------------------------

def rtf(wall_clock_s: float, audio_duration_s: float) -> float:
    if audio_duration_s <= 0:
        raise ValueError("длительность аудио должна быть положительной")
    return wall_clock_s / audio_duration_s


# ---------------------------------------------------------------------------
# VAD (sherpa-onnx-vad CLI)
# ---------------------------------------------------------------------------

_VAD_LINE_RE = re.compile(r"(\d+\.\d+)\s+--\s+(\d+\.\d+)")


def parse_vad_stdout(output: str) -> list[tuple[float, float]]:
    return [(float(a), float(b)) for a, b in _VAD_LINE_RE.findall(output)]


def run_vad(wav_path: str, vad_model_path: str) -> list[tuple[float, float]]:
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp_out:
        proc = subprocess.run(
            ["sherpa-onnx-vad", f"--silero-vad-model={vad_model_path}", wav_path, tmp_out.name],
            capture_output=True, text=True, check=True,
        )
    return parse_vad_stdout(proc.stdout + proc.stderr)


# ---------------------------------------------------------------------------
# показания железа
# ---------------------------------------------------------------------------

def read_thermal_zones(base_dir: str = "/sys/class/thermal") -> dict[str, float]:
    zones = {}
    for type_path in sorted(glob.glob(os.path.join(base_dir, "thermal_zone*", "type"))):
        zone_dir = os.path.dirname(type_path)
        try:
            with open(type_path) as f:
                name = f.read().strip()
            with open(os.path.join(zone_dir, "temp")) as f:
                millideg = int(f.read().strip())
        except OSError:
            continue
        zones[name] = millideg / 1000.0
    return zones


_CPU_DIR_RE = re.compile(r"^cpu(\d+)$")


def read_cpu_freqs(base_dir: str = "/sys/devices/system/cpu") -> dict[int, int]:
    freqs = {}
    for cpu_dir in sorted(glob.glob(os.path.join(base_dir, "cpu*"))):
        m = _CPU_DIR_RE.match(os.path.basename(cpu_dir))
        if not m:
            continue
        freq_path = os.path.join(cpu_dir, "cpufreq", "scaling_cur_freq")
        try:
            with open(freq_path) as f:
                freqs[int(m.group(1))] = int(f.read().strip())
        except OSError:
            continue
    return freqs


_VMHWM_RE = re.compile(r"VmHWM:\s+(\d+)\s*kB")


def parse_proc_status_vmhwm(status_text: str) -> int | None:
    m = _VMHWM_RE.search(status_text)
    return int(m.group(1)) if m else None


def peak_rss_kb(pid: int) -> int | None:
    try:
        with open(f"/proc/{pid}/status") as f:
            return parse_proc_status_vmhwm(f.read())
    except OSError:
        return None


def format_eta(seconds: float) -> str:
    """Человекочитаемая оценка времени: '45с', '1м 05с', '1ч 02м'."""
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}ч {m:02d}м"
    if m:
        return f"{m}м {s:02d}с"
    return f"{s}с"


def estimate_remaining_s(elapsed_s: float, done_units: float, total_units: float) -> float:
    """Линейная экстраполяция по уже пройденной скорости. NaN, если пока ничего не сделано."""
    if done_units <= 0:
        return float("nan")
    remaining_units = max(0.0, total_units - done_units)
    return elapsed_s / done_units * remaining_units


def current_max_temp_c() -> float:
    zones = read_thermal_zones()
    return max(zones.values()) if zones else 0.0


def wait_for_cooldown(
    target_c: float,
    timeout_s: float,
    poll_interval_s: float = 5.0,
    read_temp_fn=current_max_temp_c,
    sleep_fn=time.sleep,
    clock_fn=time.monotonic,
) -> float:
    """Пауза перед следующей конфигурацией, пока плата не остынет до target_c
    (или не истечёт timeout_s — вечно ждать нельзя, ночной прогон не резиновый)."""
    deadline = clock_fn() + timeout_s
    temp = read_temp_fn()
    while temp > target_c and clock_fn() < deadline:
        sleep_fn(poll_interval_s)
        temp = read_temp_fn()
    return temp


def thermal_governor_action(temp_c: float, frozen: bool, freeze_temp_c: float, resume_temp_c: float) -> str:
    """Гистерезис для SIGSTOP/SIGCONT: 'freeze' — заморозить процесс, 'resume' —
    разморозить, 'noop' — не трогать. Чистая логика, без побочных эффектов."""
    if not frozen and temp_c >= freeze_temp_c:
        return "freeze"
    if frozen and temp_c <= resume_temp_c:
        return "resume"
    return "noop"


class HardwareSampler:
    """Фоновый сэмплер температуры/частот на время прогона одной конфигурации.

    Без вентилятора один VAD-сегмент у тяжёлой модели может считаться минутами —
    проверка "между сегментами" для них бесполезна, перегрев случается посреди
    одного сегмента. Поэтому вместо (или вместе с) паузой между сегментами
    сэмплер сам держит whisper-server на SIGSTOP/SIGCONT: увидел freeze_temp_c —
    заморозил процесс ядром (вычисления встают мгновенно, без потери состояния),
    остыло до resume_temp_c — разморозил. Реагирует раз в interval_s, то есть
    даже посреди часового сегмента, а не только на его границе.

    abort_temp_c — аварийный стоп конфигурации целиком, страховка на случай,
    если заморозка почему-то не спасает (запас перед критическими 115°C)."""

    def __init__(
        self,
        interval_s: float = 2.0,
        abort_temp_c: float | None = None,
        freeze_temp_c: float | None = None,
        resume_temp_c: float | None = None,
        target_pid: int | None = None,
    ):
        self.interval_s = interval_s
        self.abort_temp_c = abort_temp_c
        self.freeze_temp_c = freeze_temp_c
        self.resume_temp_c = resume_temp_c
        self.target_pid = target_pid
        self.abort_event = threading.Event()
        self.frozen = False
        self.frozen_s = 0.0
        self._frozen_since: float | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.temp_samples: list[dict[str, float]] = []
        self.freq_samples: list[dict[int, int]] = []

    def _run(self):
        while not self._stop.is_set():
            temps = read_thermal_zones()
            self.temp_samples.append(temps)
            self.freq_samples.append(read_cpu_freqs())
            max_t = max(temps.values()) if temps else 0.0

            if self.abort_temp_c is not None and temps and max_t >= self.abort_temp_c:
                self.abort_event.set()

            if self.target_pid is not None and self.freeze_temp_c is not None and self.resume_temp_c is not None:
                action = thermal_governor_action(max_t, self.frozen, self.freeze_temp_c, self.resume_temp_c)
                if action == "freeze":
                    self._freeze()
                elif action == "resume":
                    self._resume()

            self._stop.wait(self.interval_s)

    def _freeze(self):
        try:
            os.kill(self.target_pid, signal.SIGSTOP)
        except OSError:
            return
        self.frozen = True
        self._frozen_since = time.monotonic()

    def _resume(self):
        try:
            os.kill(self.target_pid, signal.SIGCONT)
        except OSError:
            pass
        if self._frozen_since is not None:
            self.frozen_s += time.monotonic() - self._frozen_since
        self.frozen = False
        self._frozen_since = None

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join()
        # SIGTERM/SIGKILL остановленному (SIGSTOP) процессу не доставляются, пока
        # не пришёл SIGCONT — иначе server.terminate() в finally зависнет навсегда.
        if self.frozen:
            self._resume()

    def summary(self) -> dict:
        def avg_max(samples):
            vals = [v for s in samples for v in s.values()]
            if not vals:
                return None, None
            return sum(vals) / len(vals), max(vals)

        avg_t, max_t = avg_max(self.temp_samples)
        avg_f, max_f = avg_max(self.freq_samples)
        return {
            "avg_temp_c": avg_t, "max_temp_c": max_t,
            "avg_freq_khz": avg_f, "max_freq_khz": max_f,
            "frozen_s": self.frozen_s,
        }


# ---------------------------------------------------------------------------
# матрица конфигураций
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    model_name: str
    quant: str
    path: str


@dataclass
class BenchConfig:
    model_name: str
    quant: str
    model_path: str
    threads: int
    taskset_cores: str | None
    flash_attn: bool


_MODEL_FILE_RE = re.compile(r"^ggml-(.+)-(q\d(?:_\d)?)\.bin$")


def discover_models(models_dir: str) -> list[ModelConfig]:
    found = []
    for path in sorted(glob.glob(os.path.join(models_dir, "ggml-*.bin"))):
        m = _MODEL_FILE_RE.match(os.path.basename(path))
        if not m:
            continue
        found.append(ModelConfig(model_name=m.group(1), quant=m.group(2), path=path))
    return found


def build_matrix(
    models: list[ModelConfig],
    thread_options: tuple[int, ...] = (8, 4),
    flash_attn_options: tuple[bool, ...] = (True, False),
) -> list[BenchConfig]:
    """§9 Этап 0: -t 8 (все ядра) и -t 4 с пином на большие ядра (taskset -c 4-7)."""
    matrix = []
    for model in models:
        for threads in thread_options:
            taskset_cores = "4-7" if threads == 4 else None
            for flash_attn in flash_attn_options:
                matrix.append(BenchConfig(
                    model_name=model.model_name, quant=model.quant, model_path=model.path,
                    threads=threads, taskset_cores=taskset_cores, flash_attn=flash_attn,
                ))
    return matrix


# ---------------------------------------------------------------------------
# оркестрация: аудио, whisper-server, HTTP
# ---------------------------------------------------------------------------

def probe_duration_s(path: str) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


def normalize_audio(input_path: str, output_wav: str) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-i", input_path, "-ac", "1", "-ar", "16000",
         "-sample_fmt", "s16", output_wav],
        capture_output=True, check=True,
    )


def cut_segment(wav_path: str, t0: float, t1: float, out_path: str) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-ss", str(t0), "-to", str(t1), "-i", wav_path,
         "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", out_path],
        capture_output=True, check=True,
    )


def _post_multipart_file(url: str, field_name: str, file_path: str, timeout: float = 1800.0) -> bytes:
    # таймаут с большим запасом: пока whisper-server на SIGSTOP (см. HardwareSampler),
    # HTTP-ответ не придёт, и это не сбой — клиент должен просто подождать разморозки
    boundary = uuid.uuid4().hex
    with open(file_path, "rb") as f:
        file_bytes = f.read()
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{field_name}"; filename="{os.path.basename(file_path)}"\r\n'
        f"Content-Type: application/octet-stream\r\n\r\n"
    ).encode() + file_bytes + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def wait_for_server(url: str, timeout_s: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(url, timeout=2)
            return
        except Exception:
            time.sleep(0.5)
    raise TimeoutError(f"whisper-server не поднялся за {timeout_s} с: {url}")


def start_whisper_server(config: BenchConfig, port: int, prompt: str) -> subprocess.Popen:
    cmd = []
    if config.taskset_cores:
        cmd += ["taskset", "-c", config.taskset_cores]
    cmd += [
        "whisper-server", "-m", config.model_path, "--host", "127.0.0.1", "--port", str(port),
        "-l", "ru", "-t", str(config.threads),
        "-fa" if config.flash_attn else "-nfa",
    ]
    if prompt:
        cmd += ["--prompt", prompt]
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def transcribe_segment(port: int, wav_path: str) -> str:
    raw = _post_multipart_file(f"http://127.0.0.1:{port}/inference", "file", wav_path)
    data = json.loads(raw.decode("utf-8"))
    return data.get("text", "").strip()


def segment_overlaps(t0: float, t1: float, window: tuple[float, float]) -> bool:
    return t0 < window[1] and t1 > window[0]


def set_governor(governor: str) -> list[str]:
    """Возвращает список путей, где реально удалось сменить governor (для отката)."""
    changed = []
    for path in glob.glob("/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor"):
        try:
            with open(path, "w") as f:
                f.write(governor)
            changed.append(path)
        except OSError as exc:
            print(f"предупреждение: не удалось выставить governor {path}: {exc}", file=sys.stderr)
    return changed


# ---------------------------------------------------------------------------
# один прогон конфигурации
# ---------------------------------------------------------------------------

@dataclass
class BenchResult:
    config: BenchConfig
    rtf: float
    wall_clock_s: float
    peak_rss_mb: float | None
    hw: dict
    aborted: bool = False
    wer: float | None = None
    term_precision: float | None = None
    term_recall: float | None = None
    term_f1: float | None = None
    # (t0, t1, text_raw) до нормализации глоссарием — SPEC.md §9.2: пересчёт по
    # глоссарию должен быть секундами, а не повторным прогоном ASR на часы
    raw_segments: list = field(default_factory=list)


def run_one_config(
    config: BenchConfig,
    vad_segments: list[tuple[float, float]],
    wav_path: str,
    prompt: str,
    port: int,
    terms: list[GlossaryTerm] | None,
    gold: GoldTranscript | None,
    cutoff: float,
    workdir: str,
    max_temp_c: float | None = 108.0,
    freeze_temp_c: float | None = None,
    resume_temp_c: float | None = None,
) -> BenchResult:
    """freeze_temp_c/resume_temp_c — без фана плата не тянет непрерывный прогон:
    сэмплер сам держит whisper-server на SIGSTOP/SIGCONT (см. HardwareSampler),
    реагируя даже посреди одного длинного сегмента. Время заморозки не входит
    в RTF (RTF — время счёта, не время охлаждения)."""
    print(f"--- {config.model_name} {config.quant} threads={config.threads} "
          f"taskset={config.taskset_cores} flash_attn={config.flash_attn} ---", file=sys.stderr)

    server = start_whisper_server(config, port, prompt)
    sampler = HardwareSampler(
        abort_temp_c=max_temp_c, freeze_temp_c=freeze_temp_c,
        resume_temp_c=resume_temp_c, target_pid=server.pid,
    )
    aborted = False
    try:
        wait_for_server(f"http://127.0.0.1:{port}/")
        sampler.start()

        hyp_segments: list[tuple[float, float, str]] = []
        raw_segments: list[tuple[float, float, str]] = []
        wall_clock_total = 0.0
        processed_speech_s = 0.0
        total_speech_s = sum(t1 - t0 for t0, t1 in vad_segments)
        run_start = time.monotonic()
        n = len(vad_segments)
        for i, (t0, t1) in enumerate(vad_segments):
            if sampler.abort_event.is_set():
                aborted = True
                print(f"перегрев (>={max_temp_c}°C), прогон конфигурации прерван досрочно", file=sys.stderr)
                break
            seg_path = os.path.join(workdir, f"seg_{t0:.3f}_{t1:.3f}.wav")
            cut_segment(wav_path, t0, t1, seg_path)
            start = time.monotonic()
            raw_text = transcribe_segment(port, seg_path)
            wall_clock_total += time.monotonic() - start
            os.remove(seg_path)
            text = apply_glossary(raw_text, terms, cutoff) if terms else raw_text
            hyp_segments.append((t0, t1, text))
            raw_segments.append((t0, t1, raw_text))

            processed_speech_s += t1 - t0
            pure_wall_clock = max(0.0, wall_clock_total - sampler.frozen_s)
            running_rtf = pure_wall_clock / processed_speech_s if processed_speech_s else 0.0
            eta_s = estimate_remaining_s(time.monotonic() - run_start, processed_speech_s, total_speech_s)
            print(f"  [{i + 1}/{n}] +{t1 - t0:.1f}с речи, RTF~{running_rtf:.2f}, "
                  f"заморожено: {sampler.frozen_s:.0f}с, ETA конфигурации: {format_eta(eta_s)}", file=sys.stderr)
            if (i + 1) % 10 == 0 or i + 1 == n:
                print(f"прогресс: {config.model_name}/{config.quant} t={config.threads} fa={config.flash_attn} "
                      f"{i + 1}/{n} сегментов, RTF~{running_rtf:.2f}, заморожено: {sampler.frozen_s:.0f}с, "
                      f"ETA конфигурации: {format_eta(eta_s)}", file=sys.stderr)

        peak_rss = peak_rss_kb(server.pid)
    finally:
        sampler.stop()
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()

    pure_wall_clock_total = max(0.0, wall_clock_total - sampler.frozen_s)
    if sampler.frozen_s:
        print(f"суммарно заморожено (SIGSTOP): {sampler.frozen_s:.0f}с", file=sys.stderr)

    result = BenchResult(
        config=config,
        # RTF считаем к реально обработанной речи (после VAD/--max-segments), а не
        # к длительности всего файла — иначе при усечении сегментов число подтасовано.
        # Время заморозки (SIGSTOP) вычтено — RTF отражает чистую скорость счёта.
        rtf=rtf(pure_wall_clock_total, processed_speech_s) if not aborted else float("nan"),
        wall_clock_s=pure_wall_clock_total,
        peak_rss_mb=(peak_rss / 1024) if peak_rss else None,
        hw=sampler.summary(),
        aborted=aborted,
        raw_segments=raw_segments,
    )

    if not aborted and gold is not None and gold.wer_window is not None:
        hyp_window_text = " ".join(
            text for t0, t1, text in hyp_segments if segment_overlaps(t0, t1, gold.wer_window)
        )
        ref_text = gold_reference_text(gold)
        if jiwer is not None and ref_text.strip():
            result.wer = jiwer.wer(normalize_for_wer(ref_text), normalize_for_wer(hyp_window_text))
        if terms is not None:
            f1 = term_f1(terms, ref_text, hyp_window_text)
            result.term_precision = f1.precision
            result.term_recall = f1.recall
            result.term_f1 = f1.f1

    return result


def print_table(results: list[BenchResult]) -> None:
    header = ["модель", "квант", "t", "taskset", "flash-attn", "RTF", "WER", "term_P", "term_R", "term_F1", "peak_RSS_MB", "avg_temp_C"]
    rows = [header]
    for r in results:
        rows.append([
            r.config.model_name, r.config.quant, str(r.config.threads),
            r.config.taskset_cores or "-", str(r.config.flash_attn),
            "ПЕРЕГРЕВ" if r.aborted else f"{r.rtf:.3f}",
            f"{r.wer:.3f}" if r.wer is not None else "-",
            f"{r.term_precision:.3f}" if r.term_precision is not None else "-",
            f"{r.term_recall:.3f}" if r.term_recall is not None else "-",
            f"{r.term_f1:.3f}" if r.term_f1 is not None else "-",
            f"{r.peak_rss_mb:.0f}" if r.peak_rss_mb is not None else "-",
            f"{r.hw.get('avg_temp_c'):.1f}" if r.hw.get("avg_temp_c") is not None else "-",
        ])
    widths = [max(len(row[i]) for row in rows) for i in range(len(header))]
    for row in rows:
        print("  ".join(cell.ljust(w) for cell, w in zip(row, widths)))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", required=True, help="реальная запись созвона")
    parser.add_argument("--gold", help="gold.tsv (без него считается только RTF/perf)")
    parser.add_argument("--glossary", help="glossary.json")
    parser.add_argument("--models-dir", default="/srv/asr/models")
    parser.add_argument("--vad-model", default="/srv/asr/models/silero_vad.onnx")
    parser.add_argument("--workdir", default=None)
    parser.add_argument("--cutoff", type=float, default=0.8)
    parser.add_argument("--port", type=int, default=8088)
    parser.add_argument("--threads", type=int, nargs="+", default=[8, 4])
    parser.add_argument("--flash-attn", choices=["both", "on", "off"], default="both")
    parser.add_argument("--no-governor", action="store_true", help="не трогать cpufreq governor")
    parser.add_argument("--max-temp-c", type=float, default=108.0,
                         help="аварийная остановка конфигурации при превышении (запас до critical=115°C)")
    parser.add_argument("--cooldown-temp-c", type=float, default=70.0,
                         help="ждать остывания до этой температуры перед следующей конфигурацией")
    parser.add_argument("--cooldown-timeout-s", type=float, default=600.0,
                         help="не ждать остывания дольше этого времени")
    parser.add_argument("--freeze-temp-c", type=float, default=None,
                         help="без вентилятора: держать whisper-server на SIGSTOP выше этой температуры")
    parser.add_argument("--resume-temp-c", type=float, default=80.0,
                         help="SIGCONT после остывания до этой температуры")
    parser.add_argument("--no-prompt", action="store_true",
                         help="не подавать глоссарий в --prompt, оставить только постобработку")
    parser.add_argument("--dump-hypotheses", default=None,
                         help="куда сложить сырые гипотезы (JSON) для пересчёта метрик без ASR")
    parser.add_argument("--max-segments", type=int, default=None,
                         help="ограничить число VAD-сегментов на конфигурацию (для быстрой прикидки RTF)")
    args = parser.parse_args(argv)
    signal.signal(signal.SIGTERM, _raise_terminated)

    if shutil.which("whisper-server") is None:
        print("whisper-server не найден в PATH", file=sys.stderr)
        return 1

    workdir = args.workdir or tempfile.mkdtemp(prefix="asr-bench-")
    os.makedirs(workdir, exist_ok=True)

    gold = None
    terms: list[GlossaryTerm] | None = None
    prompt = ""
    if args.gold:
        with open(args.gold, encoding="utf-8") as f:
            gold = parse_gold_tsv(f.read())
    if args.glossary:
        terms = load_glossary(args.glossary)
        # промпт — первый эшелон защиты терминологии (SPEC.md §9.1), но он не
        # бесплатный: на записи 2026-08-12 промпт из 22 терминов поднял WER с
        # 0.359 до 0.399, «слыша» технику там, где её нет. Второй эшелон
        # (постобработка) при этом WER не меняет вовсе, поэтому их полезно
        # уметь включать по отдельности и мерить порознь.
        prompt = "" if args.no_prompt else glossary_prompt_text(terms)

    changed_governors = [] if args.no_governor else set_governor("performance")
    try:
        wav_path = os.path.join(workdir, "reference.wav")
        print("нормализация аудио...", file=sys.stderr)
        normalize_audio(args.audio, wav_path)
        audio_duration_s = probe_duration_s(wav_path)
        print(f"длительность записи: {audio_duration_s:.1f}с", file=sys.stderr)

        print("VAD...", file=sys.stderr)
        vad_segments = run_vad(wav_path, args.vad_model)
        speech_s = sum(t1 - t0 for t0, t1 in vad_segments)
        print(f"найдено {len(vad_segments)} речевых сегментов, {speech_s:.1f}с речи "
              f"({100 * speech_s / audio_duration_s:.0f}% файла)", file=sys.stderr)
        if args.max_segments is not None:
            vad_segments = vad_segments[:args.max_segments]
            print(f"ограничение: первые {len(vad_segments)} сегментов", file=sys.stderr)

        models = discover_models(args.models_dir)
        if not models:
            print(f"в {args.models_dir} не найдено моделей ggml-*.bin", file=sys.stderr)
            return 1

        flash_attn_options = {"both": (True, False), "on": (True,), "off": (False,)}[args.flash_attn]
        matrix = build_matrix(models, thread_options=tuple(args.threads), flash_attn_options=flash_attn_options)

        print(f"матрица: {len(matrix)} конфигураций", file=sys.stderr)
        results = []
        matrix_start = time.monotonic()
        for i, config in enumerate(matrix):
            result = run_one_config(
                config, vad_segments, wav_path, prompt,
                args.port, terms, gold, args.cutoff, workdir,
                max_temp_c=args.max_temp_c,
                freeze_temp_c=args.freeze_temp_c,
                resume_temp_c=args.resume_temp_c,
            )
            results.append(result)
            done = i + 1
            avg_config_s = (time.monotonic() - matrix_start) / done
            eta_matrix_s = avg_config_s * (len(matrix) - done)
            print(f"прогресс матрицы: {done}/{len(matrix)} конфигураций, "
                  f"ETA всей матрицы: {format_eta(eta_matrix_s)}", file=sys.stderr)
            if i < len(matrix) - 1:
                temp = current_max_temp_c()
                if temp > args.cooldown_temp_c:
                    print(f"остываю: {temp:.0f}°C -> ждём {args.cooldown_temp_c:.0f}°C "
                          f"(не дольше {args.cooldown_timeout_s:.0f}с)...", file=sys.stderr)
                    final_temp = wait_for_cooldown(args.cooldown_temp_c, args.cooldown_timeout_s)
                    print(f"остыло до {final_temp:.0f}°C, продолжаю", file=sys.stderr)

        print_table(results)
        if args.dump_hypotheses:
            dump = [
                {
                    "model": r.config.model_name, "quant": r.config.quant,
                    "threads": r.config.threads, "flash_attn": r.config.flash_attn,
                    "rtf": r.rtf, "aborted": r.aborted,
                    "segments": [
                        {"t0": t0, "t1": t1, "text_raw": text}
                        for t0, t1, text in r.raw_segments
                    ],
                }
                for r in results
            ]
            tmp = args.dump_hypotheses + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(dump, f, ensure_ascii=False)
            os.replace(tmp, args.dump_hypotheses)
            print(f"гипотезы сохранены: {args.dump_hypotheses}", file=sys.stderr)
    except Terminated:
        print("прерван SIGTERM, возвращаю governor и выхожу", file=sys.stderr)
        return 1
    finally:
        for path in changed_governors:
            try:
                with open(path, "w") as f:
                    f.write("ondemand")
            except OSError:
                pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
