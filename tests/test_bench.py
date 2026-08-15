"""Тесты чистой логики bench.py (этап 0 — замер). unittest из stdlib, без фикстур."""
import json
import os
import tempfile
import unittest

import bench


class ParseGoldTsvTests(unittest.TestCase):
    def test_valid_example_from_spec(self):
        text = (
            "# call_id: 2026-07-31-standup\n"
            "# wer_window: 300.000 420.000\n"
            "0.000\t4.320\tandrey: так, погнали, у нас VDA5050 не отдаёт orderUpdate\n"
            "4.100\t6.780\tmikhail: угу\n"
            "6.780\t11.200\tolga:\n"
        )
        gold = bench.parse_gold_tsv(text)
        self.assertEqual(gold.call_id, "2026-07-31-standup")
        self.assertEqual(gold.wer_window, (300.000, 420.000))
        self.assertEqual(len(gold.segments), 3)
        self.assertEqual(gold.segments[0].t0, 0.000)
        self.assertEqual(gold.segments[0].t1, 4.320)
        self.assertEqual(gold.segments[0].speaker, "andrey")
        self.assertEqual(gold.segments[0].text, "так, погнали, у нас VDA5050 не отдаёт orderUpdate")
        self.assertEqual(gold.segments[2].speaker, "olga")
        self.assertEqual(gold.segments[2].text, "")

    def test_missing_wer_window_is_none(self):
        text = "# call_id: x\n0.000\t1.000\tandrey: привет\n"
        gold = bench.parse_gold_tsv(text)
        self.assertIsNone(gold.wer_window)

    def test_missing_call_id_raises(self):
        text = "# wer_window: 0 10\n0.000\t1.000\tandrey: привет\n"
        with self.assertRaises(ValueError):
            bench.parse_gold_tsv(text)

    def test_malformed_line_raises(self):
        text = "# call_id: x\n0.000\tandrey: привет без второй колонки\n"
        with self.assertRaises(ValueError):
            bench.parse_gold_tsv(text)

    def test_line_without_speaker_colon_raises(self):
        text = "# call_id: x\n0.000\t1.000\tтекст без спикера\n"
        with self.assertRaises(ValueError):
            bench.parse_gold_tsv(text)

    def test_blank_lines_ignored(self):
        text = "# call_id: x\n\n0.000\t1.000\tandrey: привет\n\n"
        gold = bench.parse_gold_tsv(text)
        self.assertEqual(len(gold.segments), 1)


class GoldReferenceTextTests(unittest.TestCase):
    def test_gathers_only_segments_overlapping_window_with_text(self):
        text = (
            "# call_id: x\n"
            "# wer_window: 4.000 10.000\n"
            "0.000\t3.000\tandrey: до окна\n"
            "4.000\t6.000\tandrey: первый\n"
            "6.780\t9.200\tmikhail: второй\n"
            "9.500\t20.000\tolga:\n"
        )
        gold = bench.parse_gold_tsv(text)
        self.assertEqual(bench.gold_reference_text(gold), "первый второй")

    def test_no_window_returns_empty(self):
        text = "# call_id: x\n0.000\t1.000\tandrey: привет\n"
        gold = bench.parse_gold_tsv(text)
        self.assertEqual(bench.gold_reference_text(gold), "")


class RuNumberWordsToDigitsTests(unittest.TestCase):
    def test_simple_units(self):
        self.assertEqual(bench.ru_number_words_to_digits("пять"), "5")

    def test_teens(self):
        self.assertEqual(bench.ru_number_words_to_digits("пятнадцать"), "15")

    def test_tens_and_units(self):
        self.assertEqual(bench.ru_number_words_to_digits("двадцать один"), "21")

    def test_hundreds_tens_units(self):
        self.assertEqual(bench.ru_number_words_to_digits("сто двадцать один"), "121")

    def test_in_sentence(self):
        self.assertEqual(
            bench.ru_number_words_to_digits("у нас двадцать один участник"),
            "у нас 21 участник",
        )
        self.assertEqual(
            bench.ru_number_words_to_digits("выпустили версию пять точка два"),
            "выпустили версию 5 точка 2",
        )

    def test_unrecognized_words_untouched(self):
        self.assertEqual(bench.ru_number_words_to_digits("привет мир"), "привет мир")


class NormalizeForWerTests(unittest.TestCase):
    def test_lowercase_and_punctuation_stripped(self):
        self.assertEqual(
            bench.normalize_for_wer("Так, погнали! У нас VDA5050."),
            "так погнали у нас vda5050",
        )

    def test_hyphens_significant(self):
        self.assertEqual(bench.normalize_for_wer("бэк-энд"), "бэк-энд")
        self.assertNotEqual(bench.normalize_for_wer("бэк-энд"), bench.normalize_for_wer("бэк энд"))

    def test_numbers_converted(self):
        self.assertEqual(bench.normalize_for_wer("двадцать один участник"), "21 участник")


class GlossaryTermTests(unittest.TestCase):
    def _terms(self):
        return [
            bench.GlossaryTerm("VDA5050", ["вда 5050", "вэдэа 5050", "вда5050"], True),
            bench.GlossaryTerm("MAPF", ["мапф", "мэпф", "map f"], True),
            bench.GlossaryTerm("Isaac Sim", ["айзек сим", "изак сим"], False),
            bench.GlossaryTerm("ADG", [], True),
        ]

    def test_variant_replaced_with_canon(self):
        out = bench.apply_glossary("у нас вда 5050 не отдаёт данные", self._terms())
        self.assertIn("VDA5050", out)
        self.assertNotIn("вда 5050", out)

    def test_two_word_variant_replaced(self):
        out = bench.apply_glossary("запускаем изак сим сегодня", self._terms())
        self.assertIn("Isaac Sim", out)

    def test_declension_stripped_for_latin_acronym(self):
        out = bench.apply_glossary("вопрос по ADGу решён", self._terms())
        self.assertIn("ADG", out)
        self.assertNotIn("ADGу", out)

    def test_unrelated_text_untouched(self):
        out = bench.apply_glossary("сегодня хорошая погода", self._terms())
        self.assertEqual(out, "сегодня хорошая погода")

    def test_exact_variant_matches_even_when_short(self):
        # «сш» короче порога нечёткого сравнения, но это точный вариант
        terms = [bench.GlossaryTerm("SSH", ["ссаш", "сш"], True)]
        self.assertEqual(bench.apply_glossary("зайди по сш на хост", terms), "зайди по SSH на хост")


class GlossaryFalsePositiveTests(unittest.TestCase):
    """Регрессии на реальных ложных срабатываниях, найденных на записи 2026-08-12."""

    def test_common_word_not_replaced_by_similar_term(self):
        terms = [bench.GlossaryTerm("NetBird", ["нетборд", "нетберд"], True)]
        self.assertEqual(bench.apply_glossary("это надо сделать", terms), "это надо сделать")

    def test_personal_name_not_replaced_by_acronym(self):
        terms = [bench.GlossaryTerm("SSH", ["ссаш", "эсаш"], True)]
        for phrase in ("саш подскажи", "Саш подскажи"):
            self.assertNotIn("SSH", bench.apply_glossary(phrase, terms))

    def test_short_word_not_fuzzy_matched(self):
        terms = [bench.GlossaryTerm("LOC", ["лок", "локт"], True)]
        self.assertEqual(bench.apply_glossary("ну ло и что", terms), "ну ло и что")

    def test_unrelated_long_word_not_replaced(self):
        terms = [bench.GlossaryTerm("Unity", ["юнити", "юнутри"], True)]
        self.assertEqual(bench.apply_glossary("лежит внутри папки", terms), "лежит внутри папки")

    def test_real_distortion_still_replaced(self):
        terms = [
            bench.GlossaryTerm("NetBird", ["нетборд", "нетберд"], True),
            bench.GlossaryTerm("Unity", ["юнити", "юнутри"], True),
            bench.GlossaryTerm("LOC", ["лок", "локт"], True),
        ]
        self.assertIn("NetBird", bench.apply_glossary("порты в нетборде", terms))
        self.assertIn("Unity", bench.apply_glossary("папка юнутри проекта", terms))
        self.assertIn("LOC", bench.apply_glossary("это лок задача", terms))


class CountCanonOccurrencesTests(unittest.TestCase):
    def test_counts_word_boundary_matches(self):
        self.assertEqual(bench.count_canon_occurrences("VDA5050 и снова VDA5050", "VDA5050"), 2)

    def test_case_sensitive(self):
        self.assertEqual(bench.count_canon_occurrences("vda5050", "VDA5050"), 0)

    def test_multiword_canon(self):
        self.assertEqual(bench.count_canon_occurrences("сегодня Isaac Sim и Isaac Sim2", "Isaac Sim"), 1)


class TermF1Tests(unittest.TestCase):
    def test_perfect_match(self):
        terms = [bench.GlossaryTerm("VDA5050", [], True)]
        result = bench.term_f1(terms, "у нас VDA5050 не отдаёт", "у нас VDA5050 не отдаёт")
        self.assertEqual(result.tp, 1)
        self.assertEqual(result.fp, 0)
        self.assertEqual(result.fn, 0)
        self.assertEqual(result.precision, 1.0)
        self.assertEqual(result.recall, 1.0)
        self.assertEqual(result.f1, 1.0)

    def test_missed_term_is_false_negative(self):
        terms = [bench.GlossaryTerm("VDA5050", [], True)]
        result = bench.term_f1(terms, "у нас VDA5050 не отдаёт", "у нас вда 5050 не отдаёт")
        self.assertEqual(result.tp, 0)
        self.assertEqual(result.fn, 1)
        self.assertEqual(result.recall, 0.0)

    def test_extra_term_is_false_positive(self):
        terms = [bench.GlossaryTerm("VDA5050", [], True)]
        result = bench.term_f1(terms, "просто текст", "а тут VDA5050 упомянут зря")
        self.assertEqual(result.fp, 1)
        self.assertEqual(result.precision, 0.0)

    def test_no_terms_anywhere_is_undefined_but_safe(self):
        terms = [bench.GlossaryTerm("VDA5050", [], True)]
        result = bench.term_f1(terms, "просто текст", "просто текст")
        self.assertEqual(result.tp, 0)
        self.assertEqual(result.fp, 0)
        self.assertEqual(result.fn, 0)
        self.assertEqual(result.precision, 0.0)
        self.assertEqual(result.recall, 0.0)
        self.assertEqual(result.f1, 0.0)


class RtfTests(unittest.TestCase):
    def test_slower_than_realtime(self):
        self.assertAlmostEqual(bench.rtf(30.0, 20.0), 1.5)

    def test_faster_than_realtime(self):
        self.assertAlmostEqual(bench.rtf(10.0, 20.0), 0.5)

    def test_zero_duration_raises(self):
        with self.assertRaises(ValueError):
            bench.rtf(1.0, 0.0)


class ParseVadStdoutTests(unittest.TestCase):
    def test_parses_segments(self):
        out = (
            "какой-то мусор в начале\n"
            "1.830 -- 2.828\n"
            "3.750 -- 5.356\n"
            "Saved to /tmp/out.wav\n"
        )
        segments = bench.parse_vad_stdout(out)
        self.assertEqual(segments, [(1.830, 2.828), (3.750, 5.356)])

    def test_no_segments(self):
        self.assertEqual(bench.parse_vad_stdout("тишина, ничего не найдено\n"), [])


class ReadThermalZonesTests(unittest.TestCase):
    def test_reads_millidegree_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            zone_dir = os.path.join(tmp, "thermal_zone0")
            os.makedirs(zone_dir)
            with open(os.path.join(zone_dir, "type"), "w") as f:
                f.write("soc-thermal\n")
            with open(os.path.join(zone_dir, "temp"), "w") as f:
                f.write("45123\n")
            zones = bench.read_thermal_zones(tmp)
            self.assertEqual(zones["soc-thermal"], 45.123)

    def test_missing_dir_returns_empty(self):
        self.assertEqual(bench.read_thermal_zones("/no/such/path"), {})


class ReadCpuFreqsTests(unittest.TestCase):
    def test_reads_khz_per_cpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            for i in range(2):
                cpu_dir = os.path.join(tmp, f"cpu{i}", "cpufreq")
                os.makedirs(cpu_dir)
                with open(os.path.join(cpu_dir, "scaling_cur_freq"), "w") as f:
                    f.write(str(1800000 + i * 100000) + "\n")
            freqs = bench.read_cpu_freqs(tmp)
            self.assertEqual(freqs[0], 1800000)
            self.assertEqual(freqs[1], 1900000)


class ParseProcStatusVmHwmTests(unittest.TestCase):
    def test_parses_peak_rss_kb(self):
        status = "Name:\twhisper-server\nVmHWM:\t  412340 kB\nVmRSS:\t  400000 kB\n"
        self.assertEqual(bench.parse_proc_status_vmhwm(status), 412340)

    def test_missing_field_returns_none(self):
        self.assertIsNone(bench.parse_proc_status_vmhwm("Name:\tx\n"))


class ParseTolkTranscriptTests(unittest.TestCase):
    def test_parses_header_and_rows(self):
        text = (
            'Транскрипция записи "2026-08-12" 12 августа 2026 г\n'
            "00:00:01\tАндрей Павлов\tИтак, коллеги.\n"
            "00:01:05\tПавел Шубин\tДа, привет.\n"
        )
        rows = bench.parse_tolk_transcript(text)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0].t0, 1.0)
        self.assertEqual(rows[0].speaker, "Андрей Павлов")
        self.assertEqual(rows[0].text, "Итак, коллеги.")
        self.assertEqual(rows[1].t0, 65.0)

    def test_skips_blank_and_malformed_lines(self):
        text = (
            "заголовок\n"
            "\n"
            "00:00:01\tАндрей Павлов\tТекст.\n"
            "мусор без табов\n"
        )
        rows = bench.parse_tolk_transcript(text)
        self.assertEqual(len(rows), 1)

    def test_hours_parsed(self):
        text = "заголовок\n01:02:03\tX Y\tтекст\n"
        rows = bench.parse_tolk_transcript(text)
        self.assertEqual(rows[0].t0, 3723.0)


class ClipEndByVadTests(unittest.TestCase):
    def test_uses_last_vad_segment_inside_interval(self):
        vad = [(0.0, 1.0), (10.0, 12.5), (13.0, 14.0), (40.0, 41.0)]
        # реплика 10..30, речь внутри кончается на 14.0
        self.assertEqual(bench.clip_end_by_vad(10.0, 30.0, vad, fallback_s=2.0), 14.0)

    def test_falls_back_when_no_speech_inside(self):
        vad = [(0.0, 1.0), (40.0, 41.0)]
        self.assertEqual(bench.clip_end_by_vad(10.0, 30.0, vad, fallback_s=2.0), 12.0)

    def test_fallback_clamped_by_next_row(self):
        vad = []
        self.assertEqual(bench.clip_end_by_vad(10.0, 10.5, vad, fallback_s=2.0), 10.5)

    def test_vad_segment_crossing_next_row_is_clamped(self):
        vad = [(10.0, 45.0)]
        self.assertEqual(bench.clip_end_by_vad(10.0, 30.0, vad, fallback_s=2.0), 30.0)

    def test_never_returns_zero_length(self):
        vad = [(10.0, 10.0)]
        self.assertGreater(bench.clip_end_by_vad(10.0, 30.0, vad, fallback_s=2.0), 10.0)

    def test_simultaneous_rows_do_not_collapse(self):
        # у Толка две реплики попали на одну секунду — перебивание
        vad = [(10.0, 13.0)]
        t1 = bench.clip_end_by_vad(10.0, 10.0, vad, fallback_s=2.0)
        self.assertGreater(t1, 10.0)

    def test_next_row_before_start_does_not_collapse(self):
        t1 = bench.clip_end_by_vad(10.0, 9.0, [], fallback_s=2.0)
        self.assertGreater(t1, 10.0)


class HasLatinTests(unittest.TestCase):
    def test_pure_latin(self):
        self.assertTrue(bench.has_latin("VDA5050"))

    def test_mixed(self):
        self.assertTrue(bench.has_latin("ADGу"))

    def test_cyrillic_only(self):
        self.assertFalse(bench.has_latin("привет"))

    def test_digits_only(self):
        self.assertFalse(bench.has_latin("5050"))


class ClusterVariantsTests(unittest.TestCase):
    def test_groups_similar_spellings(self):
        word_lists = [["дейлике"], ["делике"], ["дейлик"]]
        clusters = bench.cluster_variants(word_lists, cutoff=0.7)
        self.assertEqual(len(clusters), 1)
        self.assertEqual(set(clusters[0]), {"дейлике", "делике", "дейлик"})

    def test_keeps_unrelated_words_apart(self):
        word_lists = [["привет"], ["дейлик"]]
        clusters = bench.cluster_variants(word_lists, cutoff=0.7)
        self.assertEqual(len(clusters), 2)

    def test_identical_words_form_single_cluster(self):
        word_lists = [["привет"], ["привет"]]
        clusters = bench.cluster_variants(word_lists, cutoff=0.7)
        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0], ["привет", "привет"])


class GlossaryCandidatesTests(unittest.TestCase):
    def _sources(self):
        return [
            [(1.0, "у"), (1.2, "нас"), (1.4, "лок"), (1.6, "сломался")],
            [(1.0, "у"), (1.2, "нас"), (1.4, "LOC"), (1.6, "сломался")],
        ]

    def test_latin_token_is_candidate(self):
        cands = bench.collect_glossary_candidates(self._sources(), window_s=30.0)
        forms = {frozenset(c.forms) for c in cands}
        self.assertTrue(any("loc" in f or "LOC" in f for f in forms))

    def test_cyrillic_and_latin_spellings_join_via_translit(self):
        # min_len явно: проверяется склейка через транслитерацию, не порог длины
        cands = bench.collect_glossary_candidates(self._sources(), window_s=30.0, min_len=3)
        joined = {tuple(sorted(c.forms)) for c in cands}
        self.assertTrue(any("лок" in t and "LOC" in t for t in joined))

    def test_agreeing_common_words_are_not_candidates(self):
        cands = bench.collect_glossary_candidates(self._sources(), window_s=30.0, min_len=3)
        all_forms = {f for c in cands for f in c.forms}
        self.assertNotIn("нас", all_forms)
        self.assertNotIn("сломался", all_forms)

    def test_window_separates_distant_occurrences(self):
        sources = [
            [(1.0, "локом")],
            [(600.0, "лаком")],
        ]
        cands = bench.collect_glossary_candidates(sources, window_s=30.0)
        for c in cands:
            self.assertLessEqual(len(c.forms), 1)


class FormatEtaTests(unittest.TestCase):
    def test_seconds_only(self):
        self.assertEqual(bench.format_eta(5), "5с")

    def test_zero(self):
        self.assertEqual(bench.format_eta(0), "0с")

    def test_minutes_and_seconds(self):
        self.assertEqual(bench.format_eta(65), "1м 05с")

    def test_hours_and_minutes(self):
        self.assertEqual(bench.format_eta(3725), "1ч 02м")

    def test_negative_clamped_to_zero(self):
        self.assertEqual(bench.format_eta(-5), "0с")


class EstimateRemainingTests(unittest.TestCase):
    def test_linear_extrapolation(self):
        self.assertAlmostEqual(bench.estimate_remaining_s(elapsed_s=10.0, done_units=2.0, total_units=10.0), 40.0)

    def test_nothing_done_yet_is_nan(self):
        import math
        self.assertTrue(math.isnan(bench.estimate_remaining_s(elapsed_s=10.0, done_units=0.0, total_units=10.0)))

    def test_fully_done_is_zero(self):
        self.assertAlmostEqual(bench.estimate_remaining_s(elapsed_s=10.0, done_units=10.0, total_units=10.0), 0.0)


class ThermalGovernorActionTests(unittest.TestCase):
    def test_not_frozen_below_freeze_threshold_is_noop(self):
        self.assertEqual(bench.thermal_governor_action(90.0, frozen=False, freeze_temp_c=100.0, resume_temp_c=80.0), "noop")

    def test_not_frozen_reaches_freeze_threshold_freezes(self):
        self.assertEqual(bench.thermal_governor_action(100.0, frozen=False, freeze_temp_c=100.0, resume_temp_c=80.0), "freeze")

    def test_not_frozen_above_freeze_threshold_freezes(self):
        self.assertEqual(bench.thermal_governor_action(105.0, frozen=False, freeze_temp_c=100.0, resume_temp_c=80.0), "freeze")

    def test_frozen_still_above_resume_threshold_stays_frozen(self):
        self.assertEqual(bench.thermal_governor_action(90.0, frozen=True, freeze_temp_c=100.0, resume_temp_c=80.0), "noop")

    def test_frozen_reaches_resume_threshold_resumes(self):
        self.assertEqual(bench.thermal_governor_action(80.0, frozen=True, freeze_temp_c=100.0, resume_temp_c=80.0), "resume")

    def test_frozen_below_resume_threshold_resumes(self):
        self.assertEqual(bench.thermal_governor_action(70.0, frozen=True, freeze_temp_c=100.0, resume_temp_c=80.0), "resume")

    def test_no_hysteresis_band_still_toggles_correctly(self):
        self.assertEqual(bench.thermal_governor_action(90.0, frozen=False, freeze_temp_c=90.0, resume_temp_c=90.0), "freeze")
        self.assertEqual(bench.thermal_governor_action(90.0, frozen=True, freeze_temp_c=90.0, resume_temp_c=90.0), "resume")


class WaitForCooldownTests(unittest.TestCase):
    def test_already_cool_returns_immediately(self):
        sleeps = []
        result = bench.wait_for_cooldown(
            target_c=70.0, timeout_s=100.0,
            read_temp_fn=lambda: 60.0,
            sleep_fn=lambda s: sleeps.append(s),
            clock_fn=lambda: 0.0,
        )
        self.assertEqual(result, 60.0)
        self.assertEqual(sleeps, [])

    def test_polls_until_below_target(self):
        temps = iter([95.0, 85.0, 75.0, 65.0])
        sleeps = []
        result = bench.wait_for_cooldown(
            target_c=70.0, timeout_s=100.0, poll_interval_s=5.0,
            read_temp_fn=lambda: next(temps),
            sleep_fn=lambda s: sleeps.append(s),
            clock_fn=lambda: 0.0,
        )
        self.assertEqual(result, 65.0)
        self.assertEqual(sleeps, [5.0, 5.0, 5.0])

    def test_gives_up_at_timeout_still_hot(self):
        clock = {"t": 0.0}

        def clock_fn():
            return clock["t"]

        def sleep_fn(s):
            clock["t"] += s

        result = bench.wait_for_cooldown(
            target_c=70.0, timeout_s=12.0, poll_interval_s=5.0,
            read_temp_fn=lambda: 90.0,  # никогда не остывает
            sleep_fn=sleep_fn,
            clock_fn=clock_fn,
        )
        self.assertEqual(result, 90.0)


class BuildMatrixTests(unittest.TestCase):
    def test_full_cross_product(self):
        models = [
            bench.ModelConfig("small", "q8_0", "/m/small-q8_0.bin"),
            bench.ModelConfig("small", "q5_1", "/m/small-q5_1.bin"),
        ]
        matrix = bench.build_matrix(models, thread_options=(4, 8), flash_attn_options=(True, False))
        self.assertEqual(len(matrix), 2 * 2 * 2)
        first = matrix[0]
        self.assertEqual(first.model_name, "small")
        self.assertIn(first.threads, (4, 8))

    def test_taskset_cores_pinned_to_big_cores(self):
        models = [bench.ModelConfig("small", "q8_0", "/m/small-q8_0.bin")]
        matrix = bench.build_matrix(models, thread_options=(4,), flash_attn_options=(True,))
        self.assertEqual(matrix[0].taskset_cores, "4-7")


if __name__ == "__main__":
    unittest.main()
