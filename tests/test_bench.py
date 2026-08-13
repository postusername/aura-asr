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
