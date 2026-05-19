"""
تست‌های واحد برای persian_readability.py
اجرا: python -m pytest test_persian_readability.py -v
"""
from __future__ import annotations
import math
import unittest

from persian_readability import (
    ReadabilityResult,
    _ASYL_CALIBRATION_POETRY, _ASYL_CALIBRATION_PROSE,
    _TAG_FUTURE_AUX, _TAG_NON_VERB_KHAH,
    _annotate_khah_tokens,
    _classify_khah,
    _count_en_syllables,
    _count_fa_syllables_base,
    _detect_likely_poetry,
    _is_suffix_compound_khah,
    _is_word_token,
    _split_into_sentences,
    _strip_punctuation,
    _tag_sentence_heuristic,
    compute_flesch_dayani,
    count_letters,
    count_syllables,
    interpret_score,
)

# ── متون مرجع ────────────────────────────────────────────────────────────────
_SHORT_FA = "کتاب خوب است."

_LONG_FA = (
    "زبان فارسی یکی از زبان‌های باستانی و پرغنای جهان است که تاریخی چند هزار ساله دارد. "
    "این زبان در ایران، افغانستان و تاجیکستان به عنوان زبان رسمی مورد استفاده قرار می‌گیرد. "
    "ادبیات فارسی شامل آثار برجسته‌ای از شاعران و نویسندگان بزرگی چون فردوسی، حافظ و سعدی است. "
    "دستور زبان فارسی نسبتاً ساده است و فاقد جنسیت دستوری می‌باشد. "
    "واژگان فارسی از ریشه‌های ایرانی باستان، عربی و در دوران معاصر از زبان‌های اروپایی وام گرفته شده است."
)

_SHAHNAMEH = (
    "چنین گفت کاندر نهان این سخُن\n"
    "پژوهیم تا خود چه آید به بُن\n"
    "ز پَهلو همه موبدان را بخواند\n"
    "ز سودابه چندی سخن‌ها براند\n"
    "چنین گفت موبد به شاه جهان\n"
    "که درد سپهبد نمانَد نهان\n"
    "چو خواهی که پیدا کنی گفتگوی\n"
    "بباید زدنْ سنگ را بر سبوی\n"
    "که هر چند فرزند، هست ارجمند\n"
    "دل شاه از اندیشه یابد گزند\n"
    "وزین دختر شاه هاماوران\n"
    "پُر اندیشه گشتی به دیگر کران\n"
    "ز هر در سخن چون بدین گونه گشت\n"
    "بر آتش یکی را بباید گذشت\n"
    "چنین است سوگند چرخ بلند\n"
    "که بر بی‌گناهان نیاید گزند"
)

_MIXED = "این یک test ساده است که هم Persian دارد هم English."


# ═══════════════════════════════════════════════════════════════════════════════
# ۱. _strip_punctuation
# ═══════════════════════════════════════════════════════════════════════════════
class TestStripPunctuation(unittest.TestCase):

    def test_persian_guillemets(self):
        self.assertEqual(_strip_punctuation("«مثال»"), "مثال")

    def test_parentheses(self):
        self.assertEqual(_strip_punctuation("(hello)"), "hello")

    def test_ellipsis(self):
        self.assertEqual(_strip_punctuation("...کلمه..."), "کلمه")

    def test_pure_punct_empty(self):
        self.assertEqual(_strip_punctuation("..."), "")
        self.assertEqual(_strip_punctuation("،؛؟"), "")

    def test_clean_word_unchanged(self):
        self.assertEqual(_strip_punctuation("فارسی"), "فارسی")

    def test_empty(self):
        self.assertEqual(_strip_punctuation(""), "")


# ═══════════════════════════════════════════════════════════════════════════════
# ۲. _is_word_token
# ═══════════════════════════════════════════════════════════════════════════════
class TestIsWordToken(unittest.TestCase):

    def test_persian(self):
        self.assertTrue(_is_word_token("کتاب"))

    def test_english(self):
        self.assertTrue(_is_word_token("hello"))

    def test_pure_punct_rejected(self):
        self.assertFalse(_is_word_token("..."))
        self.assertFalse(_is_word_token("،؛؟"))

    def test_number_rejected(self):
        self.assertFalse(_is_word_token("1234"))


# ═══════════════════════════════════════════════════════════════════════════════
# ۳. هجاشماری
# ═══════════════════════════════════════════════════════════════════════════════
class TestSyllableCounting(unittest.TestCase):

    def test_en_single(self):
        self.assertEqual(_count_en_syllables("cat"), 1)

    def test_en_two(self):
        self.assertEqual(_count_en_syllables("hello"), 2)
        self.assertEqual(_count_en_syllables("water"), 2)

    def test_en_silent_e(self):
        self.assertEqual(_count_en_syllables("make"), 1)

    def test_en_min_one(self):
        self.assertEqual(_count_en_syllables("gym"), 1)

    def test_fa_long_vowels(self):
        # کتابی: ا+ی = 2 واکه نوشته
        self.assertEqual(_count_fa_syllables_base("کتابی"), 2)

    def test_fa_min_one(self):
        self.assertGreaterEqual(_count_fa_syllables_base("دل"), 1)

    def test_fa_alef_counts(self):
        # آب: ا → 1
        self.assertEqual(_count_fa_syllables_base("آب"), 1)

    def test_dispatcher_persian(self):
        self.assertGreaterEqual(count_syllables("کتاب"), 1)

    def test_dispatcher_english(self):
        self.assertEqual(count_syllables("hello"), 2)

    def test_min_one_any_word(self):
        self.assertGreaterEqual(count_syllables("بل"), 1)


# ═══════════════════════════════════════════════════════════════════════════════
# ۴. count_letters
# ═══════════════════════════════════════════════════════════════════════════════
class TestCountLetters(unittest.TestCase):

    def test_pure_words(self):
        self.assertEqual(count_letters(["کتاب", "خوب"]), 7)

    def test_excludes_numbers(self):
        self.assertEqual(count_letters(["a1b"]), 2)

    def test_empty(self):
        self.assertEqual(count_letters([]), 0)

    def test_mixed(self):
        self.assertEqual(count_letters(["سلام", "world"]), 9)


# ═══════════════════════════════════════════════════════════════════════════════
# ۵. _tag_sentence_heuristic
# ═══════════════════════════════════════════════════════════════════════════════
class TestTagSentenceHeuristic(unittest.TestCase):

    def test_pure_punct_excluded(self):
        self.assertEqual(_tag_sentence_heuristic("... ،،، !!!"), [])

    def test_words_preserved(self):
        words = [w for w, _ in _tag_sentence_heuristic("کتاب خوب است")]
        self.assertIn("کتاب", words)

    def test_surrounded_punct_stripped(self):
        words = [w for w, _ in _tag_sentence_heuristic("«کتاب» خوب است.")]
        self.assertIn("کتاب", words)
        for w in words:
            self.assertNotIn("«", w)

    def test_tags_none(self):
        for _, tag in _tag_sentence_heuristic("کتاب خوب"):
            self.assertIsNone(tag)


# ═══════════════════════════════════════════════════════════════════════════════
# ۶. _split_into_sentences
# ═══════════════════════════════════════════════════════════════════════════════
class TestSplitIntoSentences(unittest.TestCase):

    def test_poetry_each_line_sentence(self):
        from persian_readability import _get_normalizer
        text = _get_normalizer().normalize(
            "چنین گفت کاندر نهان این سخن\n"
            "پژوهیم تا خود چه آید به بن\n"
            "ز پهلو همه موبدان را بخواند"
        )
        sents = _split_into_sentences(text)
        self.assertEqual(len(sents), 3)

    def test_empty_lines_skipped(self):
        from persian_readability import _get_normalizer
        text = _get_normalizer().normalize("کتاب\n\n\nخوب است")
        sents = _split_into_sentences(text)
        self.assertGreater(len(sents), 0)
        for s in sents:
            self.assertTrue(any(ch.isalpha() for ch in s))

    def test_shahnameh_16_sentences(self):
        r = compute_flesch_dayani(_SHAHNAMEH)
        self.assertEqual(r.sentences, 16)
        self.assertLess(r.asl, 15)


# ═══════════════════════════════════════════════════════════════════════════════
# ۷. _detect_likely_poetry
# ═══════════════════════════════════════════════════════════════════════════════
class TestDetectPoetry(unittest.TestCase):

    def test_shahnameh_is_poetry(self):
        r = compute_flesch_dayani(_SHAHNAMEH)
        self.assertTrue(r.is_likely_poetry)

    def test_long_prose_not_poetry(self):
        r = compute_flesch_dayani(_LONG_FA)
        self.assertFalse(r.is_likely_poetry)

    def test_poetry_warning_logged(self):
        with self.assertLogs("persian_readability", level="WARNING") as cm:
            r = compute_flesch_dayani(_SHAHNAMEH)
        if r.is_likely_poetry:
            self.assertTrue(any("verse" in m.lower() or "poetry" in m.lower() for m in cm.output))


# ═══════════════════════════════════════════════════════════════════════════════
# ۸. Classifier خواه — ۱۱ تست از سند
# ═══════════════════════════════════════════════════════════════════════════════
class TestClassifyKhah(unittest.TestCase):

    def _cls(self, tokens, target):
        return _classify_khah(tokens, tokens.index(target))

    def test_future_khaham_raft(self):
        self.assertEqual(self._cls(["من", "خواهم", "رفت"], "خواهم"), "FUTURE_AUX")

    def test_future_nakhahand(self):
        self.assertEqual(self._cls(["آنها", "نخواهند", "پذیرفت"], "نخواهند"), "FUTURE_AUX")

    def test_future_intervening_word(self):
        self.assertEqual(self._cls(["او", "خواهد", "هم", "رفت"], "خواهد"), "FUTURE_AUX")

    def test_lexical_no_next(self):
        self.assertEqual(self._cls(["او", "این", "را", "خواهد"], "خواهد"), "LEXICAL_KHASTAN")

    def test_lexical_followed_by_ke(self):
        self.assertEqual(self._cls(["او", "خواهد", "که", "برود"], "خواهد"), "LEXICAL_KHASTAN")

    def test_particle_standalone(self):
        tokens = ["خواه", "بیاید", "خواه", "نیاید"]
        self.assertEqual(_classify_khah(tokens, 0), "PARTICLE_KHAH")
        self.assertEqual(_classify_khah(tokens, 2), "PARTICLE_KHAH")

    def test_particle_nakhah(self):
        self.assertEqual(self._cls(["خواه\u200cناخواه", "باید"], "خواه\u200cناخواه"), "PARTICLE_KHAH")

    def test_nominal_khahesh(self):
        self.assertEqual(self._cls(["یک", "خواهش", "کوچک"], "خواهش"), "NOMINAL_DERIVATIVE")

    def test_nominal_khahan(self):
        self.assertEqual(self._cls(["او", "خواهان", "تغییر"], "خواهان"), "NOMINAL_DERIVATIVE")

    def test_independent_khahar(self):
        self.assertEqual(self._cls(["خواهر", "من", "پزشک"], "خواهر"), "INDEPENDENT_WORD")

    def test_suffix_azadi_khah(self):
        self.assertEqual(self._cls(["فعال", "آزادی\u200cخواه"], "آزادی\u200cخواه"), "SUFFIX_COMPOUND")


class TestAnnotateKhahTokens(unittest.TestCase):

    def test_future_gets_tag(self):
        tagged = [("من", None), ("خواهم", "V_FUT"), ("رفت", "V_PA")]
        result = _annotate_khah_tokens(tagged)
        self.assertEqual(result[1][1], _TAG_FUTURE_AUX)

    def test_non_verb_gets_tag(self):
        for tok in ("خواهش", "خواهر", "خواهان"):
            result = _annotate_khah_tokens([(tok, None)])
            self.assertEqual(result[0][1], _TAG_NON_VERB_KHAH, msg=tok)

    def test_suffix_compound_gets_non_verb(self):
        result = _annotate_khah_tokens([("خیرخواه", None)])
        self.assertEqual(result[0][1], _TAG_NON_VERB_KHAH)

    def test_lexical_keeps_original_tag(self):
        tagged = [("او", None), ("خواهد", "V_PRS"), ("که", None)]
        result = _annotate_khah_tokens(tagged)
        self.assertEqual(result[1][1], "V_PRS")

    def test_no_khah_unchanged(self):
        tagged = [("کتاب", "N_SING"), ("خوب", "ADJ")]
        self.assertEqual(_annotate_khah_tokens(tagged), tagged)


# ═══════════════════════════════════════════════════════════════════════════════
# ۹. interpret_score
# ═══════════════════════════════════════════════════════════════════════════════
class TestInterpretScore(unittest.TestCase):

    def test_very_easy(self):
        self.assertIn("بسیار آسان", interpret_score(95))
        self.assertIn("بسیار آسان", interpret_score(90))

    def test_medium(self):
        self.assertIn("متوسط", interpret_score(65))

    def test_boundary_60(self):
        self.assertIn("متوسط", interpret_score(60.0))
        self.assertIn("نسبتاً دشوار", interpret_score(59.9))

    def test_very_hard(self):
        self.assertIn("بسیار دشوار", interpret_score(10))

    def test_negative(self):
        result = interpret_score(-10)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)


# ═══════════════════════════════════════════════════════════════════════════════
# ۱۰. compute_flesch_dayani — صحت فرمول و edge cases
# ═══════════════════════════════════════════════════════════════════════════════
class TestComputeFleschDayani(unittest.TestCase):

    # ── ورودی نادرست ────────────────────────────────────────────────────────
    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            compute_flesch_dayani("")

    def test_whitespace_raises(self):
        with self.assertRaises(ValueError):
            compute_flesch_dayani("   \n\t  ")

    def test_punctuation_only_raises(self):
        with self.assertRaises(ValueError):
            compute_flesch_dayani("... ،،، !!! ؟؟؟")

    def test_numbers_only_raises(self):
        with self.assertRaises(ValueError):
            compute_flesch_dayani("1234 5678 9012")

    # ── ساختار نتیجه ────────────────────────────────────────────────────────
    def test_all_fields_present(self):
        r = compute_flesch_dayani(_LONG_FA)
        for f in ("sentences","words","letters","syllables",
                  "asl","wl","asyl","flesch_dayani","level","pos_mode"):
            self.assertIsNotNone(getattr(r, f), msg=f)

    def test_pos_enhanced_property(self):
        r = compute_flesch_dayani(_LONG_FA)
        self.assertIsInstance(r.pos_enhanced, bool)

    def test_is_likely_poetry_field(self):
        r = compute_flesch_dayani(_LONG_FA)
        self.assertIsInstance(r.is_likely_poetry, bool)

    def test_score_finite(self):
        r = compute_flesch_dayani(_LONG_FA)
        self.assertFalse(math.isnan(r.flesch_dayani))
        self.assertFalse(math.isinf(r.flesch_dayani))

    # ── صحت فرمول ───────────────────────────────────────────────────────────
    def test_asl_formula(self):
        r = compute_flesch_dayani(_LONG_FA)
        self.assertAlmostEqual(r.asl, r.words / r.sentences, places=5)

    def test_wl_formula(self):
        r = compute_flesch_dayani(_LONG_FA)
        self.assertAlmostEqual(r.wl, r.letters / r.words, places=5)

    def test_fdr_formula_prose(self):
        r = compute_flesch_dayani(_LONG_FA)
        expected = 262.835 - 84.6 * (r.asyl * _ASYL_CALIBRATION_PROSE) - 1.015 * r.asl
        self.assertAlmostEqual(r.flesch_dayani, expected, places=4)

    def test_fdr_formula_poetry(self):
        r = compute_flesch_dayani(_SHAHNAMEH)
        self.assertTrue(r.is_likely_poetry)
        expected = 262.835 - 84.6 * (r.asyl * _ASYL_CALIBRATION_POETRY) - 1.015 * r.asl
        self.assertAlmostEqual(r.flesch_dayani, expected, places=4)

    # ── کالیبراسیون معنادار ─────────────────────────────────────────────────
    def test_shahnameh_score_in_hard_range(self):
        r = compute_flesch_dayani(_SHAHNAMEH)
        # شاهنامه باید «دشوار» یا «نسبتاً دشوار» باشد — نه «بسیار آسان»
        self.assertLess(r.flesch_dayani, 70)
        self.assertGreater(r.flesch_dayani, -50)

    def test_child_story_short_text_returns_valid_result(self):
        text = 'گربه روی دیوار نشست. سگ پارس کرد. گربه پرید. سگ دوید. هر دو خوشحال شدند.'
        r = compute_flesch_dayani(text)

        self.assertIsInstance(r.flesch_dayani, float)
        self.assertGreater(r.words, 0)
        self.assertGreater(r.sentences, 0)
        self.assertIsInstance(r.level, str)

    def test_scientific_text_score_hard(self):
        text = ('سنتز پروتئین‌های ریبوزومی از طریق فرآیندهای پیچیده ترجمه‌ی mRNA صورت می‌پذیرد. '
                'تنظیم این فرآیندها در سطح اپی‌ژنتیک و پس از رونویسی از اهمیت بالینی برخوردار است. '
                'بررسی مکانیسم‌های مولکولی دخیل در تنظیم بیان ژن‌های ریبوزومی می‌تواند مسیرهای '
                'درمانی جدیدی در بیماری‌های متابولیک ایجاد کند.')
        r = compute_flesch_dayani(text)
        self.assertLess(r.flesch_dayani, 30)

    # ── فیلتر نشانه‌گذاری ───────────────────────────────────────────────────
    def test_punct_not_inflate_words(self):
        clean  = compute_flesch_dayani("کتاب خوب است و دانش مهم است.")
        punchy = compute_flesch_dayani("«کتاب» خوب است! و دانش، مهم است...")
        self.assertEqual(clean.words, punchy.words)

    def test_punct_not_inflate_syllables(self):
        clean  = compute_flesch_dayani("کتاب خوب است و دانش مهم است.")
        punchy = compute_flesch_dayani("«کتاب» خوب است! و دانش، مهم است...")
        self.assertEqual(clean.syllables, punchy.syllables)

    # ── متن کوتاه ───────────────────────────────────────────────────────────
    def test_short_warns(self):
        with self.assertLogs("persian_readability", level="WARNING") as cm:
            compute_flesch_dayani(_SHORT_FA)
        self.assertTrue(any("word" in m.lower() for m in cm.output))

    def test_mixed_text(self):
        r = compute_flesch_dayani(_MIXED)
        self.assertGreater(r.words, 0)
        self.assertFalse(math.isnan(r.flesch_dayani))




# ═══════════════════════════════════════════════════════════════════════════════
# ۱۱. _split_hemistich_line و فرمت دو مصراع در یک خط
# ═══════════════════════════════════════════════════════════════════════════════
class TestSplitHemistichLine(unittest.TestCase):

    def test_double_hemistich_splits_correctly(self):
        from persian_readability import _split_hemistich_line
        line = "سواری و تیر و کمان و کمند    عنان و رکیب و چه و چون و چند"
        parts = _split_hemistich_line(line)
        self.assertEqual(len(parts), 2)
        self.assertEqual(parts[0].strip(), "سواری و تیر و کمان و کمند")
        self.assertEqual(parts[1].strip(), "عنان و رکیب و چه و چون و چند")

    def test_single_line_unchanged(self):
        from persian_readability import _split_hemistich_line
        line = "چنین گفت کاندر نهان این سخن"
        self.assertEqual(_split_hemistich_line(line), [line])

    def test_prose_line_unchanged(self):
        from persian_readability import _split_hemistich_line
        # نثر با یک فاصله: تغییر نمی‌کند
        line = "زبان فارسی یکی از زبان‌های قدیمی است"
        self.assertEqual(_split_hemistich_line(line), [line])

    def test_both_formats_same_score(self):
        # مهم‌ترین تست: هر دو فرمت باید نتیجه یکسان دهند
        text_separate = (
            "سواری و تیر و کمان و کمند\n"
            "عنان و رکیب و چه و چون و چند\n"
            "نشستنگه و مجلس و میگسار\n"
            "همان باز و شامین و یوز و شکار"
        )
        text_paired = (
            "سواری و تیر و کمان و کمند    عنان و رکیب و چه و چون و چند\n"
            "نشستنگه و مجلس و میگسار    همان باز و شامین و یوز و شکار"
        )
        r1 = compute_flesch_dayani(text_separate)
        r2 = compute_flesch_dayani(text_paired)
        self.assertEqual(r1.sentences, r2.sentences)
        self.assertAlmostEqual(r1.flesch_dayani, r2.flesch_dayani, places=1)
        self.assertEqual(r1.is_likely_poetry, r2.is_likely_poetry)

    def test_paired_format_detected_as_poetry(self):
        text = (
            "سواری و تیر و کمان و کمند    عنان و رکیب و چه و چون و چند\n"
            "نشستنگه و مجلس و میگسار    همان باز و شامین و یوز و شکار"
        )
        r = compute_flesch_dayani(text)
        self.assertTrue(r.is_likely_poetry)

    def test_paired_format_not_very_easy(self):
        text = (
            "سواری و تیر و کمان و کمند    عنان و رکیب و چه و چون و چند\n"
            "نشستنگه و مجلس و میگسار    همان باز و شامین و یوز و شکار"
        )
        r = compute_flesch_dayani(text)
        # نباید «بسیار آسان» باشد
        self.assertNotIn("بسیار آسان", r.level)

if __name__ == "__main__":
    unittest.main(verbosity=2)
