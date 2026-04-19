"""
تست‌های واحد برای persian_readability.py
اجرا: python -m pytest test_persian_readability.py -v
"""
from __future__ import annotations
import unittest

from persian_readability import (
    ReadabilityResult,
    _TAG_FUTURE_AUX, _TAG_NON_VERB_KHAH,
    _annotate_khah_tokens,
    _classify_khah,
    _count_en_syllables,
    _count_fa_syllables_base,
    _is_suffix_compound_khah,
    _is_word_token,
    _strip_punctuation,
    _tag_sentence_heuristic,
    compute_flesch_dayani,
    count_letters,
    count_syllables,
    interpret_score,
)

_SHORT_FA = "کتاب خوب است."
_LONG_FA = (
    "زبان فارسی یکی از زبان‌های باستانی و پرغنای جهان است که تاریخی چند هزار ساله دارد. "
    "این زبان در ایران، افغانستان و تاجیکستان به عنوان زبان رسمی مورد استفاده قرار می‌گیرد. "
    "ادبیات فارسی شامل آثار برجسته‌ای از شاعران و نویسندگان بزرگی چون فردوسی، حافظ و سعدی است. "
    "دستور زبان فارسی نسبتاً ساده است و فاقد جنسیت دستوری می‌باشد. "
    "واژگان فارسی از ریشه‌های ایرانی باستان، عربی و در دوران معاصر از زبان‌های اروپایی وام گرفته شده است."
)
_MIXED = "این یک test ساده است که هم Persian دارد هم English."


# ═══════════════════════════════════════════════════════════════════════════════
# ۱. _is_suffix_compound_khah
# ═══════════════════════════════════════════════════════════════════════════════
class TestIsSuffixCompoundKhah(unittest.TestCase):

    def test_azadikkhah(self):
        self.assertTrue(_is_suffix_compound_khah("آزادی\u200cخواه"))

    def test_kheirkhah(self):
        self.assertTrue(_is_suffix_compound_khah("خیرخواه"))

    def test_dadkhah(self):
        self.assertTrue(_is_suffix_compound_khah("دادخواه"))

    def test_khودkhah(self):
        self.assertTrue(_is_suffix_compound_khah("خودخواه"))

    def test_khah_alone_not_compound(self):
        self.assertFalse(_is_suffix_compound_khah("خواه"))

    def test_future_forms_not_compound(self):
        for tok in ("خواهم", "خواهد", "نخواهند"):
            self.assertFalse(_is_suffix_compound_khah(tok), msg=tok)

    def test_khahar_not_compound(self):
        self.assertFalse(_is_suffix_compound_khah("خواهر"))


# ═══════════════════════════════════════════════════════════════════════════════
# ۲. _classify_khah — ۹ تست از سند
# ═══════════════════════════════════════════════════════════════════════════════
class TestClassifyKhah(unittest.TestCase):
    """
    تست‌های مستقیم از سند، بر اساس توکن‌های دستی.
    از word_tokenize استفاده نمی‌کنیم تا وابستگی به hazm نداشته باشیم.
    """

    def _cls(self, tokens: list[str], target: str) -> str:
        i = tokens.index(target)
        return _classify_khah(tokens, i)

    # ─── FUTURE_AUX ───────────────────────────────────────────────────────────
    def test_future_aux_khaham_raft(self):
        # من فردا خواهم رفت
        self.assertEqual(
            self._cls(["من", "فردا", "خواهم", "رفت"], "خواهم"),
            "FUTURE_AUX",
        )

    def test_future_aux_nakhahand_pazirooft(self):
        # آنها نخواهند پذیرفت
        self.assertEqual(
            self._cls(["آنها", "نخواهند", "پذیرفت"], "نخواهند"),
            "FUTURE_AUX",
        )

    def test_future_aux_with_intervening_word(self):
        # او خواهد هم رفت — next2 در mini-lexicon
        self.assertEqual(
            self._cls(["او", "خواهد", "هم", "رفت"], "خواهد"),
            "FUTURE_AUX",
        )

    # ─── LEXICAL_KHASTAN ──────────────────────────────────────────────────────
    def test_lexical_no_next(self):
        # او این را خواهد (بدون ادامه)
        self.assertEqual(
            self._cls(["او", "این", "را", "خواهد"], "خواهد"),
            "LEXICAL_KHASTAN",
        )

    def test_lexical_followed_by_ke(self):
        # او خواهد که برود
        self.assertEqual(
            self._cls(["او", "خواهد", "که", "برود"], "خواهد"),
            "LEXICAL_KHASTAN",
        )

    # ─── PARTICLE_KHAH ────────────────────────────────────────────────────────
    def test_particle_khah_standalone(self):
        # خواه بیاید خواه نیاید
        tokens = ["خواه", "بیاید", "خواه", "نیاید"]
        self.assertEqual(_classify_khah(tokens, 0), "PARTICLE_KHAH")
        self.assertEqual(_classify_khah(tokens, 2), "PARTICLE_KHAH")

    def test_particle_khah_naKhah(self):
        self.assertEqual(
            self._cls(["خواه\u200cناخواه", "باید", "برویم"], "خواه\u200cناخواه"),
            "PARTICLE_KHAH",
        )

    # ─── NOMINAL_DERIVATIVE ───────────────────────────────────────────────────
    def test_nominal_khahesh(self):
        self.assertEqual(
            self._cls(["یک", "خواهش", "کوچک"], "خواهش"),
            "NOMINAL_DERIVATIVE",
        )

    def test_nominal_khahan(self):
        self.assertEqual(
            self._cls(["او", "خواهان", "تغییر", "است"], "خواهان"),
            "NOMINAL_DERIVATIVE",
        )

    # ─── INDEPENDENT_WORD ────────────────────────────────────────────────────
    def test_independent_khahar(self):
        self.assertEqual(
            self._cls(["خواهر", "من", "پزشک", "است"], "خواهر"),
            "INDEPENDENT_WORD",
        )

    # ─── SUFFIX_COMPOUND ─────────────────────────────────────────────────────
    def test_suffix_azadikkhah(self):
        self.assertEqual(
            self._cls(["یک", "فعال", "آزادی\u200cخواه", "است"], "آزادی\u200cخواه"),
            "SUFFIX_COMPOUND",
        )

    def test_suffix_kheirkhah(self):
        self.assertEqual(
            self._cls(["انسان", "خیرخواه"], "خیرخواه"),
            "SUFFIX_COMPOUND",
        )


# ═══════════════════════════════════════════════════════════════════════════════
# ۳. _annotate_khah_tokens
# ═══════════════════════════════════════════════════════════════════════════════
class TestAnnotateKhahTokens(unittest.TestCase):

    def test_future_aux_gets_synthetic_tag(self):
        tagged = [("من", None), ("خواهم", "V_FUT"), ("رفت", "V_PA")]
        result = _annotate_khah_tokens(tagged)
        self.assertEqual(result[1][1], _TAG_FUTURE_AUX)

    def test_non_verb_gets_synthetic_tag(self):
        for tok in ("خواهش", "خواهر", "خواهان"):
            tagged = [(tok, None)]
            result = _annotate_khah_tokens(tagged)
            self.assertEqual(result[0][1], _TAG_NON_VERB_KHAH, msg=tok)

    def test_suffix_compound_gets_non_verb_tag(self):
        tagged = [("خیرخواه", None)]
        result = _annotate_khah_tokens(tagged)
        self.assertEqual(result[0][1], _TAG_NON_VERB_KHAH)

    def test_lexical_khastan_keeps_original_tag(self):
        tagged = [("او", None), ("خواهد", "V_PRS"), ("که", None)]
        result = _annotate_khah_tokens(tagged)
        # LEXICAL_KHASTAN → tag اصلی دست نمی‌خورد
        self.assertEqual(result[1][1], "V_PRS")

    def test_non_khah_tokens_unchanged(self):
        tagged = [("کتاب", "N_SING"), ("خوب", "ADJ"), ("است", "V_PRS")]
        result = _annotate_khah_tokens(tagged)
        self.assertEqual(result, tagged)

    def test_no_khah_in_text(self):
        tagged = [("زبان", None), ("فارسی", None)]
        result = _annotate_khah_tokens(tagged)
        self.assertEqual(result, tagged)

    def test_future_aux_syllable_not_inflated(self):
        # خواهم به‌عنوان FUTURE_AUX نباید هجای اضافه از قانون می/نمی بگیرد
        tagged_before = [("خواهم", "V_FUT"), ("رفت", "V_PA")]
        annotated = _annotate_khah_tokens(tagged_before)
        syl_future = count_syllables("خواهم", _TAG_FUTURE_AUX)
        syl_base   = count_syllables("خواهم")
        # هجای base و FUTURE_AUX باید یکسان باشند
        self.assertEqual(syl_future, syl_base)

    def test_non_verb_khah_syllable_same_as_base(self):
        syl_non_verb = count_syllables("خواهش", _TAG_NON_VERB_KHAH)
        syl_base     = count_syllables("خواهش")
        self.assertEqual(syl_non_verb, syl_base)


# ═══════════════════════════════════════════════════════════════════════════════
# ۴. _strip_punctuation
# ═══════════════════════════════════════════════════════════════════════════════
class TestStripPunctuation(unittest.TestCase):

    def test_removes_persian_guillemets(self):
        self.assertEqual(_strip_punctuation("«مثال»"), "مثال")

    def test_removes_parentheses(self):
        self.assertEqual(_strip_punctuation("(hello)"), "hello")

    def test_removes_ellipsis(self):
        self.assertEqual(_strip_punctuation("...کلمه..."), "کلمه")

    def test_pure_punctuation_returns_empty(self):
        self.assertEqual(_strip_punctuation("..."), "")
        self.assertEqual(_strip_punctuation("،؛؟"), "")

    def test_no_change_for_clean_word(self):
        self.assertEqual(_strip_punctuation("فارسی"), "فارسی")

    def test_empty_string(self):
        self.assertEqual(_strip_punctuation(""), "")


# ═══════════════════════════════════════════════════════════════════════════════
# ۵. _is_word_token
# ═══════════════════════════════════════════════════════════════════════════════
class TestIsWordToken(unittest.TestCase):

    def test_persian_word(self):
        self.assertTrue(_is_word_token("کتاب"))

    def test_english_word(self):
        self.assertTrue(_is_word_token("hello"))

    def test_pure_punctuation_rejected(self):
        self.assertFalse(_is_word_token("..."))
        self.assertFalse(_is_word_token("،؛؟"))
        self.assertFalse(_is_word_token("«»"))

    def test_number_only_rejected(self):
        self.assertFalse(_is_word_token("1234"))


# ═══════════════════════════════════════════════════════════════════════════════
# ۶. هجاشماری
# ═══════════════════════════════════════════════════════════════════════════════
class TestSyllableCounting(unittest.TestCase):

    def test_en_single(self):
        self.assertEqual(_count_en_syllables("cat"), 1)

    def test_en_two(self):
        self.assertEqual(_count_en_syllables("hello"), 2)
        self.assertEqual(_count_en_syllables("water"), 2)

    def test_en_silent_e(self):
        self.assertEqual(_count_en_syllables("make"), 1)
        self.assertEqual(_count_en_syllables("simple"), 1)

    def test_en_min_one(self):
        self.assertEqual(_count_en_syllables("gym"), 1)

    def test_fa_min_one(self):
        self.assertGreaterEqual(_count_fa_syllables_base("دل"), 1)

    def test_fa_written_vowels(self):
        self.assertEqual(_count_fa_syllables_base("کتابی"), 2)

    def test_fa_known_undercount(self):
        # محدودیت heuristic: کتاب → 1 (واقعی: 2)
        self.assertEqual(_count_fa_syllables_base("کتاب"), 1)

    def test_fa_daneshgah(self):
        # heuristic: دو «ا» → 2 (واقعی: 3)
        self.assertEqual(_count_fa_syllables_base("دانشگاه"), 2)

    def test_dispatcher_persian(self):
        self.assertGreaterEqual(count_syllables("کتاب"), 1)

    def test_dispatcher_english(self):
        self.assertEqual(count_syllables("hello"), 2)


# ═══════════════════════════════════════════════════════════════════════════════
# ۷. count_letters
# ═══════════════════════════════════════════════════════════════════════════════
class TestCountLetters(unittest.TestCase):

    def test_pure_words(self):
        self.assertEqual(count_letters(["کتاب", "خوب"]), 7)

    def test_excludes_numbers(self):
        self.assertEqual(count_letters(["a1b"]), 2)

    def test_empty(self):
        self.assertEqual(count_letters([]), 0)

    def test_mixed_scripts(self):
        self.assertEqual(count_letters(["سلام", "world"]), 9)


# ═══════════════════════════════════════════════════════════════════════════════
# ۸. _tag_sentence_heuristic
# ═══════════════════════════════════════════════════════════════════════════════
class TestTagSentenceHeuristic(unittest.TestCase):

    def test_pure_punctuation_excluded(self):
        self.assertEqual(_tag_sentence_heuristic("... ،،، !!!"), [])

    def test_words_preserved(self):
        words = [w for w, _ in _tag_sentence_heuristic("کتاب خوب است")]
        self.assertIn("کتاب", words)
        self.assertIn("خوب", words)

    def test_surrounded_punct_stripped(self):
        words = [w for w, _ in _tag_sentence_heuristic("«کتاب» خوب است.")]
        self.assertIn("کتاب", words)
        for w in words:
            self.assertNotIn("«", w)

    def test_all_tags_none(self):
        for _, tag in _tag_sentence_heuristic("کتاب خوب"):
            self.assertIsNone(tag)


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

    def test_negative(self):
        result = interpret_score(-10)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)


# ═══════════════════════════════════════════════════════════════════════════════
# ۱۰. compute_flesch_dayani
# ═══════════════════════════════════════════════════════════════════════════════
class TestComputeFleschDayani(unittest.TestCase):

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

    def test_short_text_returns_result(self):
        self.assertIsInstance(compute_flesch_dayani(_SHORT_FA), ReadabilityResult)

    def test_short_text_warns(self):
        with self.assertLogs("persian_readability", level="WARNING") as cm:
            compute_flesch_dayani(_SHORT_FA)
        self.assertTrue(any("word" in m.lower() for m in cm.output))

    def test_long_text(self):
        r = compute_flesch_dayani(_LONG_FA)
        self.assertGreater(r.sentences, 0)
        self.assertGreater(r.words, 0)
        self.assertGreater(r.letters, 0)
        self.assertGreater(r.syllables, 0)

    def test_all_fields_present(self):
        r = compute_flesch_dayani(_LONG_FA)
        for f in ("sentences","words","letters","syllables",
                  "asl","wl","asyl","flesch_dayani","level","pos_mode"):
            self.assertIsNotNone(getattr(r, f), msg=f)

    def test_pos_enhanced_property(self):
        r = compute_flesch_dayani(_LONG_FA)
        self.assertIsInstance(r.pos_enhanced, bool)

    def test_asl_formula(self):
        r = compute_flesch_dayani(_LONG_FA)
        self.assertAlmostEqual(r.asl, r.words / r.sentences, places=5)

    def test_wl_formula(self):
        r = compute_flesch_dayani(_LONG_FA)
        self.assertAlmostEqual(r.wl, r.letters / r.words, places=5)

    def test_fdr_formula(self):
        r = compute_flesch_dayani(_LONG_FA)
        self.assertAlmostEqual(r.flesch_dayani,
                               262.835 - 0.846 * r.asyl - 1.015 * r.asl,
                               places=5)

    def test_mixed_text(self):
        r = compute_flesch_dayani(_MIXED)
        self.assertGreater(r.words, 0)

    def test_punct_does_not_inflate_words(self):
        clean  = compute_flesch_dayani("کتاب خوب است و دانش مهم است.")
        punchy = compute_flesch_dayani("«کتاب» خوب است! و دانش، مهم است...")
        self.assertEqual(clean.words, punchy.words)

    def test_punct_does_not_inflate_syllables(self):
        clean  = compute_flesch_dayani("کتاب خوب است و دانش مهم است.")
        punchy = compute_flesch_dayani("«کتاب» خوب است! و دانش، مهم است...")
        self.assertEqual(clean.syllables, punchy.syllables)

    def test_khah_words_classified_correctly_in_pipeline(self):
        # خواهش و خواهر نباید syllable count را خراب کنند
        text_with_khah = (
            "خواهر من یک خواهش کوچک داشت. "
            "او خواهان آزادی بود و یک فعال آزادی‌خواه شناخته می‌شد. "
            "خواه برود خواه بماند، این تصمیم خودش بود."
        )
        r = compute_flesch_dayani(text_with_khah)
        self.assertIsInstance(r, ReadabilityResult)
        self.assertGreater(r.words, 0)

    def test_score_is_float(self):
        self.assertIsInstance(compute_flesch_dayani(_LONG_FA).flesch_dayani, float)


if __name__ == "__main__":
    unittest.main(verbosity=2)
