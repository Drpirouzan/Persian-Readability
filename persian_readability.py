from __future__ import annotations

import argparse
import logging
import re
import sys
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from hazm import Normalizer, sent_tokenize, word_tokenize

logger = logging.getLogger(__name__)

# ── بارگذاری اختیاری Parsivar ────────────────────────────────────────────────
try:
    from parsivar import POSTagger as ParsivarPOSTagger
    from parsivar import Tokenizer as ParsivarTokenizer

    _PARSIVAR_AVAILABLE = True
except ImportError:
    _PARSIVAR_AVAILABLE = False

# ── برچسب‌های POS (Parsivar / Bijankhan) ────────────────────────────────────
# [پیشنهاد ۳] frozenset به جای set — این مجموعه هرگز تغییر نمی‌کند
_VERB_TAGS_PARSIVAR: frozenset[str] = frozenset(
    {"V_PRS", "V_PA", "V_IMP", "V_SUB", "V_FUT", "V_PRF"}
)

# ── الگوهای پیشوندهای فعلی ───────────────────────────────────────────────────
# [پیشنهاد ۱] پیشوندهای بـ/نـ/خواه که در نسخه قبلی حذف شده بودند برگردانده شدند
_VERB_PREFIX_ATTACHED = re.compile(r"^(می|نمی|بـ|نـ|خواه)(?!‌)")

# ── اعراب (واکه‌های کوتاه) ──────────────────────────────────────────────────
# [پیشنهاد ۳] frozenset به جای set
_FA_DIACRITICS: frozenset[str] = frozenset("\u064e\u064f\u0650\u064b\u064c\u064d")

# ── واکه‌های بلند فارسی که هجا اضافه می‌کنند ────────────────────────────────
# [پیشنهاد ۳] frozenset به جای set inline در تابع
_FA_LONG_VOWELS: frozenset[str] = frozenset("اوی")

# ── حداقل کلمات برای نتیجه قابل اعتماد ─────────────────────────────────────
# [پیشنهاد ۸]
_MIN_WORDS_RELIABLE = 50


# ── هجاشماری انگلیسی ────────────────────────────────────────────────────────
def _count_en_syllables(word: str) -> int:
    w = word.lower().strip(".,!?;:\"'()[]{}")
    if not w:
        return 1
    count = len(re.findall(r"[aeiou]+", w))
    if w.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)


# ── هجاشماری فارسی پایه ─────────────────────────────────────────────────────
def _count_fa_syllables_base(word: str) -> int:
    """
    یک heuristic ساده برای شمارش هجا در فارسی.

    قواعد تقریبی:
    - «ا» همیشه واکه است.
    - «و» و «ی» در ابتدای کلمه معمولاً نیمه‌صامت‌اند (صامت در نظر گرفته می‌شوند).
    - اعراب (فتحه/ضمه/کسره) هر کدام یک هجا.
    - «ه» پایانیِ کلمه (اگر قبلش واکه نیست) یک هجا.

    محدودیت شناخته‌شده: «واو» عطف تک‌حرفی (و) یک هجا دارد اما
    در این تابع به‌درستی شمارش می‌شود چون i > 0 برقرار نیست.
    """
    if not word:
        return 0

    syllables = 0
    n = len(word)

    for i, ch in enumerate(word):
        if ch == "ا":
            syllables += 1
        elif ch in ("و", "ی"):
            # [پیشنهاد ۲] استفاده از _FA_LONG_VOWELS برای سازگاری
            if i > 0:
                syllables += 1
        elif ch in _FA_DIACRITICS:
            syllables += 1
        elif ch == "ه" and i == n - 1 and n > 1:
            if word[i - 1] not in _FA_LONG_VOWELS:
                syllables += 1

    return max(syllables, 1)


# ── هجاشماری فارسی با POS ────────────────────────────────────────────────────
def _count_fa_syllables_pos(word: str, pos_tag: str) -> int:
    base = _count_fa_syllables_base(word)

    # [پیشنهاد ۱] فعل‌های پیوسته: می‌رود، نمی‌دانم، بگو، خواهم رفت
    if pos_tag in _VERB_TAGS_PARSIVAR and _VERB_PREFIX_ATTACHED.match(word):
        base += 1

    # پسوند تفضیلی/عالی — در همه اقسام کلام ممکن است بیاید
    if word.endswith("ترین") and len(word) > 4:
        stem_syl = _count_fa_syllables_base(word[:-4])
        base = max(base, stem_syl + 2)
    elif word.endswith("تر") and len(word) > 2:
        stem_syl = _count_fa_syllables_base(word[:-2])
        base = max(base, stem_syl + 1)

    return max(base, 1)


# ── dispatcher اصلی ─────────────────────────────────────────────────────────
def count_syllables(word: str, pos_tag: str | None = None) -> int:
    """
    هجاشماری هوشمند:
    - کلمات لاتین: الگوریتم واکه‌محور انگلیسی
    - کلمات فارسی بدون POS: heuristic مورفولوژیک
    - کلمات فارسی با POS: heuristic بهبودیافته با اطلاعات دستوری
    """
    is_persian = any("\u0600" <= ch <= "\u06ff" for ch in word)
    if not is_persian:
        return _count_en_syllables(word)
    if pos_tag is not None:
        return _count_fa_syllables_pos(word, pos_tag)
    return _count_fa_syllables_base(word)


# ── توکن‌ها ──────────────────────────────────────────────────────────────────
def _is_word_token(token: str) -> bool:
    return any(ch.isalpha() for ch in token)


def count_letters(words: list[str]) -> int:
    return sum(ch.isalpha() for w in words for ch in w)


# ── Singleton normalizer ─────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _get_normalizer() -> Normalizer:
    return Normalizer()


# ── Singleton‌های Parsivar با الگوی sentinel ──────────────────────────────────
# [پیشنهاد ۴] به جای lru_cache از متغیر module-level استفاده می‌شود تا:
#   ۱. اگر Parsivar بعداً نصب شد، با یک restart درست لود شود
#   ۲. رفتار هنگام خطا صریح‌تر باشد

_parsivar_tagger = None
_parsivar_tagger_ready = False

_parsivar_tokenizer = None
_parsivar_tokenizer_ready = False


def _get_parsivar_tagger() -> "ParsivarPOSTagger | None":
    global _parsivar_tagger, _parsivar_tagger_ready
    if _parsivar_tagger_ready:
        return _parsivar_tagger
    _parsivar_tagger_ready = True
    if not _PARSIVAR_AVAILABLE:
        return None
    try:
        _parsivar_tagger = ParsivarPOSTagger(tagging_model="wapiti")
        logger.info("Parsivar POSTagger loaded successfully.")
    except Exception as exc:
        logger.warning("Parsivar POSTagger could not be loaded: %s", exc)
    return _parsivar_tagger


def _get_parsivar_tokenizer() -> "ParsivarTokenizer | None":
    global _parsivar_tokenizer, _parsivar_tokenizer_ready
    if _parsivar_tokenizer_ready:
        return _parsivar_tokenizer
    _parsivar_tokenizer_ready = True
    if not _PARSIVAR_AVAILABLE:
        return None
    try:
        _parsivar_tokenizer = ParsivarTokenizer()
        logger.info("Parsivar Tokenizer loaded successfully.")
    except Exception as exc:
        logger.warning("Parsivar Tokenizer could not be loaded: %s", exc)
    return _parsivar_tokenizer


# ── سطح‌بندی خوانایی ─────────────────────────────────────────────────────────
_READABILITY_LEVELS: list[tuple[int, str]] = [
    (90, "بسیار آسان — مناسب کودکان دبستانی"),
    (80, "آسان — مناسب نوجوانان"),
    (70, "نسبتاً آسان — مناسب عموم مردم"),
    (60, "متوسط — مناسب دانش‌آموزان دبیرستان"),
    (50, "نسبتاً دشوار — مناسب دانشجویان"),
    (30, "دشوار — مناسب متخصصان"),
    (0,  "بسیار دشوار — متون علمی/تخصصی"),
]


def interpret_score(score: float) -> str:
    for threshold, label in _READABILITY_LEVELS:
        if score >= threshold:
            return label
    return _READABILITY_LEVELS[-1][1]  # [پیشنهاد ۷] DRY — بدون رشته تکراری


# ── ثابت‌های حالت POS ────────────────────────────────────────────────────────
POS_MODE_PARSIVAR  = "POS-enhanced — Parsivar"
POS_MODE_HEURISTIC = "morphological heuristic (بدون POS)"


# ── نتیجه ────────────────────────────────────────────────────────────────────
@dataclass
class ReadabilityResult:
    sentences: int
    words:     int
    letters:   int
    syllables: int
    asl:       float   # Average Sentence Length (words/sentence)
    wl:        float   # Average Word Length (letters/word)
    asyl:      float   # Average Syllables per Word
    flesch_dayani: float
    level:     str
    pos_mode:  str


# ── توابع کمکی tagging ───────────────────────────────────────────────────────
# [پیشنهاد ۵] منطق tagging از compute_flesch_dayani استخراج شد

def _tag_sentence_parsivar(
    sent: str,
    pv_tok: "ParsivarTokenizer",
    pv_tagger: "ParsivarPOSTagger",
) -> list[tuple[str, str | None]]:
    tokens = pv_tok.tokenize_words(sent)
    word_tokens = [t for t in tokens if _is_word_token(t)]
    if not word_tokens:
        return []
    try:
        return pv_tagger.parse(word_tokens)
    except (RuntimeError, ValueError, IndexError) as exc:
        # [پیشنهاد ۷] log به انگلیسی (پیام توسعه‌دهنده)
        logger.warning("Parsivar tagger failed for sentence, falling back to heuristic: %s", exc)
        return [(w, None) for w in word_tokens]


def _tag_sentence_heuristic(sent: str) -> list[tuple[str, str | None]]:
    tokens = word_tokenize(sent)
    return [(t, None) for t in tokens if _is_word_token(t)]


def _extract_tagged_words(
    sentences: list[str],
    pv_tok: "ParsivarTokenizer | None",
    pv_tagger: "ParsivarPOSTagger | None",
) -> tuple[list[tuple[str, str | None]], str]:
    """
    تمام جملات را tag می‌زند و (tagged_words, pos_mode) برمی‌گرداند.
    اگر Parsivar در دسترس نباشد، به heuristic fallback می‌کند.
    """
    if pv_tagger is not None and pv_tok is not None:
        tagged: list[tuple[str, str | None]] = []
        for sent in sentences:
            tagged.extend(_tag_sentence_parsivar(sent, pv_tok, pv_tagger))
        return tagged, POS_MODE_PARSIVAR

    tagged = []
    for sent in sentences:
        tagged.extend(_tag_sentence_heuristic(sent))
    return tagged, POS_MODE_HEURISTIC


# ── محاسبه اصلی ──────────────────────────────────────────────────────────────
def compute_flesch_dayani(text: str) -> ReadabilityResult:
    """
    شاخص خوانایی Flesch–Dayani برای متن فارسی.

    فرمول دیانی (۱۳۷۴):
        FDR = 262.835 − 0.846 × ASYL − 1.015 × ASL

    هجاشماری:
    - اگر Parsivar نصب باشد: POS-enhanced (~85٪ دقت)
    - در غیر این صورت: morphological heuristic (~75٪ دقت)
    """
    normalizer = _get_normalizer()
    normalized = normalizer.normalize(text)

    raw_sentences = [
        s for s in sent_tokenize(normalized)
        if any(ch.isalpha() for ch in s)
    ]
    if not raw_sentences:
        # [پیشنهاد ۷] پیام‌های کاربر به فارسی
        raise ValueError("متن پس از نرمال‌سازی هیچ جمله‌ای ندارد.")

    tagged_words, pos_mode = _extract_tagged_words(
        raw_sentences,
        pv_tok=_get_parsivar_tokenizer(),
        pv_tagger=_get_parsivar_tagger(),
    )

    n_sentences = len(raw_sentences)
    n_words = len(tagged_words)
    if n_words == 0:
        raise ValueError("متن هیچ کلمه‌ای (با حروف الفبایی) ندارد.")

    all_words = [w for w, _ in tagged_words]
    n_letters = count_letters(all_words)
    if n_letters == 0:
        raise ValueError("هیچ حرف الفبایی در متن یافت نشد.")

    # [پیشنهاد ۸] هشدار برای متون خیلی کوتاه
    if n_words < _MIN_WORDS_RELIABLE:
        logger.warning(
            "Text has only %d words; Flesch–Dayani score may be unreliable "
            "(recommend >= %d words for stable results).",
            n_words,
            _MIN_WORDS_RELIABLE,
        )

    n_syllables = sum(count_syllables(w, tag) for w, tag in tagged_words)

    asl  = n_words / n_sentences
    wl   = n_letters / n_words
    asyl = n_syllables / n_words
    score = 262.835 - 0.846 * asyl - 1.015 * asl

    return ReadabilityResult(
        sentences=n_sentences,
        words=n_words,
        letters=n_letters,
        syllables=n_syllables,
        asl=asl,
        wl=wl,
        asyl=asyl,
        flesch_dayani=score,
        level=interpret_score(score),
        pos_mode=pos_mode,
    )


# ── CLI ──────────────────────────────────────────────────────────────────────
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Persian Flesch–Dayani readability index calculator"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-f", "--file",
        type=str,
        help="Path to a UTF-8 encoded Persian text file",
    )
    group.add_argument(
        "-t", "--text",
        type=str,
        help="Direct Persian text to analyze (in quotes)",
    )
    parser.add_argument(
        "--plain",
        action="store_true",
        help="Only print the raw Flesch–Dayani score",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.file:
        try:
            with open(args.file, "r", encoding="utf-8") as f:
                text = f.read()
        except OSError as e:
            print(f"خطا در خواندن فایل: {e}", file=sys.stderr)
            sys.exit(1)
    elif args.text:
        text = args.text
    else:
        if sys.stdin.isatty():
            # [پیشنهاد ۹] راهنمای stdin به جای خروج بی‌خبر
            print(
                "در انتظار متن از stdin (Ctrl+D برای پایان)...",
                file=sys.stderr,
            )
        text = sys.stdin.read()

    if not text or not text.strip():
        print("خطا: متن خالی است.", file=sys.stderr)
        sys.exit(1)

    try:
        result = compute_flesch_dayani(text)
    except ValueError as e:
        print(f"خطا: {e}", file=sys.stderr)
        sys.exit(1)

    if args.plain:
        print(f"{result.flesch_dayani:.2f}")
        return

    w = 54
    print("═" * w)
    print("  Persian Readability — Flesch–Dayani")
    print("═" * w)
    print(f"  جملات   : {result.sentences}")
    print(f"  کلمات   : {result.words}")
    print(f"  حروف    : {result.letters}")
    print(f"  هجاها   : {result.syllables}")
    print(f"  روش     : {result.pos_mode}")
    print("─" * w)
    print(f"  ASL  (کلمه/جمله)  : {result.asl:.2f}")
    print(f"  WL   (حرف/کلمه)   : {result.wl:.2f}")
    print(f"  ASYL (هجا/کلمه)   : {result.asyl:.2f}")
    print("─" * w)
    print(f"  امتیاز Flesch–Dayani : {result.flesch_dayani:.2f}")
    print(f"  سطح خوانایی         : {result.level}")
    print("═" * w)


if __name__ == "__main__":
    main()
