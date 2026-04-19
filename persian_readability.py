from __future__ import annotations

import sys
import re
import argparse
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional

from hazm import Normalizer, sent_tokenize, word_tokenize


# ── واکه‌های فارسی برای هجاشمار ─────────────────────────────────────────────
# واکه‌های بلند (حروف اصلی هجاساز در فارسی نوشتاری)
_FA_VOWELS = set("اوی")
# واکه‌های کوتاه (اعراب — معمولاً در متن عادی نیستن ولی پوشش داده می‌شن)
_FA_DIACRITICS = set("\u064e\u064f\u0650")  # فتحه، ضمه، کسره
# واکه‌های لاتین
_EN_VOWELS = set("aeiouAEIOU")


def count_persian_syllables(word: str) -> int:
    """
    هجاشمار فارسی بر اساس واکه‌های نوشتاری.

    قوانین:
    - هر «ا»، «و»، «ی» یک هجا حساب می‌شه
    - «ه» پایان کلمه = یک هجا (مثل «خانه»)
    - اعراب (فتحه/ضمه/کسره) هم هجا حساب می‌شن
    - حداقل یک هجا برای هر کلمه برگردونده می‌شه
    """
    if not word:
        return 0

    syllables = 0
    for i, ch in enumerate(word):
        if ch in _FA_VOWELS:
            syllables += 1
        elif ch in _FA_DIACRITICS:
            syllables += 1
        elif ch == "ه" and i == len(word) - 1 and len(word) > 1:
            # «ه» پایان کلمه که واکه‌ای قبلش نیست
            if i == 0 or word[i - 1] not in _FA_VOWELS:
                syllables += 1

    return max(syllables, 1)


def count_english_syllables(word: str) -> int:
    """
    هجاشمار ساده برای کلمات لاتین.
    """
    word = word.lower().strip(".,!?;:")
    if not word:
        return 0
    # شمارش گروه‌های واکه متوالی
    count = len(re.findall(r"[aeiou]+", word))
    # «e» ساکت در پایان
    if word.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)


def count_syllables(word: str) -> int:
    """
    تشخیص فارسی یا لاتین و هجاشماری مناسب.
    """
    # اگر کلمه حاوی حروف عربی/فارسی بود → هجاشمار فارسی
    if any("\u0600" <= ch <= "\u06ff" for ch in word):
        return count_persian_syllables(word)
    return count_english_syllables(word)


def _is_word_token(token: str) -> bool:
    """فقط توکن‌هایی که حداقل یک حرف الفبایی دارن."""
    return any(ch.isalpha() for ch in token)


def count_letters(words: List[str]) -> int:
    """شمارش حروف الفبایی (فارسی + لاتین)."""
    return sum(ch.isalpha() for w in words for ch in w)


@lru_cache(maxsize=1)
def _get_normalizer() -> Normalizer:
    return Normalizer()


# ── سطح‌بندی خوانایی ─────────────────────────────────────────────────────────
_READABILITY_LEVELS = [
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
    return "بسیار دشوار — متون علمی/تخصصی"


# ── نتیجه ─────────────────────────────────────────────────────────────────────
@dataclass
class ReadabilityResult:
    sentences: int
    words: int
    letters: int
    syllables: int
    asl: float          # Average Sentence Length
    wl: float           # Average Word Length (letters)
    asyl: float         # Average Syllables per Word
    flesch_dayani: float
    level: str


def compute_flesch_dayani(text: str) -> ReadabilityResult:
    """
    محاسبه شاخص خوانایی Flesch–Dayani برای متن فارسی.

    فرمول اصلی دیانی (۱۳۷۴):
        FDR = 262.835 − 0.846 × WL − 1.015 × ASL
    که در آن WL = میانگین هجا به ازای هر کلمه است.

    منبع: دیانی، م. (۱۳۷۴). سنجش خوانایی متون فارسی.
    """
    normalizer = _get_normalizer()
    normalized = normalizer.normalize(text)

    raw_sentences = sent_tokenize(normalized)
    if not raw_sentences:
        raise ValueError("متن پس از نرمال‌سازی هیچ جمله‌ای ندارد.")

    all_words: List[str] = []
    for sent in raw_sentences:
        tokens = word_tokenize(sent)
        all_words.extend(t for t in tokens if _is_word_token(t))

    n_sentences = len(raw_sentences)
    n_words = len(all_words)

    if n_words == 0:
        raise ValueError("متن هیچ کلمه‌ای (با حروف الفبایی) ندارد.")

    n_letters = count_letters(all_words)
    n_syllables = sum(count_syllables(w) for w in all_words)

    if n_letters == 0:
        raise ValueError("هیچ حرف الفبایی در متن یافت نشد.")

    asl  = n_words    / n_sentences   # کلمه به ازای جمله
    wl   = n_letters  / n_words       # حرف به ازای کلمه (نگه داشته برای مقایسه)
    asyl = n_syllables / n_words      # هجا به ازای کلمه (مقدار اصلی فرمول)

    # فرمول Flesch–Dayani با هجا (دقیق‌تر)
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
    )


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Persian Flesch–Dayani readability index calculator"
    )
    group = parser.add_mutually_exclusive_group()   # همه optional — stdin هم قبوله
    group.add_argument("-f", "--file", type=str,
                       help="Path to a UTF-8 encoded Persian text file")
    group.add_argument("-t", "--text", type=str,
                       help="Direct Persian text to analyze (in quotes)")
    parser.add_argument("--plain", action="store_true",
                        help="Only print the raw Flesch–Dayani score")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    # منبع متن: فایل → آرگومان → stdin
    if args.file:
        try:
            with open(args.file, "r", encoding="utf-8") as f:
                text = f.read()
        except OSError as e:
            print(f"Error reading file: {e}", file=sys.stderr)
            sys.exit(1)
    elif args.text:
        text = args.text
    else:
        # پشتیبانی از pipe: echo "متن" | python persian_readability.py
        if sys.stdin.isatty():
            print("خطا: متن از طریق -t، -f یا stdin وارد نشده.", file=sys.stderr)
            sys.exit(1)
        text = sys.stdin.read()

    if not text or not text.strip():
        print("متن خالی است.", file=sys.stderr)
        sys.exit(1)

    try:
        result = compute_flesch_dayani(text)
    except ValueError as e:
        print(f"خطا: {e}", file=sys.stderr)
        sys.exit(1)

    if args.plain:
        print(f"{result.flesch_dayani:.2f}")
        return

    print("═" * 50)
    print("  Persian Readability — Flesch–Dayani")
    print("═" * 50)
    print(f"  جملات   : {result.sentences}")
    print(f"  کلمات   : {result.words}")
    print(f"  حروف    : {result.letters}")
    print(f"  هجاها   : {result.syllables}")
    print("─" * 50)
    print(f"  ASL  (کلمه/جمله)  : {result.asl:.2f}")
    print(f"  WL   (حرف/کلمه)   : {result.wl:.2f}")
    print(f"  ASYL (هجا/کلمه)   : {result.asyl:.2f}")
    print("─" * 50)
    print(f"  امتیاز Flesch–Dayani : {result.flesch_dayani:.2f}")
    print(f"  سطح خوانایی         : {result.level}")
    print("═" * 50)


if __name__ == "__main__":
    main()
