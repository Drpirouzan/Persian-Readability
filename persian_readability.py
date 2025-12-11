from __future__ import annotations
import sys
import argparse
from dataclasses import dataclass
from typing import List

from hazm import Normalizer, sent_tokenize, word_tokenize

normalizer = Normalizer()


@dataclass
class ReadabilityResult:
    sentences: int
    words: int
    letters: int
    asl: float  # Average Sentence Length (words per sentence)
    wl: float   # Average Word Length (letters per word)
    flesch_dayani: float


def count_letters(words: List[str]) -> int:
    """
    شمارش حروف الفبایی (فارسی + لاتین) در لیست کلمات.
    علائم نگارشی، اعداد و فاصله‌ها شمرده نمی‌شوند.
    """
    total = 0
    for w in words:
        for ch in w:
            if ch.isalpha():
                total += 1
    return total


def compute_flesch_dayani(text: str) -> ReadabilityResult:
    # ۱) نرمال‌سازی متن فارسی
    normalized = normalizer.normalize(text)

    # ۲) جمله‌بندی
    sentences = sent_tokenize(normalized)
    n_sentences = max(len(sentences), 1)

    # ۳) کلمه‌بندی
    tokenized_sentences = [word_tokenize(s) for s in sentences]
    words = [w for sent in tokenized_sentences for w in sent if w.strip()]
    n_words = max(len(words), 1)

    # ۴) شمارش حروف
    n_letters = max(count_letters(words), 1)

    # ۵) محاسبهٔ ASL و WL
    asl = n_words / n_sentences      # کلمه به ازای هر جمله
    wl = n_letters / n_words         # حرف به ازای هر کلمه

    # ۶) امتیاز Flesch–Dayani برای فارسی
    # FDR = 262.835 - 0.846 * WL - 1.015 * SL
    score = 262.835 - 0.846 * wl - 1.015 * asl

    return ReadabilityResult(
        sentences=n_sentences,
        words=n_words,
        letters=n_letters,
        asl=asl,
        wl=wl,
        flesch_dayani=score,
    )


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Persian Flesch–Dayani readability index calculator"
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-f", "--file",
        type=str,
        help="Path to a UTF-8 encoded Persian text file"
    )
    group.add_argument(
        "-t", "--text",
        type=str,
        help="Direct Persian text to analyze (in quotes)"
    )

    parser.add_argument(
        "--plain",
        action="store_true",
        help="Only print the raw Flesch–Dayani score"
    )

    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)

    if args.file:
        try:
            with open(args.file, "r", encoding="utf-8") as f:
                text = f.read()
        except OSError as e:
            print(f"Error reading file: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        text = args.text

    if not text or not text.strip():
        print("متن خالی است.", file=sys.stderr)
        sys.exit(1)

    result = compute_flesch_dayani(text)

    if args.plain:
        print(f"{result.flesch_dayani:.2f}")
    else:
        print("----- Persian Readability (Flesch–Dayani) -----")
        print(f"Sentences: {result.sentences}")
        print(f"Words:     {result.words}")
        print(f"Letters:   {result.letters}")
        print()
        print(f"Average sentence length (ASL: words/sentence): {result.asl:.2f}")
        print(f"Average word length (WL: letters/word):        {result.wl:.2f}")
        print()
        print(f"Flesch–Dayani score: {result.flesch_dayani:.2f}")
        print()
        print("Higher scores ≈ easier Persian text (Dayani-calibrated).")


if __name__ == "__main__":
    main()
