# persian_readability.py
# -*- coding: utf-8 -*-

"""
ابزار ساده برای محاسبه شاخص خوانایی فلش–دایانی برای متن فارسی.

نکته:
- برای توکنایز و نرمال‌سازی از hazm استفاده می‌کنیم.
- برای شمارش هجا فعلاً از یک روش تقریبی استفاده می‌کنیم
  (شمارش حروفی مثل ا، آ، ی، و ... به عنوان نماینده‌ی مصوت‌ها).
"""

from __future__ import annotations
import sys
import argparse
from dataclasses import dataclass
from typing import List

from hazm import Normalizer, sent_tokenize, word_tokenize

normalizer = Normalizer()

# ----------------- مدل داده برای خروجی ----------------- #

@dataclass
class ReadabilityResult:
    sentences: int
    words: int
    syllables: int
    asl: float  # Average Sentence Length
    asw: float  # Average Syllables per Word
    flesch_dayani: float


# ----------------- توابع هجاشمار ----------------- #

def approx_syllables_in_word(word: str) -> int:
    """
    هجاشمار خیلی ساده و تقریبی برای فارسی.

    چون مصوت‌های کوتاه در خط فارسی نوشته نمی‌شن، ما فقط
    روی یه سری حروف تکیه می‌کنیم:
      ا، آ، و، ی، ۀ، ؤ، ۀ، ...
    این مدل «علمیِ نهایی» نیست، ولی برای یک اندکس نسبی جواب می‌ده.
    """
    vowels = set("اآایویۀؤءةۀۀ")  # می‌تونی بعداً این لیست رو تنظیم کنی
    count = sum(1 for ch in word if ch in vowels)
    return count or 1  # هر کلمه حداقل یک هجا


def count_syllables(words: List[str]) -> int:
    return sum(approx_syllables_in_word(w) for w in words)


# ----------------- محاسبه شاخص فلش–دایانی ----------------- #

def compute_flesch_dayani(text: str) -> ReadabilityResult:
    # ۱) نرمال‌سازی
    normalized = normalizer.normalize(text)

    # ۲) جمله‌بندی
    sentences = sent_tokenize(normalized)
    n_sentences = max(len(sentences), 1)

    # ۳) کلمه‌بندی
    tokenized_sentences = [word_tokenize(s) for s in sentences]
    words = [w for sent in tokenized_sentences for w in sent if w.strip()]
    n_words = max(len(words), 1)

    # ۴) شمارش هجا
    n_syllables = max(count_syllables(words), 1)

    # ۵) محاسبهٔ ASL و ASW
    asl = n_words / n_sentences
    asw = n_syllables / n_words

    # ۶) فرمول Flesch–Dayani (نسخه‌ی رایج برای فارسی)
    # FRES_Persian = 262.835 - 1.015 * ASL - 0.846 * ASW
    score = 262.835 - 1.015 * asl - 0.846 * asw

    return ReadabilityResult(
        sentences=n_sentences,
        words=n_words,
        syllables=n_syllables,
        asl=asl,
        asw=asw,
        flesch_dayani=score,
    )


# ----------------- CLI ----------------- #

def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Persian Flesch–Dayani readability index calculator"
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-f", "--file",
        type=str,
        help="مسیر فایل متنی فارسی"
    )
    group.add_argument(
        "-t", "--text",
        type=str,
        help="متن فارسی به صورت مستقیم (داخل کوتیشن)"
    )

    parser.add_argument(
        "--plain",
        action="store_true",
        help="فقط امتیاز فلش–دایانی را به صورت عدد خام چاپ کن"
    )

    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)

    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            text = f.read()
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
        print(f"جمله‌ها (sentences): {result.sentences}")
        print(f"کلمه‌ها (words):    {result.words}")
        print(f"هجاها (syllables): {result.syllables}")
        print()
        print(f"میانگین طول جمله (ASL): {result.asl:.2f}")
        print(f"میانگین هجا به ازای هر کلمه (ASW): {result.asw:.2f}")
        print()
        print(f"امتیاز Flesch–Dayani: {result.flesch_dayani:.2f}")
        print()
        print("تفسیر خیلی کلی (هرچی امتیاز بالاتر → متن ساده‌تر):")
        print("  ~200 به بالا: خیلی ساده")
        print("  150–200: نسبتاً ساده")
        print("  100–150: متوسط / دانشگاهی")
        print("  زیر 100: سنگین / تخصصی")

if __name__ == "__main__":
    main()
