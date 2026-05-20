# Persian Readability (Flesch–Dayani)

[![PyPI](https://img.shields.io/pypi/v/persian-readability)](https://pypi.org/project/persian-readability/)
![Python](https://img.shields.io/pypi/pyversions/persian-readability)
[![License](https://img.shields.io/github/license/Drpirouzan/Persian-Readability)](https://github.com/Drpirouzan/Persian-Readability/blob/main/LICENSE)

A lightweight Python package and command-line tool to calculate the **Flesch–Dayani readability score** for Persian (Farsi) text — with an optional POS-enhanced syllable counter for higher accuracy.

---

## Features

- Persian text normalization and tokenization via `hazm`
- **Punctuation-aware tokenization** — علائم نشانه‌گذاری از شمارش کلمات و هجاها حذف می‌شوند
- **Two-tier syllable counting:**
  - **POS-enhanced** (Better Accuracy) — if `parsivar` is installed, uses part-of-speech tags to correctly count syllables in verbs with attached prefixes (`میرود`، `نمی‌دانم`) and comparative adjectives (`بهتر`، `بزرگ‌ترین`)
  - **Morphological heuristic** (Good Accuracy) — used automatically if `parsivar` is not installed
- **Context-aware خواه classifier** — three-layer disambiguation prevents confusing `خواهش`, `خواهر`, `آزادی‌خواه`, and `خواه ... خواه ...` with the future auxiliary (`خواهم رفت`)
- Computes:
  - Number of sentences, words, letters, and syllables
  - **ASL** — Average Sentence Length (words per sentence)
  - **WL** — Average Word Length (letters per word)
  - **ASYL** — Average Syllables per Word *(used in the original Dayani formula)*
  - Flesch–Dayani readability score
  - **Human-readable level** (e.g. *متوسط — مناسب دانش‌آموزان دبیرستان*)
- Accepts input from a file, a command-line argument, or **stdin** (pipe-friendly)
- `--plain` flag for scripting and pipeline use
- `--verbose` flag for debug logging
- Warns when text is too short for a reliable score (< 50 words)

---

## Readability Levels

| Score | Level |
|-------|-------|
| ≥ 90  | بسیار آسان — مناسب کودکان دبستانی |
| ≥ 80  | آسان — مناسب نوجوانان |
| ≥ 70  | نسبتاً آسان — مناسب عموم مردم |
| ≥ 60  | متوسط — مناسب دانش‌آموزان دبیرستان |
| ≥ 50  | نسبتاً دشوار — مناسب دانشجویان |
| ≥ 30  | دشوار — مناسب متخصصان |
| < 30  | بسیار دشوار — متون علمی/تخصصی |

---

## Installation

Install from PyPI after release:

```bash
pip install persian-readability
```

For local development:

```bash
pip install -e ".[dev]"
```

For optional POS-enhanced syllable counting:

```bash
pip install "persian-readability[pos]"
```

---

## Requirements

### Required

- Python **3.10** or newer
- [`hazm`](https://github.com/roshan-research/hazm) — Persian NLP library

```
pip install hazm
```

### Optional (for higher syllable accuracy)

- [`parsivar`](https://github.com/ICTRC/Parsivar) — Persian preprocessing toolkit with POS tagger

```
pip install parsivar
```

> If `parsivar` is not installed, the script falls back to the morphological heuristic automatically — no configuration needed.

---

## Usage

**Direct text:**

```
persian-readability -t "متن فارسی شما"
```

**From a file:**

```
persian-readability -f sample.txt
```

**From stdin (pipe):**

```
echo "متن فارسی شما" | persian-readability
cat article.txt | persian-readability
```

**Raw score only (for scripting):**

```
persian-readability -f sample.txt --plain
```

**With debug logging:**

```
persian-readability -f sample.txt --verbose
```

---

## Python API Usage

```python
from persian_readability import calculate_readability

result = calculate_readability("برای پیشگیری از پوسیدگی دندان، روزی دو بار مسواک بزنید.")
print(result)
```

---

## Real-World Examples

### Example 1 — Public health text

**Input:**

```bash
persian-readability -t "برای پیشگیری از پوسیدگی دندان، بهتر است روزی دو بار مسواک بزنید و مصرف مواد قندی را کاهش دهید."
```

**Possible use case:**

This can help public health educators check whether patient-facing Persian health messages are simple enough for the general public.

---

### Example 2 — Academic text

**Input:**

```bash
persian-readability -t "شاخص‌های زیستی بزاقی می‌توانند در تشخیص زودهنگام برخی بیماری‌های دهان و فک و صورت نقش مهمی داشته باشند."
```

**Possible use case:**

Researchers can compare the readability of Persian academic summaries, abstracts, or educational materials.

---

### Example 3 — Pipeline use

**Input:**

```bash
cat article.txt | persian-readability --plain
```

**Possible use case:**

Developers can integrate the readability score into larger Persian NLP or content-quality workflows.

---

## Sample Output

```text
══════════════════════════════════════════════════════
  Persian Readability — Flesch–Dayani
══════════════════════════════════════════════════════
  جملات   : 5
  کلمات   : 87
  حروف    : 412
  هجاها   : 201
  روش     : POS-enhanced — Parsivar
────────────────────────────────────────────────────
  ASL  (کلمه/جمله)  : 17.40
  WL   (حرف/کلمه)   : 4.74
  ASYL (هجا/کلمه)   : 2.31
────────────────────────────────────────────────────
  امتیاز Flesch–Dayani : 58.34
  سطح خوانایی         : متوسط — مناسب دانش‌آموزان دبیرستان
══════════════════════════════════════════════════════
```

---

## Formula

```
FDR = 262.835 − 0.846 × ASYL − 1.015 × ASL
```

Where **ASYL** = average syllables per word and **ASL** = average words per sentence.
Higher scores indicate easier text.

---

## How Syllable Accuracy Tiers Work

| Mode | Accuracy | How |
|------|----------|-----|
| POS-enhanced | ~85% | Parsivar POSTagger (wapiti CRF, Bijankhan corpus) detects verb/adjective tags; prefix/suffix rules applied per POS |
| Morphological heuristic | ~75% | Counts written long vowels (ا و ی), diacritics, and word-final ه; no POS context |

Main cases where POS tagging improves accuracy:

- Verbs with attached `می`/`نمی` prefix (no half-space): `میرود` → +1 syllable
- Comparative/superlative adjectives: `بهترین` → suffix `ترین` = 2 syllables

### خواه Classifier

The word `خواه` has multiple roles in Persian. A three-layer classifier resolves ambiguity **before** syllable counting:

| Label | Examples | Treatment |
|-------|---------|-----------|
| `FUTURE_AUX` | خواهم رفت، نخواهند پذیرفت | syllable count unchanged (هجاشماری base درست است) |
| `LEXICAL_KHASTAN` | خواهد که برود، این را خواهد | tag اصلی حفظ می‌شود |
| `PARTICLE_KHAH` | خواه بیاید خواه نیاید | treated as non-verb |
| `NOMINAL_DERIVATIVE` | خواهش، خواهان، خواهنده | treated as non-verb |
| `INDEPENDENT_WORD` | خواهر، خواهران | treated as non-verb |
| `SUFFIX_COMPOUND` | آزادی‌خواه، خیرخواه، دادخواه | treated as non-verb |

The classifier uses exact lexical sets (layer 1), suffix-compound detection (layer 2), and a 2-token context window (layer 3) — never a simple prefix regex.

---

## Notes

- **Minimum text length:** The Flesch–Dayani formula is designed for running prose. Texts shorter than ~50 words produce unstable scores. A warning is emitted in this case (visible with `--verbose`).
- **Punctuation filtering:** علائم نشانه‌گذاری فارسی و لاتین (گیومه، نقطه، ویرگول، ...) از لبه‌های هر توکن پاک می‌شوند و توکن‌های تمام‌علامت از شمارش حذف می‌شوند.
- **stdin:** When running interactively without `-t` or `-f`, the script waits for input and prints a prompt. Press `Ctrl+D` to signal end of input.
- **Log messages:** All warnings go to stderr and do not affect `--plain` output.

---

## Running Tests

```
pip install pytest hazm
python -m pytest tests/test_core.py -v
```

76 tests covering: خواه classifier (all 9 document cases), punctuation filtering,
syllable counting, heuristic limitations, formula verification, and edge cases.

---

## References

- Dayani, M. (1374/1995). *سنجش خوانایی متون فارسی*. Persian adaptation of the Flesch Reading Ease formula.
- Mohtaj et al. (2018). [Parsivar: A Language Processing Toolkit for Persian](https://github.com/ICTRC/Parsivar). LREC 2018.
- Mohammadi & Khasteh (2020). [A Machine Learning Approach to Persian Text Readability](https://arxiv.org/abs/1810.06639).
- Sobhe. [hazm — Persian NLP library](https://github.com/roshan-research/hazm).

---

## Author

**Dr. Mohammad Pirouzan** — [@Drpirouzan](https://github.com/Drpirouzan)

---

## License

MIT License — see [LICENSE](LICENSE) for details.
