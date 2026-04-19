# Persian Readability (Flesch–Dayani)

A lightweight Python script to calculate the **Flesch–Dayani readability score** for Persian (Farsi) text — with a real Persian syllable counter for higher accuracy.

---

## Features

- Persian text normalization and tokenization via `hazm`
- **Real Persian syllable counter** (long vowels `ا و ی`, diacritics, word-final `ه`, and mixed Persian/Latin words)
- Computes:
  - Number of sentences, words, letters, and syllables
  - **ASL** — Average Sentence Length (words per sentence)
  - **WL** — Average Word Length (letters per word)
  - **ASYL** — Average Syllables per Word *(used in the original Dayani formula)*
  - Flesch–Dayani readability score
  - **Human-readable level** (e.g. *متوسط — مناسب دانش‌آموزان دبیرستان*)
- Accepts input from a file, a command-line argument, or **stdin** (pipe-friendly)
- `--plain` flag for scripting and pipeline use

---

## Readability Levels

| Score | Level |
|-------|-------|
| ≥ 90 | بسیار آسان — مناسب کودکان دبستانی |
| ≥ 80 | آسان — مناسب نوجوانان |
| ≥ 70 | نسبتاً آسان — مناسب عموم مردم |
| ≥ 60 | متوسط — مناسب دانش‌آموزان دبیرستان |
| ≥ 50 | نسبتاً دشوار — مناسب دانشجویان |
| ≥ 30 | دشوار — مناسب متخصصان |
| < 30 | بسیار دشوار — متون علمی/تخصصی |

---

## Requirements

- Python **3.7** or newer
- [`hazm`](https://github.com/sobhe/hazm) — Persian NLP library

```bash
pip install hazm
```

---

## Usage

**Direct text:**
```bash
python persian_readability.py -t "متن فارسی شما"
```

**From a file:**
```bash
python persian_readability.py -f sample.txt
```

**From stdin (pipe):**
```bash
echo "متن فارسی شما" | python persian_readability.py
cat article.txt | python persian_readability.py
```

**Raw score only (for scripting):**
```bash
python persian_readability.py -f sample.txt --plain
```

---

## Sample Output

```
══════════════════════════════════════════════════
  Persian Readability — Flesch–Dayani
══════════════════════════════════════════════════
  جملات   : 5
  کلمات   : 87
  حروف    : 412
  هجاها   : 201
──────────────────────────────────────────────────
  ASL  (کلمه/جمله)  : 17.40
  WL   (حرف/کلمه)   : 4.74
  ASYL (هجا/کلمه)   : 2.31
──────────────────────────────────────────────────
  امتیاز Flesch–Dayani : 58.34
  سطح خوانایی         : متوسط — مناسب دانش‌آموزان دبیرستان
══════════════════════════════════════════════════
```

---

## Formula

```
FDR = 262.835 − 0.846 × ASYL − 1.015 × ASL
```

Where **ASYL** = average syllables per word and **ASL** = average words per sentence.
Higher scores indicate easier text.

> **Note:** The original Dayani (1374/1995) formula uses syllables per word (ASYL).
> Earlier implementations approximated this with letter count (WL); this version uses a real Persian syllable counter for improved accuracy.

---

## References

- Dayani, M. (1374/1995). *سنجش خوانایی متون فارسی*. Persian adaptation of the Flesch Reading Ease formula.
- Mohammadi & Khasteh (2020). [A Machine Learning Approach to Persian Text Readability](https://arxiv.org/abs/1810.06639).
- Sobhe. [hazm — Persian NLP library](https://github.com/sobhe/hazm).

---

## Author

**Dr. Mohammad Pirouzan** — [@Drpirouzan](https://github.com/Drpirouzan)

---

## License

MIT License — see [LICENSE](LICENSE) for details.
