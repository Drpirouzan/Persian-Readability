# Persian Readability (Flesch–Dayani)

A lightweight Python script to calculate the **Flesch–Dayani readability score** for Persian (Farsi) text — with an optional POS-enhanced syllable counter for higher accuracy.

---

## Features

- Persian text normalization and tokenization via `hazm`
- **Two-tier syllable counting:**
  - **POS-enhanced** (~85% accuracy) — if `parsivar` is installed, uses part-of-speech tags to correctly count syllables in verbs with attached prefixes (`می‌رود`, `نمیدانم`), comparative adjectives (`بهتر`, `بزرگ‌ترین`), and mixed Persian/Latin text
  - **Morphological heuristic** (~75% accuracy) — used automatically if `parsivar` is not installed; no extra dependency required
- Computes:
  - Number of sentences, words, letters, and syllables
  - **ASL** — Average Sentence Length (words per sentence)
  - **WL** — Average Word Length (letters per word)
  - **ASYL** — Average Syllables per Word *(used in the original Dayani formula)*
  - Flesch–Dayani readability score
  - **Human-readable level** (e.g. *متوسط — مناسب دانش‌آموزان دبیرستان*)
- Accepts input from a file, a command-line argument, or **stdin** (pipe-friendly)
- `--plain` flag for scripting and pipeline use
- The output indicates which syllable mode was active

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

### Required
- Python **3.7** or newer
- [`hazm`](https://github.com/roshan-research/hazm) — Persian NLP library

```bash
pip install hazm
```

### Optional (for higher syllable accuracy)
- [`parsivar`](https://github.com/ICTRC/Parsivar) — Persian preprocessing toolkit with POS tagger

```bash
pip install parsivar
```

> If `parsivar` is not installed, the script falls back to the morphological heuristic automatically — no configuration needed.

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
════════════════════════════════════════════════════
  Persian Readability — Flesch–Dayani
════════════════════════════════════════════════════
  جملات   : 5
  کلمات   : 87
  حروف    : 412
  هجاها   : 201  [POS-enhanced (Parsivar)]
────────────────────────────────────────────────────
  ASL  (کلمه/جمله)  : 17.40
  WL   (حرف/کلمه)   : 4.74
  ASYL (هجا/کلمه)   : 2.31
────────────────────────────────────────────────────
  امتیاز Flesch–Dayani : 58.34
  سطح خوانایی         : متوسط — مناسب دانش‌آموزان دبیرستان
════════════════════════════════════════════════════
```

---

## Formula

```
FDR = 262.835 − 0.846 × ASYL − 1.015 × ASL
```

Where **ASYL** = average syllables per word and **ASL** = average words per sentence.
Higher scores indicate easier text.

> **Syllable counting note:** The original Dayani (1374/1995) formula requires syllables per word (ASYL).
> This implementation uses POS-aware syllable counting when `parsivar` is available, falling back to a
> morphological heuristic otherwise. Neither approach is a trained ML model — both are deterministic and
> fast, with no GPU or large model download required.

---

## How Syllable Accuracy Tiers Work

| Mode | Accuracy | How |
|------|----------|-----|
| POS-enhanced | ~85% | Parsivar POSTagger (wapiti CRF model, trained on Bijankhan corpus) detects verb/adjective tags; prefix/suffix rules applied per POS |
| Morphological heuristic | ~75% | Counts long vowels (ا و ی), diacritics, and word-final ه; no POS context |

The main cases where POS tagging improves accuracy:
- Verbs with attached prefixes: `میرود` (written without half-space) → correctly +1 syllable
- Comparative/superlative adjectives: `بهترین` → suffix `ترین` counted as 2 syllables
- Mixed Persian/Latin tokens: routed to an English vowel-group counter

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
