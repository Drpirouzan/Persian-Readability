# Persian Readability (Flesch–Dayani)

This repository provides a simple Python script to calculate the **Flesch–Dayani readability score** for Persian (Farsi) text.

## Features

- Uses `hazm` for Persian normalization and tokenization.
- Approximates syllable counts for Persian words.
- Computes:
  - Number of sentences
  - Number of words
  - Approximate number of syllables
  - ASL (Average Sentence Length)
  - ASW (Average Syllables per Word)
  - Flesch–Dayani readability score

## Requirements

- Python 3.9 or newer
- [`hazm`](https://github.com/sobhe/hazm) for Persian text normalization and tokenization (install with: `pip install hazm`)

## Usage

- Run with direct text:
\python persian_readability.py -t "امروز هوا خوب است و من به دانشگاه رفتم."

- Run with a file:
\python persian_readability.py -f sample.txt

- Just print the raw score:
\python persian_readability.py -f sample.txt --plain

- Higher scores ≈ easier text (for Persian calibration).
