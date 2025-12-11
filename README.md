# Persian Readability (Flesch–Dayani)

This repository provides a simple Python script to calculate the **Flesch–Dayani readability score** for Persian (Farsi) text.

## Features

- Uses `hazm` for Persian normalization and tokenization.
- Computes basic length-based features for Persian text.
- Computes:
  - Number of sentences
  - Number of words
  - Number of letters
  - ASL (Average Sentence Length = words per sentence)
  - WL (Average Word Length = letters per word)
  - Flesch–Dayani readability score (Dayani-calibrated for Persian)
Based on Dayani (2000) Persian adaptation of the Flesch Reading Ease formula.

## Requirements

- Python 3.9 or newer
- [`hazm`](https://github.com/sobhe/hazm) for Persian text normalization and tokenization (install with: `pip install hazm`)

## Usage

- Run with direct text:
`python persian_readability.py -t "متن فارسی شما"`

- Run with a file:
`python persian_readability.py -f sample.txt`

- Just print the raw score:
`python persian_readability.py -f sample.txt`

- Higher scores ≈ easier text (for Persian calibration).

## References

- Dayani (2000) – Persian adaptation of the Flesch Reading Ease.

- Mohammadi & Khasteh (2020) – A Machine Learning Approach to Persian Text Readability.

## Author

- Dr. Mohammad Pirouzan (GitHub: @Drpirouzan)
