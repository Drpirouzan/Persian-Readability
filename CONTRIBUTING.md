# Contributing to Persian-Readability

Thank you for your interest in contributing!  
This project welcomes contributions from the Persian NLP community.

## Ways to Contribute

- **Bug reports** — open an Issue describing the problem and sample text
- **Accuracy improvements** — better syllable counting rules or POS handling
- **New features** — support for Dari, Pashto, or other Persian-script languages
- **Documentation** — Persian translations, usage examples, blog posts
- **Tests** — add test cases covering edge cases in Persian text

## Getting Started

```bash
git clone https://github.com/Drpirouzan/Persian-Readability.git
cd Persian-Readability
pip install -e ".[pos]"   # installs hazm + parsivar
```

## Submitting a Pull Request

1. Fork the repository
2. Create a branch: `git checkout -b feature/your-feature-name`
3. Make your changes and add tests
4. Run tests: `pytest tests/`
5. Open a Pull Request with a clear description

## Reporting Issues

Please include:
- The Persian text that caused the problem (or a minimal example)
- Expected vs actual output
- Python version and OS

## Code Style

- Follow PEP 8
- Add docstrings to new functions
- Comments in English or Persian are both welcome

## Community

This project serves the Persian-speaking developer and research community.  
We especially welcome contributions that improve accuracy for literary,
journalistic, and academic Persian text.
