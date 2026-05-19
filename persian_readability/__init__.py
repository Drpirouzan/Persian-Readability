"""
Persian Readability — Flesch–Dayani readability score for Persian/Farsi text.
"""

from . import core as _core

# Re-export core functions/classes, including existing internal helpers,
# so older tests/imports keep working after converting the project to a package.
for _name in dir(_core):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_core, _name)

__version__ = "0.1.1"


class PersianReadability:
    """Small convenience wrapper around the core readability calculator."""

    def analyze(self, text: str, mode="auto"):
        return compute_flesch_dayani(text, mode=mode)

    def calculate(self, text: str, mode="auto") -> dict:
        return calculate_readability(text, mode=mode)


def calculate_readability(text: str, mode="auto") -> dict:
    """
    Calculate Persian/Farsi readability and return a simple dictionary.

    This wrapper is useful for users who prefer a JSON-like output instead of
    the ReadabilityResult dataclass.
    """
    result = compute_flesch_dayani(text, mode=mode)

    return {
        "score": result.flesch_dayani,
        "level": result.level,
        "sentences": result.sentences,
        "words": result.words,
        "letters": result.letters,
        "syllables": result.syllables,
        "asl": result.asl,
        "wl": result.wl,
        "asyl": result.asyl,
        "pos_mode": result.pos_mode,
        "pos_enhanced": result.pos_enhanced,
        "is_likely_poetry": result.is_likely_poetry,
        "diacritics_mode": result.diacritics_mode,
        "diacritic_ratio": result.diacritic_ratio,
    }


__all__ = [
    "PersianReadability",
    "calculate_readability",
    "compute_flesch_dayani",
    "ReadabilityResult",
    "InputMode",
    "count_syllables",
    "count_letters",
    "interpret_score",
    "analyze_diacritics",
]
