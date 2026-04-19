from __future__ import annotations

import argparse
import logging
import re
import sys
from dataclasses import dataclass
from functools import lru_cache

from hazm import Normalizer, sent_tokenize, word_tokenize

logger = logging.getLogger(__name__)

# ── بارگذاری اختیاری Parsivar ────────────────────────────────────────────────
try:
    from parsivar import POSTagger as ParsivarPOSTagger
    from parsivar import Tokenizer as ParsivarTokenizer
    _PARSIVAR_AVAILABLE = True
except ImportError:
    _PARSIVAR_AVAILABLE = False

# ── برچسب‌های POS (Parsivar / Bijankhan) ────────────────────────────────────
_VERB_TAGS_PARSIVAR: frozenset[str] = frozenset(
    {"V_PRS", "V_PA", "V_IMP", "V_SUB", "V_FUT", "V_PRF"}
)

# ── الگوی پیشوندهای فعلی (می/نمی پیوسته) ────────────────────────────────────
# فقط می/نمی بدون نیم‌فاصله.
# خواه با classifier سه‌لایه‌ی جداگانه مدیریت می‌شود.
_VERB_PREFIX_ATTACHED = re.compile(r"^(می|نمی)(?!\u200c)")

# ── اعراب و واکه‌های بلند ────────────────────────────────────────────────────
_FA_DIACRITICS:  frozenset[str] = frozenset("\u064e\u064f\u0650\u064b\u064c\u064d")
_FA_LONG_VOWELS: frozenset[str] = frozenset("اوی")

# ── حداقل کلمات برای نتیجه قابل اعتماد ─────────────────────────────────────
_MIN_WORDS_RELIABLE = 50


# ═══════════════════════════════════════════════════════════════════════════════
# Classifier سه‌لایه برای توکن‌های «خواه»
#
# خواه در فارسی چند هویت دارد:
#   FUTURE_AUX       → کمکیِ آینده:       خواهم رفت / نخواهند پذیرفت
#   LEXICAL_KHASTAN  → فعل اصلیِ خواستن: خواهد که برود / این را خواهد
#   PARTICLE_KHAH    → ادات چه/یا:        خواه بیاید خواه نیاید
#   NOMINAL_DERIV    → مشتق اسمی/صفتی:   خواهش / خواهان / خواهنده
#   INDEPENDENT_WORD → واژه مستقل:        خواهر / خواهران
#   SUFFIX_COMPOUND  → پسوند ترکیبی:     آزادی‌خواه / خیرخواه / دادخواه
# ═══════════════════════════════════════════════════════════════════════════════

_FUTURE_AUX_FORMS: frozenset[str] = frozenset({
    "خواهم",  "خواهی",  "خواهد",  "خواهیم",  "خواهید",  "خواهند",
    "نخواهم", "نخواهی", "نخواهد", "نخواهیم", "نخواهید", "نخواهند",
})

_PARTICLE_KHAH_FORMS: frozenset[str] = frozenset({
    "خواه", "خواه\u200cناخواه", "خواهناخواه",
})

_NOMINAL_KHAH_DERIVATIVES: frozenset[str] = frozenset({
    "خواهش", "خواهشمند", "خواهشمندانه", "خواهان", "خواهنده",
})

_INDEPENDENT_KHAH_WORDS: frozenset[str] = frozenset({
    "خواهر", "خواهران",
})

# mini-lexicon: افعال گذشته‌ای که در ساخت آینده بعد از خواه* می‌آیند
_FUTURE_MAIN_VERB_STEMS: frozenset[str] = frozenset({
    "رفت", "کرد", "شد", "داد", "گفت", "آمد", "خواند", "نوشت",
    "دید", "گرفت", "پذیرفت", "ساخت", "برد", "خورد", "زد",
    "افتاد", "ماند", "بست", "ریخت", "فروخت", "خرید", "شکست",
    "بود", "توانست", "خواست", "دانست", "پرسید", "فهمید",
    "کشت", "سوخت", "آموخت", "یافت", "باخت", "انداخت",
    "نشست", "برخاست", "پرداخت", "شناخت", "فرستاد", "برگشت",
})

# tag های مصنوعی — پیشوند _ برای تمایز از tag های واقعی Parsivar
_TAG_FUTURE_AUX    = "_FUTURE_AUX"
_TAG_NON_VERB_KHAH = "_NON_VERB_KHAH"


def _is_suffix_compound_khah(token: str) -> bool:
    """آیا token یک ترکیب پسوندی ـخواه است؟ مثل آزادی‌خواه، خیرخواه"""
    stripped = token.replace("\u200c", "")
    return (
        len(stripped) > 4
        and stripped.endswith("خواه")
        and token not in _PARTICLE_KHAH_FORMS
    )


def _classify_khah(tokens: list[str], i: int) -> str:
    """
    توکن tokens[i] را در context جمله classify می‌کند.

    خروجی یکی از:
        FUTURE_AUX | LEXICAL_KHASTAN | PARTICLE_KHAH |
        NOMINAL_DERIVATIVE | INDEPENDENT_WORD | SUFFIX_COMPOUND | OTHER

    سه‌لایه:
      لایه ۱ — استثناهای واژگانی دقیق
      لایه ۲ — ترکیب‌های پسوندی ـخواه
      لایه ۳ — صورت‌های آینده + پنجره context
    """
    tok       = tokens[i]
    next_tok  = tokens[i + 1] if i + 1 < len(tokens) else None
    next2_tok = tokens[i + 2] if i + 2 < len(tokens) else None

    # لایه ۱: استثناهای واژگانی
    if tok in _PARTICLE_KHAH_FORMS:
        return "PARTICLE_KHAH"
    if tok in _NOMINAL_KHAH_DERIVATIVES:
        return "NOMINAL_DERIVATIVE"
    if tok in _INDEPENDENT_KHAH_WORDS or tok.startswith("خواهر"):
        return "INDEPENDENT_WORD"

    # لایه ۲: ترکیب‌های پسوندی
    if _is_suffix_compound_khah(tok):
        return "SUFFIX_COMPOUND"

    # لایه ۳: صورت‌های آینده + context window
    if tok in _FUTURE_AUX_FORMS:
        if next_tok == "که":
            return "LEXICAL_KHASTAN"
        if next_tok in _FUTURE_MAIN_VERB_STEMS:
            return "FUTURE_AUX"
        if next2_tok in _FUTURE_MAIN_VERB_STEMS:
            return "FUTURE_AUX"
        # fallback محافظه‌کارانه: overcount بدتر از undercount است
        return "LEXICAL_KHASTAN"

    return "OTHER"


def _annotate_khah_tokens(
    tagged_words: list[tuple[str, str | None]],
) -> list[tuple[str, str | None]]:
    """
    annotation step: توکن‌های خواه را قبل از حلقه هجاشماری classify می‌کند.
    tag مصنوعی می‌گذارد تا count_syllables بتواند تصمیم context-aware بگیرد.
    """
    words  = [w for w, _ in tagged_words]
    result = list(tagged_words)

    for i, (tok, tag) in enumerate(tagged_words):
        if "خواه" not in tok:
            continue
        cls = _classify_khah(words, i)
        if cls == "FUTURE_AUX":
            result[i] = (tok, _TAG_FUTURE_AUX)
        elif cls in ("PARTICLE_KHAH", "NOMINAL_DERIVATIVE",
                     "INDEPENDENT_WORD", "SUFFIX_COMPOUND"):
            result[i] = (tok, _TAG_NON_VERB_KHAH)
        # LEXICAL_KHASTAN و OTHER → tag اصلی حفظ می‌شود

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# هجاشماری
# ═══════════════════════════════════════════════════════════════════════════════

def _count_en_syllables(word: str) -> int:
    w = word.lower().strip(".,!?;:\"'()[]{}")
    if not w:
        return 1
    count = len(re.findall(r"[aeiou]+", w))
    if w.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)


def _count_fa_syllables_base(word: str) -> int:
    """
    Heuristic ساده برای شمارش هجا در فارسی (~75% دقت).
    محدودیت شناخته‌شده: هجاهای بدون واکه نوشتاری از دست می‌روند.
    """
    if not word:
        return 0

    syllables = 0
    n = len(word)

    for i, ch in enumerate(word):
        if ch == "ا":
            syllables += 1
        elif ch in ("و", "ی"):
            if i > 0:
                syllables += 1
        elif ch in _FA_DIACRITICS:
            syllables += 1
        elif ch == "ه" and i == n - 1 and n > 1:
            if word[i - 1] not in _FA_LONG_VOWELS:
                syllables += 1

    return max(syllables, 1)


def _count_fa_syllables_pos(word: str, pos_tag: str) -> int:
    """
    هجاشماری فارسی با اطلاعات POS + tag های مصنوعی classifier خواه.
    """
    base = _count_fa_syllables_base(word)

    # tag های مصنوعی از _annotate_khah_tokens
    if pos_tag == _TAG_NON_VERB_KHAH:
        return max(base, 1)  # خواهش/خواهر/آزادی‌خواه — فعل نیست
    if pos_tag == _TAG_FUTURE_AUX:
        pass  # خواهم/خواهد — base هجاشماری درستی دارد (خا+هم = 2)

    # فعل‌های پیوسته با می/نمی
    if pos_tag in _VERB_TAGS_PARSIVAR and _VERB_PREFIX_ATTACHED.match(word):
        base += 1

    # پسوندهای تفضیلی/عالی
    if word.endswith("ترین") and len(word) > 4:
        base = max(base, _count_fa_syllables_base(word[:-4]) + 2)
    elif word.endswith("تر") and len(word) > 2:
        base = max(base, _count_fa_syllables_base(word[:-2]) + 1)

    return max(base, 1)


def count_syllables(word: str, pos_tag: str | None = None) -> int:
    """dispatcher اصلی هجاشماری."""
    is_persian = any("\u0600" <= ch <= "\u06ff" for ch in word)
    if not is_persian:
        return _count_en_syllables(word)
    if pos_tag is not None:
        return _count_fa_syllables_pos(word, pos_tag)
    return _count_fa_syllables_base(word)


# ═══════════════════════════════════════════════════════════════════════════════
# توکن‌ها و حروف
# ═══════════════════════════════════════════════════════════════════════════════

def _strip_punctuation(token: str) -> str:
    """کاراکترهای غیرالفبایی را از لبه‌های توکن پاک می‌کند."""
    i, j = 0, len(token)
    while i < j and not token[i].isalpha():
        i += 1
    while j > i and not token[j - 1].isalpha():
        j -= 1
    return token[i:j]


def _is_word_token(token: str) -> bool:
    return any(ch.isalpha() for ch in token)


def count_letters(words: list[str]) -> int:
    return sum(ch.isalpha() for w in words for ch in w)


# ═══════════════════════════════════════════════════════════════════════════════
# Singleton‌ها
# ═══════════════════════════════════════════════════════════════════════════════

@lru_cache(maxsize=1)
def _get_normalizer() -> Normalizer:
    return Normalizer()


_parsivar_tagger       = None
_parsivar_tagger_ready = False
_parsivar_tokenizer       = None
_parsivar_tokenizer_ready = False


def _get_parsivar_tagger() -> "ParsivarPOSTagger | None":
    global _parsivar_tagger, _parsivar_tagger_ready
    if _parsivar_tagger_ready:
        return _parsivar_tagger
    _parsivar_tagger_ready = True
    if not _PARSIVAR_AVAILABLE:
        return None
    try:
        _parsivar_tagger = ParsivarPOSTagger(tagging_model="wapiti")
        logger.info("Parsivar POSTagger loaded successfully.")
    except Exception as exc:
        logger.warning("Parsivar POSTagger could not be loaded: %s", exc)
    return _parsivar_tagger


def _get_parsivar_tokenizer() -> "ParsivarTokenizer | None":
    global _parsivar_tokenizer, _parsivar_tokenizer_ready
    if _parsivar_tokenizer_ready:
        return _parsivar_tokenizer
    _parsivar_tokenizer_ready = True
    if not _PARSIVAR_AVAILABLE:
        return None
    try:
        _parsivar_tokenizer = ParsivarTokenizer()
        logger.info("Parsivar Tokenizer loaded successfully.")
    except Exception as exc:
        logger.warning("Parsivar Tokenizer could not be loaded: %s", exc)
    return _parsivar_tokenizer


# ═══════════════════════════════════════════════════════════════════════════════
# سطح‌بندی خوانایی
# ═══════════════════════════════════════════════════════════════════════════════

_READABILITY_LEVELS: list[tuple[int, str]] = [
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
    return _READABILITY_LEVELS[-1][1]


POS_MODE_PARSIVAR  = "POS-enhanced — Parsivar"
POS_MODE_HEURISTIC = "morphological heuristic (بدون POS)"


@dataclass
class ReadabilityResult:
    sentences: int
    words:     int
    letters:   int
    syllables: int
    asl:       float
    wl:        float
    asyl:      float
    flesch_dayani: float
    level:     str
    pos_mode:  str

    @property
    def pos_enhanced(self) -> bool:
        """Backward-compatible alias."""
        return self.pos_mode.startswith("POS-enhanced")


# ═══════════════════════════════════════════════════════════════════════════════
# Tagging pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def _tag_sentence_parsivar(sent, pv_tok, pv_tagger):
    tokens = pv_tok.tokenize_words(sent)
    word_tokens = [
        cleaned
        for t in tokens
        if _is_word_token(t)
        for cleaned in (_strip_punctuation(t),)
        if cleaned
    ]
    if not word_tokens:
        return []
    try:
        return pv_tagger.parse(word_tokens)
    except Exception as exc:
        logger.warning("Parsivar tagger failed, falling back to heuristic: %s", exc)
        return [(w, None) for w in word_tokens]


def _tag_sentence_heuristic(sent: str) -> list[tuple[str, str | None]]:
    result = []
    for t in word_tokenize(sent):
        if not _is_word_token(t):
            continue
        cleaned = _strip_punctuation(t)
        if cleaned:
            result.append((cleaned, None))
    return result


def _extract_tagged_words(sentences, pv_tok, pv_tagger):
    if pv_tagger is not None and pv_tok is not None:
        tagged = []
        for sent in sentences:
            tagged.extend(_tag_sentence_parsivar(sent, pv_tok, pv_tagger))
        return tagged, POS_MODE_PARSIVAR
    tagged = []
    for sent in sentences:
        tagged.extend(_tag_sentence_heuristic(sent))
    return tagged, POS_MODE_HEURISTIC


# ═══════════════════════════════════════════════════════════════════════════════
# محاسبه اصلی
# ═══════════════════════════════════════════════════════════════════════════════

def compute_flesch_dayani(text: str) -> ReadabilityResult:
    """
    شاخص خوانایی Flesch–Dayani برای متن فارسی.

    فرمول دیانی (۱۳۷۴):
        FDR = 262.835 − 0.846 × ASYL − 1.015 × ASL

    pipeline:
        normalize → sent_tokenize → tag → annotate_khah → syllable_count → score
    """
    normalizer = _get_normalizer()
    normalized = normalizer.normalize(text)

    raw_sentences = [
        s for s in sent_tokenize(normalized)
        if any(ch.isalpha() for ch in s)
    ]
    if not raw_sentences:
        raise ValueError("متن پس از نرمال‌سازی هیچ جمله‌ای ندارد.")

    tagged_words, pos_mode = _extract_tagged_words(
        raw_sentences,
        pv_tok=_get_parsivar_tokenizer(),
        pv_tagger=_get_parsivar_tagger(),
    )

    n_sentences = len(raw_sentences)
    n_words     = len(tagged_words)

    if n_words == 0:
        raise ValueError(
            "پس از پاک‌سازی علائم نشانه‌گذاری، هیچ کلمه‌ای در متن یافت نشد."
        )

    all_words = [w for w, _ in tagged_words]
    n_letters = count_letters(all_words)
    if n_letters == 0:
        raise ValueError("هیچ حرف الفبایی در متن یافت نشد.")

    if n_words < _MIN_WORDS_RELIABLE:
        logger.warning(
            "Text has only %d words; score may be unreliable (recommend >= %d).",
            n_words, _MIN_WORDS_RELIABLE,
        )

    # annotation step — context-aware خواه classification
    tagged_words = _annotate_khah_tokens(tagged_words)

    n_syllables = sum(count_syllables(w, tag) for w, tag in tagged_words)

    asl  = n_words / n_sentences
    wl   = n_letters / n_words
    asyl = n_syllables / n_words
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
        pos_mode=pos_mode,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Persian Flesch–Dayani readability index calculator"
    )
    g = parser.add_mutually_exclusive_group()
    g.add_argument("-f", "--file",  type=str, help="Path to a UTF-8 Persian text file")
    g.add_argument("-t", "--text",  type=str, help="Persian text to analyze")
    parser.add_argument("--plain",   action="store_true", help="Print raw score only")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.file:
        try:
            with open(args.file, "r", encoding="utf-8") as f:
                text = f.read()
        except OSError as e:
            print(f"خطا در خواندن فایل: {e}", file=sys.stderr)
            sys.exit(1)
    elif args.text:
        text = args.text
    else:
        if sys.stdin.isatty():
            print("در انتظار متن از stdin (Ctrl+D برای پایان)...", file=sys.stderr)
        text = sys.stdin.read()

    if not text or not text.strip():
        print("خطا: متن خالی است.", file=sys.stderr)
        sys.exit(1)

    try:
        result = compute_flesch_dayani(text)
    except ValueError as e:
        print(f"خطا: {e}", file=sys.stderr)
        sys.exit(1)

    if args.plain:
        print(f"{result.flesch_dayani:.2f}")
        return

    W = 54
    print("═" * W)
    print("  Persian Readability — Flesch–Dayani")
    print("═" * W)
    print(f"  جملات   : {result.sentences}")
    print(f"  کلمات   : {result.words}")
    print(f"  حروف    : {result.letters}")
    print(f"  هجاها   : {result.syllables}")
    print(f"  روش     : {result.pos_mode}")
    print("─" * W)
    print(f"  ASL  (کلمه/جمله)  : {result.asl:.2f}")
    print(f"  WL   (حرف/کلمه)   : {result.wl:.2f}")
    print(f"  ASYL (هجا/کلمه)   : {result.asyl:.2f}")
    print("─" * W)
    print(f"  امتیاز Flesch–Dayani : {result.flesch_dayani:.2f}")
    print(f"  سطح خوانایی         : {result.level}")
    print("═" * W)


if __name__ == "__main__":
    main()
