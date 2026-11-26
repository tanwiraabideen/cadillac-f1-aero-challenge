"""
Text normalisation, applied in exactly the same way to reference and hypothesis.

It works one word at a time and keeps each word's speaker and timing, so a
single word can become several tokens ("remote-control" -> "remote", "control")
or none ("." -> nothing), and every token still knows when it was said.
"""
import re

from num2words import num2words

import config
from words import Word

# Anything that isn't a letter, digit, apostrophe or space.
_PUNCT = re.compile(r"[^\w\s']")
_DIGITS = re.compile(r"\d+")


def normalise_text(text: str) -> list[str]:
    """One word's text -> zero or more scoring tokens. Order of steps matters."""
    t = text
    if config.LOWERCASE:
        t = t.lower()
    if config.HYPHEN_TO_SPACE:
        t = t.replace("-", " ")
    if config.DIGITS_TO_WORDS:
        t = _DIGITS.sub(lambda m: " " + num2words(int(m.group())).replace("-", " ") + " ", t)
        t = t.replace(",", " ")            # num2words(1200) = "one thousand, two hundred"
    if config.STRIP_PUNCTUATION:
        t = _PUNCT.sub(" ", t)
        # apostrophes are only kept *inside* words: "don't" stays, "'cause" -> "cause"
        t = " ".join(tok.strip("'") for tok in t.split())

    tokens = []
    for tok in t.split():
        tok = config.SPELLING_MAP.get(tok, tok)
        if not config.KEEP_FILLERS and tok in config.FILLERS:
            continue
        tokens.append(tok)
    return tokens


def normalise_words(words: list[Word]) -> list[Word]:
    """Normalise a whole transcript, keeping speaker and timing on every token."""
    out = []
    for w in words:
        for tok in normalise_text(w.text):
            out.append(Word(speaker=w.speaker, start=w.start, end=w.end, text=tok))
    return out
