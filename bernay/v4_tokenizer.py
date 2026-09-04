"""
V4 tokenizer — a FIXED 96-character vocabulary, independent of corpus content.

V1–V3 derived their vocab from `sorted(set(text))` of a single clean book.
V4's prime corpus is multi-source (Gutenberg, archive.org OCR scans,
sacred-texts HTML) and its inference inputs are ad-platform exports, so the
vocab must be a constant, not a function of whatever junk arrived today:

    VOCAB = "\\n" + printable ASCII 32..126   (96 chars, ids 0..95, fits uint8)

`normalize()` folds everything into that whitelist: NFKD decomposition,
diacritic stripping, smart-punctuation mapping, CRLF -> LF, tab -> space,
anything else -> space. `vocab_hash()` is stamped into every results JSON
and asserted at every checkpoint load so vocab drift is impossible.

No torch import here — callers convert the id lists themselves, which also
guarantees the BLAS thread caps are always set before torch loads.
"""

import hashlib
import unicodedata

VOCAB = "\n" + "".join(chr(i) for i in range(32, 127))
VOCAB_SIZE = len(VOCAB)
assert VOCAB_SIZE == 96

STOI = {c: i for i, c in enumerate(VOCAB)}
ITOS = {i: c for i, c in enumerate(VOCAB)}

# smart punctuation and common typographic marks -> ASCII equivalents
_PUNCT = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "‐": "-", "‑": "-", "‒": "-", "–": "-",
    "—": "-", "―": "-", "−": "-",
    "…": "...",
    " ": " ", " ": " ", " ": " ", " ": " ",
    " ": " ", " ": " ",
    "\r\n": "\n", "\r": "\n", "\t": " ",
    "×": "x", "÷": "/",
}


def normalize(text):
    """Fold arbitrary unicode text into the 96-char VOCAB whitelist."""
    for k, v in _PUNCT.items():
        text = text.replace(k, v)
    # NFKD splits accented chars into base + combining mark; drop the marks
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return "".join(c if c in STOI else " " for c in text)


def encode(s, normalized=False):
    """Text -> list of ids in [0, 95]. Set normalized=True to skip cleanup."""
    if not normalized:
        s = normalize(s)
    return [STOI[c] for c in s]


def decode(ids):
    return "".join(ITOS[int(i)] for i in ids)


def vocab_hash():
    """Short sha256 of the vocab — stamp into results, assert at ckpt load."""
    return hashlib.sha256(VOCAB.encode("utf-8")).hexdigest()[:16]


if __name__ == "__main__":
    print(f"VOCAB_SIZE = {VOCAB_SIZE}, vocab_hash = {vocab_hash()}")

    # round-trip on already-clean text
    clean = "In the beginning God created the heaven and the earth.\n"
    assert decode(encode(clean)) == clean

    # deliberately dirty input: smart quotes, em dash, accents, emoji, CRLF
    dirty = ("“Désire” — café ‘angle’"
             " \U0001f525 100×\r\nnext")
    out = decode(encode(dirty))
    print(f"dirty in : {dirty!r}")
    print(f"normalized: {out!r}")
    assert all(0 <= i < VOCAB_SIZE for i in encode(dirty))
    assert "Desire" in out and "cafe" in out and "100x" in out
    assert "\r" not in out

    # whitelist is total: every codepoint maps somewhere valid
    probe = "".join(chr(i) for i in range(0x20, 0x3000, 7))
    assert all(0 <= i < VOCAB_SIZE for i in encode(probe))

    print("self-test OK — round trip, dirty-input fold, totality all pass")
