"""
v4_ocr_multi — multilingual OCR via EasyOCR, for NON-LATIN scripts the default
RapidOCR (ch/en) can't read (Devanagari/Hindi, etc.). EasyOCR is CRAFT
detection + CRNN/CTC recognition — feed-forward, NO autoregressive generate
loop, so it does NOT hit the torch-generate segfault that killed the VLMs on
this box. Subprocess-isolated (the torch models' RAM is reclaimed on exit and
never coexists with bernay's resident model).

Escalated from v4_vision ONLY when the fast default OCR returns text with
non-Latin characters (a script it mangled) — so the common English/European
path never pays EasyOCR's cost.

Language set via V4_OCR_MULTI_LANGS (default 'hi,en' — Hindi+English; EasyOCR
requires script-compatible languages in one reader). Set e.g. 'ar,en' for
Arabic, 'ru,en' for Cyrillic, 'th,en' for Thai.

  run(paths) -> recognized text (all frames, de-duped), or "" on failure.
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LANGS = [s.strip() for s in
         os.environ.get("V4_OCR_MULTI_LANGS", "hi,en").split(",") if s.strip()]


def _worker(paths):
    import easyocr
    reader = easyocr.Reader(LANGS, gpu=False, verbose=False)
    lines, seen = [], set()
    for p in paths:
        if not os.path.exists(p):
            continue
        for t in reader.readtext(p, detail=0, paragraph=True):
            t = (t or "").strip()
            if t and t not in seen:
                seen.add(t)
                lines.append(t)
    return "\n".join(lines)


def run(paths, timeout=600):
    real = [p for p in (paths or []) if p and os.path.exists(p)]
    if not real:
        return ""
    try:
        out = subprocess.run([sys.executable, os.path.abspath(__file__),
                              "--ocr", *real], capture_output=True,
                             text=True, encoding="utf-8", timeout=timeout)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    return ""


if __name__ == "__main__":
    import contextlib
    if len(sys.argv) > 1 and sys.argv[1] == "--ocr":
        with contextlib.redirect_stdout(sys.stderr):   # easyocr progress noise
            txt = _worker(sys.argv[2:])
        sys.stdout.write(txt)
    else:
        print(run(sys.argv[1:]))
