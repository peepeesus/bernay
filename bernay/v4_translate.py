"""
v4_translate — offline LOCAL translation (argostranslate / CTranslate2).

A foreign-language lander decomposes badly because the motif scorer, the KB
cue regex, and the demographic markers are all ENGLISH-trained. Translating
the copy to English first fixes awareness / archetype / desires / painpoints
wholesale. CTranslate2 is a C++ inference engine (NOT torch's generate loop),
so it can't hit the torch-generate SEGFAULT that killed the VLMs on this box,
and it's fully offline ($0, no API — unlike Gemini). Subprocess-isolated so
the model RAM is reclaimed and never coexists with bernay's resident model.

  to_english(text, region=...) -> English text, or "" if unavailable.

Language packages (~100MB each) download lazily on first use of a pair.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# v4_stats.detect_market region name -> language code
REGION_LANG = {"Sweden": "sv", "Germany": "de", "France": "fr",
               "Italy": "it", "Netherlands": "nl", "Denmark": "da",
               "Norway": "nb", "Spain": "es", "India": "hi",
               "Indonesia": "id", "Bangladesh": "bn", "Portugal": "pt",
               "Poland": "pl", "Turkey": "tr", "Vietnam": "vi",
               "Thailand": "th", "Russia": "ru", "Japan": "ja",
               "Korea": "ko", "China": "zh", "Arabia": "ar",
               "Greece": "el", "Israel": "he", "Tamil": "ta",
               "Telugu": "te", "Ukraine": "uk"}

# ---------------------------------------------------------------------------
# NLLB-200 — Meta's open-source "Google Translate equivalent": ONE model that
# translates 200 languages into English, so Bernay is not limited to a
# hand-maintained region table (argos needed a separate ~100MB package per
# PAIR, and only 9 regions were ever mapped — Indonesian/Bengali ads decomposed
# as noise because no route existed). Runs LOCAL/offline on CPU, no API, $0.
# distilled-600M is the quality/size sweet spot (~2.5GB) on this box.
# Subprocess-isolated like the rest, so its RAM is reclaimed on exit.
# ---------------------------------------------------------------------------
NLLB_MODEL = os.environ.get("V4_NLLB_MODEL",
                            "facebook/nllb-200-distilled-600M")

# ISO-639-1 (what detect_market/REGION_LANG speak) -> NLLB FLORES-200 code.
_NLLB_CODE = {
    "sv": "swe_Latn", "de": "deu_Latn", "fr": "fra_Latn", "it": "ita_Latn",
    "nl": "nld_Latn", "da": "dan_Latn", "nb": "nob_Latn", "es": "spa_Latn",
    "hi": "hin_Deva", "id": "ind_Latn", "bn": "ben_Beng", "pt": "por_Latn",
    "pl": "pol_Latn", "tr": "tur_Latn", "vi": "vie_Latn", "th": "tha_Thai",
    "ru": "rus_Cyrl", "ja": "jpn_Jpan", "ko": "kor_Hang", "zh": "zho_Hans",
    "ar": "arb_Arab", "uk": "ukr_Cyrl", "fa": "pes_Arab", "he": "heb_Hebr",
    "el": "ell_Grek", "ro": "ron_Latn", "hu": "hun_Latn", "cs": "ces_Latn",
    "fi": "fin_Latn", "ms": "zsm_Latn", "tl": "tgl_Latn", "ur": "urd_Arab",
    "ta": "tam_Taml", "te": "tel_Telu", "mr": "mar_Deva", "gu": "guj_Gujr",
    "sw": "swh_Latn", "bg": "bul_Cyrl", "sr": "srp_Cyrl", "hr": "hrv_Latn",
}


def _nllb(text, src):
    """Translate with NLLB-200 (200 languages -> English). Returns "" if the
    model or the language code is unavailable, so the caller can fall back."""
    code = _NLLB_CODE.get(src)
    if not code:
        return ""
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    import torch
    tok = AutoTokenizer.from_pretrained(NLLB_MODEL, src_lang=code)
    model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL)
    model.eval()
    eng_id = tok.convert_tokens_to_ids("eng_Latn")
    # NLLB is a 512-token seq2seq: translate SENTENCE-CHUNKS, not one blob, so
    # long ad copy isn't silently truncated to its first few lines.
    import re as _re
    parts, cur = [], ""
    for s in _re.split(r"(?<=[.!?。！？])\s+|\n+", text):
        s = s.strip()
        if not s:
            continue
        if len(cur) + len(s) + 1 <= 900:
            cur = (cur + " " + s).strip()
        else:
            parts.append(cur)
            cur = s[:900]
    if cur:
        parts.append(cur)
    out = []
    for chunk in parts[:24]:                    # bound worst-case VSL cost
        enc = tok(chunk, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            gen = model.generate(**enc, forced_bos_token_id=eng_id,
                                 max_new_tokens=512, num_beams=1)
        out.append(tok.batch_decode(gen, skip_special_tokens=True)[0])
    return " ".join(o for o in out if o).strip()


def _ensure_pkg(src):
    import argostranslate.package as P
    if any(p.from_code == src and p.to_code == "en"
           for p in P.get_installed_packages()):
        return True
    P.update_package_index()
    pkg = next((p for p in P.get_available_packages()
                if p.from_code == src and p.to_code == "en"), None)
    if pkg is None:
        return False
    P.install_from_path(pkg.download())
    return True


def _argos(text, src):
    if not _ensure_pkg(src):
        return ""
    import argostranslate.translate as T
    return T.translate(text, src, "en")


def _worker(text, src):
    """NLLB-200 first (200 languages); argos as the fallback (works offline
    once its per-pair package is cached, and covers a few codes NLLB lacks)."""
    if os.environ.get("V4_TRANSLATE_ENGINE", "nllb") == "nllb":
        try:
            eng = _nllb(text, src)
            if eng:
                return eng
        except Exception:  # noqa: BLE001
            pass
    try:
        return _argos(text, src)
    except Exception:  # noqa: BLE001
        return ""


def to_english(text, region=None, src=None, timeout=600):
    """Translate `text` to English. `region` is a v4_stats.detect_market name
    (mapped to a language code) or pass `src` directly. Returns "" on any
    failure so the caller keeps the original copy."""
    src = src or (REGION_LANG.get(region) if region else None)
    if not src or not (text or "").strip():
        return ""
    try:
        out = subprocess.run([sys.executable, os.path.abspath(__file__),
                              "--tr", src], input=text, capture_output=True,
                             text=True, encoding="utf-8", timeout=timeout)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    return ""


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--tr":
        import contextlib
        src = sys.argv[2]
        payload = sys.stdin.read()
        with contextlib.redirect_stdout(sys.stderr):  # argos/stanza noise
            eng = _worker(payload, src)
        sys.stdout.write(eng)
    else:
        sv = ("Mounjaslim Officiell webbplats. Svenska läkare slår larm om "
              "viktminskning. Det här är inte som andra piller och du behöver "
              "inte träna för att gå ner i vikt. Köp nu här.")
        print(to_english(sv, region="Sweden"))
