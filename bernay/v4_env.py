"""
Load Downloads/.env into os.environ — the missing plumbing that made the
Gemini reasoning path permanently dark.

The hard rule is "secrets live in .env ONLY", but nothing actually loaded
.env into the environment: gethookd/meta parse the file themselves, while
the VISION path (`v4_vision.backend`, `v4_gemini._api_key`) reads
`os.environ['GEMINI_API_KEY']`. So a key sitting in .env never reached it
and every media ad silently fell back to local OCR + regex.

`load()` parses `.env` (KEY=VALUE, '#' comments, optional surrounding
quotes/spaces) and sets each var with `setdefault` — a real environment
variable always WINS, so nothing here can clobber an explicit override.
Idempotent and import-safe (never raises).
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
_ENV = os.path.join(HERE, ".env")
_LOADED = False


def load(path=_ENV):
    """Populate os.environ from a .env file (setdefault). Returns the set of
    keys it provided. No-op after the first successful load."""
    global _LOADED
    keys = set()
    if _LOADED or not os.path.exists(path):
        return keys
    try:
        with open(path, encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                if key.startswith(("#", "export ")):
                    key = key.lstrip("#").replace("export ", "").strip()
                val = val.strip().strip('"').strip("'")
                if key and val:
                    os.environ.setdefault(key, val)
                    keys.add(key)
    except Exception:  # noqa: BLE001 — never let env loading break a run
        pass
    _LOADED = True
    return keys


# load on import so a simple `import v4_env` is enough
load()
