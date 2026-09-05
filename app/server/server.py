"""
Bernay desktop app — backend service.

FastAPI wrapper around the live Schwartz stack (same venv + env as
`bernay.cmd`). The model pipeline is NOT modified here: this module only
routes inputs exactly like the `bernay` REPL (text | media/URL ->
v4_media.ingest_structured -> v4_admix.analyze) and exposes the structured
result over localhost HTTP for the desktop UI.

Hard rails honored (BERNAY.md):
  - stack loads ONCE at startup (background thread, /api/health reports it)
  - NO external LLMs in the model path (Gemini stays input-side, inside
    the existing v4_distill route)
  - secrets stay wherever the existing modules read them (.env)
  - every analysis appends to logs/gpt2_sessions.log like the REPL does
    (except in a sandbox instance — see BERNAY_SANDBOX below)

Run:  run_server.cmd   (uvicorn on 127.0.0.1:8756, localhost only)
      sandbox.py       (throwaway instance on a free port, for test loops)
"""
import os
import re
import sys

# --- where the code and data live -----------------------------------------
# Both roots are resolved, never hardcoded, so a clone runs anywhere:
#   BERNAY_ROOT  the restructured package tree this server imports from
#                (communities_*/). Default: a sibling "Bernay" next to the flat
#                source tree, which is the layout this was developed in.
#   BERNAY_SRC   the flat v4_*.py tree. Default: the parent of bernay-app/,
#                derived from this file, so moving the checkout moves both.
# Set either to relocate; nothing else in the file needs changing.
_HERE = os.path.dirname(os.path.abspath(__file__))            # .../server
_APP = os.path.dirname(_HERE)                                 # .../<app dir>
_PARENT = os.path.dirname(_APP)                               # checkout root


def _find_src():
    """The tree holding the model modules.

    In the distribution the app sits beside the model as `<repo>/app` and
    `<repo>/bernay`; in the development layout the app lives INSIDE the flat
    tree. Identify the model by a marker file rather than by directory name,
    so neither layout needs configuring."""
    for cand in (os.path.join(_PARENT, "bernay"), _PARENT, _APP):
        if os.path.exists(os.path.join(cand, "v4_admix.py")):
            return cand
    return _PARENT


BERNAY_SRC = os.environ.get("BERNAY_SRC", _find_src())
BERNAY_ROOT = os.environ.get(
    "BERNAY_ROOT", os.path.join(os.path.dirname(_PARENT), "Bernay"))
# Checkpoints live with whichever tree actually has them.
_CKPTS = (os.path.join(BERNAY_ROOT, "checkpoints")
          if os.path.isdir(os.path.join(BERNAY_ROOT, "checkpoints"))
          else os.path.join(BERNAY_SRC, "checkpoints"))

# --- instance identity ----------------------------------------------------
# One box can host several Bernay servers at once: the real app on the default
# port, plus throwaway sandbox instances driven by test loops. Only WRITTEN
# state is per-instance — checkpoints and the sourced KB are large and
# read-only, so every instance shares them.
PORT = int(os.environ.get("BERNAY_PORT", "8756"))
SANDBOX = re.sub(r"[^A-Za-z0-9_-]", "-", os.environ.get("BERNAY_SANDBOX", ""))

# --- environment BEFORE any model import (mirrors bernay.cmd: Schwartz-4) ---
os.environ.setdefault("V4_N_EMBD", "128")
os.environ.setdefault("V4_N_LAYER", "4")
os.environ.setdefault("V4_CKPT", os.path.join(_CKPTS, "v4_big_ckpt.pt"))
os.environ.setdefault("V4_MOTIF_CACHE",
                      os.path.join(_CKPTS, "v4_big_motif_cache.pt"))
os.environ.setdefault("V4_PV_CKPT", os.path.join(_CKPTS, "v4_big_pv_ckpt.pt"))
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("BERNAY_NO_OPEN", "1")   # never pop browsers from a service

# Both trees go on the path: the restructured one wins the import (see _model
# below) when it is present, and the flat one carries the distribution.
sys.path.insert(0, BERNAY_SRC)
if os.path.isdir(BERNAY_ROOT):
    sys.path.insert(0, BERNAY_ROOT)
# KB JSONs / logs resolve relative to the project root
os.chdir(BERNAY_ROOT if os.path.isdir(BERNAY_ROOT) else BERNAY_SRC)

# The model is importable in TWO layouts and this server must not care which:
# the restructured package tree (communities_*/, tried first so an install that
# has both keeps loading the live tree), or a flat directory of v4_*.py, which
# is how the model is distributed. Every model import below goes through
# _model(), never a bare package path.
#
# v4_quiet MUST precede any fastembed/HF import (silences loguru bleed).
import importlib  # noqa: E402


def _model(pkg, name):
    """Import module `name` from the communities package `pkg`, else flat."""
    try:
        return importlib.import_module("%s.%s" % (pkg, name))
    except ImportError:
        return importlib.import_module(name)


v4_quiet = _model("communities_70_79.community_74_v4_quiet",
                  "v4_quiet")  # noqa: F401

import base64               # noqa: E402
import contextlib          # noqa: E402
import datetime            # noqa: E402
import io                  # noqa: E402
import json                # noqa: E402
import queue               # noqa: E402
import re                  # noqa: E402
import subprocess          # noqa: E402 — starts the GPU image half (see below)
import threading           # noqa: E402
import time                # noqa: E402

from fastapi import FastAPI, HTTPException                    # noqa: E402
from fastapi.middleware.cors import CORSMiddleware            # noqa: E402
from fastapi.responses import StreamingResponse               # noqa: E402
from fastapi.staticfiles import StaticFiles                   # noqa: E402
from pydantic import BaseModel                                # noqa: E402

import requests                                                     # noqa: E402

# Langfuse is OPTIONAL tracing, not part of the model. A hard import made a
# third-party observability SaaS a requirement for the app to start at all.
try:
    from langfuse import observe, Langfuse                     # noqa: E402
except ImportError:                       # no tracing installed: no-op wrapper
    Langfuse = None

    def observe(*a, **kw):                                     # noqa: D401
        def _wrap(fn):
            return fn
        return _wrap(a[0]) if a and callable(a[0]) else _wrap

admix = _model("communities_00_09.community_9_v4_admix_v4_categorize_batch",
               "v4_admix")
v4_media = _model("communities_00_09.community_0_v4_media_v4_verify",
                  "v4_media")
v4_visualize = _model("communities_00_09.community_8_v4_visualize",
                      "v4_visualize")

# ---------------------------------------------------------------------------

_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

STATE = {"status": "loading", "stack": None, "error": None,
         "model": "?", "params": 0, "last_result": None,
         "last_maslow_png": None, "last_maslow_painpoint_png": None,
         "imagegen": "unknown",
         # Every analysis keeps its own renderable record, addressed by id.
         # /api/viz used to render ONLY the newest, so clicking an older entry
         # in the UI's History restored that result on the left while the deck
         # on the right still showed the newest ad — two different ads on
         # screen at once, with nothing saying so.
         "results": {}, "result_seq": 0}
RESULT_RING = 24          # how many analyses stay renderable
ANALYZE_LOCK = threading.Lock()   # one model, serialize inferences


def _model_label():
    n = int(os.environ.get("V4_N_EMBD", "48"))
    return {128: "Schwartz-4.5", 72: "Schwartz-3", 48: "Schwartz-2.5"}.get(
        n, f"Schwartz-{n}d")


# Bernay is ONE model, not a pile of services someone has to remember to start.
# The Maslow image-gen half lives in its own process only because it needs the
# GPU venv (torch/diffusers) which must not be imported into this CPU process —
# that is an implementation constraint, not a reason for the app to come up
# half-loaded. It used to fail SILENTLY (a log line, then a deck rendered with
# no avatar/painpoint image), which is how an entire arm of the architecture sat
# dark without anyone noticing. So the server now starts it as part of its own
# startup and reports its state in /api/health.
# The GPU service needs the OTHER interpreter (torch/diffusers), which this CPU
# process must never import. Point V4_MASLOW_PY at it; the default looks for a
# .venv-bench beside the source tree and otherwise falls back to this
# interpreter, so a single-venv install still starts (it will simply report the
# image arm as unavailable rather than failing to launch).
def _default_gpu_python():
    for cand in (os.path.join(BERNAY_SRC, ".venv-bench", "Scripts",
                              "python.exe"),
                 os.path.join(BERNAY_SRC, ".venv-bench", "bin", "python")):
        if os.path.exists(cand):
            return cand
    return sys.executable


MASLOW_VENV_PY = os.environ.get("V4_MASLOW_PY", _default_gpu_python())


def _find_maslow_script():
    """The image server lives in whichever tree actually has it — it is only
    present in the flat source tree today, which is exactly how this half of
    the architecture went missing. Resolve across both instead of assuming."""
    for root in (BERNAY_ROOT, BERNAY_SRC):
        p = os.path.join(root, "v4_maslow_server.py")
        if os.path.exists(p):
            return p
    return ""


MASLOW_SCRIPT = _find_maslow_script()


def _maslow_health(timeout=2.0):
    """Maslow's own report on BOTH its arms.

    That service now carries the image pipeline AND the language reader (the
    audience-and-market half), so 'is it up' is no longer one bit. Returns the
    service's dict, or {} when it cannot be reached."""
    try:
        r = requests.get(f"{MASLOW_IMAGEGEN_URL}/health", timeout=timeout)
        if r.status_code < 500:
            try:
                got = r.json()
                return got if isinstance(got, dict) else {"imagegen": "ready"}
            except ValueError:
                return {"imagegen": "ready"}
    except requests.RequestException:
        pass
    try:                       # older build with no /health route
        requests.get(MASLOW_IMAGEGEN_URL, timeout=timeout)
        return {"imagegen": "ready"}
    except requests.RequestException:
        return {}


def _imagegen_alive(timeout=2.0):
    return bool(_maslow_health(timeout))


def _start_imagegen_bg():
    """Bring the GPU image half of the model up with the rest of it."""
    if _imagegen_alive():
        STATE["imagegen"] = "ready"
        return
    if not (os.path.exists(MASLOW_VENV_PY) and os.path.exists(MASLOW_SCRIPT)):
        STATE["imagegen"] = "unavailable (missing .venv-bench or script)"
        print(f"[maslow] cannot start: {MASLOW_VENV_PY} / {MASLOW_SCRIPT}",
              flush=True)
        return
    try:
        STATE["imagegen"] = "starting"
        subprocess.Popen(
            [MASLOW_VENV_PY, MASLOW_SCRIPT],
            cwd=os.path.dirname(MASLOW_SCRIPT),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        for _ in range(60):          # SDXL takes a while to load its weights
            time.sleep(2)
            if _imagegen_alive():
                STATE["imagegen"] = "ready"
                print("[maslow] image service ready on "
                      f"{MASLOW_IMAGEGEN_URL}", flush=True)
                return
        STATE["imagegen"] = "timeout"
    except Exception as e:  # noqa: BLE001 — never block the CPU stack
        STATE["imagegen"] = f"error: {type(e).__name__}"


def _load_stack_bg():
    try:
        stack = admix.load_stack()
        STATE["params"] = sum(
            p.numel() for p in stack["scorer"].model.parameters())
        STATE["stack"] = stack
        STATE["model"] = _model_label()
        STATE["status"] = "ready"
    except Exception as e:  # noqa: BLE001
        STATE["status"] = "error"
        STATE["error"] = f"{type(e).__name__}: {e}"
    # the other half of the same model — started here so "ready" means the
    # WHOLE architecture is up, not just the parts this process imports.
    _start_imagegen_bg()


def _plain(x):
    """Recursively coerce to JSON-safe primitives (torch/numpy scalars -> py)."""
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()
                if not str(k).startswith("_")}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, (str, int, bool)) or x is None:
        return x
    if isinstance(x, float):
        return x
    try:
        return float(x)          # torch/numpy scalar
    except (TypeError, ValueError):
        return str(x)


def _session_log_path():
    """Where this instance's analyses land.

    The real app appends to the shared REPL log, as it always has. A sandbox
    instance must NOT: gpt2_sessions.log is read as a record of real
    analyses, and a test loop firing hundreds of synthetic ads through it
    would bury the real ones.
    """
    if SANDBOX:
        d = os.path.join(BERNAY_ROOT, "logs", "sandbox")
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, f"{SANDBOX}_sessions.log")
    return os.path.join(BERNAY_ROOT, "logs", "gpt2_sessions.log")


def _log_session(text):
    """Same diagnostic log the REPL writes — the app is just another surface."""
    try:
        with open(_session_log_path(), "a", encoding="utf-8") as logf:
            stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            logf.write(f"\n===== {stamp} | {len(text)} chars =====\n")
            logf.write(text + "\n")
    except OSError:
        pass


# painpoint name -> citations, best-effort from the sourced KB (display only).
# KB shape: {painpoints: [{id, name, ...}], correlations: {id: [record, ...]}}
_CITATIONS = {}
try:
    with open(os.path.join(BERNAY_ROOT, "data", "v4_correlations.json"),
              encoding="utf-8") as f:
        _kb = json.load(f)
    _id2name = {p.get("id"): p.get("name") for p in _kb.get("painpoints", [])
                if isinstance(p, dict)}
    for _pid, _recs in (_kb.get("correlations") or {}).items():
        _name = _id2name.get(_pid) or _pid
        _cits = [{
            "factor": _c.get("factor", ""),
            "magnitude": _c.get("magnitude", ""),
            "finding": _c.get("finding", ""),
            "source": _c.get("source_name", ""),
            "url": _c.get("source_url", ""),
        } for _c in (_recs or []) if isinstance(_c, dict)]
        if _name and _cits:
            _CITATIONS[str(_name).lower()] = _cits
except Exception:  # noqa: BLE001 — citations are a nicety, never fatal
    _CITATIONS = {}


@observe(name="bernay-desktop-analysis")
def run_analysis(source, progress=None, generate_maslow=False):
    """Route exactly like the REPL, return the normalized API payload.

    generate_maslow gates the Maslow image-gen step: Schwartz and Maslow are
    separate models the caller explicitly chose between, not a shared pipeline
    where analyzing under one silently also produces the other's output.
    Every call resets both cached images first, so a Schwartz-only run never
    leaves a PREVIOUS Maslow run's images sitting behind for /api/viz to
    render if the user switches tabs afterward."""
    def note(msg):
        if progress:
            progress(msg)

    src = (source or "").strip().strip('"')
    if not src:
        raise ValueError("empty input")

    brief, text, kind = None, src, None
    try:
        kind = v4_media.is_media(src)
    except Exception:  # noqa: BLE001
        kind = None
    if kind:
        note(f"ingesting {kind} (scrape / transcribe / distill)…")
        brief, text = v4_media.ingest_structured(src)

    _log_session(text)
    # Name the model the USER PICKED, not the one that happens to be loaded.
    # STATE["model"] is _model_label() — set once at startup from V4_N_EMBD and
    # always "Schwartz-4.5" — so choosing Maslow still reported "decomposing
    # with Schwartz-4.5…" while the badge two inches above it said Maslow. The
    # UI already resolves this correctly for the badge (App.tsx:187); the
    # progress line was the one place the lens was ignored. `generate_maslow`
    # is exactly "the user chose Maslow", so no new plumbing is needed.
    note("decomposing with "
         + ("Maslow" if generate_maslow else STATE["model"]) + "…")

    # MASLOW reads the copy with its own language model (v4_maslow_server
    # /read) and names the painpoint open-set, in the buyer's words; the KB
    # only grounds that name. SCHWARTZ gets no reader — its job is critiquing
    # the ad as built, against weights tuned for exactly that, so its path is
    # unchanged. Two models, two questions.
    reader = None
    if generate_maslow and _maslow_health(timeout=1.5).get("reader") == "ready":
        try:
            try:                       # restructured live tree
                from communities_00_09.community_9_v4_admix_v4_categorize_batch \
                    .v4_reason_loop import MaslowReader
            except ImportError:        # flat Downloads tree
                from v4_reason_loop import MaslowReader
            reader = MaslowReader(MASLOW_IMAGEGEN_URL)
            note("reading the copy (Maslow)…")
        except Exception:  # noqa: BLE001 — fall back to KB matching
            reader = None

    buf = io.StringIO()
    with ANALYZE_LOCK, contextlib.redirect_stdout(buf):
        result = admix.analyze(text, STATE["stack"], brief=brief,
                               return_result=True, reader=reader)
    pretty = _ANSI.sub("", buf.getvalue())
    # Maslow (v4_visualize) presents this SAME raw result dict — stash it so
    # GET /api/viz can render it without re-running analysis.
    STATE["last_result"] = result
    STATE["last_maslow_png"] = None
    STATE["last_maslow_painpoint_png"] = None
    # Abstention: analyze established nothing (see v4_admix._abstain). Do not
    # generate an avatar/painpoint visual — it would invent a buyer and a
    # problem out of the fallback strings in _maslow_decomp and present them
    # as what the model read.
    if result.get("insufficient_evidence"):
        generate_maslow = False
    if generate_maslow:
        note("generating avatar visual (Maslow)…")
        STATE["last_maslow_png"] = _generate_maslow_png(result, kind="avatar")
        note("generating painpoint visual (Maslow)…")
        STATE["last_maslow_painpoint_png"] = _generate_maslow_png(result, kind="painpoint")

    # Keep this analysis renderable by id, so /api/viz can present the ad the
    # user is actually looking at rather than whichever ran most recently.
    STATE["result_seq"] += 1
    rid = STATE["result_seq"]
    STATE["results"][rid] = {
        "result": result,
        "avatar": STATE["last_maslow_png"],
        "painpoint": STATE["last_maslow_painpoint_png"],
    }
    for old in sorted(STATE["results"])[:-RESULT_RING]:
        STATE["results"].pop(old, None)

    pains = result.get("painpoints") or []
    payload = {
        "model": STATE["model"],
        "input_kind": kind or "text",
        # Addresses this exact analysis in /api/viz. The UI keeps it per
        # history entry so restoring an old one restores its deck too.
        "result_id": rid,
        "pv": {
            "desire": result.get("dsr"),
            "t": result.get("T"),
            "problem_gain": result.get("gain"),
            "top_primal": result.get("top_primal"),
            "pv": result.get("PV"),
            # None (not 0.00) when the analysis abstained — a printed
            # "0.00dsr x 0.00T = 0.00 PV" is a measurement, and no
            # measurement was made.
            "equation_string": (
                f"{result['dsr']:.2f}dsr x {result['T']:.2f}T = "
                f"{result['PV']:.2f} PV"
                if result.get("PV") is not None else None),
        },
        "insufficient_evidence": bool(result.get("insufficient_evidence")),
        "abstain_reason": result.get("abstain_reason"),
        "awareness_spread": [
            {"stage": s, "share": w}
            for s, w in result.get("awareness_spread", [])],
        "awareness_journey": result.get("awareness_journey") or [],
        "sophistication_spread": [
            {"stage": s, "share": w}
            for s, w in result.get("sophistication_spread", [])],
        "archetype_angles": [
            {"id": n, "score": v} for n, v in result.get("angles", [])],
        "sophistication": [
            {"id": n, "score": v}
            for n, v in result.get("sophistication", [])],
        "psych_center": [
            {"id": n, "score": v}
            for n, v in result.get("psych_center", [])],
        "maslow_level": [
            {"id": n, "score": v}
            for n, v in result.get("maslow_level", [])],
        "desires": result.get("desires") or [],
        "problem": result.get("problem"),
        "painpoints": [
            {"name": p, "citations": _CITATIONS.get(str(p).lower(), [])}
            for p in pains],
        # The MECHANISM the ad blames ("DHT", "Cortisol"). Only /api/viz ever
        # saw this, because it renders the raw result dict — so the Maslow
        # deck showed the angle while the Schwartz view had no access to it
        # at all. NOT the same as `archetype_angles` (everyman/sage) above.
        "painpoint_angles": result.get("painpoint_angles") or [],
        "audience": {
            "age": result.get("age"),
            "gender": result.get("gender"),
            "income": result.get("income"),
            "income_by_age": result.get("income_by_age"),
            "life_stage": result.get("life_stage"),
            "ethnicity": result.get("ethnicity"),
            "presenter": result.get("presenter"),
        },
        "product": result.get("product"),
        "visual_category": result.get("visual_category"),
        "selling_beats": (brief or {}).get("selling_stages") or [],
        "win_prob": result.get("winner_prob"),
        "vision": _plain({
            "avatar": (brief or {}).get("avatar"),
            "accent_region": (brief or {}).get("accent_region"),
            "cta": (brief or {}).get("cta"),
            "core_desires": (brief or {}).get("core_desires"),
        }) if brief else None,
        "pretty_text": pretty,
        "text_analyzed_chars": len(text),
    }
    if Langfuse is not None:            # tracing is optional; see the import
        Langfuse().flush()
    return _plain(payload)


# ---------------------------------------------------------------------------

app = FastAPI(title="Bernay", docs_url=None, redoc_url=None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173",
                   f"http://localhost:{PORT}", f"http://127.0.0.1:{PORT}"],
    allow_methods=["*"], allow_headers=["*"],
)


@app.middleware("http")
async def _no_cache_shell(request, call_next):
    """Never let the browser/WebView2 pin a stale SPA shell across rebuilds.

    Hashed files under /assets/* are content-addressed (safe to cache
    forever); index.html is not, and the desktop shell's WebView2 profile
    persists across app restarts (unlike an ephemeral browser tab), so a
    rebuilt ui/dist could otherwise keep serving yesterday's JS bundle.
    """
    response = await call_next(request)
    # /api/viz renders STATE["last_result"], so its body changes on every
    # analysis while its URL does not. Without this the webview served the
    # first analysis's deck for the rest of the session — the user pastes a
    # new ad, the analysis runs, and the presentation never changes.
    if request.url.path in ("/", "/index.html", "/api/viz"):
        response.headers["Cache-Control"] = "no-store"
    return response


class AnalyzeBody(BaseModel):
    input: str
    model: str = "Schwartz-4.5"   # which lens was selected when Analyze ran


_IG_CACHE = {"at": 0.0, "value": None, "reader": "unknown"}


def _imagegen_status(ttl=5.0):
    """LIVE state of the GPU image half, not the value it had at boot.

    STATE["imagegen"] is written once by _start_imagegen_bg() and never
    revisited, so /api/health kept reporting "ready" long after the SDXL
    server had died — which is precisely the failure the field exists to make
    visible. Caught live: v4_maslow_server was gone (no process at all, VRAM
    taken by another job) while health still said ready, so the app would have
    rendered a Maslow deck with no generated images and no warning.

    Probed with a short TTL because the UI polls this endpoint; without it
    every poll would pay an HTTP round-trip to the image service.
    """
    if STATE.get("imagegen") in ("starting", "unavailable"):
        return STATE["imagegen"], "unknown"  # boot states are authoritative
    now = time.time()
    if now - _IG_CACHE["at"] > ttl:
        got = _maslow_health(timeout=1.5)
        _IG_CACHE["value"] = got.get("imagegen", "ready") if got else "down"
        # The reader is the OTHER arm of the same service and loads AFTER the
        # image pipeline, so it is routinely still loading while imagegen is
        # already ready. Report it separately or that gap is invisible.
        _IG_CACHE["reader"] = got.get("reader", "unknown") if got else "down"
        _IG_CACHE["at"] = now
    return _IG_CACHE["value"], _IG_CACHE.get("reader", "unknown")


@app.get("/api/health")
def health():
    imagegen, reader = _imagegen_status()
    return {"ok": STATE["status"] == "ready", "status": STATE["status"],
            "model": STATE["model"], "params": STATE["params"],
            "error": STATE["error"],
            # Both GPU arms of Maslow, so a dark one is VISIBLE instead of
            # silently rendering a deck with no generated images, or falling
            # back to KB matching while claiming the reader is in play.
            "imagegen": imagegen, "reader": reader}


@app.post("/api/analyze")
def analyze_endpoint(body: AnalyzeBody):
    if STATE["status"] == "loading":
        raise HTTPException(503, "model is still loading — poll /api/health")
    if STATE["status"] == "error":
        raise HTTPException(500, f"stack failed to load: {STATE['error']}")
    try:
        return run_analysis(body.input, generate_maslow=body.model == "Maslow")
    except ValueError as e:      # guiding errors (page URLs etc.) -> 400
        raise HTTPException(400, str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"analysis failed: {type(e).__name__}: {e}")


@app.get("/api/viz")
def viz_endpoint(format: str = "html",  # noqa: A002 — matches the wire contract
                 id: int | None = None):  # noqa: A002 — wire contract
    """Maslow — the visual-presentation model.

    format=html (default): self-contained slide deck (v4_visualize.to_html),
    same output the REPL's `/viz` command opens in a browser — rendered
    inline, since it's designed to auto-render.

    format=excalidraw: the editable board (v4_visualize.to_excalidraw), same
    JSON the REPL's `/viz --excalidraw` writes to a `.excalidraw` file.
    Excalidraw boards can't auto-render in a plain webview (per
    v4_visualize's own comment: "excalidraw needs excalidraw.com and can't
    auto-render") — served as a download instead, for the user to open at
    excalidraw.com or a local Excalidraw instance.
    """
    if STATE["last_result"] is None:
        raise HTTPException(404, "no analysis yet — run /api/analyze first")
    # `id` addresses ONE analysis. Without it (or once it has aged out of the
    # ring) fall back to the newest, which is the old behaviour.
    entry = STATE["results"].get(id) if id is not None else None
    if entry is None:
        entry = {"result": STATE["last_result"],
                 "avatar": STATE.get("last_maslow_png"),
                 "painpoint": STATE.get("last_maslow_painpoint_png")}
    if format == "excalidraw":
        from fastapi.responses import JSONResponse
        scene = v4_visualize.to_excalidraw(entry["result"])
        return JSONResponse(
            scene,
            headers={"Content-Disposition":
                     'attachment; filename="bernay_board.excalidraw"'},
        )
    if format != "html":
        raise HTTPException(400, f"unknown format {format!r} (html|excalidraw)")
    from fastapi.responses import HTMLResponse
    html = v4_visualize.to_html(entry["result"])

    def splice(html, marker, png, alt):
        # Splice an auto-generated image right into the card v4_visualize
        # already renders, so the presentation shows a generated visual with
        # no separate action — never a manually-triggered, bolted-on extra.
        if not png:
            return html
        b64 = base64.b64encode(png).decode("ascii")
        img_tag = (f'<img src="data:image/png;base64,{b64}" alt="{alt}" '
                   'style="width:100%;border-radius:6px;margin-top:10px;">')
        return html.replace(marker, marker + img_tag, 1)

    # This analysis's OWN images, not the newest run's.
    html = splice(html, "Avatar</h3>", entry.get("avatar"),
                 "Maslow-generated avatar")
    html = splice(html, "Painpoints</h3>", entry.get("painpoint"),
                 "Maslow-generated painpoint scene")
    return HTMLResponse(html)


MASLOW_IMAGEGEN_URL = os.environ.get("V4_MASLOW_URL", "http://127.0.0.1:8799")


def _maslow_decomp(result):
    """Map a just-completed analysis onto the {painpoint, avatar, product}
    shape v4_maslow_imagegen.build_prompt() expects — the SAME fields the
    avatar/painpoints cards in v4_visualize already read off `result`."""
    pains = result.get("painpoints") or []
    # 'unclear' is the demographic layer ABSTAINING, not a description of a
    # person — passing it through put the literal word into the SDXL prompt
    # ("unclear aged 35-44, promoting ..."). Drop it and let the prompt stay
    # generic, which is what an abstention actually means.
    _g = result.get("gender")
    _a = result.get("age")
    bits = [_g if _g and _g != "unclear" else None,
            f"aged {_a}" if _a and _a != "unclear" else None]
    avatar = " ".join(b for b in bits if b) or "adult"
    return {
        "painpoint": str(pains[0]) if pains else "joint pain arthritis",
        "avatar": avatar,
        "product": result.get("product") or "health supplement",
    }


def _generate_maslow_png(result, kind="avatar"):
    """Maslow — real image generation (SDXL-base + the DR-creative style
    LoRA) of what THIS analysis detected: kind="avatar" is a portrait of the
    buyer, kind="painpoint" is an editorial scene of the problem itself (see
    v4_maslow_imagegen.build_prompt / build_painpoint_prompt). Runs
    automatically as the last step of every analysis (see run_analysis) so
    the presentation (/api/viz) always shows both — never a manual,
    separately-triggered action. Proxies to the standalone GPU service
    (v4_maslow_server.py, .venv-bench) instead of importing torch/diffusers
    here, so this process stays the light CPU Schwartz stack.
    Never raises: a down image service should not fail the analysis, it
    should just mean the presentation renders without that image.
    """
    try:
        body = dict(_maslow_decomp(result), kind=kind)
        r = requests.post(f"{MASLOW_IMAGEGEN_URL}/generate", json=body, timeout=120)
        if r.status_code == 200:
            return r.content
        print(f"[maslow] image service returned {r.status_code}: "
              f"{r.text[:200]}", flush=True)
    except requests.RequestException as e:
        print(f"[maslow] image service not reachable at "
              f"{MASLOW_IMAGEGEN_URL} ({e}) — presentation will render "
              f"without the {kind} image", flush=True)
    return None


@app.get("/api/analyze/stream")
def analyze_stream(input: str, model: str = "Schwartz-4.5"):  # noqa: A002 — matches the wire contract
    """SSE: `progress` events while the pipeline runs, then `result`/`error`."""
    if STATE["status"] != "ready":
        raise HTTPException(503, "model not ready — poll /api/health")
    q = queue.Queue()

    def work():
        try:
            payload = run_analysis(input, progress=lambda m: q.put(
                ("progress", {"message": m})), generate_maslow=model == "Maslow")
            q.put(("result", payload))
        except Exception as e:  # noqa: BLE001
            q.put(("error", {"message": f"{type(e).__name__}: {e}"}))
        q.put(None)

    threading.Thread(target=work, daemon=True).start()

    def gen():
        while True:
            item = q.get()
            if item is None:
                break
            event, data = item
            yield (f"event: {event}\n"
                   f"data: {json.dumps(data, default=str)}\n\n")

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache"})


# serve the built UI when it exists (desktop shell loads http://127.0.0.1:8756/)
_UI_DIST = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                 "ui", "dist"))
if os.path.isdir(_UI_DIST):
    app.mount("/", StaticFiles(directory=_UI_DIST, html=True), name="ui")
else:
    from fastapi.responses import HTMLResponse

    @app.get("/", response_class=HTMLResponse)
    def _no_ui():
        return ("<body style='background:#0d0d12;color:#d8d8e0;"
                "font-family:Consolas,monospace;display:flex;height:96vh;"
                "align-items:center;justify-content:center'><div>"
                "<h2 style='color:#7f9cf5'>Bernay</h2>"
                "<p>API is up — the UI bundle isn't built yet "
                "(<code>ui/dist</code> missing).</p>"
                "<p>Try <code>GET /api/health</code> or "
                "<code>POST /api/analyze</code>.</p></div></body>")

# load the model in the background so /api/health answers immediately
threading.Thread(target=_load_stack_bg, daemon=True).start()
