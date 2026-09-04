"""
gpt2.5 media ingestion — turn a social-video URL (YouTube, TikTok,
Instagram, X/Twitter, Reddit, Facebook, … — anything yt-dlp/reclip can
fetch), a local video, or an image into the same [VISUAL]/[TRANSCRIPT] text
blob the user used to paste by hand, ready for v4_admix.analyze.

Speech-to-text is LOCAL whisper at all times:
  - audio -> transcript ALWAYS runs locally: ffmpeg extracts a 16k mono
    wav, then whisper.cpp (tools/whisper). We never inline audio to Gemini
    and never use Gemini for transcription (keeps STT free + offline, and
    costs no audio tokens). This includes YouTube — we transcribe the
    pulled audio with whisper rather than its caption track, so there is
    exactly one STT path.
  - visual analysis uses Gemini (v4_gemini): images directly, videos via
    ~6 evenly-spaced ffmpeg keyframes. Gemini sees frames + the whisper
    transcript text — never the raw audio.

The multi-site download breadth follows reclip (github.com/averygan/reclip),
a self-hosted yt-dlp + ffmpeg downloader; we call the same yt-dlp engine
inline so any site it supports works here too.

External binaries only (ffmpeg, whisper-cli, yt-dlp) — no torch/vision
Python deps, so it survives Python 3.14's missing wheels.
"""

import glob
import hashlib
import os
import re
import subprocess
import tempfile
import urllib.parse
import urllib.request

import v4_distill
import v4_vision
from v4_fetch_transcripts import video_id

_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

HERE = os.path.dirname(os.path.abspath(__file__))
WDIR = os.path.join(HERE, "tools", "whisper")
VIDEO_EXT = (".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v")
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif")

# Social video hosts to route through the shared yt-dlp downloader, the same
# breadth reclip (github.com/averygan/reclip) exposes. YouTube/TikTok have
# their own dedicated branches above this; these get the generic "social"
# treatment, with a graceful landing-page fallback when there's no video.
_SOCIAL_HOSTS = (
    "instagram.com", "twitter.com", "x.com", "reddit.com", "redd.it",
    "facebook.com", "fb.watch", "vimeo.com", "dailymotion.com",
    "twitch.tv", "streamable.com", "bilibili.com",
)


# Transcription model preference — most accurate first. Drop a bigger
# ggml-*.bin into tools/whisper to upgrade quality; it's picked up
# automatically, no code change (and deleting it falls back to the next).
_WHISPER_MODELS = (
    "ggml-large-v3.bin", "ggml-large-v3-turbo.bin", "ggml-large-v2.bin",
    "ggml-large.bin", "ggml-medium.bin", "ggml-medium.en.bin",
    "ggml-small.bin", "ggml-small.en.bin", "ggml-base.bin",
    "ggml-base.en.bin", "ggml-tiny.bin", "ggml-tiny.en.bin",
)


def _whisper_cli():
    """Back-compat: (cli, best_model) — best_model is just the first entry
    of _whisper_candidates(), kept for anything still calling this form."""
    cli, models = _whisper_candidates()
    return cli, (models[0] if models else None)


def _whisper_candidates():
    """-> (cli_path, [model_paths, best-first]). Existence + a >1MB size
    floor is only a SANITY check, not a completeness guarantee — a
    partially-downloaded or truncated .bin can sail past both and still
    fail to load (whisper-cli exits non-zero on a bad checkpoint). Caught
    live 2026-07-18: an in-progress ggml-large-v3-turbo.bin download got
    picked as 'best available' the instant it crossed 1MB, whisper-cli
    failed to load it, and the OLD code just returned the single picked
    model with no fallback — transcribe_audio() silently degraded to
    '(whisper produced no transcript)' on a real, fully-transcribable ad
    with no error surfaced anywhere. Returning the FULL ranked list lets
    the caller retry the next-best model instead of failing outright."""
    hits = glob.glob(os.path.join(WDIR, "**", "whisper-cli.exe"),
                     recursive=True)
    if not hits:
        return None, []
    ranked = [os.path.join(WDIR, name) for name in _WHISPER_MODELS]
    present = [p for p in ranked
              if os.path.exists(p) and os.path.getsize(p) > 1_000_000]
    # any other ggml model present (not in the known list) -> append,
    # largest-first, after the named candidates
    named = set(ranked)
    others = sorted(
        (p for p in glob.glob(os.path.join(WDIR, "ggml-*.bin"))
         if p not in named and os.path.getsize(p) > 1_000_000),
        key=os.path.getsize, reverse=True)
    return hits[0], present + others


def _run(cmd, timeout=600):
    return subprocess.run(cmd, capture_output=True, text=True,
                          timeout=timeout)


def _scratch(name, key=""):
    """A temp path that is UNIQUE PER SOURCE, and empty before we write it.

    Every artifact here used to have a FIXED name — v4_dl.mp4, v4_audio.wav,
    v4_frame_0.png — shared by every ad the app ever analysed. That is a
    cross-ad leak, and it bit exactly the way you would expect: the caller's
    only test that a step succeeded is `os.path.exists(...)`, which cannot
    tell "ffmpeg just wrote this" from "this survived from a different ad six
    days ago". So when a download returned a placeholder and frame extraction
    failed, keyframes() happily returned the PREVIOUS ad's frames and the
    whole vision read described the wrong ad — silently, with no error.
    Caught live 2026-09-02: v4_dl.mp4 was 49KB from that morning while
    v4_frame_0..5.png were six days old, and every newly pasted ad came back
    with the first ad's decomposition.

    Keying on the source makes collisions impossible; unlinking first makes
    `os.path.exists` mean what its callers already assume it means.
    """
    h = (hashlib.sha1(str(key).encode("utf-8", "replace")).hexdigest()[:12]
         if key else "none")
    stem, ext = os.path.splitext(name)
    out = os.path.join(tempfile.gettempdir(), f"{stem}_{h}{ext}")
    try:
        os.remove(out)
    except OSError:
        pass
    return out


def extract_wav(media_path):
    out = _scratch("v4_audio.wav", media_path)
    _run(["ffmpeg", "-y", "-i", media_path, "-ar", "16000", "-ac", "1",
          "-f", "wav", out])
    return out if os.path.exists(out) else None


def keyframes(media_path, n=6):
    """n evenly-spaced frames as PNGs (skips the very start/end)."""
    pr = _run(["ffprobe", "-v", "error", "-show_entries",
               "format=duration", "-of", "default=nw=1:nk=1", media_path])
    try:
        dur = float(pr.stdout.strip())
    except ValueError:
        dur = 0.0
    frames = []
    for i in range(n):
        ts = dur * (i + 0.5) / n if dur > 1 else 0
        # _scratch unlinks first, so a frame surviving from ANOTHER ad can
        # never be picked up here as if this call had produced it.
        fp = _scratch(f"v4_frame_{i}.png", media_path)
        _run(["ffmpeg", "-y", "-ss", f"{ts:.2f}", "-i", media_path,
              "-frames:v", "1", "-q:v", "3", fp])
        if os.path.exists(fp):
            frames.append(fp)
    return frames


def _is_degenerate(text, min_repeats=4):
    """True if `text` ends in a repetition loop — a REAL whisper.cpp failure
    mode (not hypothetical): caught live 2026-07-18 on a real ad, the
    large-v3-turbo model got stuck repeating 'The fatigue is still there.'
    TEN times in a row and then just stopped, silently truncating the
    transcript before the brand name and all four ingredient names it names
    later in the ad — a technically-successful (exit 0, non-empty) run that
    is nonetheless much WORSE than a smaller model's complete transcript.
    Detects any sentence repeated >=min_repeats times consecutively
    anywhere in the text (loops don't always land at the very end)."""
    import re
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    run_val, run_len = None, 0
    for s in sentences:
        if s == run_val:
            run_len += 1
            if run_len >= min_repeats:
                return True
        else:
            run_val, run_len = s, 1
    return False


_FASTER_WHISPER_MODEL = os.environ.get("V4_FASTER_WHISPER_MODEL", "small")
_FW = {}


def _faster_whisper_transcribe(wav, model_size=None):
    """faster-whisper (CTranslate2), a proper pip-installable Python STT
    engine — PRIMARY path, tried before shelling out to the whisper.cpp
    binary. Lazy/optional import: faster-whisper lives in .venv-vlm
    (py3.12) alongside torch/transformers, NOT under bare system py3.14,
    where this module's other functions are deliberately dependency-free
    (see the module docstring) so they still run. A missing/failed import
    here just means 'try whisper.cpp next', not a crash.

    Measured 2026-07-18 head-to-head on a real ad against whisper.cpp
    (v4_vision_model_search.md): faster-whisper 'small' produced a MORE
    complete transcript (3401 vs ~2900 chars) than whisper.cpp's OWN small
    model, in comparable time (32s), and neither faster-whisper size hit
    the repetition-loop failure whisper.cpp's turbo model did. 'medium' is
    slightly more accurate on obscure proper nouns (herb names) but ~3x
    slower for a ~4-char difference in total length -- not worth the
    default cost; still selectable via V4_FASTER_WHISPER_MODEL."""
    size = model_size or _FASTER_WHISPER_MODEL
    try:
        if _FW.get("size") != size:
            from faster_whisper import WhisperModel
            _FW["model"] = WhisperModel(size, device="cpu", compute_type="int8")
            _FW["size"] = size
        segments, _info = _FW["model"].transcribe(wav, beam_size=5)
        return " ".join(s.text.strip() for s in segments).strip()
    except Exception:  # noqa: BLE001 — not installed / model load failed / etc.
        return ""


def transcribe_audio(media_path):
    """ffmpeg -> 16k wav -> speech-to-text, LOCAL ONLY. Whisper at all times:
    we never fall back to Gemini for speech-to-text, so transcription stays
    free, offline, and costs no audio tokens.

    PRIMARY: faster-whisper (see _faster_whisper_transcribe). FALLBACK: the
    whisper.cpp binary, tried across EVERY available model, best-first,
    falling back to the next on a failed/empty/DEGENERATE run instead of
    giving up (or worse, silently keeping a bad result) — see
    _whisper_candidates() for the truncated-download failure mode and
    _is_degenerate() for the repetition-loop one. If everything degrades,
    returns the LONGEST candidate seen (a partial real transcript beats an
    empty one) rather than the generic failure message."""
    wav = extract_wav(media_path)
    if not wav:
        return "(audio extraction failed)"

    fw_text = _faster_whisper_transcribe(wav)
    if fw_text and not _is_degenerate(fw_text):
        return fw_text

    cli, models = _whisper_candidates()
    if not cli:
        return fw_text or (
            "(whisper not set up — put whisper-cli.exe + ggml-tiny.bin "
            "under tools/whisper, e.g. via v4_setup_media.py)")
    if not models:
        return fw_text or "(no whisper model found under tools/whisper)"
    last_err = ""
    best_degenerate = fw_text  # a degenerate faster-whisper run still counts
    for model in models:
        try:
            out = _run([cli, "-m", model, "-f", wav, "-nt", "-l", "auto"])
        except Exception as e:  # noqa: BLE001 — e.g. a timeout on this model
            last_err = f"{type(e).__name__}: {e}"
            continue
        if out.returncode != 0:
            last_err = f"exit {out.returncode} on {os.path.basename(model)}"
            continue                                # try the next model
        text = " ".join(line.strip() for line in out.stdout.splitlines()
                        if line.strip())
        if text and not _is_degenerate(text):
            return text                             # clean result, done
        if text:
            last_err = f"repetition loop from {os.path.basename(model)}"
            if len(text) > len(best_degenerate):
                best_degenerate = text               # keep as a fallback
            continue                                # try the next model
        last_err = f"empty output from {os.path.basename(model)}"
    if best_degenerate:
        return best_degenerate    # a partial/looping transcript beats none
    return f"(whisper produced no transcript — {last_err})"


def _combine(visual, transcript):
    parts = []
    if visual:
        parts.append("[VISUAL]\n" + visual.strip())
    if transcript:
        parts.append("[TRANSCRIPT]\n" + transcript.strip())
    return "\n\n".join(parts)


def ingest_image(path):
    return _combine(v4_vision.describe_images([path]), "")


def ingest_video(path):
    frames = keyframes(path)
    visual = (v4_vision.describe_images(frames) if frames
              else "(no frames extracted)")
    return _combine(visual, transcribe_audio(path))


def _ytdlp_fetch(url, stem, height=480):
    """Download a small copy of any yt-dlp-supported URL (YouTube, TikTok,
    …) to a temp file and return the local path, or None on failure. Shared
    by the YouTube and TikTok ingest paths so the download logic lives once.
    height caps the resolution to keep frame-sampling fast."""
    tmp = os.path.join(tempfile.gettempdir(), f"{stem}.mp4")
    try:
        import yt_dlp
        opts = {"format": f"mp4[height<={height}]/best[height<={height}]/best",
                "outtmpl": tmp, "quiet": True, "no_warnings": True,
                "noprogress": True, "overwrites": True}
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
    except Exception:  # noqa: BLE001  (yt-dlp missing, private/geo-locked, …)
        return None
    if os.path.exists(tmp):
        return tmp
    return next(iter(glob.glob(os.path.join(
        tempfile.gettempdir(), f"{stem}.*"))), None)


def _url_stem(url):
    """A stable-ish temp-file stem from a social-video URL — the numeric
    id when present (TikTok /video/, /photo/), else a sanitized tail of the
    path (handles vm./vt. and other short links, which yt-dlp resolves on
    download)."""
    m = re.search(r"/(?:video|photo|reel|p|status|watch)/(\d+)", url) \
        or re.search(r"/(?:video|photo)/(\d+)", url)
    if m:
        return m.group(1)
    tail = urllib.parse.urlparse(url).path.strip("/").split("/")[-1]
    return "".join(c for c in tail if c.isalnum())[-32:] or "clip"


def ingest_youtube(url):
    vid = video_id(url)
    # download a small copy, sample frames, transcribe locally with whisper
    dl = _ytdlp_fetch(url, f"v4_yt_{vid}", height=360)
    if not dl:
        return _combine("(video download failed)", "")
    frames = keyframes(dl)
    visual = v4_vision.describe_images(frames) if frames else "(no frames)"
    return _combine(visual, transcribe_audio(dl))   # whisper, always (local)


def ingest_tiktok(url):
    """TikTok has no caption API, so always pull the clip via yt-dlp and let
    whisper transcribe the audio (same shape as ingest_video)."""
    dl = _ytdlp_fetch(url, f"v4_tt_{_url_stem(url)}")
    if not dl:
        raise ValueError(
            "couldn't fetch that TikTok (it may be private/region-locked, "
            "or yt-dlp is out of date — try: pip install -U yt-dlp)")
    return ingest_video(dl)


def ingest_social(url):
    """Any reclip/yt-dlp-supported social video (Instagram, X, Reddit,
    Facebook, …) -> frames + local whisper transcript. If there's no
    downloadable video (e.g. a text-only post), fall back to the landing
    scrape so the link still yields something."""
    dl = _ytdlp_fetch(url, f"v4_sm_{_url_stem(url)}")
    if not dl:
        import v4_landing
        got = v4_landing.fetch_guarded(url)
        # A landing page already yields real page copy; only the Gemini
        # backend adds value by also narrating the screenshot. Local mode
        # relies on the page text (OCR of a full-page shot is weak/redundant).
        visual = (v4_vision.describe_images([got["screenshot"]])
                  if got.get("screenshot")
                  and v4_vision.backend() == "gemini" else "")
        return _combine(visual, got.get("text", ""))
    return ingest_video(dl)


def _url_ext(url):
    """Extension of a URL's path, ignoring the query string (fbcdn links
    carry the real extension in the path, then a long signed query)."""
    return os.path.splitext(urllib.parse.urlparse(url).path)[1].lower()


def is_media(source):
    # Windows "Copy as path" wraps the path in double quotes; drag-drop
    # sometimes adds single quotes. Strip a matching surrounding pair.
    s = source.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        s = s[1:-1].strip()
    low = s.lower()
    if low.startswith(("http://", "https://")):
        if "youtube.com" in low or "youtu.be" in low:
            return "youtube"
        if "tiktok.com" in low:
            return "tiktok"           # TikTok video — yt-dlp downloads it
        if "facebook.com/ads/" in low or "/ads/library" in low \
                or "/ads/archive" in low:
            return "adlib"            # Meta Ad Library page (Playwright)
        if "trendtrack.io" in low:
            return "trendtrack"       # TrendTrack share/ads page (Playwright)
        if "gethookd.ai" in low and "/share/board/" in low:
            return "gethookd_board"   # a BOARD of ads -> resolve via board API
        if "gethookd.ai" in low and "/share/ad/" in low:
            return "gethookd"         # gethookd share link -> resolve via get_ad
        if any(h in low for h in _SOCIAL_HOSTS):
            return "social"           # reclip/yt-dlp social video (IG, X, …)
        ext = _url_ext(s)
        if ext in VIDEO_EXT:
            return "video_url"        # direct CDN video (e.g. fbcdn .mp4)
        if ext in IMAGE_EXT:
            return "image_url"        # direct CDN image (Copy image addr)
        return "landing"              # any other web/landing page (Playwright)
    if os.path.exists(s) and low.endswith(VIDEO_EXT):
        return "video"
    if os.path.exists(s) and low.endswith(IMAGE_EXT):
        return "image"
    return None


def _clean_source(source):
    s = source.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        s = s[1:-1].strip()
    return s


# What each detected kind ACTUALLY does — so the REPL banner tells the truth.
# Video speech is transcribed LOCALLY by whisper (never inlined to Gemini);
# Gemini only ever sees frames + that transcript text. A landing page has no
# audio (page copy + a screenshot); an image is a single frame.
_KIND_DESC = {
    "youtube":    "video — frames + transcript (captions/whisper), distilling via Gemini (~30-90s)",
    "tiktok":     "TikTok video — yt-dlp download, frames + local whisper transcript, distilling via Gemini (~30-90s)",
    "social":     "social video (yt-dlp/reclip) — frames + local whisper transcript, distilling via Gemini (~30-90s)",
    "video":      "video — frames + local whisper transcript, distilling via Gemini (~30-90s)",
    "video_url":  "video — frames + local whisper transcript, distilling via Gemini (~30-90s)",
    "image":      "image frame — distilling via Gemini (~10-30s)",
    "image_url":  "image frame — distilling via Gemini (~10-30s)",
    "adlib":      "Meta ad — capturing the creative, distilling via Gemini (~30-90s)",
    "trendtrack": "TrendTrack ad — capturing the creative, distilling via Gemini (~30-90s)",
    "gethookd":   "gethookd ad — resolving the share link to the ad's real body "
                  "copy + transcript + landing page via the API (~2-5s, no model)",
    "gethookd_board": "gethookd BOARD — resolving the signed board API and "
                  "analysing its FIRST ad's copy + transcript (~2-5s, no model)",
    "landing":    "landing page — reading copy + screenshot, distilling via Gemini "
                  "(~20-45s, no audio)",
}


# Same kinds, LOCAL backend (no Gemini): on-screen text via OCR, page copy
# read directly, speech via local whisper — all free + offline.
_KIND_DESC_LOCAL = {
    "youtube":    "video — frames -> on-screen text (OCR) + local whisper transcript (~20-60s, free/offline)",
    "tiktok":     "TikTok video — yt-dlp download, frames -> on-screen text (OCR) + local whisper transcript (~20-60s, free)",
    "social":     "social video (yt-dlp/reclip) — frames -> on-screen text (OCR) + local whisper transcript (~20-60s, free)",
    "video":      "video — frames -> on-screen text (OCR) + local whisper transcript (~20-60s, free)",
    "video_url":  "video — frames -> on-screen text (OCR) + local whisper transcript (~20-60s, free)",
    "image":      "image — reading on-screen text (OCR) (~5-15s, free)",
    "image_url":  "image — reading on-screen text (OCR) (~5-15s, free)",
    "adlib":      "Meta ad — capturing the creative -> on-screen text (OCR) (~15-40s, free)",
    "trendtrack": "TrendTrack ad — capturing the creative -> on-screen text (OCR) (~15-40s, free)",
    "gethookd":   "gethookd ad — resolving the share link to the ad's real body "
                  "copy + transcript + landing page via the API (~2-5s, no model)",
    "gethookd_board": "gethookd BOARD — resolving the signed board API and "
                  "analysing its FIRST ad's copy + transcript (~2-5s, no model)",
    "landing":    "landing page — reading page copy directly (~10-30s, free, no model)",
}


def describe_kind(kind):
    """Human, HONEST one-liner for what ingest will do for `kind` — used by
    the REPL so it announces the real work per input type AND the active
    backend (local OCR/whisper vs Gemini)."""
    if v4_vision.backend() != "gemini":
        return _KIND_DESC_LOCAL.get(
            kind, "web page — reading copy locally (~15-40s, free)")
    return _KIND_DESC.get(kind, "web page — fetching + distilling via Gemini (~30-90s)")


def download_url(url, kind):
    """Download a direct media URL to a temp file with the right ext."""
    ext = _url_ext(url) or (".mp4" if kind == "video_url" else ".jpg")
    out = _scratch(f"v4_dl{ext}", url)
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=120) as r, \
            open(out, "wb") as f:
        f.write(r.read())
    return out


def _no_creative_message(got):
    """Why the capture yielded nothing, in the fetcher's own words.

    This used to be one fixed sentence blaming Meta's bot placeholder, which
    is only one of the reasons -- and it hid the most important one. An ad id
    that is not in the Library at all renders an empty-state page whose only
    scontent asset is the advertiser's 80x80 profile photo; the pipeline
    analysed that avatar as the creative and produced a full, confident
    decomposition of nothing. Say which failure actually happened."""
    if got.get("unavailable"):
        return (got.get("reason") or "this ad is not in the Meta Ad Library")\
            + ". Nothing was analysed. Check the id, or paste the ad's " \
              "creative (image/video file path) or its copy directly."
    reason = got.get("reason") or "no creative media was served"
    return (
        f"couldn't capture this ad's creative -- {reason}. Meta often serves "
        "a headless browser a placeholder instead of the real ad (it shows "
        "the creative to you because you're logged in). Save or screenshot "
        "the creative from your browser and paste the IMAGE FILE PATH here "
        "instead (e.g. C:\\Pictures\\ad.png) -- that reads the "
        "real pixels (on-screen text + the people in the image).")


def ingest_adlib(url):
    """Meta Ad Library page -> Playwright captures the creative -> the
    normal video/image pipeline, PLUS the ad's own displayed body copy
    (v4_adlib.fetch's body_text — captured from the page DOM, independent of
    OCR/CLIP) appended as its own tagged section. For a video/image-led ad
    whose on-screen text is sparse, this copy is often the only real
    targeting-language signal available."""
    import v4_adlib
    got = v4_adlib.fetch_guarded(url)
    body_text = (got.get("body_text") or "").strip()
    if got.get("video"):
        out = ingest_video(got["video"])
    elif got.get("image"):
        out = ingest_image(got["image"])
    elif got.get("screenshot"):
        out = _combine(v4_vision.describe_images([got["screenshot"]]),
                       "(no downloadable media found; described a "
                       "screenshot of the ad)")
    elif body_text:
        out = ""            # no creative at all, but we still have the copy
    else:
        raise ValueError(_no_creative_message(got))
    if body_text:
        out = (out + "\n\n" if out else "") + "[AD COPY]\n" + body_text
    return out


# ---- structured-brief path (Gemini distillation; the v4_distill route) -----
def _local_materials(path):
    """A local media file -> (frames, None, transcript). Images have no
    audio; a video's speech is always transcribed LOCALLY by whisper (the
    middle slot is kept as None — audio is never inlined to Gemini)."""
    if path.lower().endswith(IMAGE_EXT):
        return [path], None, None
    frames = keyframes(path)
    transcript = transcribe_audio(path)        # whisper, always (local + free)
    return frames, None, transcript


def _youtube_materials(url):
    # Whisper everywhere: transcribe the pulled audio locally rather than
    # using YouTube's caption track, so STT is one consistent path.
    vid = video_id(url)
    frames, transcript = [], None
    dl = _ytdlp_fetch(url, f"v4_yt_{vid}", height=360)
    if dl:
        frames = keyframes(dl)
        transcript = transcribe_audio(dl)      # whisper, always (local)
    return frames, None, transcript


def _tiktok_materials(url):
    """TikTok clip -> (frames, None, transcript). No caption API, so the
    audio is transcribed LOCALLY by whisper (never inlined to Gemini)."""
    dl = _ytdlp_fetch(url, f"v4_tt_{_url_stem(url)}")
    if not dl:
        raise ValueError(
            "couldn't fetch that TikTok (it may be private/region-locked, "
            "or yt-dlp is out of date — try: pip install -U yt-dlp)")
    frames = keyframes(dl)
    transcript = transcribe_audio(dl)          # whisper, always (local)
    return frames, None, transcript


def _social_materials(url):
    """A reclip/yt-dlp social URL -> (frames, None, transcript, meta).
    Whisper transcribes locally. If the link has no downloadable video
    (e.g. a text post), fall back to the landing scrape so it still yields
    copy + a screenshot."""
    dl = _ytdlp_fetch(url, f"v4_sm_{_url_stem(url)}")
    if not dl:
        return _landing_materials(url)         # (frames, None, text, title)
    frames = keyframes(dl)
    transcript = transcribe_audio(dl)          # whisper, always (local)
    return frames, None, transcript, None


def _page_materials(url, kind):
    """Ad Library / TrendTrack share page -> capture the creative via the
    matching Playwright fetcher, then resolve it like any local file.
    Returns (frames, wav, transcript, meta) where meta carries any
    scraped performance/'booming' signal."""
    if kind == "adlib":
        import v4_adlib
        got = v4_adlib.fetch_guarded(url)
    else:
        import v4_trendtrack
        got = v4_trendtrack.fetch_guarded(url)
    path = got.get("video") or got.get("image") or got.get("screenshot")
    if not path:
        raise ValueError(_no_creative_message(got))
    frames, wav, transcript = _local_materials(path)
    # got['body_text'] (adlib) is the ad's OWN displayed copy, captured from
    # the Ad Library page's DOM — independent of OCR/whisper/CLIP, and often
    # the ONLY real signal for a video/image-led creative whose on-screen
    # text is sparse or stylized (its actual targeting language, e.g. 'beer
    # belly'/'man boobs', lives in this copy, not on the video's pixels).
    # Folded into the transcript slot so it flows into the brief's
    # spoken_transcript -> ground_text -> painpoint/demographic matching the
    # same way a real transcript would.
    body_text = (got.get("body_text") or "").strip()
    if body_text:
        transcript = ("\n".join([transcript, body_text]) if transcript
                      else body_text)
    # got['landing_page'] (adlib) is the ad's real outbound destination — its
    # domain names the product, the signal the creative itself withholds.
    return frames, wav, transcript, got.get("meta"), got.get("landing_page")


def _landing_materials(url):
    """A web/landing/sales page -> (frames=[top-of-page shot], no audio,
    page copy as transcript, page title as meta) for v4_distill."""
    import v4_landing
    got = v4_landing.fetch_guarded(url)
    frames = [got["screenshot"]] if got.get("screenshot") else []
    return frames, None, (got.get("text") or None), got.get("title")


def _brief_from_gethookd_ad(ad):
    """Build a local (non-Gemini) brief from ONE gethookd ad object's real
    fields — title + body (the hook/copy), the whisper transcript, and the true
    destination landing page (its domain names the product). Shared by the
    single-ad (get_ad) and board (get-shared-board) resolvers."""
    if isinstance(ad, dict) and isinstance(ad.get("data"), dict):
        ad = ad["data"]
    title = (ad.get("title") or "").strip()
    body = (ad.get("body") or "").strip()
    tr = ad.get("transcripts")
    transcript = (" ".join(str(x) for x in tr if x) if isinstance(tr, list)
                  else str(tr or "")).strip()
    onscreen = "\n".join(x for x in (title, body) if x)
    brief = v4_distill._normalize({
        "onscreen_text": onscreen[:20000],
        "spoken_transcript": transcript[:8000],
        "product": "",
        "avatar": {},
        "_source_meta": "",      # never leak the source label as the product
        "_local": True,          # carries copy but no structured brief fields
    })
    landing = (ad.get("landing_page") or "").strip()
    if landing.lower().startswith(("http://", "https://")):
        brief["landing_page"] = landing   # domain -> brand (extract_product)
    return brief, v4_distill.brief_to_text(brief)


def _gethookd_brief(url):
    """A gethookd SHARE link (app.gethookd.ai/share/ad/<id>) is NOT the ad's
    landing page — scraping it yields gethookd's own app chrome ('Sign up',
    location, 'Unlock % off'), which poisons the decomposition. Instead resolve
    the ad id via the gethookd API (get_ad) and build the brief from the ad's
    REAL fields, so v4_admix keyword-scans the actual ad words."""
    import v4_gethookd
    m = re.search(r"/share/ad/(\d+)", url)
    aid = m.group(1) if m else None
    if not aid:
        raise ValueError("couldn't read the ad id out of that gethookd link")
    ad = v4_gethookd.Client().call("get_ad", {"ad_id": aid})
    if isinstance(ad, dict) and isinstance(ad.get("data"), dict):
        ad = ad["data"]
    if not isinstance(ad, dict) or not (ad.get("body") or ad.get("transcripts")):
        raise ValueError(f"gethookd get_ad({aid}) returned no usable ad copy")
    return _brief_from_gethookd_ad(ad)


def _gethookd_board_brief(url):
    """A gethookd BOARD link (app.gethookd.ai/share/board/<id>?signature=…) is a
    COLLECTION of ads, not one creative and NOT a landing page. Scraping it
    yields gethookd's JS SPA shell (no ad copy), which used to fall through to
    the 'landing' path and decompose into pure noise (a hair-regrowth board read
    as dog conditions). Resolve the board via its public SIGNED JSON API and
    analyse its FIRST usable ad — the app shows one result. The signature rides
    in the URL the user pasted, so no key is needed."""
    import json
    m = re.search(r"/share/board/(\d+)", url)
    bid = m.group(1) if m else None
    q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
    sig = (q.get("signature") or [""])[0]
    if not bid or not sig:
        raise ValueError("that gethookd board link is missing its id or "
                         "?signature= — copy the full share URL.")
    api = (f"https://app.gethookd.ai/api/get-shared-board/{bid}"
           f"?signature={urllib.parse.quote(sig)}&page=1&per_page=12")
    req = urllib.request.Request(
        api, headers={"User-Agent": _UA, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.loads(r.read().decode("utf-8", "replace"))
    if isinstance(data, dict) and data.get("errors"):
        raise ValueError(f"gethookd board {bid}: {data['errors']}")
    block = (data or {}).get("ads") or {}
    ads = block.get("data") or []
    ad = next((a for a in ads if isinstance(a, dict)
               and (a.get("body") or a.get("transcripts"))), None)
    if not ad:
        raise ValueError(f"gethookd board {bid} returned no usable ad copy "
                         "(is the share link still valid?)")
    return _brief_from_gethookd_ad(ad)


def ingest_structured(source):
    """The distillation route: resolve `source` to frames+audio, ask
    Gemini for a structured brief, and return (brief, readable_text).
    The text is the brief rendered for the session log + as a still-useful
    fallback feed to the motif scorer."""
    kind = is_media(source)
    s = _clean_source(source)
    meta = None
    landing = None
    if kind == "gethookd":
        return _gethookd_brief(s)     # structured ad from the API, no scrape
    if kind == "gethookd_board":
        return _gethookd_board_brief(s)   # board API -> first ad, no scrape
    if kind == "youtube":
        frames, wav, transcript = _youtube_materials(s)
    elif kind == "tiktok":
        frames, wav, transcript = _tiktok_materials(s)
    elif kind == "social":
        frames, wav, transcript, meta = _social_materials(s)
    elif kind in ("adlib", "trendtrack"):
        frames, wav, transcript, meta, landing = _page_materials(s, kind)
    elif kind == "landing":
        frames, wav, transcript, meta = _landing_materials(s)
    elif kind in ("video", "image"):
        frames, wav, transcript = _local_materials(s)
    elif kind in ("video_url", "image_url"):
        frames, wav, transcript = _local_materials(download_url(s, kind))
    elif s.lower().startswith(("http://", "https://")):
        raise ValueError(
            "that looks like a web PAGE, not a direct media link. For a "
            "Meta Ad Library ad, right-click the creative -> 'Copy image "
            "address' / 'Copy video address' and paste THAT (an fbcdn.net "
            "link), or save/screenshot it and give the file path.")
    else:
        raise ValueError(f"not a recognized media source: {source[:60]}")
    brief = v4_distill.distill(frames, audio_path=wav, transcript=transcript)
    if meta:
        brief["_source_meta"] = meta
    # A captured DESTINATION url (ad-library / share page) is the real product
    # signal — the advertiser's domain names the brand, the same ~95% path as a
    # pasted landing URL. Set it BEFORE the fallback below so the aggregator
    # source URL (facebook.com/ads/library/...) can't claim landing_page first.
    if isinstance(brief, dict) and landing:
        brief.setdefault("landing_page", landing)
    # The pasted URL is the strongest product signal — a sales page's domain IS
    # the brand (tryelvera.com -> Elvera). Thread it into the brief so
    # v4_admix.extract_product can read the brand off the domain. Harmless for
    # youtube/tiktok/cdn URLs (brand_from_url skips those hosts).
    if isinstance(brief, dict) and s.lower().startswith(("http://", "https://")):
        brief.setdefault("landing_page", s)
    return brief, v4_distill.brief_to_text(brief)


def ingest(source):
    kind = is_media(source)
    s = _clean_source(source)
    if kind == "youtube":
        return ingest_youtube(s)
    if kind == "tiktok":
        return ingest_tiktok(s)
    if kind == "social":
        return ingest_social(s)
    if kind == "adlib":
        return ingest_adlib(s)
    if kind == "video":
        return ingest_video(s)
    if kind == "image":
        return ingest_image(s)
    if kind == "video_url":
        return ingest_video(download_url(s, kind))
    if kind == "image_url":
        return ingest_image(download_url(s, kind))
    if kind in ("gethookd", "gethookd_board"):
        brief, text = (_gethookd_brief(s) if kind == "gethookd"
                       else _gethookd_board_brief(s))
        return text
    if kind == "landing":
        import v4_landing
        got = v4_landing.fetch_guarded(s)
        # A landing page already yields real page copy; only the Gemini
        # backend adds value by also narrating the screenshot. Local mode
        # relies on the page text (OCR of a full-page shot is weak/redundant).
        visual = (v4_vision.describe_images([got["screenshot"]])
                  if got.get("screenshot")
                  and v4_vision.backend() == "gemini" else "")
        return _combine(visual, got.get("text", ""))
    if s.lower().startswith(("http://", "https://")):
        raise ValueError(
            "that looks like a web PAGE, not a direct media link. For a "
            "Meta Ad Library ad, right-click the creative -> 'Copy image "
            "address' / 'Copy video address' and paste THAT (an fbcdn.net "
            "link), or save/screenshot it and give the file path.")
    raise ValueError(f"not a recognized media source: {source[:60]}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: python v4_media.py <image|video path | youtube url>")
        raise SystemExit(1)
    print(ingest(sys.argv[1]))
