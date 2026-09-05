# Bernay

A from-scratch marketing-psychology model. It reads an ad — copy, creative, or
a landing page — and decomposes it into the things that decide whether the ad
works: the painpoint it sells against, the mechanism it blames, the awareness
stage it addresses, the buyer it implies, and a perceived-value score.

Bernay runs on a CPU. There is no API call in the inference path.

## What it produces

For one ad:

| field | what it is |
| --- | --- |
| painpoint | the problem the ad sells against |
| angle | the mechanism the ad blames for that problem |
| awareness | where the reader sits on the Schwartz ladder, as a journey |
| avatar | sex, age, life stage, income — each able to abstain |
| PV | perceived value, `PV = Desire × T` |

Every field can come back empty. Abstention is a designed output: an ad with no
evidence for an age returns `unclear` rather than a confident guess, and a
painpoint the knowledge base cannot ground is reported as ungrounded rather
than snapped to the nearest catalogued row.

## Architecture

```
             ┌─ copy ──────────────┐
 ad input ───┼─ creative frames ───┼──► perception ──┐
             └─ landing page ──────┘                 │
                                                     ▼
                                   ┌──────── decomposition ────────┐
                                   │  painpoint · angle · awareness │
                                   │  avatar · desires · PV         │
                                   └───────────────┬───────────────┘
                                                   ▼
                              READ → GROUND → CHECK → REVISE → EMIT
```

**Backbone** — `gpt2_pv_v4.py`, a character-level GPT-2 variant trained from
scratch on a domain corpus. `v4_tokenizer.py` is its tokenizer.

**PV head** — `v4_pv_engine.py`. `PV = Dsr × T`, where desire is a learned
function of the decomposition and `T` decays with how long the ad has been
running relative to its cohort.

**Heads** — condition (`v4_condition_head.py`), awareness recognition
(`v4_recognition_head.py`), visual product category (`v4_vision_head.py`),
motif scoring (`v4_motif_scorer.py`), winner probability
(`v4_winner_score.py`).

**Decomposition** — `v4_admix.py` is the entry point. `v4_correlations.py`
holds the knowledge-base interface (painpoints, mechanisms, domains),
`v4_demographics.py` the audience inference, `v4_semantic_painpoint.py` the
grounding search.

**Reasoning loop** — `v4_reason_loop.py`. An open-set read, grounded against
the knowledge base, checked by critics (evidence, grounding, species, role,
singularity), revised, and only then emitted. A read that fails its critics is
discarded rather than published.

**Perception** — `v4_vision.py` reads creatives offline: OCR for on-screen
text, CLIP/SigLIP zero-shot scene tags, and face detection for the people
shown. `v4_distill.py` turns frames plus transcript into a structured brief.

**Ingest** — `v4_media.py` routes an input to the right reader: a local image
or video (ffmpeg keyframes, whisper transcript), or a web page.
`v4_landing.py` renders a landing page in headless chromium, dismisses consent
walls, and returns the page copy plus viewport screenshots at successive
scroll offsets — deliberately viewport-sized, because a single tall capture
squashed for a face detector loses the people on it. The landing page is the
brand's own pitch to its buyer, so faces found there are treated as a stronger
signal of who buys than the creative's cautious read.

### Two design rules worth stating

**Open-set, closed-grounding.** The painpoint is read from the copy in open
vocabulary, then *grounded* against the knowledge base. The knowledge base is a
grounding table, not the vocabulary. An ad about something uncatalogued is
named honestly and reported as ungrounded.

**Who is shown is not who buys.** A face detected in the creative is recorded
as the on-screen presenter, deliberately not the target buyer — a young
demonstrator is not the audience for the product they are demonstrating.
Faces on the advertiser's own landing page are treated as the stronger signal,
because that is the brand depicting its own customer.

## Performance

Measured on a Ryzen 5 7500F (6c/12t), 32 GB DDR5-6000, **no GPU**:

```
import + model load          1.6 s + 3.0 s   one-time
short pasted ad (1.1 KB)     1.6 s
20 KB advertorial            11.7 s
resident memory              211 MB loaded, 278 MB peak
```

It is single-threaded: 2 threads and 6 threads measure the same, so it behaves
the same on modest hardware.

## What is and is not in this repository

**Included** — the architecture, the heads, the decomposition, the reasoning
loop, the perception path, the full ingest layer, the GPU service, and the
desktop app. 37 modules, ~617 KB of source. Every input the model accepts
works: pasted copy, a local image or video, a landing page, a Meta Ad Library
or TrendTrack link, a gethookd share link, YouTube, TikTok, or a direct CDN
media URL.

Verified from a clean clone in a fresh environment, with artifacts supplied
only through the variables below: 37/37 modules import, text decomposes, OCR
and face reads run on a real creative, a live landing page is rendered and
scraped, ffmpeg pulls keyframes, the reader returns an open-set painpoint, and
the app's server imports against the flat layout.

**Not included** — the trained weights and the sourced knowledge base. As
published this is the model's code, not its parameters; you can read exactly
how it decides, and train your own. This mirrors how open-weight releases ship
without their training data.

Also absent: the training and evaluation harnesses, and the regression corpus.

## Running it

`torch` is the only module-level dependency — everything else is imported at
the point of use, so you install exactly what your inputs need:

| for | install |
| --- | --- |
| **core** — import and decompose text | `torch` (brings numpy) |
| creatives — OCR, scene tags, faces | `rapidocr-onnxruntime fastembed opencv-python insightface pillow` |
| web pages | `playwright` then `playwright install chromium` |
| video | `yt-dlp faster-whisper`, plus `ffmpeg` on PATH |
| YouTube transcripts | `youtube-transcript-api` |
| non-English copy | `argostranslate easyocr` |
| the visual heads | `transformers open_clip_torch` |
| brand extraction | `wordfreq` |
| the desktop app | `fastapi uvicorn pywebview requests` |

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

```python
import v4_admix, v4_media

stack = v4_admix.load_stack()

# pasted ad copy
v4_admix.analyze(open("ad.txt").read(), stack)

# any URL or local file — ingest resolves it and returns the structured brief
brief, text = v4_media.ingest_structured("https://example.com/offer")
v4_admix.analyze(text, stack, brief=brief)
```

`v4_media.is_media()` routes the input: `landing`, `adlib`, `trendtrack`,
`gethookd`, `youtube`, `tiktok`, `video`/`image`, or a direct media URL. Each
fetcher is imported only when its path is taken, so you install browser and
video tooling only for the sources you actually use. Sources that need an API
key or a logged-in session read it from `.env`; none is bundled.

## The desktop app

`app/` is the native shell: a FastAPI server that wraps the model and a React
UI rendered in a real window (pywebview / Edge WebView2 — no Electron).

```bash
pip install fastapi uvicorn pywebview requests
cd app/ui && npm install && npm run build
cd .. && python desktop.py
```

`desktop.py` reuses an already-running server if it finds one and otherwise
boots its own, then shuts it down again when the window closes. The server
finds the model by marker file, so it works both as distributed
(`<repo>/app` beside `<repo>/bernay`) and in a checkout where the app sits
inside the model tree — no configuration either way.

`app/server/sandbox.py` starts throwaway instances on free ports so a test loop
never touches the real one on 8756. Langfuse tracing is optional: if the
package isn't installed, `observe` degrades to a no-op instead of refusing to
start.

## Maslow — the GPU half

Schwartz decomposes and critiques. Maslow reads the audience, and it is a
separate process on purpose: it needs CUDA, and the decomposition stack must
stay a light CPU process.

```bash
# in a SEPARATE environment with a CUDA build of torch
pip install transformers diffusers accelerate bitsandbytes peft fastapi uvicorn
python v4_maslow_server.py          # serves 127.0.0.1:8799
```

It carries two arms, both loading their weights from HuggingFace at first run —
nothing is bundled here:

- **the reader** — an open-set painpoint read. `v4_reason_loop.py` is the
  client: READ → GROUND → CHECK → REVISE, with critics that discard a read
  rather than publish one they can't defend. Configure with
  `V4_MASLOW_READER`, `V4_MASLOW_READER_QUANT` (`nf4` / `int8` / `bf16` /
  `off`), `V4_MASLOW_READER_ADAPTER`.
- **image generation** — `v4_maslow_imagegen.py` builds the avatar and
  painpoint scenes for a decomposition.

The app starts this service itself and reports both arms in `/api/health`, so a
dark GPU half is visible rather than silently producing a deck with no images.
Everything on the CPU side runs without it.

### What you have to supply

The taxonomy ships. The trained artifacts do not — `load_stack()` names every
one that is missing, with the variable that relocates it, rather than failing
on whichever file it happened to open first:

| artifact | variable | what it is |
| --- | --- | --- |
| `v4_taxonomy.json` | — | desire tags and principles; **ships with the code** |
| motif cache `.pt` | `V4_MOTIF_CACHE` | trained |
| PV engine `.pt` | `V4_PV_CKPT` | trained |
| backbone `.pt` | `V4_CKPT` | trained (`V4_N_EMBD`, `V4_N_LAYER` describe it) |
| knowledge base `.json` | `V4_KB` | the sourced painpoint/mechanism table |

Everything that does not need weights — ingest, landing pages, OCR, scene
tags, faces, the routing in `is_media()` — works without any of them.

Other paths resolve from the module location and can be overridden:

| variable | meaning |
| --- | --- |
| `BERNAY_SRC` | the source tree |
| `BERNAY_ROOT` | the data/checkpoint tree |
| `V4_VISION_BACKEND` | `local` (default), `florence`, `off` |

## License

Apache License 2.0 — see [LICENSE](LICENSE).
