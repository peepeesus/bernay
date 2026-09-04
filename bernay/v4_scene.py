"""
v4_scene — CLIP zero-shot SCENE tagging (onnxruntime via fastembed).

This is the scene-understanding capability the torch VLMs (SmolVLM/Florence-2)
couldn't deliver: they SEGFAULT in torch's autoregressive generate() on this
py3.14 CPU build. CLIP is a single forward pass (image tower + text tower, no
generation loop), so it CANNOT hit that segfault, runs on the same onnxruntime
stack as RapidOCR/insightface/fastembed, and is light enough for the tight RAM
here. Subprocess-isolated so its model RAM is reclaimed and never coexists with
bernay's resident model.

It reads what OCR (text) and insightface (faces) can't — the ACTIVITY, SETTING
and SUBJECT of the creative ("applying skincare", "exercising", "holding a
supplement bottle", "outdoors", "a pet dog"). It's zero-SHOT CLASSIFICATION
against a curated phrase vocabulary (not free-form captioning) — the robust
version that actually runs on this box; free-form captioners need a torch
generate loop (segfaults) or ~1GB+ to export (OOM at current free RAM).

  scene_tags(paths, topk=3, margin=0.02) -> [(tag, score)]
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
VIS_MODEL = "Qdrant/clip-ViT-B-32-vision"
TXT_MODEL = "Qdrant/clip-ViT-B-32-text"

# Curated marketing-relevant scene vocabulary: (short tag, CLIP prompt). CLIP
# scores a photo against each prompt; we keep the top few. Grouped by what the
# other channels (OCR=text, insightface=faces) genuinely miss.
SCENE_VOCAB = [
    # --- activity / product-in-use ---
    ("applying skincare", "a person applying skincare cream or lotion to their face"),
    ("applying makeup", "a person applying makeup or cosmetics"),
    ("exercising", "a person exercising or working out at a gym"),
    ("cooking", "a person cooking or preparing food in a kitchen"),
    ("eating", "a person eating food"),
    ("drinking", "a person drinking a beverage or shake"),
    ("taking a supplement", "a person taking a pill, capsule or supplement"),
    ("holding a product", "a hand holding a product bottle, jar or package"),
    ("sleeping", "a person sleeping or lying in bed"),
    ("cleaning", "a person cleaning a surface or doing chores"),
    ("hair care", "a person washing, brushing or styling their hair"),
    ("talking to camera", "a person talking directly to the camera, a selfie video"),
    ("showing results", "a before-and-after body or skin transformation"),
    # --- setting ---
    ("in a kitchen", "the inside of a kitchen"),
    ("in a bathroom", "the inside of a bathroom"),
    ("in a gym", "the inside of a gym or fitness studio"),
    ("outdoors", "an outdoor scene, nature, park or street"),
    ("in a bedroom", "a bedroom"),
    ("in an office", "an office or workplace"),
    # --- subject ---
    ("a baby or child", "a baby, toddler or young child"),
    ("an older adult", "an elderly or older adult person"),
    ("a couple", "a romantic couple together"),
    ("a group of people", "a group of friends or family together"),
    ("a pet dog or cat", "a pet dog or cat"),
    # --- product type shown ---
    ("skincare product", "a skincare or cosmetic product, bottle or jar"),
    ("supplement bottle", "a bottle of vitamins, supplements or pills"),
    ("food or drink product", "a food or drink product package"),
    ("a beauty device", "a beauty gadget or skincare device"),
    ("packaged product shot", "a product package on a plain studio background"),
    ("text-only graphic", "a graphic with only text and no people or scene"),
]


def _cos(a, b):
    import math
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-9
    nb = math.sqrt(sum(y * y for y in b)) or 1e-9
    return dot / (na * nb)


def _worker(paths, topk):
    from fastembed import ImageEmbedding, TextEmbedding
    vis = ImageEmbedding(model_name=VIS_MODEL)
    txt = TextEmbedding(model_name=TXT_MODEL)
    # average the image vector over up to 4 frames (robust for video creatives)
    ivecs = [list(map(float, v)) for v in vis.embed(paths[:4])]
    n = len(ivecs[0])
    img = [sum(v[i] for v in ivecs) / len(ivecs) for i in range(n)]
    pvecs = [list(map(float, v))
             for v in txt.embed([p for _, p in SCENE_VOCAB])]
    scored = [(SCENE_VOCAB[i][0], _cos(img, pv)) for i, pv in enumerate(pvecs)]
    scored.sort(key=lambda t: -t[1])
    return scored[:topk], scored


def scene_tags(paths, topk=3):
    """Top scene tags for the creative (subprocess; RAM reclaimed). Returns
    [(tag, score)] or [] if unavailable."""
    real = [p for p in (paths or []) if p and os.path.exists(p)]
    if not real:
        return []
    try:
        out = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--tag", str(topk),
             *real], capture_output=True, text=True, timeout=600)
        if out.returncode == 0 and out.stdout.strip():
            return [(t, round(s, 4)) for t, s in json.loads(out.stdout)]
    except Exception:  # noqa: BLE001
        pass
    return []


def caption(paths, topk=3):
    """A short human phrase from the top scene tags, e.g.
    'talking to camera; outdoors; an older adult'. '' if unavailable."""
    tags = scene_tags(paths, topk=topk)
    return "; ".join(t for t, _ in tags)


if __name__ == "__main__":
    import contextlib
    args = sys.argv[1:]
    if args and args[0] == "--tag":
        topk = int(args[1])
        paths = args[2:]
        with contextlib.redirect_stdout(sys.stderr):   # fastembed noise
            top, alls = _worker(paths, topk)
        sys.stdout.write(json.dumps(top))
    else:
        # self-test on any image paths passed
        for p in args:
            print(p, "->", scene_tags([p]))
