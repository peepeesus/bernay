"""
v4_vision_cognition — the model's NATIVE EYES.

Until now Bernay was text-only: every "visual" signal (OCR text, scene tags,
the insightface gender guess) was flattened to TEXT and bolted onto the input.
The model never saw pixels. This module gives it real visual cognition,
SELF-CONTAINED and OFFLINE — local CLIP (fastembed, the same encoder v4_scene
already uses), NOT an external LLM, NEVER Gemini.

Two capabilities:
  image_vector(paths)        -> a 512-dim L2-normalized CLIP embedding of the
                                creative (mean-pooled over frames). This is the
                                feature that feeds the learned categorization
                                head (RecogHead) so a TRAINED model can read the
                                creative — the multimodal upgrade path.
  zeroshot(paths, prompts)   -> {prompt: similarity} via CLIP image<->text
                                cosine. Lets the model CATEGORIZE what it sees
                                with NO training data (works today), the same
                                contrastive trick v4_scene tags scenes with.

CLIP runs as a single forward pass (no autoregressive generate(), which
SEGFAULTS on this py3.14 CPU build — see v4_scene), so it is safe in-process.
"""

import os

import v4_quiet  # noqa: F401 — silence fastembed/HF logging before it loads

HERE = os.path.dirname(os.path.abspath(__file__))
_VIS_MODEL = "Qdrant/clip-ViT-B-32-vision"
_TXT_MODEL = "Qdrant/clip-ViT-B-32-text"
_M = {}


def _vis():
    if "vis" not in _M:
        from fastembed import ImageEmbedding
        _M["vis"] = ImageEmbedding(_VIS_MODEL)
    return _M["vis"]


def _txt():
    if "txt" not in _M:
        from fastembed import TextEmbedding
        _M["txt"] = TextEmbedding(_TXT_MODEL)
    return _M["txt"]


def available():
    try:
        import importlib.util
        return importlib.util.find_spec("fastembed") is not None
    except Exception:  # noqa: BLE001
        return False


def image_vector(paths):
    """Mean-pooled, L2-normalized 512-dim CLIP embedding over up to 4 frames.
    Returns a torch.FloatTensor[512], or None if it can't run / no images.
    THIS is the pixel-derived feature the learned head ingests."""
    import torch
    paths = [p for p in (paths or []) if p and os.path.exists(p)][:4]
    if not paths:
        return None
    try:
        import numpy as np
        vecs = [np.asarray(v, dtype="float32") for v in _vis().embed(paths)]
        if not vecs:
            return None
        v = torch.tensor(np.mean(vecs, axis=0))
        return v / (v.norm() + 1e-8)
    except Exception:  # noqa: BLE001
        return None


def zeroshot(paths, prompts):
    """CLIP zero-shot: cosine similarity of the creative to each text prompt.
    Returns {prompt: float} (higher = better match), or {} on failure. No
    training — the model categorizes what it sees by image<->text alignment."""
    iv = image_vector(paths)
    if iv is None or not prompts:
        return {}
    try:
        import numpy as np
        import torch
        tv = [np.asarray(v, dtype="float32") for v in _txt().embed(list(prompts))]
        out = {}
        for p, t in zip(prompts, tv):
            t = torch.tensor(t)
            t = t / (t.norm() + 1e-8)
            out[p] = float((iv * t).sum())
        return out
    except Exception:  # noqa: BLE001
        return {}


def classify(paths, options):
    """Pick the best-matching option for a category. `options` is
    {label: prompt}. Returns (best_label, score, all_scores) or (None, 0, {})."""
    scores = zeroshot(paths, list(options.values()))
    if not scores:
        return None, 0.0, {}
    inv = {v: k for k, v in options.items()}
    by_label = {inv[p]: s for p, s in scores.items()}
    best = max(by_label, key=by_label.get)
    return best, by_label[best], by_label


# Category prompt banks — what the model "asks" the image. Extend freely.
WHO_SHOWN = {                       # the PERSON depicted (presenter, not target)
    "woman": "a photo of a woman",
    "man": "a photo of a man",
    "older adult": "a photo of an older person, 55 or older",
    "young adult": "a photo of a young adult in their twenties",
    "no person": "a product photo with no people",
}
CREATIVE_KIND = {
    "product shot": "a clean product photo on a plain background",
    "ugc selfie": "a casual selfie-style user-generated ad video",
    "before/after": "a before and after comparison photo",
    "lifestyle": "a lifestyle photo of someone using a product",
    "text card": "an image that is mostly text on a colored background",
}


if __name__ == "__main__":
    import glob
    import sys
    imgs = sys.argv[1:] or sorted(glob.glob(os.path.join(HERE, "*.jpg")))[:1]
    print("available:", available())
    print("images   :", imgs)
    v = image_vector(imgs)
    print("image_vector:", None if v is None else tuple(v.shape),
          "(the pixel feature for the learned head)")
    who, s, alls = classify(imgs, WHO_SHOWN)
    print("who is shown (zero-shot):", who, round(s, 3),
          {k: round(x, 3) for k, x in sorted(alls.items(), key=lambda kv: -kv[1])})
    kind, s2, allk = classify(imgs, CREATIVE_KIND)
    print("creative kind (zero-shot):", kind, round(s2, 3))
