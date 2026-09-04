"""
v4_vision_head — the model's NATIVE VISUAL CATEGORIZER.

Trains a small head on LOCAL CLIP image embeddings (v4_vision_cognition) of the
harvested gethookd creatives (v4_creatives.jsonl) so the model READS a creative
and names its product category from PIXELS — learned, not zero-shot, no Gemini.

Primary target = niche (product category), the dense label every harvested ad
has. (gender/age labels are too sparse/skewed to train here; they stay
marker/text-driven.) This is the proof that the model can natively categorize
what it sees; the same 512-dim image feature is what later augments RecogHead
for multimodal awareness/demographics once denser labels exist.

    python v4_vision_head.py --cv     # honest k-fold CV vs majority baseline
    python v4_vision_head.py          # train on all + save v4_vision_head.pt
"""

import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")

import v4_quiet  # noqa: F401 — silence fastembed/HF logging before it loads

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(HERE, "v4_creatives.jsonl")
# SigLIP-base replaced CLIP-ViT-B-32 as the production image encoder
# 2026-07-18 (v4_vision_model_search.md): honest same-methodology 5-fold CV
# on the current 586-creative/26-category set — CLIP 48.0%, SigLIP-base
# 60.8% (+12.8pp), DINOv2-base 33.3% (worse; self-supervised features lack
# the text-alignment that makes category naming work at all). SigLIP's own
# cache/checkpoint are kept under distinct names so the old CLIP artifacts
# aren't silently overwritten (rollback stays possible).
EMB_CACHE = os.path.join(HERE, "v4_creature_embeds_siglip_siglip_base_patch16_224.pt")
EMB_CACHE_CLIP = os.path.join(HERE, "v4_creature_embeds.pt")   # kept for reference/rollback
CKPT = os.path.join(HERE, "v4_vision_head.pt")


def _rows():
    return [json.loads(l) for l in open(MANIFEST, encoding="utf-8") if l.strip()]


def _embed_all(rows):
    """SigLIP-embed every creative once (768-dim), cached to disk by ad_id so
    re-runs are instant. Returns {ad_id: tensor[768]}. See the EMB_CACHE
    comment above for why this replaced the original CLIP embedder."""
    import v4_vision_siglip as VS
    cache = {}
    if os.path.exists(EMB_CACHE):
        cache = torch.load(EMB_CACHE, weights_only=False)
    todo = [r for r in rows if r["ad_id"] not in cache
            and os.path.exists(os.path.join(HERE, r["image"]))]
    if todo:
        print(f"  embedding {len(todo)} creatives via SigLIP "
              f"({VS._MODEL_NAME}) ...", flush=True)
        bad = 0
        for i, r in enumerate(todo):            # one at a time: a single
            p = os.path.join(HERE, r["image"])  # corrupt image can't kill the
            v = VS.image_vector([p])            # whole batch
            if v is None:
                bad += 1
                continue
            cache[r["ad_id"]] = v
            if (i + 1) % 50 == 0:
                torch.save(cache, EMB_CACHE)     # checkpoint (resume-safe)
        if bad:
            print(f"  skipped {bad} unreadable creative(s)", flush=True)
        torch.save(cache, EMB_CACHE)
    return cache


class VisHead(nn.Module):
    def __init__(self, d_in, n_cls):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, 64), nn.Tanh(),
                                 nn.Dropout(0.2), nn.Linear(64, n_cls))

    def forward(self, x):
        return self.net(x)


TXT_CACHE = os.path.join(HERE, "v4_creative_text_embeds.pt")


def _text_embed_all(rows):
    """CLIP TEXT embedding (512-dim) of each creative's copy (body+transcript),
    cached by ad_id. Same CLIP family as the image side, so image+text live in
    a comparable space for fusion. Self-contained, no LLM."""
    cache = torch.load(TXT_CACHE, weights_only=False) \
        if os.path.exists(TXT_CACHE) else {}
    todo = [r for r in rows if r["ad_id"] not in cache]
    if todo:
        import numpy as np
        import v4_vision_cognition as VC
        txt = VC._txt()
        texts = [((r.get("body") or "") + " " + (r.get("transcript") or "")
                  ).strip()[:1500] or "ad" for r in todo]
        for r, v in zip(todo, txt.embed(texts)):
            v = torch.tensor(np.asarray(v, dtype="float32"))
            cache[r["ad_id"]] = v / (v.norm() + 1e-8)
        torch.save(cache, TXT_CACHE)
    return cache


def _xy(rows, cache, tcache=None):
    """Feature matrix. With tcache, fuse image(512)+text(512)=1024-dim."""
    niches = sorted({r["niche"] for r in rows})
    idx = {n: i for i, n in enumerate(niches)}
    X, y = [], []
    for r in rows:
        v = cache.get(r["ad_id"])
        if v is None:
            continue
        if tcache is not None:
            t = tcache.get(r["ad_id"])
            if t is None:
                continue
            v = torch.cat([v, t])
        X.append(v)
        y.append(idx[r["niche"]])
    return torch.stack(X), torch.tensor(y), niches


def _train(X, y, n_cls, tr, steps=1200, seed=0):
    torch.manual_seed(seed)
    head = VisHead(X.size(1), n_cls)
    opt = torch.optim.AdamW(head.parameters(), lr=2e-3, weight_decay=1e-3)
    g = torch.Generator().manual_seed(seed + 1)
    for _ in range(steps):
        ix = tr[torch.randint(len(tr), (min(64, len(tr)),), generator=g)]
        loss = F.cross_entropy(head(X[ix]), y[ix])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    head.eval()
    return head


def cross_validate(k=5, mode="image"):
    rows = _rows()
    cache = _embed_all(rows)
    tcache = _text_embed_all(rows) if mode in ("text", "fuse") else None
    if mode == "text":
        X, y, niches = _xy(rows, {r["ad_id"]: tcache[r["ad_id"]]
                                  for r in rows if r["ad_id"] in tcache})
    else:
        X, y, niches = _xy(rows, cache, tcache)
    n = len(y)
    print(f"[{mode}] {n} creatives, {len(niches)} product categories",
          flush=True)
    g = torch.Generator().manual_seed(7)
    perm = torch.randperm(n, generator=g)
    folds = [perm[i::k] for i in range(k)]
    pred = torch.full((n,), -1, dtype=torch.long)
    for f in range(k):
        te = folds[f]
        tr = torch.cat([folds[j] for j in range(k) if j != f])
        head = _train(X, y, len(niches), tr, seed=f)
        with torch.no_grad():
            pred[te] = head(X[te]).argmax(-1)
    acc = (pred == y).float().mean().item()
    maj = y.bincount().argmax()
    base = (y == maj).float().mean().item()
    print(f"  [{mode:5}] product-category: {acc*100:.0f}%  (n={n}, baseline "
          f"{base*100:.0f}% = always '{niches[int(maj)]}', chance "
          f"{100/len(niches):.0f}%)  "
          f"{'>>> beats baseline' if acc > base + 1e-6 else '(<= baseline)'}")
    return acc, base


def train_save():
    rows = _rows()
    cache = _embed_all(rows)
    X, y, niches = _xy(rows, cache)
    head = _train(X, y, len(niches), torch.arange(len(y)), steps=2000)
    torch.save({"model": head.state_dict(), "niches": niches,
                "d_in": X.size(1)}, CKPT)
    print(f"saved {os.path.basename(CKPT)} ({len(niches)} categories, "
          f"{len(y)} creatives)")


TXT_CKPT = os.path.join(HERE, "v4_vision_head_text.pt")


def train_save_text():
    """Companion text-only category head (CLIP-text embeddings of the ad's
    own body+transcript) -- see predict_category_routed. Trained/saved
    separately from the image head so ROUTING can pick one head's output
    outright (never blend the two): v4_vision_model_search.md measured
    early/feature fusion (concatenating embeddings before one head) at 78%,
    WORSE than text alone (81%); naive late-fusion averaging measured 81.7%
    in aggregate but was caught dragging a correct confident image read to
    a wrong answer on a real case (see predict_category_routed) -- routing
    avoids that failure mode entirely by never averaging in the first
    place."""
    rows = _rows()
    tcache = _text_embed_all(rows)
    X, y, niches = _xy(rows, tcache)
    head = _train(X, y, len(niches), torch.arange(len(y)), steps=2000)
    torch.save({"model": head.state_dict(), "niches": niches,
                "d_in": X.size(1)}, TXT_CKPT)
    print(f"saved {os.path.basename(TXT_CKPT)} ({len(niches)} categories, "
          f"{len(y)} creatives)")


_INFER = {}
_INFER_TXT = {}


def predict_category(image_paths):
    """The model's NATIVE visual product-category read for a creative:
    {category: prob}, or None if the head/encoder isn't available.
    Backed by SigLIP-base (see EMB_CACHE comment above), not CLIP."""
    if not os.path.exists(CKPT):
        return None
    try:
        if "head" not in _INFER:
            ck = torch.load(CKPT, weights_only=False)
            h = VisHead(ck["d_in"], len(ck["niches"]))
            h.load_state_dict(ck["model"])
            h.eval()
            _INFER.update(head=h, niches=ck["niches"])
        import v4_vision_siglip as VS
        v = VS.image_vector(image_paths)
        if v is None:
            return None
        with torch.no_grad():
            p = torch.softmax(_INFER["head"](v.unsqueeze(0))[0], -1)
        return {n: float(p[i]) for i, n in enumerate(_INFER["niches"])}
    except Exception:  # noqa: BLE001
        return None


def predict_category_text(text):
    """Text-only category read via the companion CLIP-text head — {category:
    prob}, or None if unavailable/no text. Same niches as predict_category
    (both trained on the same manifest); used internally by
    predict_category_routed, but callable standalone."""
    if not os.path.exists(TXT_CKPT) or not (text or "").strip():
        return None
    try:
        if "head" not in _INFER_TXT:
            ck = torch.load(TXT_CKPT, weights_only=False)
            h = VisHead(ck["d_in"], len(ck["niches"]))
            h.load_state_dict(ck["model"])
            h.eval()
            _INFER_TXT.update(head=h, niches=ck["niches"])
        import numpy as np
        import v4_vision_cognition as VC
        tv = next(iter(VC._txt().embed([text[:1500]])))
        tv = torch.tensor(np.asarray(tv, dtype="float32"))
        tv = tv / (tv.norm() + 1e-8)
        with torch.no_grad():
            p = torch.softmax(_INFER_TXT["head"](tv.unsqueeze(0))[0], -1)
        return {n: float(p[i]) for i, n in enumerate(_INFER_TXT["niches"])}
    except Exception:  # noqa: BLE001
        return None


# Per-class 5-fold recall with SigLIP (v4_vision_model_search.md): trust
# only the niches that score >=~50% -- on the generic-bottle niches (gut
# health/anxiety-sleep/weight loss/supplement, ~25-46% even under SigLIP)
# the image read is close to a coin flip. This is the single canonical set
# (v4_admix.py imports it rather than keeping its own copy) since it's a
# property of the IMAGE classifier specifically, not a pipeline concern.
RELIABLE_IMAGE_CLASSES = {"prostate", "eye vision", "skincare", "tinnitus",
                          "blood sugar", "hair loss", "teeth",
                          "joint pain", "menopause"}


# --- MetaCLIP zero-shot: the generalizing layer BENEATH the trained head ---
# When the head is NOT on its reliable set (i.e. it's guessing), a zero-shot
# open-vocabulary read via MetaCLIP-H generalizes to unseen brands far better
# than the head does off that set. Honest held-out benchmark (v4_visbench/):
# trained head 48% vs MetaCLIP zero-shot 69% on 42 unseen-brand frames; and a
# grouped cascade (head-on-top, MetaCLIP-beneath) never regressed below
# zero-shot. Loaded once, CPU, ~1s/frame; only consulted on the fallback path,
# so the fast reliable-image case is untouched. Scores over the head's OWN
# niche labels, so no taxonomy remap is needed.
_META = {}
_META_PROMPT_TPL = "an advertisement for {} products"
_META_PROMPTS = {
    "menopause": "an ad for menopause or hormone relief for women",
    "prostate": "an ad for prostate or men's urinary health",
    "blood sugar": "an ad for blood sugar, glucose or diabetes support",
    "skincare": "an ad for skincare, anti-aging or wrinkle cream",
    "eye vision": "an ad for eye health or vision support",
    "teeth": "an ad for teeth, gum or oral health",
    "joint pain": "an ad for joint pain, arthritis or mobility relief",
    "tinnitus": "an ad for tinnitus or ringing-in-the-ears relief",
    "hair loss": "an ad for hair loss or hair regrowth",
    "gut health": "an ad for gut health, digestion or bloating",
    "energy fatigue": "an ad for energy, fatigue or tiredness",
    "anxiety sleep": "an ad for anxiety, stress or sleep support",
    "weight loss": "an ad for weight loss or fat burning",
    "supplement": "an ad for a general dietary supplement or vitamin",
}


def _meta_niches():
    if not os.path.exists(CKPT):
        return None
    try:
        return torch.load(CKPT, weights_only=False)["niches"]
    except Exception:  # noqa: BLE001
        return None


def metaclip_zeroshot(image_paths, niches):
    """Zero-shot MetaCLIP-H over ANY niche label set — the shared generalizing
    fallback used beneath BOTH the category and condition heads. Loads the
    encoder once; caches text features per niche-set. {niche: prob} or None."""
    if not niches:
        return None
    try:
        import open_clip
        from PIL import Image, ImageFile
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        if "model" not in _META:
            m, _, pp = open_clip.create_model_and_transforms(
                "ViT-H-14-quickgelu", pretrained="metaclip_fullcc")
            m.eval()
            _META.update(model=m, preprocess=pp,
                         tok=open_clip.get_tokenizer("ViT-H-14-quickgelu"),
                         tfeat={})
        s = _META
        key = tuple(niches)
        if key not in s["tfeat"]:
            prompts = [_META_PROMPTS.get(n, _META_PROMPT_TPL.format(n)) for n in niches]
            with torch.no_grad():
                tf = s["model"].encode_text(s["tok"](prompts))
                s["tfeat"][key] = tf / tf.norm(dim=-1, keepdim=True)
        tfeat = s["tfeat"][key]
        feats = []
        for p in (image_paths or []):
            if p and os.path.exists(p):
                img = Image.open(p).convert("RGB")
                with torch.no_grad():
                    f = s["model"].encode_image(s["preprocess"](img).unsqueeze(0))
                    feats.append(f / f.norm(dim=-1, keepdim=True))
        if not feats:
            return None
        with torch.no_grad():
            fm = torch.stack(feats).mean(0)
            fm = fm / fm.norm(dim=-1, keepdim=True)
            pr = torch.softmax((fm @ tfeat.T)[0] * 100.0, -1)
        return {n: float(pr[i]) for i, n in enumerate(niches)}
    except Exception:  # noqa: BLE001 — fallback is best-effort
        return None


def predict_category_metaclip(image_paths):
    """Category-head wrapper: MetaCLIP zero-shot over the head's own niches."""
    return metaclip_zeroshot(image_paths, _meta_niches())


def predict_category_routed(image_paths, text=None):
    """Mixture-of-experts ROUTING, not blending: consult the image and text
    classifiers, then TRUST WHICHEVER ONE HAS DEMONSTRATED RELIABILITY for
    this specific prediction, rather than averaging their outputs together.

    Tried averaging first (predict_category_fused, since removed) and
    caught a real failure live: on a real prostate-niche image with generic
    ad copy, the image head was confidently RIGHT (prostate, 0.998) but the
    text head was confidently WRONG (supplement, 0.979) — averaged at
    0.3/0.7 the wrong-but-confident text guess dragged the blend to
    'supplement' too (0.686), losing a read that was correct on its own.
    Averaging can't tell a confident right answer from a confident wrong
    one; it just does arithmetic on both.

    Routing instead: if the image head's own top class is one of the ~9
    niches ITS honest per-class CV showed genuinely visually distinctive
    (RELIABLE_IMAGE_CLASSES) — trust image outright, don't consult text at
    all. Otherwise trust text (81% overall vs image's 60.8%, and
    specifically the stronger tool everywhere outside that reliable set).
    This is what 'use the best tool for the segment' means mechanically:
    a gate that picks a SOURCE, never a blend of two sources.

    Returns (probs, source) where source is 'image' | 'text' | None."""
    pi = predict_category(image_paths)
    if pi:
        top = max(pi, key=pi.get)
        if top in RELIABLE_IMAGE_CLASSES:
            return pi, "image"
    # BENEATH our head: the MetaCLIP zero-shot generalizing fallback, consulted
    # only when the head is OFF its reliable set (i.e. guessing). Preferred over
    # the text head here because the honest held-out benchmark showed it both
    # stronger and generalizing to unseen brands; text stays as a further
    # fallback when MetaCLIP is unavailable.
    pm = predict_category_metaclip(image_paths)
    if pm:
        return pm, "metaclip"
    pt = predict_category_text(text)
    if pt:
        return pt, "text"
    if pi:
        return pi, "image"          # no text available -- image is all we have
    return None, None


if __name__ == "__main__":
    if "--compare" in sys.argv:
        print("=== k-fold CV: vision vs text vs fused (every creative once) ===")
        for m in ("image", "text", "fuse"):
            cross_validate(mode=m)
    elif "--cv" in sys.argv:
        cross_validate()
    elif "--train-text" in sys.argv:
        train_save_text()
    else:
        train_save()
