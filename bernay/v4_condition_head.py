"""
v4_condition_head — a DEDICATED classifier for KB CONDITION categorization,
separate from v4_vision_head's general 26-niche PRODUCT-category classifier.

Why this exists: v4_vision_head's classifier is trained across all 26
harvested niches, including 'supplement' (a product-TYPE label with no
consistent visual signature -- 33% precision, an attractor of confusion in
both directions) and twelve near-empty ingredient niches (n=4 each, ~0%
recall -- chance-level, pure softmax noise). Only 13 of the 26 niches map to
a KB condition at all (v4_admix.VISUAL_TO_CONDITION) -- the rest can only
ever hurt the `problem` decision, never help it, since predicting them can't
feed VISUAL_TO_CONDITION either way. Dropping them concentrates all the
model's capacity on the classes that actually matter for this decision.

Measured effect (5-fold CV, same 586-creative manifest, same SigLIP/CLIP
embeddings -- no re-embedding needed, this reuses v4_vision_head's caches):
run `python v4_condition_head.py --cv` for current numbers.

    python v4_condition_head.py --cv     # honest k-fold CV, both modalities
    python v4_condition_head.py          # train + save both heads
"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")

import v4_quiet  # noqa: F401 — silence fastembed/HF logging before it loads

import torch
import torch.nn.functional as F

import v4_vision_head as _base

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "v4_condition_head.pt")
TXT_CKPT = os.path.join(HERE, "v4_condition_head_text.pt")

# The 13 niches v4_admix.VISUAL_TO_CONDITION actually maps to a sourced KB
# condition. Everything else in the 26-niche manifest ('supplement' + 12
# n=4 ingredient niches) is dropped -- it cannot improve this decision and
# measurably dilutes it (see module docstring).
CONDITION_NICHES = {
    "joint pain", "prostate", "menopause", "hair loss", "blood sugar",
    "gut health", "tinnitus", "eye vision", "teeth", "weight loss",
    "anxiety sleep", "energy fatigue", "skincare",
}


def _rows():
    return [r for r in _base._rows() if r["niche"] in CONDITION_NICHES]


def cross_validate(k=5, mode="image", verbose=True):
    rows = _rows()
    cache = _base._embed_all(rows)
    tcache = _base._text_embed_all(rows) if mode in ("text", "fuse") else None
    if mode == "text":
        X, y, niches = _base._xy(rows, tcache)
    else:
        X, y, niches = _base._xy(rows, cache, tcache)
    n = len(y)
    if verbose:
        print(f"[{mode}] {n} creatives, {len(niches)} condition niches "
              f"(dropped 'supplement' + 12 near-empty ingredient niches)",
              flush=True)
    g = torch.Generator().manual_seed(7)
    perm = torch.randperm(n, generator=g)
    folds = [perm[i::k] for i in range(k)]
    pred = torch.full((n,), -1, dtype=torch.long)
    for f in range(k):
        te = folds[f]
        tr = torch.cat([folds[j] for j in range(k) if j != f])
        head = _base._train(X, y, len(niches), tr, seed=f)
        with torch.no_grad():
            pred[te] = head(X[te]).argmax(-1)
    acc = (pred == y).float().mean().item()
    maj = y.bincount().argmax()
    base_acc = (y == maj).float().mean().item()
    if verbose:
        print(f"  [{mode:5}] condition: {acc*100:.0f}%  (n={n}, baseline "
              f"{base_acc*100:.0f}% = always '{niches[int(maj)]}', chance "
              f"{100/len(niches):.0f}%)  "
              f"{'>>> beats baseline' if acc > base_acc + 1e-6 else '(<= baseline)'}")
    return pred, y, niches


def per_class_report(pred, y, niches, label):
    import collections
    y_l, pred_l = y.tolist(), pred.tolist()
    support = collections.Counter(niches[t] for t in y_l)
    tp, fp = collections.Counter(), collections.Counter()
    for t, p in zip(y_l, pred_l):
        tn, pn = niches[t], niches[p]
        if tn == pn:
            tp[tn] += 1
        else:
            fp[pn] += 1
    print(f"\n  {label} per-class (n / recall / precision):")
    reliable = []
    for niche in sorted(niches, key=lambda nm: -support[nm]):
        n_ = support[niche]
        rec = tp[niche] / n_ if n_ else 0.0
        prec_denom = tp[niche] + fp[niche]
        prec = tp[niche] / prec_denom if prec_denom else float("nan")
        flag = ""
        if rec >= 0.5 and (prec >= 0.5 or prec != prec):
            reliable.append(niche)
            flag = "  <- reliable (recall & precision both >=50%)"
        print(f"    {niche:<16}{n_:>4}  {rec*100:>5.0f}%{prec*100:>10.0f}%{flag}")
    return reliable


def train_save():
    rows = _rows()
    cache = _base._embed_all(rows)
    X, y, niches = _base._xy(rows, cache)
    head = _base._train(X, y, len(niches), torch.arange(len(y)), steps=2000)
    torch.save({"model": head.state_dict(), "niches": niches,
                "d_in": X.size(1)}, CKPT)
    print(f"saved {os.path.basename(CKPT)} ({len(niches)} conditions, "
          f"{len(y)} creatives)")


def train_save_text():
    rows = _rows()
    tcache = _base._text_embed_all(rows)
    X, y, niches = _base._xy(rows, tcache)
    head = _base._train(X, y, len(niches), torch.arange(len(y)), steps=2000)
    torch.save({"model": head.state_dict(), "niches": niches,
                "d_in": X.size(1)}, TXT_CKPT)
    print(f"saved {os.path.basename(TXT_CKPT)} ({len(niches)} conditions, "
          f"{len(y)} creatives)")


_INFER = {}
_INFER_TXT = {}


def predict_condition(image_paths):
    if not os.path.exists(CKPT):
        return None
    try:
        if "head" not in _INFER:
            ck = torch.load(CKPT, weights_only=False)
            h = _base.VisHead(ck["d_in"], len(ck["niches"]))
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


def predict_condition_text(text):
    if not text or not os.path.exists(TXT_CKPT):
        return None
    try:
        if "head" not in _INFER_TXT:
            ck = torch.load(TXT_CKPT, weights_only=False)
            h = _base.VisHead(ck["d_in"], len(ck["niches"]))
            h.load_state_dict(ck["model"])
            h.eval()
            _INFER_TXT.update(head=h, niches=ck["niches"])
        import numpy as np
        import v4_vision_cognition as VC
        v = next(iter(VC._txt().embed([text[:1500]])))
        v = torch.tensor(np.asarray(v, dtype="float32"))
        v = v / (v.norm() + 1e-8)
        with torch.no_grad():
            p = torch.softmax(_INFER_TXT["head"](v.unsqueeze(0))[0], -1)
        return {n: float(p[i]) for i, n in enumerate(_INFER_TXT["niches"])}
    except Exception:  # noqa: BLE001
        return None


# Measured 5-fold CV, 485-creative condition-only manifest (2026-07-19,
# `python v4_condition_head.py --cv`): image 71% overall (up from the general
# 26-niche classifier's 60.8%), text 91% overall (up from 81%). Conservative
# on purpose -- recall AND precision both >=50% -- after the general
# classifier's 'menopause' was found at 72% recall but only 60% PRECISION:
# recall alone let a real over-attractor class through (see
# v4_vision_model_search.md for the full confusion-matrix writeup). Excluded
# here: gut health (33%/33%), anxiety sleep (35%/46%), weight loss (19%/60%)
# -- all still routed via text (91%/83-91% on those same three).
RELIABLE_IMAGE_CLASSES = {
    "blood sugar", "energy fatigue", "eye vision", "hair loss",
    "joint pain", "menopause", "prostate", "skincare", "teeth", "tinnitus",
}


# RELIABLE_IMAGE_CLASSES is a per-CLASS historical guarantee (recall AND
# precision >=50% averaged over many predictions); it says nothing about how
# confident any ONE prediction is. Caught live: fed abstract song lyrics
# (not real ad copy) for a gender-neutral device, the text head's top guess
# was 'menopause' at just 34% -- a near-tie against 'prostate' (30%) and
# 'teeth' (20%) -- yet was trusted outright because 'text source' had no
# confidence floor at all. MIN_CONF is a per-PREDICTION sanity check on top
# of the per-class gate, so a genuinely uncertain call can't override.
MIN_CONF = 0.5


def predict_condition_routed(image_paths, text=None):
    """Mixture-of-experts routing, same architecture as
    v4_vision_head.predict_category_routed: pick a SOURCE, never blend."""
    pi = predict_condition(image_paths) if image_paths else None
    if pi:
        top = max(pi, key=pi.get)
        if top in RELIABLE_IMAGE_CLASSES and pi[top] >= MIN_CONF:
            return pi, "image"
    pt = predict_condition_text(text) if text else None
    if pt:
        top_t = max(pt, key=pt.get)
        if pt[top_t] >= MIN_CONF:
            return pt, "text"
    if pi:
        return pi, "image"
    return None, None


if __name__ == "__main__":
    if "--cv" in sys.argv:
        print("=" * 70)
        pred_i, y_i, niches_i = cross_validate(mode="image")
        rel_i = per_class_report(pred_i, y_i, niches_i, "IMAGE")
        print("=" * 70)
        pred_t, y_t, niches_t = cross_validate(mode="text")
        rel_t = per_class_report(pred_t, y_t, niches_t, "TEXT")
        print("=" * 70)
        print(f"\nRELIABLE_IMAGE_CLASSES (recall & precision both >=50%): "
              f"{sorted(rel_i)}")
    else:
        train_save()
        train_save_text()
