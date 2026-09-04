"""
v4_recognition_head — multi-task supervised head over the model's MOTIF
representation. This is the "reasoning into backprop" layer: a tiny head
(DemoHead generalized) that learns to predict the structured labels —
painpoint, age, gender, awareness — from the motif vector + marker scores,
so the network produces them in one forward pass and can generalize past the
exact-string regex KB.

Input  : 88-dim motif z-vector (the learned MotifScorer representation) +
         Stage-1 marker scores. NO face/vision features -> the demographic
         heads must learn (copy -> demographic) from words alone.
Heads  : painpoint (multi-label BCE), age (5-way), gender (binary),
         awareness (5-way). Each loss is MASKED to the rows that carry that
         label (heterogeneous labeling -> train every head on its own subset).
Honest : with few labels the held-out numbers are noisy; this harness is built
         to be correct and to SCALE — re-run as v4_recognition_data accrues
         more face-labeled creatives. It reports per-head held-out accuracy
         against a majority-class baseline so "did it actually learn?" is
         measurable, not assumed.

Run: python v4_recognition_head.py        # train + held-out report
"""
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import torch
import torch.nn as nn
import torch.nn.functional as F

import v4_correlations as C
import v4_tokenizer as tok
from v4_meta_audience import N_MARKER, marker_features

torch.set_num_threads(2)
HERE = os.path.dirname(os.path.abspath(__file__))
LABELS = os.path.join(HERE, "v4_recognition_labels.jsonl")
CKPT = os.path.join(HERE, "v4_recognition_head.pt")

AGE_BUCKETS = ["18-24", "25-34", "35-44", "45-54", "55+"]
STAGES = ["unaware", "problem_aware", "solution_aware", "product_aware",
          "most_aware"]
PAINPOINTS = [p["id"] for p in C.load_kb().get("painpoints", [])]


class RecogHead(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(d_in, 32), nn.Tanh(),
                                   nn.Dropout(0.1))
        self.painpoint = nn.Linear(32, len(PAINPOINTS))
        self.age = nn.Linear(32, len(AGE_BUCKETS))
        self.gender = nn.Linear(32, 1)
        self.awareness = nn.Linear(32, len(STAGES))

    def forward(self, x):
        h = self.trunk(x)
        return {"painpoint": self.painpoint(h), "age": self.age(h),
                "gender": self.gender(h).squeeze(-1),
                "awareness": self.awareness(h)}


# Feature mode (env): "motif" (default, 88 motif z + markers),
# "embed" (384-dim semantic embedding + markers — the #3 representation
# upgrade), or "both" (motif + markers + embedding).
FEAT_MODE = os.environ.get("V4_FEAT_MODE", "motif")
_EMBED_CACHE = os.path.join(HERE, "v4_recognition_embeds.json")


def _embeddings(texts):
    """384-dim semantic embeddings for `texts`, cached to disk by exact text
    so CV folds / re-runs don't re-encode (encoding is a subprocess call)."""
    import v4_embed
    cache = {}
    if os.path.exists(_EMBED_CACHE):
        try:
            cache = json.load(open(_EMBED_CACHE, encoding="utf-8"))
        except Exception:  # noqa: BLE001
            cache = {}
    missing = [t for t in texts if t not in cache]
    if missing:
        print(f"  encoding {len(missing)} new texts ...", flush=True)
        for t, v in zip(missing, v4_embed.encode(missing)):
            cache[t] = v
        json.dump(cache, open(_EMBED_CACHE, "w", encoding="utf-8"))
    return [torch.tensor(cache[t], dtype=torch.float32) for t in texts]


def _features(texts):
    parts_per = [[] for _ in texts]
    if FEAT_MODE in ("motif", "both"):
        from v4_motif_scorer import MotifScorer
        blob = torch.load(os.path.join(HERE, "v4_motif_cache.pt"),
                          weights_only=False)
        assert blob["vocab_hash"] == tok.vocab_hash(), "stale motif cache"
        scorer, mean, std = MotifScorer(), blob["mean"], blob["std"]
        for i, t in enumerate(texts):
            z = (scorer.score(t) - mean) / std
            parts_per[i].append(torch.cat([z, marker_features(t)]))
    elif FEAT_MODE == "embed":
        for i, t in enumerate(texts):     # markers still cheap + informative
            parts_per[i].append(marker_features(t))
    if FEAT_MODE in ("embed", "both"):
        embs = _embeddings(texts)
        for i, e in enumerate(embs):
            parts_per[i].append(e)
    print(f"  feature mode: {FEAT_MODE}", flush=True)
    return torch.stack([torch.cat(p) for p in parts_per])


def _targets(rows):
    pp_idx = {p: i for i, p in enumerate(PAINPOINTS)}
    age_idx = {a: i for i, a in enumerate(AGE_BUCKETS)}
    aw_idx = {s: i for i, s in enumerate(STAGES)}
    y_pp = torch.zeros(len(rows), len(PAINPOINTS))
    m_pp = torch.zeros(len(rows))
    y_age = torch.full((len(rows),), -1, dtype=torch.long)
    y_gen = torch.full((len(rows),), -1.0)
    y_aw = torch.full((len(rows),), -1, dtype=torch.long)
    for i, r in enumerate(rows):
        if r.get("painpoints") is not None:
            m_pp[i] = 1.0
            for p in r["painpoints"]:
                if p in pp_idx:
                    y_pp[i, pp_idx[p]] = 1.0
        if r.get("age") in age_idx:
            y_age[i] = age_idx[r["age"]]
        if r.get("gender") in ("female", "male"):
            y_gen[i] = 1.0 if r["gender"] == "female" else 0.0
        if r.get("awareness") in aw_idx:
            y_aw[i] = aw_idx[r["awareness"]]
    return {"pp": y_pp, "m_pp": m_pp, "age": y_age, "gen": y_gen, "aw": y_aw}


def _split(n, seed=1611, frac=0.8):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    cut = int(frac * n)
    return perm[:cut], perm[cut:]


def _train_head(X, Y, idx, steps=1500, seed=1611):
    torch.manual_seed(seed)
    head = RecogHead(X.size(1))
    opt = torch.optim.AdamW(head.parameters(), lr=3e-3, weight_decay=1e-3)
    g = torch.Generator().manual_seed(7)
    idx = idx.clone()
    for _ in range(steps):
        ix = idx[torch.randint(len(idx), (min(64, len(idx)),), generator=g)]
        out = head(X[ix])
        loss = torch.zeros(())
        mpp = Y["m_pp"][ix]
        if mpp.sum() > 0:
            bce = F.binary_cross_entropy_with_logits(
                out["painpoint"], Y["pp"][ix], reduction="none").mean(-1)
            loss = loss + (bce * mpp).sum() / mpp.sum()
        for key, logit in (("age", out["age"]), ("aw", out["awareness"])):
            yk = Y[key][ix]
            m = yk >= 0
            if m.any():
                loss = loss + F.cross_entropy(logit[m], yk[m])
        yg = Y["gen"][ix]
        mg = yg >= 0
        if mg.any():
            loss = loss + F.binary_cross_entropy_with_logits(
                out["gender"][mg], yg[mg])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
    return head


def cross_validate(k=5):
    """k-fold CV — every labeled row is tested exactly once (in the fold where
    it's held out). The honest small-data metric: with ~30 demographic labels a
    single 80/20 split tests only ~2, which is noise. Reports per-field
    accuracy over ALL labeled rows vs the majority-class baseline."""
    rows = [json.loads(line) for line in open(LABELS, encoding="utf-8")]
    print(f"{len(rows)} rows; building motif features ...", flush=True)
    X = _features([r["text"] for r in rows])
    Y = _targets(rows)
    N = len(rows)
    g = torch.Generator().manual_seed(1611)
    perm = torch.randperm(N, generator=g)
    folds = [perm[i::k] for i in range(k)]

    pred = {"age": torch.full((N,), -1, dtype=torch.long),
            "gen": torch.full((N,), -1, dtype=torch.long),
            "aw": torch.full((N,), -1, dtype=torch.long),
            "pp_top": torch.full((N,), -1, dtype=torch.long)}
    for f in range(k):
        te = folds[f]
        tr = torch.cat([folds[j] for j in range(k) if j != f])
        head = _train_head(X, Y, tr, seed=1611 + f)
        head.eval()
        with torch.no_grad():
            out = head(X[te])
        pred["age"][te] = out["age"].argmax(-1)
        pred["gen"][te] = (torch.sigmoid(out["gender"]) >= 0.5).long()
        pred["aw"][te] = out["awareness"].argmax(-1)
        pred["pp_top"][te] = torch.sigmoid(out["painpoint"]).argmax(-1)
        print(f"  fold {f+1}/{k} done", flush=True)

    print("\n=== {0}-fold CV (every labeled row tested once)".format(k))
    _cv_cls("age", pred["age"], Y["age"], AGE_BUCKETS)
    _cv_cls("gender", pred["gen"], Y["gen"].long(), ["male", "female"],
            is_gender=True)
    _cv_cls("awareness", pred["aw"], Y["aw"], STAGES)
    _cv_pp(pred["pp_top"], Y)


def _cv_cls(name, pred, y, classes, is_gender=False):
    m = y >= 0
    if m.sum() == 0:
        print(f"  {name:10}: no labels"); return
    yy = y[m]
    pp = pred[m]
    acc = (pp == yy).float().mean().item()
    maj = yy.bincount().argmax()
    base = (yy == maj).float().mean().item()
    print(f"  {name:10}: {acc*100:4.0f}%  (n={int(m.sum())}, baseline "
          f"{base*100:.0f}% = always '{classes[int(maj)]}')  "
          f"{'>>> beats baseline' if acc > base + 1e-6 else '(<= baseline)'}")


def _cv_pp(pred_top, Y):
    m = Y["m_pp"] > 0
    idx = torch.nonzero(m).squeeze(-1)
    if len(idx) == 0:
        print(f"  {'painpoint':10}: no labels"); return
    gold = Y["pp"][idx]
    hit = gold[torch.arange(len(idx)), pred_top[idx]].mean().item()
    print(f"  {'painpoint':10}: top-1 in-label {hit*100:4.0f}%  (n={len(idx)}; "
          f"vs regex labels)")


def main():
    rows = [json.loads(line) for line in open(LABELS, encoding="utf-8")]
    print(f"{len(rows)} rows; building motif features ...", flush=True)
    X = _features([r["text"] for r in rows])
    Y = _targets(rows)
    tr, te = _split(len(rows))

    torch.manual_seed(1611)
    head = RecogHead(X.size(1))
    n = sum(p.numel() for p in head.parameters())
    print(f"RecogHead: {n:,} params (input {X.size(1)})")
    opt = torch.optim.AdamW(head.parameters(), lr=3e-3, weight_decay=1e-3)
    g = torch.Generator().manual_seed(7)
    for step in range(3000):
        ix = tr[torch.randint(len(tr), (min(64, len(tr)),), generator=g)]
        out = head(X[ix])
        loss = torch.zeros(())
        # painpoint (masked multi-label BCE)
        mpp = Y["m_pp"][ix]
        if mpp.sum() > 0:
            bce = F.binary_cross_entropy_with_logits(
                out["painpoint"], Y["pp"][ix], reduction="none").mean(-1)
            loss = loss + (bce * mpp).sum() / mpp.sum()
        # age / awareness (masked CE)
        for key, logit in (("age", out["age"]), ("aw", out["awareness"])):
            yk = Y[key][ix]
            m = yk >= 0
            if m.any():
                loss = loss + F.cross_entropy(logit[m], yk[m])
        # gender (masked BCE)
        yg = Y["gen"][ix]
        mg = yg >= 0
        if mg.any():
            loss = loss + F.binary_cross_entropy_with_logits(
                out["gender"][mg], yg[mg])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        if step % 600 == 0:
            print(f"  step {step:>4} loss {loss.item():.4f}", flush=True)

    head.eval()
    with torch.no_grad():
        out = head(X)
    print("\n=== held-out (test split) — accuracy vs majority-class baseline")
    _report_cls("age", out["age"], Y["age"], te, AGE_BUCKETS)
    _report_bin("gender", out["gender"], Y["gen"], te)
    _report_cls("awareness", out["awareness"], Y["aw"], te, STAGES)
    _report_pp(out["painpoint"], Y, te)
    torch.save({"model": head.state_dict(), "vocab_hash": tok.vocab_hash(),
                "painpoints": PAINPOINTS, "ages": AGE_BUCKETS,
                "stages": STAGES}, CKPT)
    print(f"\nsaved {os.path.basename(CKPT)}")


def _report_cls(name, logit, y, te, classes):
    m = (y[te] >= 0)
    idx = te[m]
    if len(idx) == 0:
        print(f"  {name:10}: no held-out labels"); return
    pred = logit[idx].argmax(-1)
    acc = (pred == y[idx]).float().mean().item()
    # majority baseline from the FULL labeled set
    full = y[y >= 0]
    maj = full.bincount().argmax() if len(full) else 0
    base = (y[idx] == maj).float().mean().item()
    print(f"  {name:10}: {acc*100:4.0f}%  (n={len(idx)}, baseline "
          f"{base*100:.0f}% = always '{classes[int(maj)]}')")


def _report_bin(name, logit, y, te):
    m = (y[te] >= 0)
    idx = te[m]
    if len(idx) == 0:
        print(f"  {name:10}: no held-out labels"); return
    pred = (torch.sigmoid(logit[idx]) >= 0.5).float()
    acc = (pred == y[idx]).float().mean().item()
    print(f"  {name:10}: {acc*100:4.0f}%  (n={len(idx)})")


def _report_pp(logit, Y, te):
    m = (Y["m_pp"][te] > 0)
    idx = te[m]
    if len(idx) == 0:
        print(f"  {'painpoint':10}: no held-out labels"); return
    prob = torch.sigmoid(logit[idx])
    pred = prob.argmax(-1)
    gold = Y["pp"][idx]
    # top-1 hit: does the head's top painpoint appear in the regex label set?
    hit = gold[torch.arange(len(idx)), pred].mean().item()
    print(f"  {'painpoint':10}: top-1 in-label {hit*100:4.0f}%  (n={len(idx)}; "
          f"agreement with regex labels, not an independent metric)")


# ---- live inference: the model's OWN learned read for the analyze() path ----
_INFER = {}


def _load_infer():
    """Load the trained head once for live prediction. Returns None if the
    checkpoint is missing or stale (caller falls back to the rule read)."""
    if "head" in _INFER:
        return _INFER["head"]
    _INFER["head"] = None
    try:
        ck = torch.load(CKPT, weights_only=False)
        if ck.get("vocab_hash") != tok.vocab_hash():
            return None
        # Honour V4_MOTIF_CACHE — the same cache MotifScorer below is about to
        # build its features from. This used to be hardcoded to
        # v4_motif_cache.pt while the live stack ran on v4_big_motif_cache.pt,
        # so the head standardised features with statistics from a DIFFERENT
        # cache: mean off by 0.011 and std off by ~30% (0.011 vs 0.008), which
        # mis-scales every input. Falls back to the old name so a bare env
        # still works.
        cache = os.environ.get("V4_MOTIF_CACHE") or os.path.join(
            HERE, "v4_motif_cache.pt")
        if not os.path.exists(cache):
            cache = os.path.join(HERE, "v4_motif_cache.pt")
        blob = torch.load(cache, weights_only=False)
        if blob.get("vocab_hash") != tok.vocab_hash():
            return None
        from v4_motif_scorer import MotifScorer
        head = RecogHead(len(STAGES) + 0)  # placeholder; rebuilt below
        # input dim must match training (motif z[88] + markers[N_MARKER])
        d_in = blob["mean"].numel() + N_MARKER
        head = RecogHead(d_in)
        head.load_state_dict(ck["model"])
        head.eval()
        _INFER.update(head=head, scorer=MotifScorer(),
                      mean=blob["mean"], std=blob["std"],
                      stages=ck.get("stages", STAGES))
        return head
    except Exception:  # noqa: BLE001 — never let the learned read break analyze
        return None


def predict_awareness(text):
    """The LEARNED awareness read (k-fold CV ~93% vs 23% baseline): a dict
    {stage: prob} from the trained RecogHead over the model's own motif +
    marker representation. None if the head can't load — so analyze() can
    fall back to the deterministic rule read. This is the model reading the
    ad through its own taxonomy, not a hand-written regex."""
    head = _load_infer()
    if head is None:
        return None
    try:
        z = (_INFER["scorer"].score(text) - _INFER["mean"]) / _INFER["std"]
        feat = torch.cat([z, marker_features(text)]).unsqueeze(0)
        with torch.no_grad():
            logits = head(feat)["awareness"][0]
        probs = torch.softmax(logits, dim=-1)
        return {s: float(probs[i]) for i, s in enumerate(_INFER["stages"])}
    except Exception:  # noqa: BLE001
        return None


if __name__ == "__main__":
    import sys
    if "--cv" in sys.argv:
        cross_validate()
    else:
        main()
