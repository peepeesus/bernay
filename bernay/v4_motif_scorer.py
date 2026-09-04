"""
V4 Stage B — motif scorer: the prime weights put to work.

The pretrained backbone (Jung + Hopkins + chakra + kabbalah + Maslow + KJV)
is used to recognize archetypal/desire motifs in market inputs — customer
avatar descriptions, market-trend summaries, competitor ad copy. Per the
plan this is a HYBRID scorer, because 31k-param representations are weak on
their own:

  embedding channel: each taxonomy category's anchor_texts are pushed
      through the trained backbone (post-ln_f hidden states, mean-pooled
      over sliding 99-char windows, unit-normalized); an input text is
      scored by cosine similarity to each category's anchor centroid.
  keyword channel: normalized hit-rate of each category's keywords in the
      input text (hits per 1,000 chars, sqrt-squashed).

44 categories x 2 channels = an 88-dim motif vector per text. Z-score
calibration stats over the full synthetic set go to v4_motif_stats.json;
every scored entity is cached in v4_motif_cache.pt so Stage C never re-runs
the backbone.

Self-test (gates Stage C): leave-one-out anchor recovery — each anchor text
scored against all 44 categories with itself excluded from its own
category's centroid. Target top-1 >= 60% on the embedding channel alone
(chance 2.3%). The keyword channel is reported separately: it is ~perfect
by construction, which is the floor, not the proof.
"""

import json
import math
import os
import re

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import torch

import gpt2_pv_v4 as pv
import v4_tokenizer as tok

torch.set_num_threads(2)
HERE = os.path.dirname(os.path.abspath(__file__))

TAXONOMY_PATH = os.path.join(HERE, "v4_taxonomy.json")
# motif cache path is env-overridable so gpt3 (V4_MOTIF_CACHE=...gpt3...)
# and the gpt2.5 fallback keep separate caches during the retrain
CACHE_PATH = os.environ.get("V4_MOTIF_CACHE") or \
    os.path.join(HERE, "v4_motif_cache.pt")
STATS_PATH = os.path.join(HERE, "v4_motif_stats.json")

WINDOW = pv.v2.BLOCK_SIZE      # 99
STRIDE = 64

# WHICH ENCODER produces the 44 anchor-cosine dims. Default stays "backbone"
# so nothing changes without an explicit opt-in.
#
# Measured 2026-08-23 — the backbone's own embeddings do not separate the
# taxonomy: leave-one-out anchor recovery 1.5% top-1 against 2.3% chance,
# same-vs-different-category cosine separation +0.0002, and 45 pooling x
# postprocessing variants top out at 5.3% (~2 sigma). bge-small on the identical
# metric gets 14.4% / 34.8% with separation +0.0560 — 280x. On v4_realad_eval
# (real gethookd grades, NICHE-grouped folds) bge beats the backbone
# 0.1634 vs 0.1109, higher in 4 of 5 folds.
#
# The synthetic v4_ads_truth benchmark CANNOT see this difference (swapping the
# encoder there moves PV by -0.004), which is why it went unnoticed: on that
# data the regex channel alone reaches 0.8741 of an available 0.8836. Gate any
# change here on v4_realad_eval.py, never on the synthetic PV numbers or the
# regression suite (both channel ablations pass it 32/32).
MOTIF_ENCODER = os.environ.get("V4_MOTIF_ENCODER", "backbone")


def load_taxonomy():
    with open(TAXONOMY_PATH, encoding="utf-8") as f:
        tax = json.load(f)
    cats = tax["categories"]
    assert len(cats) == 44, f"expected 44 categories, got {len(cats)}"
    return tax, cats


def pick_checkpoint():
    full = os.path.join(HERE, "v4_prime_ckpt.pt")
    smoke = os.path.join(HERE, "v4_prime_ckpt_smoke.pt")
    path = os.environ.get("V4_CKPT") or (full if os.path.exists(full)
                                         else smoke)
    if path == smoke:
        print("WARNING: using SMOKE checkpoint — scores are low-quality")
    return path


def load_model():
    torch.manual_seed(1611)
    model = pv.GPT2PV4()
    step = pv.load_ckpt(pick_checkpoint(), model)
    model.eval()
    print(f"backbone loaded at step {step}")
    return model


_BGE_CACHE = {}


def _bge_embed(text):
    """384-dim unit-norm bge-small vector (memoised — anchors repeat)."""
    if text in _BGE_CACHE:
        return _BGE_CACHE[text]
    import v4_embed
    v = torch.tensor(v4_embed.encode([text])[0], dtype=torch.float32)
    v = v / (v.norm() + 1e-8)
    if len(_BGE_CACHE) < 20000:
        _BGE_CACHE[text] = v
    return v


@torch.no_grad()
def embed_text(model, text):
    """Unit-norm pooled hidden state over sliding windows of the text.

    With V4_MOTIF_ENCODER=bge this delegates to bge-small instead; the
    returned vector is still unit-norm, so every downstream cosine against a
    centroid built the same way is unchanged in kind (only the dimensionality
    differs, and nothing downstream depends on it being N_EMBD).
    """
    if MOTIF_ENCODER == "bge":
        return _bge_embed(text)
    ids = tok.encode(text)
    if len(ids) < 8:                      # degenerate input
        ids = ids + tok.encode(" " * (8 - len(ids)))
    windows = [ids[i: i + WINDOW]
               for i in range(0, max(1, len(ids) - WINDOW + 1), STRIDE)]
    if len(ids) > WINDOW and (len(ids) - WINDOW) % STRIDE:
        windows.append(ids[-WINDOW:])     # tail window
    pooled = []
    for w in windows:
        h = model.hidden(torch.tensor([w], dtype=torch.long))
        pooled.append(h.mean(dim=1).squeeze(0))
    P = torch.stack(pooled)
    # Long inputs (a full VSL transcript = hundreds of windows) wash out:
    # averaging every window regresses to the global mean and distinct
    # ads collapse to the same vector. Average only the most DISTINCTIVE
    # windows (largest deviation from the centroid) — the hook and pitch
    # carry the motif signal; filler clusters at the mean and dilutes it.
    # Short inputs (anchors, synthetic ad copies — all <~12 windows) keep
    # the plain mean, so the trained cache stays exactly valid.
    if P.size(0) > 12:
        dev = (P - P.mean(dim=0)).norm(dim=1)
        k = min(24, P.size(0))          # small fixed budget of the MOST
        P = P[dev.topk(k).indices]      # distinctive windows; averaging
        # hundreds of windows (n//3) still regressed to the generic mean
        # and made distinct VSLs collapse to one vector.
    u = P.mean(dim=0)
    return u / (u.norm() + 1e-8)


def keyword_rate(text, keywords):
    """sqrt(hits per 1,000 chars) for one category's keyword list."""
    low = tok.normalize(text).lower()
    hits = 0
    for kw in keywords:
        hits += len(re.findall(r"\b" + re.escape(kw.lower()) + r"\b", low))
    return math.sqrt(1000.0 * hits / max(len(low), 1))


class MotifScorer:
    def __init__(self, model=None):
        self.tax, self.cats = load_taxonomy()
        self.model = model if model is not None else load_model()
        self.anchor_vecs = {}             # cat id -> (n_anchors, d) tensor
        for c in self.cats:
            vecs = [embed_text(self.model, a) for a in c["anchor_texts"]]
            self.anchor_vecs[c["id"]] = torch.stack(vecs)
        self.centroids = torch.stack([
            self._centroid(self.anchor_vecs[c["id"]]) for c in self.cats])

    @staticmethod
    def _centroid(vecs):
        m = vecs.mean(dim=0)
        return m / (m.norm() + 1e-8)

    def score(self, text):
        """88-dim raw motif vector: [44 cosine sims | 44 keyword rates]."""
        u = embed_text(self.model, text)
        cos = self.centroids @ u
        kw = torch.tensor([keyword_rate(text, c["keywords"])
                           for c in self.cats])
        return torch.cat([cos, kw])

    def anchor_recovery(self):
        """Leave-one-out top-1/top-3 on the embedding channel."""
        top1 = top3 = total = 0
        for k, c in enumerate(self.cats):
            vecs = self.anchor_vecs[c["id"]]
            for i in range(len(vecs)):
                cents = self.centroids.clone()
                rest = torch.cat([vecs[:i], vecs[i + 1:]])
                cents[k] = self._centroid(rest)
                sims = cents @ vecs[i]
                order = sims.argsort(descending=True)
                top1 += int(order[0] == k)
                top3 += int(k in order[:3].tolist())
                total += 1
        return top1 / total, top3 / total, total


def read_csv_dicts(path):
    import csv
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


if __name__ == "__main__":
    scorer = MotifScorer()
    print(f"taxonomy: {len(scorer.cats)} categories, "
          f"{sum(len(v) for v in scorer.anchor_vecs.values())} anchors\n")

    # ---- self-test: leave-one-out anchor recovery (embedding channel) ----
    t1, t3, n = scorer.anchor_recovery()
    print(f"anchor recovery (embedding channel, n={n}): "
          f"top-1 {100 * t1:.1f}%  top-3 {100 * t3:.1f}%  (chance 2.3%)")

    # keyword channel sanity, reported separately (near-perfect by design)
    kw_top1 = 0
    for k, c in enumerate(scorer.cats):
        for a in c["anchor_texts"]:
            rates = torch.tensor([keyword_rate(a, cc["keywords"])
                                  for cc in scorer.cats])
            kw_top1 += int(rates.argmax() == k)
    print(f"anchor recovery (keyword channel): top-1 {100 * kw_top1 / n:.1f}%"
          f"  [floor by construction]\n")

    # ---- score and cache every synthetic entity --------------------------
    cache = {}
    for fname, id_col, text_col in [("v4_avatars.csv", "avatar_id",
                                     "description"),
                                    ("v4_trends.csv", "trend_id",
                                     "summary_text"),
                                    ("v4_angles.csv", "angle_id",
                                     "ad_copy_text")]:
        path = os.path.join(HERE, fname)
        if not os.path.exists(path):
            print(f"WARNING: {fname} missing — skipped")
            continue
        rows = read_csv_dicts(path)
        for r in rows:
            cache[r[id_col]] = scorer.score(r[text_col])
        print(f"scored {len(rows):>4} texts from {fname}")

    if cache:
        all_vecs = torch.stack(list(cache.values()))
        mean, std = all_vecs.mean(dim=0), all_vecs.std(dim=0) + 1e-8
        torch.save({"cache": cache, "mean": mean, "std": std,
                    "vocab_hash": tok.vocab_hash()}, CACHE_PATH)
        with open(STATS_PATH, "w") as f:
            # record WHICH encoder produced these numbers — the file used to
            # say only "anchor_top1: 0.0227" (= chance) with no way to tell
            # whether that was the backbone, a fallback, or a stale run
            json.dump({"n_texts": len(cache), "dim": int(all_vecs.size(1)),
                       "encoder": MOTIF_ENCODER,
                       "cache": os.path.basename(CACHE_PATH),
                       "block_size": WINDOW,
                       "vocab_hash": tok.vocab_hash(),
                       "anchor_top1": t1, "anchor_top3": t3,
                       "anchor_chance": 1.0 / len(scorer.cats)}, f, indent=1)
        print(f"\ncached {len(cache)} motif vectors (dim "
              f"{all_vecs.size(1)}) -> {os.path.basename(CACHE_PATH)}")

    # ---- 3 hand-written probes for eyeballing ----------------------------
    probes = {
        "status-driven founder": (
            "Ambitious startup founder, obsessed with growth and market "
            "dominance. Wants power, status and recognition among peers. "
            "Fears being ordinary. Tracks every competitor and wants to "
            "win the category."),
        "safety-first parent": (
            "A careful mother of two who worries about her family's "
            "security and health. She wants protection, stability and "
            "comfort, reads every label, and distrusts flashy claims."),
        "seeker of meaning": (
            "Burned-out professional searching for purpose and inner "
            "peace. Meditates, reads psychology and spiritual texts, "
            "wants transcendence, clarity and a sense of higher calling."),
    }
    for name, text in probes.items():
        s = scorer.score(text)
        z = (s - mean) / std if cache else s
        comb = z[:44] + z[44:]
        order = comb.argsort(descending=True)[:5]
        cats = ", ".join(scorer.cats[i]["id"] for i in order.tolist())
        print(f"probe '{name}': top-5 -> {cats}")
