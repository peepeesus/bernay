"""
V4 demographic avatar inference — Stage 2: learned head on real data.

THE DATA CONTRACT. The Meta Ad Library (facebook.com/ads/library, free)
publishes EU transparency data per ad including the REACHED AUDIENCE
broken down by age bracket and gender — i.e., labeled (ad text ->
demographics) pairs at scale. Export rows into:

    v4_meta_ads.csv with columns:
        ad_text            the creative text/transcript
        age_18_24 .. age_65_plus    fraction of reach per bracket
        gender_female      fraction of reach (male = 1 - female approx)

(The Ad Library API field is `age_country_gender_reach_breakdown`;
aggregate countries, normalize to fractions. Nationality/ethnicity are
deliberately NOT part of this contract.)

THE MODEL. DemoHead — a PVEngine-sized sibling (a few thousand params,
NOT a big LM): input = 88-dim motif vector of the ad text plus the
Stage-1 marker scores; outputs = softmax over 5 age brackets + a gender
fraction. Stage 1's deterministic markers remain the prior; this head
learns the residual real-world signal the markers can't see.

Run `python v4_meta_audience.py` any time: with no CSV present it
prints the contract and exits; with data it trains, evaluates against
the Stage-1-only baseline, and saves v4_demo_head.pt.
"""

import csv
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import torch
import torch.nn as nn
import torch.nn.functional as F

import v4_demographics as demo_mod
import v4_tokenizer as tok

torch.set_num_threads(2)
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "v4_meta_ads.csv")
CKPT = os.path.join(HERE, "v4_demo_head.pt")

AGE_COLS = ["age_18_24", "age_25_34", "age_35_44", "age_45_54",
            "age_65_plus"]
MARKER_DIMS = ["gender", "age", "life_stage", "income"]


def marker_features(text):
    """Flatten Stage-1 marker scores into a fixed feature vector."""
    demo = demo_mod.infer_demographics(text)
    feats = []
    for dim in MARKER_DIMS:
        for val in sorted(demo_mod.MARKERS[dim]):
            feats.append(demo[dim]["scores"][val])
    return torch.tensor(feats)


N_MARKER = sum(len(v) for v in demo_mod.MARKERS.values())


class DemoHead(nn.Module):
    """88-dim motif vector + Stage-1 marker scores -> age distribution
    (5 brackets) and female-fraction. PVEngine-sized on purpose."""

    def __init__(self):
        super().__init__()
        d_in = 88 + N_MARKER
        self.trunk = nn.Sequential(nn.Linear(d_in, 24), nn.Tanh())
        self.age = nn.Linear(24, len(AGE_COLS))
        self.gender = nn.Linear(24, 1)

    def forward(self, x):
        h = self.trunk(x)
        return (F.log_softmax(self.age(h), dim=-1),
                torch.sigmoid(self.gender(h)).squeeze(-1))


def load_rows():
    if not os.path.exists(DATA):
        return None
    with open(DATA, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


if __name__ == "__main__":
    rows = load_rows()
    if rows is None:
        print(__doc__)
        print(f"\nno {os.path.basename(DATA)} found — export Meta Ad "
              "Library transparency data into the contract above, then "
              "re-run. Stage 1 markers keep working without it.")
        raise SystemExit(0)

    print(f"{len(rows)} labeled ads — building features...")
    from v4_motif_scorer import MotifScorer
    blob = torch.load(os.path.join(HERE, "v4_motif_cache.pt"),
                      weights_only=False)
    assert blob["vocab_hash"] == tok.vocab_hash()
    scorer = MotifScorer()
    mean, std = blob["mean"], blob["std"]

    X, y_age, y_gen = [], [], []
    for r in rows:
        z = (scorer.score(r["ad_text"]) - mean) / std
        X.append(torch.cat([z, marker_features(r["ad_text"])]))
        y_age.append([float(r[c]) for c in AGE_COLS])
        y_gen.append(float(r["gender_female"]))
    X = torch.stack(X)
    y_age = torch.tensor(y_age)
    y_age = y_age / y_age.sum(-1, keepdim=True).clamp(min=1e-8)
    y_gen = torch.tensor(y_gen)

    g = torch.Generator().manual_seed(1611)
    perm = torch.randperm(len(X), generator=g)
    cut = int(0.8 * len(X))
    tr, te = perm[:cut], perm[cut:]

    torch.manual_seed(1611)
    head = DemoHead()
    n = sum(p.numel() for p in head.parameters())
    print(f"DemoHead: {n:,} params (input {X.size(1)})")
    opt = torch.optim.AdamW(head.parameters(), lr=3e-3)
    for step in range(2_000):
        ix = tr[torch.randint(len(tr), (min(128, len(tr)),),
                              generator=g)]
        la, gen = head(X[ix])
        loss = (F.kl_div(la, y_age[ix], reduction="batchmean")
                + F.mse_loss(gen, y_gen[ix]))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        if step % 400 == 0:
            print(f"step {step:>4} loss {loss.item():.4f}", flush=True)

    head.eval()
    with torch.no_grad():
        la, gen = head(X[te])
    top1 = (la.argmax(-1) == y_age[te].argmax(-1)).float().mean()
    mae = (gen - y_gen[te]).abs().mean()
    print(f"\nheld-out: age-bracket top-1 {100 * top1:.1f}%  "
          f"gender-fraction MAE {mae:.3f}")
    torch.save({"model": head.state_dict(),
                "vocab_hash": tok.vocab_hash()}, CKPT)
    print(f"saved {os.path.basename(CKPT)}")
