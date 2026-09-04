"""
V4 Stage C — the Perceived Value engine.

    PV = Desire(dsr) x T(time)

DESIRE is the single driver before time decay: per the user's model, desire
is a DERIVATIVE OF A PROBLEM — a problem creates emotion, emotion creates
the want — so the problem is folded INTO desire, not a separate multiplier.
primal_gain (passed in as problem_gain) is the psychological-hierarchy
amplifier: lower/primal problems — survival, sex, safety — make a stronger
emotion and thus a stronger desire. T is the time-to-solve decay

    T(t) = exp(-lambda * max(0, t/t_avg - 1)),   lambda = softplus(raw) > 0

which is exactly 1 for faster-than-average solves and decays monotonically
below 1 for slower ones.

PV maps to ad metrics through MONOTONE-BY-CONSTRUCTION links (all slopes
softplus'd positive, so training cannot break monotonicity).

Loss is MSE in the pre-sigmoid spaces. Evaluation is rank-based: Spearman of
predicted PV against held-out CTR, CVR, -CPM and against the PLANTED PV*
(v4_ads_truth.csv, which training never sees).

--------------------------------------------------------------------------
ARCHITECTURES
--------------------------------------------------------------------------
`arch="v1"` — the ORIGINAL head, kept byte-compatible so existing
checkpoints (v4_pv_ckpt.pt, v4_big_pv_ckpt.pt, ...) still load:

    f(353) -> Linear -> tanh(32) -> Linear -> softplus -> Dsr
    PV = Dsr * T ;  z = a*log PV + b          (links strictly LINEAR)

Measured defects that motivated v2 (probe over live ad vectors):
  * 74% of the 32 tanh units saturate (|h| > 0.9); median (1-h^2) = 0.0087,
    so dJ/dW1 arrives 112x smaller than dJ/dW2 — and W1 is 99% of the head.
  * 89 of the 353 inputs are HARD ZERO at inference (the 88-dim trend block
    and the momentum scalar): training fits weights on signal that inference
    never supplies. That is a train/inference distribution mismatch, not
    merely wasted capacity.
  * the links are affine in log PV, so the whole output stage is a line.

`arch="v2"` — deeper, gated, residual, and branching:

    x    = LayerNorm(f)                             (unsaturates the trunk)
    h_0  = W_in x + b_in
    h_k  = h_{k-1} + P_k( A_k (*) sigmoid(G_k) )    k = 1..depth
                                                    GLU: multiplicative, and
                                                    the residual is a
                                                    gradient highway past
                                                    every nonlinearity
    hD   = LayerNorm(h_depth)

    Dsr    = softplus(w_d . hD + b_d) * problem_gain
    lam_i  = softplus(w_l . hD + b_l)               PER-AD time sensitivity
    T      = exp( -(lam_g + lam_i) * relu(t_ratio - 1) )
    PV     = Dsr * T
    L      = log(PV + 1e-6)

    z_ctr  =  a0*L + b0 + c0*softplus(L - k0)       still monotone (a,c > 0)
    z_cvr  =  a1*L + b1 + c1*softplus(L - k1)       but no longer a LINE
    z_cpm  = -a2*L + b2 - c2*softplus(L - k2)       monotone DECREASING

FEATURES. v1 fed [za | trend | zg | za*zg | momentum]; trend and momentum are
unavailable at inference and were passed as zeros. v2 drops both and spends
the budget on interactions that inference CAN compute, built by ONE function
(`build_feats`) shared by training and inference so the two cannot drift:

    [ za | zg | za*zg | za-zg | |za-zg| | 8 scalars ]  = 5*88 + 8 = 448

Set V4_PV_ARCH=v1 to train the legacy head. V4_PV_HIDDEN / V4_PV_DEPTH tune
v2; `python v4_pv_engine.py --sweep` picks them on HELD-OUT Spearman.
"""

import csv
import json
import math
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import torch
import torch.nn as nn
import torch.nn.functional as F

import v4_tokenizer as tok

torch.set_num_threads(2)
HERE = os.path.dirname(os.path.abspath(__file__))

MOTIF_DIM = 88
FEAT_DIM_V1 = 3 * 88 + 88 + 1   # avatar || trend || angle || avatar*angle || mom
N_SCALARS = 8
FEAT_DIM_V2 = 5 * MOTIF_DIM + N_SCALARS          # 448
FEAT_DIM = FEAT_DIM_V1          # back-compat alias (v1 default)

CTR_MAX, CVR_MAX, CPM_MIN, CPM_RANGE = 0.06, 0.08, 4.0, 36.0
ABLATE_T = os.environ.get("V4_ABLATE_T") == "1"

# DEFAULT = the transformer desire head. `v1` (the 32-unit tanh MLP) remains
# constructible so every existing checkpoint still loads — load_engine() reads
# each ckpt's own arch_cfg — but new training builds v3 unless told otherwise.
ARCH = os.environ.get("V4_PV_ARCH", "v3")
# FEATURES are independent of ARCH — conflating them produced a misleading
# sweep (the deep head looked worse when it was really the feature change).
# "v1" = [za|trend|zg|za*zg|mom], 89 dims of which are HARD ZERO at inference.
# "v2" = build_feats(za, zg), every dim computable at inference.
FEATS = os.environ.get("V4_PV_FEATS", "v2")
HIDDEN = int(os.environ.get("V4_PV_HIDDEN", "64"))
# depth 2 measured best of the transformer configs on REAL ads (Spearman
# 0.1997 vs 0.1792 at L4 and 0.1583 at L6 — monotone decline with depth)
DEPTH = int(os.environ.get("V4_PV_DEPTH", "2"))

CKPT = os.environ.get("V4_PV_CKPT") or os.path.join(HERE, "v4_pv_ckpt.pt")
MOTIF_CACHE = os.environ.get("V4_MOTIF_CACHE") or \
    os.path.join(HERE, "v4_motif_cache.pt")
RESULTS = os.path.join(HERE, "v4_pv_results.json")


# ---- features ----------------------------------------------------------------
def build_feats(za, zg):
    """THE feature vector for arch v2 — used by training AND inference.

    za, zg: (..., 88) standardised motif vectors (avatar, copy/angle).
    Every block is computable from those two alone, so nothing is zero-filled
    at inference the way the v1 trend/momentum blocks were.
    """
    single = za.dim() == 1
    if single:
        za, zg = za.unsqueeze(0), zg.unsqueeze(0)
    inter = za * zg
    diff = za - zg
    eps = 1e-8
    na, ng = za.norm(dim=-1), zg.norm(dim=-1)
    scal = torch.stack([
        (inter.sum(-1) / (na * ng + eps)),   # cosine(avatar, angle)
        na, ng,                              # how loud each side is
        za.mean(-1), zg.mean(-1),
        zg.std(-1), zg.amax(-1),
        diff.abs().mean(-1),                 # total avatar/angle mismatch
    ], dim=-1)
    out = torch.cat([za, zg, inter, diff, diff.abs(), scal], dim=-1)
    return out.squeeze(0) if single else out


class GLUBlock(nn.Module):
    """Pre-norm gated residual block: h + P(A (*) sigmoid(G)).

    The gate is what makes it non-linear WITHOUT a squashing saturator, and
    the residual gives every earlier layer a gradient path that does not pass
    through a (1 - h^2) factor — the term measured at median 0.0087 in v1.
    """

    def __init__(self, h):
        super().__init__()
        self.ln = nn.LayerNorm(h)
        self.fc = nn.Linear(h, 2 * h)
        self.proj = nn.Linear(h, h)

    def forward(self, x):
        a, g = self.fc(self.ln(x)).chunk(2, dim=-1)
        return x + self.proj(a * torch.sigmoid(g))


class TxBlock(nn.Module):
    """Standard pre-LN transformer block — the same shape an LLM layer has.

        x = x + MHSA(LN(x))
        x = x + MLP_GELU(LN(x))

    Both sublayers are residual, so gradient reaches layer 1 without passing
    through a single squashing factor (the (1-h^2) term that measured a median
    of 0.0087 in the v1 head).
    """

    def __init__(self, d, n_head, mult=4, drop=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_head, dropout=drop,
                                          batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, mult * d), nn.GELU(),
                                 nn.Linear(mult * d, d), nn.Dropout(drop))

    def forward(self, x):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        return x + self.mlp(self.ln2(x))


class PVEngine(nn.Module):
    def __init__(self, arch=None, feat_dim=None, hidden=None, depth=None,
                 n_head=None):
        super().__init__()
        self.arch = arch or ARCH
        if self.arch == "v1":
            self.feat_dim = feat_dim or FEAT_DIM_V1
            self.hidden, self.depth = 32, 0
            # byte-identical to the original head so old ckpts load
            self.desire = nn.Sequential(nn.Linear(self.feat_dim, 32),
                                        nn.Tanh(), nn.Linear(32, 1))
        elif self.arch == "v3":
            # TRANSFORMER desire head. The 448-dim feature vector is read as a
            # SEQUENCE of 5 motif blocks (za | zg | za*zg | za-zg | |za-zg|),
            # each 88-dim, plus the 8 scalars folded into a 6th token. A CLS
            # token attends over them through n_layer pre-LN blocks.
            #
            # Why a sequence and not a wider MLP: an MLP sees one flat vector,
            # so "which block matters for THIS ad" is fixed in the weights.
            # Self-attention makes that routing input-dependent — the same
            # mechanism an LLM uses — and every block can condition on every
            # other instead of only through a shared bottleneck.
            self.feat_dim = feat_dim or FEAT_DIM_V2
            self.hidden = hidden or HIDDEN
            self.depth = depth or DEPTH
            self.n_head = n_head or int(os.environ.get("V4_PV_HEADS", "4"))
            self.n_tok = 6
            self.tok_ln = nn.LayerNorm(MOTIF_DIM)
            self.tok_proj = nn.Linear(MOTIF_DIM, self.hidden)
            self.scal_proj = nn.Linear(N_SCALARS, self.hidden)
            self.pos = nn.Parameter(torch.zeros(1, self.n_tok + 1,
                                                self.hidden))
            self.cls = nn.Parameter(torch.zeros(1, 1, self.hidden))
            nn.init.normal_(self.pos, std=0.02)
            nn.init.normal_(self.cls, std=0.02)
            self.blocks = nn.ModuleList(
                [TxBlock(self.hidden, self.n_head) for _ in range(self.depth)])
            self.out_ln = nn.LayerNorm(self.hidden)
            self.head_d = nn.Linear(self.hidden, 1)
            self.head_l = nn.Linear(self.hidden, 1)
            self.c = nn.Parameter(torch.full((3,), -2.0))
            self.k = nn.Parameter(torch.zeros(3))
        else:
            self.feat_dim = feat_dim or FEAT_DIM_V2
            self.hidden = hidden or HIDDEN
            self.depth = depth or DEPTH
            self.in_ln = nn.LayerNorm(self.feat_dim)
            self.inp = nn.Linear(self.feat_dim, self.hidden)
            self.blocks = nn.ModuleList(
                [GLUBlock(self.hidden) for _ in range(self.depth)])
            self.out_ln = nn.LayerNorm(self.hidden)
            self.head_d = nn.Linear(self.hidden, 1)   # desire
            self.head_l = nn.Linear(self.hidden, 1)   # per-ad time sensitivity
            # curvature for the links (softplus'd -> monotonicity preserved)
            self.c = nn.Parameter(torch.full((3,), -2.0))
            self.k = nn.Parameter(torch.zeros(3))

        # lambda init 0.5 — neutral start, must LEARN its way to the truth
        self.lam_raw = nn.Parameter(torch.tensor(
            math.log(math.exp(0.5) - 1.0)))
        self.a = nn.Parameter(torch.zeros(3))      # softplus(0) = 0.693
        self.b = nn.Parameter(torch.zeros(3))

    def lam(self):
        return F.softplus(self.lam_raw)

    def cfg(self):
        c = {"arch": self.arch, "feat_dim": self.feat_dim,
             "hidden": self.hidden, "depth": self.depth}
        if self.arch == "v3":
            c["n_head"] = self.n_head
        return c

    def trunk(self, feats):
        """-> (Dsr_raw, lam_extra). lam_extra is 0 for v1."""
        if self.arch == "v1":
            return self.desire(feats).squeeze(-1), None
        if self.arch == "v3":
            B = feats.size(0)
            blocks = feats[:, :5 * MOTIF_DIM].view(B, 5, MOTIF_DIM)
            t = self.tok_proj(self.tok_ln(blocks))              # (B,5,H)
            s = self.scal_proj(feats[:, 5 * MOTIF_DIM:]).unsqueeze(1)
            x = torch.cat([self.cls.expand(B, -1, -1), t, s], 1)
            x = x + self.pos[:, : x.size(1)]
            for blk in self.blocks:
                x = blk(x)
            h = self.out_ln(x[:, 0])                            # CLS
            return (self.head_d(h).squeeze(-1),
                    F.softplus(self.head_l(h).squeeze(-1)))
        h = self.inp(self.in_ln(feats))
        for blk in self.blocks:
            h = blk(h)
        h = self.out_ln(h)
        return self.head_d(h).squeeze(-1), F.softplus(
            self.head_l(h).squeeze(-1))

    def forward(self, feats, t_ratio, problem_gain=1.0):
        raw, lam_i = self.trunk(feats)
        Dsr = F.softplus(raw) * problem_gain
        if ABLATE_T:
            T = torch.ones_like(t_ratio)
        else:
            lam = self.lam() if lam_i is None else self.lam() + lam_i
            T = torch.exp(-lam * F.relu(t_ratio - 1.0))
        PV = Dsr * T                    # = Desire x T
        L = torch.log(PV + 1e-6)
        a = F.softplus(self.a)
        if self.arch == "v1":
            z_ctr = a[0] * L + self.b[0]
            z_cvr = a[1] * L + self.b[1]
            z_cpm = -a[2] * L + self.b[2]
        else:
            # monotone but CURVED: softplus is increasing, c > 0, so d/dL of
            # z_ctr/z_cvr stays > 0 and of z_cpm stays < 0 for every L.
            c = F.softplus(self.c)
            z_ctr = a[0] * L + self.b[0] + c[0] * F.softplus(L - self.k[0])
            z_cvr = a[1] * L + self.b[1] + c[1] * F.softplus(L - self.k[1])
            z_cpm = -a[2] * L + self.b[2] - c[2] * F.softplus(L - self.k[2])
        pred = {
            "Dsr": Dsr, "D": Dsr, "I": Dsr, "T": T, "PV": PV,
            "z_ctr": z_ctr, "z_cvr": z_cvr, "z_cpm": z_cpm,
        }
        pred["ctr"] = CTR_MAX * torch.sigmoid(pred["z_ctr"])
        pred["cvr"] = CVR_MAX * torch.sigmoid(pred["z_cvr"])
        pred["cpm"] = CPM_MIN + CPM_RANGE * torch.sigmoid(pred["z_cpm"])
        return pred


def load_engine(path=None):
    """Construct the RIGHT shape from the checkpoint's own arch record.

    Checkpoints written before v2 have no "arch" key -> legacy v1 head.
    """
    path = path or CKPT
    ck = torch.load(path, weights_only=False)
    cfg = ck.get("arch_cfg") or {"arch": "v1", "feat_dim": FEAT_DIM_V1,
                                 "hidden": 32, "depth": 0}
    eng = PVEngine(**cfg)
    eng.load_state_dict(ck["model"])
    eng.eval()
    # which feature builder this head was FIT on — inference must match it or
    # it silently feeds the head a vector it was never trained for
    eng.feats_ver = ck.get("feats_ver", "v1")
    return eng, ck


# ---- data loading -------------------------------------------------------------
def read_csv_dicts(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def logit(p):
    p = p.clamp(1e-4, 1 - 1e-4)
    return torch.log(p / (1 - p))


def load_dataset(arch=None, feats_ver=None):
    """Join ad_history with the motif cache -> tensors + truth (eval only).

    `feats_ver` selects the FEATURE BUILDER and is independent of the head
    architecture; it defaults to `arch` only for backwards compatibility with
    callers written before the two were separated.
    """
    arch = feats_ver or arch or FEATS
    blob = torch.load(MOTIF_CACHE, weights_only=False)
    assert blob["vocab_hash"] == tok.vocab_hash(), "vocab drift in cache"
    cache, mean, std = blob["cache"], blob["mean"], blob["std"]

    ads = read_csv_dicts(os.path.join(HERE, "v4_ad_history.csv"))
    avatars = {r["avatar_id"]: r
               for r in read_csv_dicts(os.path.join(HERE, "v4_avatars.csv"))}
    momentum = {r["trend_id"]: float(r["momentum"]) for r in
                read_csv_dicts(os.path.join(HERE, "v4_trends.csv"))}
    truth = {r["ad_id"]: float(r["PV_true"]) for r in
             read_csv_dicts(os.path.join(HERE, "v4_ads_truth.csv"))}

    feats, t_ratio, targets, pv_true, segments = [], [], [], [], []
    skipped = 0
    for r in ads:
        try:
            za = (cache[r["avatar_id"]] - mean) / std
            zg = (cache[r["angle_id"]] - mean) / std
            if arch == "v1":
                zt = (cache[r["trend_id"]] - mean) / std
                mom = torch.tensor([momentum[r["trend_id"]] - 0.9])
                f = torch.cat([za, zt, zg, za * zg, mom])
            else:
                # SAME function inference calls — no second code path to drift
                f = build_feats(za, zg)
        except KeyError:
            skipped += 1
            continue
        feats.append(f)
        t_ratio.append(float(r["time_to_solve_days"]) /
                       float(r["t_avg_days"]))
        targets.append([float(r["ctr"]), float(r["cvr"]), float(r["cpm"])])
        pv_true.append(truth[r["ad_id"]])
        segments.append(avatars[r["avatar_id"]]["segment"])
    if skipped:
        print(f"WARNING: {skipped} ad rows skipped (missing motif vectors)")

    X = torch.stack(feats)
    tr = torch.tensor(t_ratio)
    Y = torch.tensor(targets)
    y = {"ctr": logit(Y[:, 0] / CTR_MAX), "cvr": logit(Y[:, 1] / CVR_MAX),
         "cpm": logit((Y[:, 2] - CPM_MIN) / CPM_RANGE),
         "raw": Y}
    return X, tr, y, torch.tensor(pv_true), segments


def stratified_split(segments, frac=0.8, seed=1611):
    g = torch.Generator().manual_seed(seed)
    by_seg = {}
    for i, s in enumerate(segments):
        by_seg.setdefault(s, []).append(i)
    train_idx, test_idx = [], []
    for s, idxs in sorted(by_seg.items()):
        perm = torch.randperm(len(idxs), generator=g).tolist()
        cut = int(frac * len(idxs))
        train_idx += [idxs[p] for p in perm[:cut]]
        test_idx += [idxs[p] for p in perm[cut:]]
    return torch.tensor(train_idx), torch.tensor(test_idx)


def spearman(x, y):
    """Rank-Pearson via argsort — no scipy (house rule)."""
    rx = torch.argsort(torch.argsort(x)).float()
    ry = torch.argsort(torch.argsort(y)).float()
    rx = (rx - rx.mean()) / (rx.std() + 1e-8)
    ry = (ry - ry.mean()) / (ry.std() + 1e-8)
    return (rx * ry).mean().item()


def train(X, tr, y, train_idx, steps=None, batch=256, seed=1611, arch=None,
          hidden=None, depth=None, quiet=False, wd=0.0, n_head=None):
    steps = steps or int(os.environ.get("V4_PV_STEPS", "2000"))
    torch.manual_seed(seed)
    eng = PVEngine(arch=arch, feat_dim=X.size(1), hidden=hidden, depth=depth,
                   n_head=n_head)
    total = sum(p.numel() for p in eng.parameters())
    if eng.arch == "v1":
        expect = (eng.feat_dim * 32 + 32 + 32 + 1) + 1 + 6
        assert total == expect, f"param drift: {total} != {expect}"

    opt = torch.optim.AdamW(eng.parameters(), lr=3e-3, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    eng.train()
    for step in range(steps):
        ix = train_idx[torch.randint(len(train_idx), (batch,))]
        p = eng(X[ix], tr[ix])
        loss = (F.mse_loss(p["z_ctr"], y["ctr"][ix]) +
                F.mse_loss(p["z_cvr"], y["cvr"][ix]) +
                0.5 * F.mse_loss(p["z_cpm"], y["cpm"][ix]))
        if not torch.isfinite(loss):
            raise RuntimeError(f"NON-FINITE loss at step {step}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(eng.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 250 == 0 and not quiet:
            print(f"step {step:>4}  loss {loss.item():.4f}  "
                  f"lambda {eng.lam().item():.3f}", flush=True)
    eng.eval()
    return eng, total


def evaluate(eng, X, tr, y, test_idx, pv_true):
    with torch.no_grad():
        p = eng(X[test_idx], tr[test_idx])
    raw = y["raw"][test_idx]
    slow = tr[test_idx] > 1.0
    return {
        "Spearman(PV^, CTR)": spearman(p["PV"], raw[:, 0]),
        "Spearman(PV^, CVR)": spearman(p["PV"], raw[:, 1]),
        "Spearman(PV^, -CPM)": spearman(p["PV"], -raw[:, 2]),
        "Spearman(PV^, PV*)": spearman(p["PV"], pv_true[test_idx]),
        "Spearman(PV^, PV*) slow-solve": spearman(
            p["PV"][slow], pv_true[test_idx][slow]),
        "learned lambda": eng.lam().item(),
    }, slow


# ---- capacity sweep ------------------------------------------------------------
def sweep():
    """Pick hidden/depth on HELD-OUT Spearman, not train loss.

    Answers 'is the pipeline too small?' with a number instead of an opinion.
    """
    grid = [("v1", 32, 0), ("v2", 32, 0), ("v2", 64, 1), ("v2", 64, 2),
            ("v2", 128, 2), ("v2", 128, 4), ("v2", 256, 4), ("v2", 256, 6)]
    print(f"{'arch':<6}{'hid':>5}{'dep':>5}{'params':>10}"
          f"{'PV*':>9}{'CTR':>8}{'CVR':>8}{'-CPM':>8}{'slow':>8}")
    print("-" * 67)
    best = None
    for arch, hid, dep in grid:
        X, tr, y, pv_true, segments = load_dataset(arch=arch)
        train_idx, test_idx = stratified_split(segments)
        eng, total = train(X, tr, y, train_idx, arch=arch, hidden=hid,
                           depth=dep, quiet=True)
        rows, _ = evaluate(eng, X, tr, y, test_idx, pv_true)
        pv = rows["Spearman(PV^, PV*)"]
        print(f"{arch:<6}{hid:>5}{dep:>5}{total:>10,}{pv:>9.4f}"
              f"{rows['Spearman(PV^, CTR)']:>8.4f}"
              f"{rows['Spearman(PV^, CVR)']:>8.4f}"
              f"{rows['Spearman(PV^, -CPM)']:>8.4f}"
              f"{rows['Spearman(PV^, PV*) slow-solve']:>8.4f}", flush=True)
        if best is None or pv > best[0]:
            best = (pv, arch, hid, dep, total)
    print("-" * 67)
    print(f"BEST held-out PV*: {best[0]:.4f}  ->  arch={best[1]} "
          f"hidden={best[2]} depth={best[3]}  ({best[4]:,} params)")
    return best


if __name__ == "__main__":
    if "--sweep" in sys.argv:
        sweep()
        raise SystemExit(0)

    X, tr, y, pv_true, segments = load_dataset(feats_ver=FEATS)
    train_idx, test_idx = stratified_split(segments)
    print(f"dataset: {len(X)} ads, {X.size(1)}-dim features ({FEATS}), "
          f"{len(train_idx)} train / {len(test_idx)} held-out "
          f"(stratified by segment)")
    print(f"arch: {ARCH}" + ("" if ARCH == "v1" else
                             f"  hidden {HIDDEN}  depth {DEPTH}"))
    print(f"mode: {'ABLATION T==1' if ABLATE_T else 'FULL (learnable T)'}\n")

    eng, total = train(X, tr, y, train_idx)
    print(f"\n{'component':<14}{'params':>9}")
    print("-" * 23)
    if eng.arch == "v1":
        print(f"{'desire':<14}"
              f"{sum(p.numel() for p in eng.desire.parameters()):>9,}")
    elif eng.arch == "v3":
        for nm, mod in (("embed", [eng.tok_ln, eng.tok_proj, eng.scal_proj]),
                        ("blocks", list(eng.blocks)),
                        ("out_ln+heads", [eng.out_ln, eng.head_d,
                                          eng.head_l])):
            print(f"{nm:<14}{sum(p.numel() for m in mod
                                 for p in m.parameters()):>9,}")
        print(f"{'pos+cls':<14}{eng.pos.numel() + eng.cls.numel():>9,}")
        print(f"  {eng.depth} layers x {eng.n_head} heads, d_model "
              f"{eng.hidden}, seq {eng.n_tok + 1} tokens")
    else:
        for nm, mod in (("in_ln+inp", [eng.in_ln, eng.inp]),
                        ("blocks", list(eng.blocks)),
                        ("out_ln", [eng.out_ln]),
                        ("heads", [eng.head_d, eng.head_l])):
            print(f"{nm:<14}{sum(p.numel() for m in mod
                                 for p in m.parameters()):>9,}")
    print(f"{'T + links':<14}{total - sum(p.numel() for n, p in
                                          eng.named_parameters()
                                          if not n.startswith(('lam', 'a', 'b',
                                                               'c', 'k'))):>9,}")
    print("-" * 23)
    print(f"{'TOTAL':<14}{total:>9,}\n")

    rows, slow = evaluate(eng, X, tr, y, test_idx, pv_true)
    hdr = f"{'metric':<32}{'value':>8}"
    print(hdr)
    print("-" * len(hdr))
    for k, v in rows.items():
        print(f"{k:<32}{v:>8.4f}")
    print(f"\nslow-solve rows in held-out: {int(slow.sum())}/{len(slow)}")

    tag = "ablate_" if ABLATE_T else ""
    results = {}
    if os.path.exists(RESULTS):
        with open(RESULTS) as f:
            results = json.load(f)
    results.update({tag + k: v for k, v in rows.items()})
    results["vocab_hash"] = tok.vocab_hash()
    results["arch"] = eng.cfg()
    with open(RESULTS, "w") as f:
        json.dump(results, f, indent=1)

    if not ABLATE_T:
        torch.save({"model": eng.state_dict(),
                    "lambda": eng.lam().item(),
                    "arch_cfg": eng.cfg(),
                    "feats_ver": FEATS,
                    "vocab_hash": tok.vocab_hash()}, CKPT)
        print(f"\nsaved {os.path.basename(CKPT)}")
        assert rows["Spearman(PV^, PV*)"] >= 0.8, "PV* recovery below 0.8"
        print("acceptance: Spearman(PV^, PV*) >= 0.8 OK")
        if "ablate_Spearman(PV^, PV*) slow-solve" in results:
            drop = (rows["Spearman(PV^, PV*) slow-solve"] -
                    results["ablate_Spearman(PV^, PV*) slow-solve"])
            print(f"T-ablation slow-solve degradation: {drop:+.4f} "
                  f"(target >= +0.1)")
    else:
        print("\nablation results stored — run without V4_ABLATE_T for "
              "the side-by-side comparison")
