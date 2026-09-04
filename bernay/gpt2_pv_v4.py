"""
GPT2PV V4 — the V2 backbone repurposed for marketing psychology.

V3 wired a Fibonacci gematria cipher into the parameter equation. V4 wires in
PERCEIVED VALUE:    PV = (Intensity x Desire) x T(time)
where Intensity and Desire are learned from market-segment data (customer
avatars, market trends, competitor ad angles — see v4_pv_engine.py) and T is
a time-to-solve decay. This module holds the shared pieces:

1. THE BACKBONE, reused verbatim. gpt2_bible_v2.py is spec-loaded and its
   module global VOCAB_SIZE is set to 96 (the fixed v4_tokenizer vocab)
   BEFORE any model is constructed — the V2 classes resolve VOCAB_SIZE from
   module globals at call time, so wte / lm_head_bias / the loss reshape all
   re-size with zero code duplication. Param count moves from 31,102 to
   exactly 31,102 + 22*(32+1) = 31,828 (22 extra vocab slots), asserted.

2. THE PV GATE, V3's pattern with one deliberate change: V3 baked one global
   gate into training; V4's gate is per-campaign, so the model TRAINS with a
   neutral gate (g = 1, numerically identical to V2) and the gate is swapped
   in at generation time, applied to the tied LM head only:
       logits = F.linear(x, diag(g) @ W_te, b)
   Same squash as V3, g = 1 + (phi-1)*tanh(z/phi) in [phi^-2, phi], zero new
   parameters. z comes from PV-weighted term frequencies (pv_char_gate).

3. SHARED TRAIN/EVAL MACHINERY — train_loop with the V3-25k lessons baked in
   (clip_grad_norm_ 1.0, isfinite abort with the offending-parameter dump),
   per-step-resumable checkpointing, and the eval_holdout.py bpc protocol.
   Checkpoints carry the tokenizer's vocab_hash and refuse to load on drift.
"""

import importlib.util
import json
import math
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import torch
import torch.nn.functional as F

import v4_tokenizer as tok

torch.set_num_threads(2)

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- backbone: spec-load V2, override vocab BEFORE constructing anything ----
_spec = importlib.util.spec_from_file_location(
    "gpt2_bible_v2", os.path.join(HERE, "gpt2_bible_v2.py"))
v2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v2)

v2.VOCAB_SIZE = tok.VOCAB_SIZE          # 74 -> 96, the load-bearing line
# Backbone width/depth are ENV-CONFIGURABLE so "gpt3" (the ~2x model) and
# the gpt2.5 fallback share one module: set V4_N_EMBD / V4_N_LAYER. The
# expansion 32 -> 48 (Collier+Hill corpus) cut held-out 2.398 -> 2.205
# bpc; gpt3 pushes width/depth further on the stats+correlations-grown
# corpus until held-out beats 2.205 ("size lands where it must"). The
# downstream interfaces (88-dim motif vectors, 353-dim PV feats) are
# unchanged; only the internal hidden width/depth grow.
v2.N_EMBD = int(os.environ.get("V4_N_EMBD", "48"))
v2.N_LAYER = int(os.environ.get("V4_N_LAYER", str(v2.N_LAYER)))
# CONTEXT LENGTH. 99 chars ~= 15 words: the backbone physically cannot hold a
# claim in paragraph 1 and its payoff in paragraph 6 in the same forward pass,
# and v4_motif_scorer then mean-pools the windows, discarding their order. That
# is the ceiling on how abstract any downstream read can be. Read at CONSTRUCTION
# time (wpe = nn.Embedding(BLOCK_SIZE, ...) and the tril mask buffer live in
# __init__), so this override must precede GPT2PV4(). WINDOW in v4_motif_scorer
# is `pv.v2.BLOCK_SIZE`, so the encoder follows automatically.
# A checkpoint is only loadable at the BLOCK_SIZE it was trained at (wpe shape).
v2.BLOCK_SIZE = int(os.environ.get("V4_BLOCK_SIZE", str(v2.BLOCK_SIZE)))
# baseline param count is pinned only for the original 48-wide/2-deep
# config (drift guard); other configs are measured + reported, not pinned
TARGET_PARAMS = 64_564 if (v2.N_EMBD == 48 and v2.N_LAYER == 2) else None
assert v2.N_EMBD % v2.N_HEAD == 0, "head_dim must stay integer"

PHI = (1 + 5 ** 0.5) / 2


class GPT2PV4(v2.GPT2BibleV2):
    """V2 backbone + a swappable per-character PV gate on the tied LM head.

    With pv_gate = ones (the training default) the forward pass is
    numerically identical to GPT2BibleV2."""

    def __init__(self):
        super().__init__()
        self.register_buffer("pv_gate", torch.ones(tok.VOCAB_SIZE))

    def set_gate(self, gate):
        """Swap in a per-campaign gate (or ones to neutralize). In-place so
        the buffer stays registered and checkpoint-compatible."""
        self.pv_gate.copy_(gate)

    def hidden(self, idx):
        """Post-ln_f hidden states (B, T, N_EMBD) — the representations the
        motif scorer pools. forward() builds its logits from these."""
        B, T = idx.size()
        assert T <= v2.BLOCK_SIZE
        pos = torch.arange(T, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)                    # input ungated
        X = x.unsqueeze(2).expand(B, T, v2.N_HC, v2.N_EMBD).contiguous()
        for layer in self.h:
            X = layer(X)
        return self.ln_f(X.mean(dim=2))

    def forward(self, idx, targets=None):
        x = self.hidden(idx)
        w = self.wte.weight * self.pv_gate.unsqueeze(1)      # diag(g) @ W_te
        logits = F.linear(x, w, self.lm_head_bias)           # gated tied head

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, tok.VOCAB_SIZE), targets.view(-1))
        return logits, loss


# ---- PV -> per-character gate (mirrors v3.gematria_gate) ---------------------
def pv_char_gate(term_weights, base_text, strength=1.0):
    """Per-vocab-character gate from PV-weighted terms, V3's squash shape.

    term_weights: {term: weight>=0} — taxonomy keywords / avatar desire terms
    weighted by the campaign's PV-relevant motif scores.
    base_text: reference corpus text giving the baseline char distribution.
    strength: scales the gate's deviation from 1. V3 trained WITH its gate,
    so the model adapted to the full [phi^-2, phi] range; V4 applies the
    gate at inference only, where full strength wrecks fluency (it crushes
    the space character and words fuse). ~0.35 biases without breaking.

    LETTERS ONLY: space, digits, and punctuation always gate at 1.0 — they
    carry sentence structure, not campaign meaning.

    v(ch) = PV-weighted char frequency of the term bag minus the baseline
    char frequency; standardized over letters; squashed like V3.
    """
    letters = torch.tensor([c.isalpha() for c in tok.VOCAB])
    term_counts = torch.zeros(tok.VOCAB_SIZE)
    for term, wgt in term_weights.items():
        if wgt <= 0:
            continue
        for i in tok.encode(term.lower()):
            term_counts[i] += float(wgt)
    base_counts = torch.zeros(tok.VOCAB_SIZE)
    sample = base_text[:200_000]
    for i in tok.encode(sample, normalized=True):
        base_counts[i] += 1.0

    term_counts[~letters] = 0.0
    base_counts[~letters] = 0.0
    p_term = term_counts / term_counts.sum().clamp(min=1e-8)
    p_base = base_counts / base_counts.sum().clamp(min=1e-8)
    v = p_term - p_base
    lv = v[letters]
    z = torch.zeros(tok.VOCAB_SIZE)
    z[letters] = (lv - lv.mean()) / (lv.std() + 1e-8)
    return 1.0 + strength * (PHI - 1.0) * torch.tanh(z / PHI)


# ---- data helpers ------------------------------------------------------------
def load_corpus_u8(path):
    """Corpus file -> uint8 id tensor (4 MB instead of 32 MB for 4M chars)."""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    ids = tok.encode(text, normalized=True)   # corpus files are pre-normalized
    return torch.frombuffer(bytearray(ids), dtype=torch.uint8).clone()


def get_batch(data_u8, batch_size=64):
    ix = torch.randint(len(data_u8) - v2.BLOCK_SIZE - 1, (batch_size,))
    x = torch.stack([data_u8[i: i + v2.BLOCK_SIZE].long() for i in ix])
    y = torch.stack([data_u8[i + 1: i + 1 + v2.BLOCK_SIZE].long() for i in ix])
    return x, y


# ---- checkpointing with vocab-drift protection -------------------------------
def save_ckpt(path, model, muon, adamw, step):
    torch.save({"vocab_hash": tok.vocab_hash(), "step": step,
                "model": model.state_dict(), "muon": muon.state_dict(),
                "adamw": adamw.state_dict()}, path)


def load_ckpt(path, model, muon=None, adamw=None):
    # map_location="cpu" is REQUIRED, not defensive: checkpoints trained in
    # .venv-bench (CUDA) carry cuda storage tags, and the app itself runs on
    # .venv-vlm, which is CPU-only torch. Without this a GPU-trained backbone
    # raises "Attempting to deserialize object on a CUDA device" and cannot
    # ship at all. Loading to CPU then .to(dev) is correct on both.
    ckpt = torch.load(path, weights_only=False, map_location="cpu")
    assert ckpt["vocab_hash"] == tok.vocab_hash(), \
        f"vocab drift: ckpt {ckpt['vocab_hash']} != {tok.vocab_hash()}"
    # gate-folded checkpoints (the production backbone) carry plain v2
    # weights with no pv_gate buffer; pv_gate stays at its neutral init
    # (ones), which is exactly right for a folded model. Anything else
    # missing/unexpected is a real error.
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    assert not unexpected, f"unexpected keys: {unexpected}"
    assert set(missing) <= {"pv_gate"}, f"missing keys: {missing}"
    if muon is not None and "muon" in ckpt:
        muon.load_state_dict(ckpt["muon"])
    if adamw is not None and "adamw" in ckpt:
        adamw.load_state_dict(ckpt["adamw"])
    return ckpt["step"]


# ---- shared training loop (V3-25k lessons baked in) ---------------------------
def train_loop(model, data_u8, steps, batch_size=64, lr_muon=1.5e-2,
               lr_adamw=3e-3, log_every=100, ckpt_path=None, ckpt_every=500,
               resume=True, clip=1.0):
    """Returns (final_step, last_loss). Aborts on non-finite loss with the
    param-name diagnostic from v3_25k_diagnose.py. Resumable per ckpt_every
    if ckpt_path is given."""
    muon, adamw = v2.build_optimizers(model, lr_adamw=lr_adamw,
                                      lr_muon=lr_muon)
    start = 0
    if resume and ckpt_path and os.path.exists(ckpt_path):
        start = load_ckpt(ckpt_path, model, muon, adamw)
        # load_state_dict restores the checkpoint's param_groups (incl. lr);
        # re-apply the lrs requested for THIS run
        for g in muon.param_groups:
            g["lr"] = lr_muon
        for g in adamw.param_groups:
            g["lr"] = lr_adamw
        print(f"resumed from {os.path.basename(ckpt_path)} at step {start}",
              flush=True)
    if start >= steps:
        print(f"already trained to step {start} >= {steps}", flush=True)
        return start, float("nan")

    model.train()
    t0, loss = time.time(), None
    for step in range(start, steps):
        x, y = get_batch(data_u8, batch_size)
        _, loss = model(x, y)
        if not torch.isfinite(loss):
            bad = [n for n, p in model.named_parameters()
                   if not torch.isfinite(p).all()]
            raise RuntimeError(
                f"NON-FINITE loss at step {step}; bad params: {bad or 'none'}")
        muon.zero_grad(set_to_none=True)
        adamw.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        # NaN/inf-gradient guard: one bad batch makes clip_grad_norm_ return a
        # non-finite norm, whose scale poisons EVERY weight into NaN (the deep
        # big-model late Muon spikes that crashed the run at 6387/8409). Skip
        # the update on a non-finite norm so weights stay finite and training
        # continues instead of aborting. Healthy steps are unaffected.
        if torch.isfinite(gnorm):
            muon.step()
            adamw.step()
        if step % log_every == 0:
            rate = (step - start + 1) / (time.time() - t0)
            print(f"step {step:>5}  loss {loss.item():.4f}  "
                  f"({rate:.2f} it/s)", flush=True)
        if ckpt_path and (step + 1) % ckpt_every == 0:
            save_ckpt(ckpt_path, model, muon, adamw, step + 1)
    if ckpt_path:
        save_ckpt(ckpt_path, model, muon, adamw, steps)
    return steps, loss.item() if loss is not None else float("nan")


@torch.no_grad()
def eval_bpc(model, data_u8, n_batches=200, batch_size=64):
    """Held-out bits/char, eval_holdout.py protocol (seed 7)."""
    model.eval()
    torch.manual_seed(7)
    losses = [model(*get_batch(data_u8, batch_size))[1].item()
              for _ in range(n_batches)]
    model.train()
    return (sum(losses) / len(losses)) / math.log(2)


def stamp_results(path, extra):
    """Merge extra into a results JSON, always stamping the vocab hash."""
    results = {}
    if os.path.exists(path):
        with open(path) as f:
            results = json.load(f)
    results.update(extra)
    results["vocab_hash"] = tok.vocab_hash()
    with open(path, "w") as f:
        json.dump(results, f, indent=1)
    return results


if __name__ == "__main__":
    print(f"vocab: {tok.VOCAB_SIZE} chars, hash {tok.vocab_hash()}\n")

    torch.manual_seed(1611)
    m = GPT2PV4()
    groups = {}
    for name, p in m.named_parameters():
        key = name.split(".")[0] if not name.startswith("h.") else \
              "blocks (mHC+attn+mlp)"
        groups[key] = groups.get(key, 0) + p.numel()
    print(f"{'component group':<22}{'params':>8}")
    print("-" * 30)
    for k, v in groups.items():
        print(f"{k:<22}{v:>8,}")
    total = sum(p.numel() for p in m.parameters())
    print("-" * 30)
    print(f"{'TOTAL':<22}{total:>8,}")
    if TARGET_PARAMS is not None:
        assert total == TARGET_PARAMS, f"param drift: {total:,}"
        print(f"= pinned baseline {TARGET_PARAMS:,} OK\n")
    else:
        print(f"= measured for V4_N_EMBD={v2.N_EMBD}, "
              f"V4_N_LAYER={v2.N_LAYER} (not pinned)\n")

    # neutral-gate equivalence: same seed => identical weights => identical
    # logits between GPT2PV4(gate=1) and the raw V2 model
    torch.manual_seed(1611)
    ref = v2.GPT2BibleV2()
    x = torch.randint(0, tok.VOCAB_SIZE, (2, 32))
    la, _ = m(x)
    lb, _ = ref(x)
    assert torch.allclose(la, lb, atol=1e-6), "neutral gate != V2"
    print("neutral-gate equivalence vs V2: OK")
    del ref

    # a non-neutral gate must change the head and stay in V3's bounds
    g = pv_char_gate({"desire": 3.0, "power": 2.0, "value": 1.0},
                     "the quick brown fox " * 500)
    assert (g.min() >= PHI ** -2 - 1e-4) and (g.max() <= PHI + 1e-4)
    m.set_gate(g)
    lc, _ = m(x)
    assert not torch.allclose(la, lc), "gate had no effect"
    m.set_gate(torch.ones(tok.VOCAB_SIZE))
    print(f"pv_char_gate bounds OK: [{g.min():.3f}, {g.max():.3f}] "
          f"(phi^-2={PHI**-2:.3f}, phi={PHI:.3f})")

    # 20-step smoke train on random data: clipping + finiteness path
    data = torch.randint(0, tok.VOCAB_SIZE, (5_000,), dtype=torch.uint8)
    _, last = train_loop(m, data, steps=20, batch_size=16, log_every=10)
    print(f"\n20-step smoke train OK, last loss {last:.4f} "
          f"(random data, ~ln96={math.log(96):.3f} expected)")
    print("\ngpt2_pv_v4 self-test: ALL OK")
