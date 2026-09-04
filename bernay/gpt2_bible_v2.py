"""
GPT2Bible V2 — same 31,102 parameters, rebuilt with DeepSeek-V4 techniques.

What changed vs. the original (and where it comes from in the paper):

1. RESIDUAL PROPAGATION REDONE — Manifold-Constrained Hyper-Connections (§2.2)
   Ordinary equation:   x_{l+1} = x_l + F(LN(x_l))
   mHC equation (Eq.1): X_{l+1} = B_l X_l + C_l F(A_l X_l)
   where the residual stream is widened to n_hc=2 copies, B_l is projected
   onto the manifold of doubly stochastic matrices with Sinkhorn-Knopp
   (Eq. 8, t_max=20) so ||B_l||_2 <= 1 (non-expansive propagation), and
   A_l = sigmoid(~A), C_l = 2*sigmoid(~C) (Eqs. 6-7). All three are
   dynamically generated from the input (Eqs. 3-5).

2. PARAMETER DEBLOAT — the original spent 22,496/31,102 params (72%) on a
   1,406-token embedding while the data uses 74 characters. V2 right-sizes
   the vocab and reinvests the budget in compute:
        vocab 1406 -> 74,  d_model 16 -> 32,  heads 2 -> 4,  context 38 -> 99
   Attention uses Shared-KV MQA (§2.3.1: "each compressed KV entry serves
   as both attention key and value"), halving KV projection params, plus
   RMSNorm on q/k to prevent exploding logits (§2.4) and per-head learnable
   gains (cf. the indexer head weights, Eq. 15-16).

3. MUON OPTIMIZER (§2.4, Algorithm 1) — momentum + Nesterov + hybrid
   Newton-Schulz orthogonalization (8 steps with a,b,c = 3.4445, -4.7750,
   2.0315, then 2 steps with 2, -1.5, 0.5; Eq. 28), update rescaled by
   sqrt(max(n,m)) * gamma. AdamW is kept for embeddings, biases, norms,
   and the mHC static/gating params, exactly as the paper prescribes.

Total parameter count: exactly 31,102. Still one per KJV verse. Still no heresy.
"""

import json
import math
import os
import time

# cap BLAS thread pools BEFORE torch loads — unbounded OpenBLAS thread
# buffers exhaust memory on this machine ("Memory allocation still failed")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(2)
torch.manual_seed(1611)

# ---- V2 config: same budget, spent on compute instead of vocab --------------
VOCAB_SIZE = 74     # actual KJV character vocabulary (was 1406)
BLOCK_SIZE = 99     # context length (was 38) — "more info can run"
N_LAYER    = 2
N_HEAD     = 4      # head_dim = 8
N_EMBD     = 32     # d_model (was 16)
N_HC       = 2      # mHC residual stream width
SINKHORN_T = 20     # paper's t_max

TARGET_PARAMS = 31_102


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), self.weight)


class SharedKVAttention(nn.Module):
    """Causal attention with shared key/value projection (DeepSeek-V4 §2.3:
    each KV entry serves as both key and value), QK-RMSNorm (§2.4), and
    per-head learnable gains (cf. indexer head weights, Eq. 15-16)."""

    def __init__(self):
        super().__init__()
        self.ln = nn.LayerNorm(N_EMBD)
        self.w_q = nn.Linear(N_EMBD, N_EMBD)
        self.w_kv = nn.Linear(N_EMBD, N_EMBD)     # one projection: K and V
        hd = N_EMBD // N_HEAD
        self.q_norm = RMSNorm(hd)
        self.k_norm = RMSNorm(hd)
        self.head_gain = nn.Parameter(torch.ones(N_HEAD))
        self.c_proj = nn.Linear(N_EMBD, N_EMBD)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE))
                 .view(1, 1, BLOCK_SIZE, BLOCK_SIZE),
        )

    def forward(self, x):
        B, T, C = x.size()
        x = self.ln(x)
        hd = C // N_HEAD
        q = self.w_q(x).view(B, T, N_HEAD, hd).transpose(1, 2)
        kv = self.w_kv(x).view(B, T, N_HEAD, hd).transpose(1, 2)
        q, k = self.q_norm(q), self.k_norm(kv)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ kv                                  # value IS the kv entry
        y = y * self.head_gain.view(1, N_HEAD, 1, 1)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = nn.LayerNorm(N_EMBD)
        self.c_fc = nn.Linear(N_EMBD, 4 * N_EMBD)
        self.c_proj = nn.Linear(4 * N_EMBD, N_EMBD)

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(self.ln(x)), approximate="tanh"))


class MHC(nn.Module):
    """Manifold-Constrained Hyper-Connections wrapping one sublayer F.
    Implements Eqs. 1-8 of the DeepSeek-V4 report with n_hc = N_HC."""

    def __init__(self, sublayer):
        super().__init__()
        self.f = sublayer
        n, d = N_HC, N_EMBD
        # dynamic components (Eqs. 3-5)
        self.w_pre  = nn.Linear(n * d, n, bias=False)
        self.w_res  = nn.Linear(n * d, n * n, bias=False)
        self.w_post = nn.Linear(n * d, n, bias=False)
        # static biases; S_res starts strongly diagonal so the Sinkhorn
        # projection of exp(S_res) is near-identity (stable start = plain
        # residual), per the paper's stability goal
        self.s_pre  = nn.Parameter(torch.zeros(n))
        self.s_res  = nn.Parameter(3.0 * torch.eye(n))
        self.s_post = nn.Parameter(torch.zeros(n))
        # gating factors "initialized to small values"
        self.alpha_pre  = nn.Parameter(torch.tensor(0.01))
        self.alpha_res  = nn.Parameter(torch.tensor(0.01))
        self.alpha_post = nn.Parameter(torch.tensor(0.01))

    @staticmethod
    def sinkhorn(logits):
        """Project exp(logits) onto doubly stochastic matrices (Eq. 8)."""
        logits = logits - logits.amax(dim=(-2, -1), keepdim=True)
        m = torch.exp(logits)
        for _ in range(SINKHORN_T):
            m = m / (m.sum(dim=-2, keepdim=True) + 1e-8)   # column norm
            m = m / (m.sum(dim=-1, keepdim=True) + 1e-8)   # row norm
        return m

    def forward(self, X):                          # X: (B, T, n_hc, d)
        B, T, n, d = X.size()
        x_hat = F.rms_norm(X.reshape(B, T, n * d), (n * d,))

        A = torch.sigmoid(self.alpha_pre * self.w_pre(x_hat) + self.s_pre)
        C = 2 * torch.sigmoid(self.alpha_post * self.w_post(x_hat) + self.s_post)
        B_raw = (self.alpha_res * self.w_res(x_hat)).view(B, T, n, n) + self.s_res
        B_mat = self.sinkhorn(B_raw)

        layer_in = torch.einsum("btn,btnd->btd", A, X)      # A_l X_l
        f_out = self.f(layer_in)                            # F_l(A_l X_l)
        # X_{l+1} = B_l X_l + C_l F_l(A_l X_l)
        return torch.einsum("btnm,btmd->btnd", B_mat, X) + \
               torch.einsum("btn,btd->btnd", C, f_out)


class GPT2BibleV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(VOCAB_SIZE, N_EMBD)
        self.wpe = nn.Embedding(BLOCK_SIZE, N_EMBD)
        layers = []
        for _ in range(N_LAYER):
            layers.append(MHC(SharedKVAttention()))
            layers.append(MHC(MLP()))
        self.h = nn.ModuleList(layers)
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.lm_head_bias = nn.Parameter(torch.zeros(VOCAB_SIZE))

        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * N_LAYER))

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= BLOCK_SIZE
        pos = torch.arange(T, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)
        X = x.unsqueeze(2).expand(B, T, N_HC, N_EMBD).contiguous()  # widen stream
        for layer in self.h:
            X = layer(X)
        x = self.ln_f(X.mean(dim=2))                                # collapse
        logits = F.linear(x, self.wte.weight, self.lm_head_bias)    # tied head

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -BLOCK_SIZE:])
            probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
            idx = torch.cat((idx, torch.multinomial(probs, 1)), dim=1)
        return idx


# ---- Muon optimizer (Algorithm 1) -------------------------------------------
def hybrid_newton_schulz(M):
    """Eq. 28 with the paper's two-stage coefficients (8 fast + 2 stable)."""
    X = M / (M.norm() + 1e-7)
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T
    for a, b, c in [(3.4445, -4.7750, 2.0315)] * 8 + [(2.0, -1.5, 0.5)] * 2:
        XXt = X @ X.T
        X = a * X + b * (XXt @ X) + c * (XXt @ XXt @ X)
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=1.5e-2, momentum=0.95, weight_decay=0.01,
                 gamma=0.2):
        super().__init__(params, dict(lr=lr, momentum=momentum,
                                      weight_decay=weight_decay, gamma=gamma))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            mu, lr = group["momentum"], group["lr"]
            lam, gamma = group["weight_decay"], group["gamma"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "m" not in state:
                    state["m"] = torch.zeros_like(g)
                state["m"].mul_(mu).add_(g)                  # M_t
                u = mu * state["m"] + g                      # Nesterov
                o = hybrid_newton_schulz(u)
                o = o * math.sqrt(max(p.size(0), p.size(1))) * gamma
                p.mul_(1 - lr * lam).add_(o, alpha=-lr)


_MHC_STATIC_NAMES = ("s_pre", "s_res", "s_post", "alpha_")


def build_optimizers(model, lr_adamw=3e-3, lr_muon=1.5e-2):
    """Paper's routing: AdamW for embedding/head/norm/static/gating params,
    Muon for everything else (the 2D linear weights). s_res is a (N_HC,N_HC)
    matrix, so it must be excluded by name — the ndim==2 check alone would
    route it to Muon like a linear weight, contradicting the paper (and
    s_pre/s_post, its 1D siblings, which the ndim check already excludes)."""
    muon_params, adamw_params = [], []
    for name, p in model.named_parameters():
        if (p.ndim == 2 and "wte" not in name and "wpe" not in name
                and not any(s in name for s in _MHC_STATIC_NAMES)):
            muon_params.append(p)
        else:
            adamw_params.append(p)
    return (Muon(muon_params, lr=lr_muon),
            torch.optim.AdamW(adamw_params, lr=lr_adamw))


# ---- capacity test (identical protocol to the baseline run) ------------------
if __name__ == "__main__":
    model = GPT2BibleV2()
    total = sum(p.numel() for p in model.parameters())
    print(f"{'component group':<22}{'params':>8}")
    print("-" * 30)
    groups = {}
    for name, p in model.named_parameters():
        key = name.split(".")[0] if not name.startswith("h.") else \
              "blocks (mHC+attn+mlp)"
        groups[key] = groups.get(key, 0) + p.numel()
    for k, v in groups.items():
        print(f"{k:<22}{v:>8,}")
    print("-" * 30)
    print(f"{'TOTAL':<22}{total:>8,}")
    print(f"{'KJV verses':<22}{TARGET_PARAMS:>8,}")
    assert total == TARGET_PARAMS, f"heresy detected: {total:,}"
    print("still 31,102 — one parameter per verse, now better spent\n")

    HERE = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(HERE, "kjv.txt"), encoding="utf-8") as f:
        raw = f.read()
    start = raw.find("In the beginning God created")
    assert start != -1, "KJV header anchor not found in kjv.txt"
    end = raw.rfind("*** END OF THE PROJECT GUTENBERG EBOOK")
    text = raw[start:end if end != -1 else None]
    chars = sorted(set(text))
    assert len(chars) == VOCAB_SIZE, f"vocab mismatch: {len(chars)}"
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    encode = lambda s: torch.tensor([stoi[c] for c in s], dtype=torch.long)

    def train_on(n_chars, steps=2000, batch_size=64):
        torch.manual_seed(1611)
        data = encode(text[:n_chars])
        m = GPT2BibleV2()
        muon, adamw = build_optimizers(m)

        def get_batch():
            ix = torch.randint(len(data) - BLOCK_SIZE - 1, (batch_size,))
            x = torch.stack([data[i: i + BLOCK_SIZE] for i in ix])
            y = torch.stack([data[i + 1: i + 1 + BLOCK_SIZE] for i in ix])
            return x, y

        m.train()
        for _ in range(steps):
            x, y = get_batch()
            _, loss = m(x, y)
            muon.zero_grad(set_to_none=True)
            adamw.zero_grad(set_to_none=True)
            loss.backward()
            muon.step()
            adamw.step()

        m.eval()
        with torch.no_grad():
            losses = [m(*get_batch())[1].item() for _ in range(20)]
        return m, sum(losses) / len(losses)

    BASELINE = {1_000: 0.1629, 5_000: 0.6569, 25_000: 1.3627, 100_000: 1.6389}
    SIZES = [1_000, 5_000, 25_000, 100_000]

    # resumable: completed sizes are recorded here and skipped on restart
    RESULTS_PATH = os.path.join(HERE, "v2_results.json")
    results = {}
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            results = json.load(f)

    print(f"{'data (chars)':>13} {'V2 loss':>9} {'V1 loss':>9} {'improvement':>12}  time")
    print("-" * 62)
    for n in SIZES:
        if str(n) in results:
            loss = results[str(n)]
            imp = 100 * (BASELINE[n] - loss) / BASELINE[n]
            print(f"{n:>13,} {loss:>9.4f} {BASELINE[n]:>9.4f} {imp:>+11.1f}%  (done)",
                  flush=True)
            continue
        t0 = time.time()
        m, loss = train_on(n)
        dt = time.time() - t0
        imp = 100 * (BASELINE[n] - loss) / BASELINE[n]
        print(f"{n:>13,} {loss:>9.4f} {BASELINE[n]:>9.4f} {imp:>+11.1f}%  {dt:>6.1f}s",
              flush=True)
        torch.save(m.state_dict(), os.path.join(HERE, f"v2_ckpt_{n}.pt"))
        results[str(n)] = loss
        with open(RESULTS_PATH, "w") as f:
            json.dump(results, f)

    for n in (1_000, 100_000):
        ckpt = os.path.join(HERE, f"v2_ckpt_{n}.pt")
        if not os.path.exists(ckpt):
            print(f"\n--- no checkpoint for {n:,} chars, skipping sample ---")
            continue
        m = GPT2BibleV2()
        m.load_state_dict(torch.load(ckpt))
        m.eval()
        idx = encode("In the beginning").unsqueeze(0)
        out = m.generate(idx, max_new_tokens=300, temperature=0.8)
        print(f"\n--- V2 sample (trained on {n:,} chars) ---")
        print("".join(itos[int(i)] for i in out[0]))
