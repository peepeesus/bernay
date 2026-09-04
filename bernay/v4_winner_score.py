"""
v4_winner_score — score ad copy -> WIN-LIKELIHOOD using the gethookd-trained
winner head (v4_gethookd_winner_head.pt, held-out AUC ~0.76). Turns the trained
classifier into a callable Bernay can show alongside PV: "how winner-like is
this copy", grounded in real performance grades.

    from v4_winner_score import win_prob
    p = win_prob(ad_copy)   # 0..1, higher = more winner-like

Loads lazily + caches; returns None if the head isn't trained yet (so callers
degrade gracefully). HONEST: this is a COARSE copy-quality signal (copy is one
factor in performance, not all) — see v4_gethookd_score_head notes.
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "v4_gethookd_winner_head.pt")
_C = {}


def _load():
    if "head" in _C or _C.get("missing"):
        return
    if not os.path.exists(CKPT):
        _C["missing"] = True
        return
    import torch
    import torch.nn as nn
    import v4_tokenizer as tok
    from v4_motif_scorer import MotifScorer
    mc = os.environ.get("V4_MOTIF_CACHE") or os.path.join(
        HERE, "v4_big_motif_cache.pt")
    blob = torch.load(mc, weights_only=False)
    if blob.get("vocab_hash") != tok.vocab_hash():
        _C["missing"] = True
        return
    d = torch.load(CKPT, weights_only=False)
    head = nn.Sequential(nn.Linear(d["in_dim"], 32), nn.Tanh(),
                         nn.Dropout(0.3), nn.Linear(32, 1))
    head.load_state_dict(d["model"])
    head.eval()
    _C.update(head=head, scorer=MotifScorer(), mean=blob["mean"],
              std=blob["std"], torch=torch)


def win_prob(text):
    """0..1 win-likelihood for an ad's copy, or None if the head isn't ready."""
    if not (text or "").strip():
        return None
    _load()
    if _C.get("missing"):
        return None
    torch = _C["torch"]
    z = (_C["scorer"].score(text) - _C["mean"]) / _C["std"]
    with torch.no_grad():
        return float(torch.sigmoid(_C["head"](z).squeeze(-1)))


if __name__ == "__main__":
    import sys
    t = " ".join(sys.argv[1:]) or \
        "Stop wasting money on cheap pans. Invest in cookware that lasts."
    p = win_prob(t)
    print(f"win-likelihood: {p:.0%}" if p is not None
          else "winner head not trained yet (run v4_gethookd_winner_head.py)")
