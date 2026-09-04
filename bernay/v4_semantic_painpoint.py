"""
v4_semantic_painpoint — semantic painpoint recognizer (option #3, the piece
that actually pays off). Instead of exact-string regex cue matching (which
breaks on OCR space-loss 'gut support'->'gutsupport' and paraphrase), embed
the copy and each KB painpoint's name+description+cues, and match by COSINE
similarity. Generalizes past the literal strings.

Used as a GAP-FILLER: regex match_painpoints stays the high-precision primary;
this fires when regex finds nothing (or weakly), tagged so it's auditable.

  match(copy, topk=2, thresh=0.45) -> [(painpoint_id, name, score)]
  compare_to_regex()                -> prints regex vs semantic on the gathered
                                       creatives (where regex most often misses)
"""
import json
import os

import v4_correlations as C
import v4_embed

HERE = os.path.dirname(os.path.abspath(__file__))
ANCHORS = os.path.join(HERE, "v4_painpoint_anchors.json")


def _anchor_text(p):
    cues = ", ".join(p.get("cues", [])[:12])
    return f"{p['name']}. {p.get('description', '')} Mentions: {cues}"


def build_anchors():
    pps = C.load_kb().get("painpoints", [])
    vecs = v4_embed.encode([_anchor_text(p) for p in pps])
    anchors = [{"id": p["id"], "name": p["name"], "vec": v}
               for p, v in zip(pps, vecs)]
    json.dump(anchors, open(ANCHORS, "w", encoding="utf-8"))
    print(f"built {len(anchors)} painpoint anchors -> "
          f"{os.path.basename(ANCHORS)}")
    return anchors


def _load_anchors():
    if not os.path.exists(ANCHORS):
        return build_anchors()
    return json.load(open(ANCHORS, encoding="utf-8"))


def _cos(a, b):
    import math
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb + 1e-9)


def match(copy, topk=2, thresh=0.45, _anchors=None, _qvec=None, allow_ids=None):
    """allow_ids: optional set of painpoint ids to restrict the search to. Used
    by match_painpoints to confine the search to the ad's likely vertical when
    the gate tokens are one-sided (kills health-anchor class-imbalance drag —
    987 health anchors vs 56 non-health would otherwise out-score the correct
    non-health painpoint for a wealth/dating/beauty ad)."""
    anchors = _anchors or _load_anchors()
    if allow_ids is not None:
        anchors = [a for a in anchors if a["id"] in allow_ids]
    q = _qvec or v4_embed.encode([copy or ""])[0]
    scored = [(a["id"], a["name"], _cos(q, a["vec"])) for a in anchors]
    scored.sort(key=lambda t: -t[2])
    return [(i, n, round(s, 3)) for i, n, s in scored[:topk] if s >= thresh]


def compare_to_regex():
    """Side-by-side regex vs semantic on the gathered creatives — focus on the
    rows where regex finds NOTHING (the OCR-garble/paraphrase misses)."""
    import glob
    anchors = _load_anchors()
    rows, seen = [], set()
    for f in glob.glob(os.path.join(HERE, "v4_*creatives*.json")):
        try:
            data = json.load(open(f, encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        for r in data:
            br = r.get("brief") or {}
            copy = ((br.get("onscreen_text") or "") + " "
                    + (br.get("spoken_transcript") or "")).strip()
            if len(copy) < 25 or copy[:100] in seen:
                continue
            seen.add(copy[:100])
            rows.append(copy)
    qvecs = v4_embed.encode([c[:1800] for c in rows])
    miss_recovered = 0
    print(f"\n{'='*78}\nREGEX vs SEMANTIC painpoint matching "
          f"({len(rows)} distinct creatives)\n{'='*78}")
    for copy, qv in zip(rows, qvecs):
        rgx = [p["name"] for p, h in C.match_painpoints(copy)]
        sem = match(copy, topk=2, thresh=0.45, _anchors=anchors, _qvec=qv)
        flag = ""
        if not rgx and sem:
            flag = "  <-- regex MISS, semantic recovered"
            miss_recovered += 1
        print(f"\ncopy: {copy[:80]!r}")
        print(f"  regex   : {rgx or '(none)'}")
        print(f"  semantic: {[(n, s) for _, n, s in sem] or '(none)'}{flag}")
    print(f"\n{'='*78}\nregex-miss rows where semantic recovered a painpoint: "
          f"{miss_recovered}")


if __name__ == "__main__":
    import sys
    if "--build" in sys.argv:
        build_anchors()
    else:
        compare_to_regex()
