"""
v4_gethookd — connector for the gethookd ad-intelligence MCP (Streamable HTTP).
The clean, GRADED winner source: every ad carries a real performance_score
(1-100; testing 1-40, scaling 41-60, growing 61-80, optimized 81-90,
winning 91+), full body copy + transcripts, brand, spend range, demographics.

This is the label-grounded basis for "recognize a winning ad" — pull ads across
the score spectrum and supervise on the real grade (no scrape, no rate-limit,
no budget-proxy confound).

KEY: GETHOOKD_KEY in Downloads/.env. search_ads is CREDIT-METERED (per result)
— every call reports used_credits / remaining_credits; we surface them so
spend is visible.
"""

import json
import os
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
URL = os.environ.get("GETHOOKD_URL", "https://app.gethookd.ai/api/mcp/v1")
ENV = os.path.join(HERE, ".env")


def _key():
    k = os.environ.get("GETHOOKD_KEY")
    if not k and os.path.exists(ENV):
        for line in open(ENV, encoding="utf-8"):
            if line.strip().startswith("GETHOOKD_KEY="):
                k = line.split("=", 1)[1].strip()
    if not k:
        raise RuntimeError("GETHOOKD_KEY not in Downloads/.env")
    return k


def _post(body, sid=None):
    h = {"Authorization": f"Bearer {_key()}", "Content-Type": "application/json",
         "Accept": "application/json, text/event-stream"}
    if sid:
        h["Mcp-Session-Id"] = sid
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers=h, method="POST")
    try:
        r = urllib.request.urlopen(req, timeout=120)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"gethookd {e.code}: {e.read().decode()[:300]}")
    sid2 = r.headers.get("Mcp-Session-Id")
    ct = r.headers.get("Content-Type", "")
    raw = r.read().decode()
    if not raw.strip():                       # notifications -> empty 202 body
        return None, sid2
    if "event-stream" in ct:
        out = None
        for ln in raw.splitlines():
            if ln.startswith("data:"):
                try:
                    out = json.loads(ln[5:].strip())
                except Exception:  # noqa: BLE001
                    pass
        return out, sid2
    try:
        return json.loads(raw), sid2
    except Exception:  # noqa: BLE001
        return {"_raw": raw[:300]}, sid2


class Client:
    def __init__(self):
        init = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
                "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                           "clientInfo": {"name": "bernay", "version": "1.0"}}}
        _, self.sid = _post(init)
        _post({"jsonrpc": "2.0", "method": "notifications/initialized"},
              self.sid)
        self._id = 1

    def call(self, name, args):
        self._id += 1
        res, _ = _post({"jsonrpc": "2.0", "id": self._id, "method": "tools/call",
                        "params": {"name": name, "arguments": args}}, self.sid)
        content = ((res or {}).get("result") or {}).get("content") or []
        for c in content:
            if c.get("type") == "text":
                try:
                    return json.loads(c["text"])
                except Exception:  # noqa: BLE001
                    return {"_text": c["text"]}
        return res

    def search_ads(self, query="", performance_scores=None, limit=50, page=1,
                   collapse_variants=True, **extra):
        args = {"query": query, "limit": limit, "page": page,
                "collapse_variants": collapse_variants}
        if performance_scores:
            args["performance_scores"] = performance_scores
        args.update(extra)
        d = self.call("search_ads", args)
        ads = d.get("data") or d.get("ads") or []
        meta = {"used_credits": d.get("used_credits"),
                "remaining_credits": d.get("remaining_credits"),
                **(d.get("meta") or {})}
        return ads, meta


NICHES = ["supplement", "skincare", "weight loss", "joint pain", "tinnitus",
          "prostate", "menopause", "hair loss", "gut health", "blood sugar",
          "anxiety sleep", "energy fatigue", "eye vision", "teeth"]
OUT = os.path.join(HERE, "v4_gethookd_ads.jsonl")


def _flat(a, niche, bucket):
    br = a.get("brand") or {}
    tr = a.get("transcripts")
    transcript = ""
    if isinstance(tr, list):
        transcript = " ".join(str(x) for x in tr if x)
    elif isinstance(tr, str):
        transcript = tr
    return {
        "ad_id": a.get("id"), "niche": niche, "bucket": bucket,
        "score": a.get("performance_score"),
        "score_title": a.get("performance_score_title"),
        "spend_score": a.get("ad_spend_range_score_title"),
        "brand": br.get("name"), "brand_id": br.get("id"),
        "platform": a.get("platform"), "asset_type": a.get("asset_type"),
        "days_active": a.get("days_active"),
        "title": a.get("title"), "body": (a.get("body") or "").strip(),
        "transcript": transcript.strip(),
        "cta": a.get("cta_text"), "landing": a.get("landing_page"),
        "age_min": a.get("age_audience_min"), "age_max": a.get("age_audience_max"),
        "gender": a.get("gender_audience"), "eu_reach": a.get("eu_total_reach"),
    }


def harvest(queries=None,
            buckets=("winning", "optimized", "growing", "scaling", "testing"),
            per_bucket=40, credit_floor=200):
    """Pull a balanced copy->score set across niches x score buckets. Stops if
    remaining credits drop below credit_floor (safety). Appends to OUT."""
    queries = queries or NICHES
    c = Client()
    seen = set()
    if os.path.exists(OUT):
        for line in open(OUT, encoding="utf-8"):
            try:
                seen.add(json.loads(line).get("ad_id"))
            except Exception:  # noqa: BLE001
                pass
    rows, rem = [], None
    with open(OUT, "a", encoding="utf-8") as f:
        for q in queries:
            for bucket in buckets:
                got, page = 0, 1
                while got < per_bucket:
                    try:
                        ads, meta = c.search_ads(
                            query=q, performance_scores=bucket,
                            limit=min(50, per_bucket - got), page=page)
                    except RuntimeError as e:        # transient 504 etc.
                        print(f"  {q}/{bucket} p{page}: {str(e)[:45]} — skip",
                              flush=True)
                        break
                    rem = meta.get("remaining_credits") or rem
                    if not ads:
                        break
                    new = 0
                    for a in ads:
                        aid = a.get("id")
                        if aid in seen:
                            continue
                        seen.add(aid)
                        r = _flat(a, q, bucket)
                        if len((r["body"] or "") + (r["transcript"] or "")) >= 40:
                            f.write(json.dumps(r, ensure_ascii=False) + "\n")
                            rows.append(r)
                            new += 1
                            got += 1
                    page += 1
                    if new == 0 or len(ads) < 50:
                        break
                    if rem is not None and rem < credit_floor:
                        print(f"credit floor hit ({rem}) — stopping", flush=True)
                        return rows, rem
                print(f"  {q:14} {bucket:8} +{got}  (rem {rem})", flush=True)
    return rows, rem


if __name__ == "__main__":
    import sys
    if "--harvest" in sys.argv:
        pb = int(sys.argv[sys.argv.index("--harvest") + 1]) \
            if len(sys.argv) > sys.argv.index("--harvest") + 1 \
            and sys.argv[sys.argv.index("--harvest") + 1].isdigit() else 60
        rows, rem = harvest(per_bucket=pb)
        from collections import Counter
        print(f"\n{len(rows)} new ads -> {os.path.basename(OUT)} | "
              f"remaining credits {rem}")
        print("by bucket:", dict(Counter(r["bucket"] for r in rows)))
        raise SystemExit(0)
    c = Client()
    q = sys.argv[1] if len(sys.argv) > 1 else "supplement"
    bucket = sys.argv[2] if len(sys.argv) > 2 else "winning"
    ads, meta = c.search_ads(query=q, performance_scores=bucket, limit=3)
    print(f"query={q!r} bucket={bucket} -> {len(ads)} ads")
    print(f"CREDITS: used {meta.get('used_credits')} | "
          f"remaining {meta.get('remaining_credits')}")
    for a in ads:
        b = (a.get("brand") or {}).get("name")
        print(f"  score {a.get('performance_score')} | {a.get('days_active')}d "
              f"| {b} | {(a.get('body') or '')[:60]!r}")
