"""
App-parity QA runner — feeds every ad in v4_regression_ads.jsonl through the
LIVE HTTP surface (POST /api/analyze) and asserts the same locked
expectations v4_regression.py checks against the raw pipeline. The desktop
app is a new surface over the same Schwartz-4 stack, not a new model — so
parity between "python v4_regression.py" and "curl /api/analyze" is the
whole test. This must NEVER patch expectations to make the app pass; a
mismatch is a bug in the app's routing/serialization, not in the fixture.

Coverage note (read before "fixing" a skip):
  v4_regression_ads.jsonl has two case shapes:
    mode="text"        -> raw ad copy, no creative. The API takes exactly
                           this shape as {"input": <text>} -> POST /api/analyze.
                           Fully replayable over HTTP. (6/8 cases)
    mode="local_brief"  -> a SYNTHETIC brief (hand-labeled OCR/transcript/
                           face) injected directly at the Python level via
                           distill._normalize(...) + admix.analyze(brief=...),
                           bypassing v4_media entirely. It exists specifically
                           so the regression harness can lock demographic/
                           avatar/grounding-guard behaviour WITHOUT a real
                           media file or a paid Gemini/gethookd call.
                           The API has no endpoint that accepts a raw brief —
                           POST /api/analyze takes one string and ALWAYS
                           re-derives the brief via v4_media.ingest_structured,
                           which needs a real image/video/URL. No such asset
                           backs these 2 fixtures, and manufacturing one
                           would violate the "no paid credits" rail and
                           would exercise a different code path (real OCR/
                           Gemini) than the one the fixture locks.
                           These 2 cases (bioblade_microcurrent,
                           grounding_guard_phantom_domain) are therefore
                           reported as SKIP, not silently dropped — see the
                           summary. They stay covered at the CLI level by
                           v4_regression.py (8/8 there).

Run:  .venv-vlm\\Scripts\\python.exe bernay-app\\server\\test_parity.py
      (server must already be up on 127.0.0.1:8756 — see run_server.cmd)
"""

import json
import os
import re
import sys
import time

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
CASES = os.path.normpath(os.path.join(HERE, "..", "..", "v4_regression_ads.jsonl"))
BASE = os.environ.get("BERNAY_API_BASE", "http://127.0.0.1:8756")


def _wait_ready(timeout=120):
    deadline = time.time() + timeout
    last_status = None
    while time.time() < deadline:
        try:
            r = requests.get(f"{BASE}/api/health", timeout=5)
            h = r.json()
            last_status = h.get("status")
            if last_status == "ready":
                return h
            if last_status == "error":
                raise SystemExit(f"API stack failed to load: {h.get('error')}")
        except requests.RequestException as e:
            last_status = f"unreachable ({e})"
        time.sleep(2)
    raise SystemExit(f"API never became ready within {timeout}s "
                     f"(last status: {last_status}). Is the Bernay API "
                     f"server running on {BASE}?")


def _check(expect, payload):
    """Same assertion set as v4_regression.py's _check(), re-pointed at the
    API's JSON shape instead of the raw analyze() result dict."""
    fails = []
    audience = payload.get("audience") or {}
    aw = payload.get("awareness_journey") or []
    product = payload.get("product") or ""
    pains = " | ".join(p.get("name", "") for p in (payload.get("painpoints") or []))
    problem = payload.get("problem") or ""

    if "product" in expect and product != expect["product"]:
        fails.append(f"product: got {product!r}, want {expect['product']!r}")
    if "product_re" in expect and not re.search(expect["product_re"], product, re.I):
        fails.append(f"product: got {product!r}, want ~/{expect['product_re']}/")
    for field in ("gender", "age"):
        got = audience.get(field)
        if field in expect and got != expect[field]:
            fails.append(f"{field}: got {got!r}, want {expect[field]!r}")
        if f"{field}_not" in expect and got == expect[f"{field}_not"]:
            fails.append(f"{field}: got {got!r}, must NOT be {expect[f'{field}_not']!r}")
    if "age_in" in expect and audience.get("age") not in expect["age_in"]:
        fails.append(f"age: got {audience.get('age')!r}, want one of {expect['age_in']}")
    for stage in expect.get("awareness_has", []):
        if stage not in aw:
            fails.append(f"awareness: missing {stage!r} (got {aw})")
    for stage in expect.get("awareness_not", []):
        if stage in aw:
            fails.append(f"awareness: {stage!r} should be absent (got {aw})")
    if "problem_re" in expect and not re.search(expect["problem_re"], problem, re.I):
        fails.append(f"problem: got {problem!r}, want ~/{expect['problem_re']}/")
    if "painpoint_re" in expect and not re.search(expect["painpoint_re"], pains, re.I):
        fails.append(f"painpoints: got {pains!r}, want ~/{expect['painpoint_re']}/")
    return fails


def _analyze(text):
    r = requests.post(f"{BASE}/api/analyze", json={"input": text}, timeout=180)
    if not r.ok:
        detail = None
        try:
            detail = r.json().get("detail")
        except Exception:  # noqa: BLE001
            pass
        raise RuntimeError(f"HTTP {r.status_code}: {detail or r.text[:200]}")
    return r.json()


def main():
    if not os.path.exists(CASES):
        raise SystemExit(f"no cases file: {CASES}")
    cases = [json.loads(line) for line in open(CASES, encoding="utf-8")
             if line.strip()]

    health = _wait_ready()
    print(f"API ready — model={health.get('model')} "
          f"params={health.get('params')} base={BASE}\n")

    tested = [c for c in cases if c.get("mode", "text") == "text"]
    skipped = [c for c in cases if c.get("mode", "text") != "text"]

    passed = 0
    failures = []
    for c in tested:
        name, text = c["name"], c.get("text", "")
        try:
            payload = _analyze(text)
            fails = _check(c.get("expect", {}), payload)
        except Exception as e:  # noqa: BLE001
            fails = [f"EXCEPTION: {e!r}"]
            payload = None
        if not fails:
            passed += 1
            print(f"  PASS  {name}")
        else:
            print(f"  FAIL  {name}")
            for f in fails:
                print(f"          - {f}")
            failures.append({"name": name, "input": text, "expect": c.get("expect", {}),
                             "got": payload, "fails": fails})

    for c in skipped:
        print(f"  SKIP  {c['name']}  (mode={c.get('mode')!r} — no HTTP-replayable "
              f"brief; see module docstring; covered at CLI level)")

    print(f"\n{passed}/{len(tested)} API-replayable cases pass "
          f"({len(skipped)} skipped: {', '.join(c['name'] for c in skipped)})")

    if failures:
        print("\n--- FAILURE DETAIL (for bug tickets) ---")
        for f in failures:
            print(json.dumps(f, indent=2, default=str))

    sys.exit(0 if passed == len(tested) else 1)


if __name__ == "__main__":
    main()
