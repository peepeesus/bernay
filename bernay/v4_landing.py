"""
Landing-page fetcher (headless Playwright) — lets Bernay analyze a normal
web/landing/sales page URL, not just an ad creative. Renders the page in
chromium, dismisses cookie/consent walls, captures the visible COPY
(body inner_text) and a bounded screenshot of the top of the page (the
hero + first few screens carry the pitch; the full text covers the rest).

v4_media routes any generic http(s) page here, then v4_distill turns the
screenshot + copy into the same structured ad brief — so a landing page
is decomposed (avatar, awareness journey, desires, painpoints, PV) exactly
like an ad. Best-effort and defensive, like v4_adlib / v4_trendtrack.
"""

import os
import re
import tempfile

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
COOKIE_BTN = re.compile(
    r"accept all|allow all|accept|agree|got it|i understand|continue|"
    r"only essential|reject", re.I)
# bounded capture height — the pitch lives up top; the full inner_text
# carries the rest, so we don't ship a 12000px base64 image to Gemini.
SHOT_H = 2600


def fetch(url):
    """-> {'text': page copy, 'screenshot': path|None, 'title': str}."""
    from playwright.sync_api import sync_playwright

    out = {"text": "", "screenshot": None, "title": ""}
    shot = os.path.join(tempfile.gettempdir(), "v4_landing_shot.png")
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True)
        ctx = b.new_context(user_agent=UA, locale="en-US",
                            viewport={"width": 1280, "height": SHOT_H})
        page = ctx.new_page()
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=60000)
        except Exception:  # noqa: BLE001
            pass
        for _ in range(3):                     # dismiss consent walls
            try:
                btn = page.get_by_role("button", name=COOKIE_BTN)
                if btn.count():
                    btn.first.click(timeout=3000)
                    break
            except Exception:  # noqa: BLE001
                pass
            page.wait_for_timeout(800)
        try:
            page.wait_for_load_state("networkidle", timeout=8000)
        except Exception:  # noqa: BLE001
            pass
        page.wait_for_timeout(1500)
        try:
            out["title"] = page.title()
        except Exception:  # noqa: BLE001
            pass
        try:
            out["text"] = page.inner_text("body").strip()
        except Exception:  # noqa: BLE001
            pass
        try:                                   # bounded top-of-page shot
            page.screenshot(path=shot, full_page=False)
            out["screenshot"] = shot
        except Exception:  # noqa: BLE001
            out["screenshot"] = None
        b.close()
    return out


def fetch_guarded(url, timeout=None):
    """fetch(url), but run in an isolated child process with a HARD
    wall-clock timeout — see v4_fetch_guard.py. Any generic web/landing
    page a user pastes routes here, so this can never be allowed to hang
    the caller indefinitely."""
    import v4_fetch_guard
    t = timeout or v4_fetch_guard.DEFAULT_TIMEOUT
    here = os.path.dirname(os.path.abspath(__file__))
    return v4_fetch_guard.run_guarded(
        os.path.join(here, "v4_landing.py"), [url], timeout=t)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--json":
        u = sys.argv[2] if len(sys.argv) > 2 else "https://www.ridge.com"
        import json
        print(json.dumps(fetch(u)))
        raise SystemExit(0)
    u = sys.argv[1] if len(sys.argv) > 1 else "https://www.ridge.com"
    print(f"fetching: {u}")
    r = fetch(u)
    print(f"  title:      {r['title']}")
    print(f"  text chars: {len(r['text']):,}")
    print(f"  screenshot: {r['screenshot']}")
    print("\n--- first 500 chars of copy ---")
    print(r["text"][:500])
