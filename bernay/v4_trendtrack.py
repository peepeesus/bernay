"""
TrendTrack share-page fetcher (headless Playwright) — makes an
app.trendtrack.io/share/ads/<slug> PAGE url work directly, the same way
v4_adlib does for Meta Ad Library. TrendTrack is an ad-spy SaaS; its
share pages render the creative (image or video) plus performance signal
('booming' = many active ads / long run-time). The creative is served
from a CDN (fbcdn or TrendTrack's own), so we watch network responses
for media rather than parse the JS-rendered DOM, with a screenshot
fallback. We also scrape any visible performance text into `meta`.

Returns {'video','image','screenshot','meta'} for v4_media to distill.
Best-effort and defensive — degrades to a screenshot Gemini can read.
"""

import os
import re
import tempfile
import urllib.request

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
COOKIE_BTN = re.compile(
    r"accept all|allow all|accept|agree|got it|i understand|continue", re.I)
MEDIA_EXT = (".mp4", ".webm", ".m4v", ".jpg", ".jpeg", ".png", ".webp")
# lines that look like ad-spy performance signal -> the 'booming' read
METRIC = re.compile(
    r"\b(active|running|ads?|impressions?|spend|reach|likes?|shares?|days?|"
    r"weeks?|months?|since|started|launched|engagement|comments?)\b", re.I)
MIN_IMAGE_BYTES = 20_000           # below this it's UI chrome, not a creative


def _download(url, ext):
    out = os.path.join(tempfile.gettempdir(), f"v4_trendtrack{ext}")
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=120) as r, \
            open(out, "wb") as f:
        f.write(r.read())
    return out if os.path.getsize(out) > 1000 else None


def _ext_of(url):
    for e in MEDIA_EXT:
        if e in url.lower():
            return e
    return ""


def fetch(url):
    """-> {'video','image','screenshot','meta'}. Captures the creative
    from network media responses; falls back to a screenshot."""
    from playwright.sync_api import sync_playwright

    images, videos = [], []      # images: (size,url) ; videos: url

    def on_response(resp):
        u = resp.url
        ct = (resp.headers or {}).get("content-type", "")
        try:
            size = int((resp.headers or {}).get("content-length", "0"))
        except ValueError:
            size = 0
        is_vid = (".mp4" in u.lower() or ".webm" in u.lower()
                  or ct.startswith("video/"))
        is_img = (ct.startswith("image/")
                  or _ext_of(u) in (".jpg", ".jpeg", ".png", ".webp"))
        if is_vid:
            videos.append(u)
        elif is_img and size >= MIN_IMAGE_BYTES:
            images.append((size, u))

    shot = os.path.join(tempfile.gettempdir(), "v4_trendtrack_shot.png")
    meta = None
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True)
        ctx = b.new_context(user_agent=UA, locale="en-US",
                            viewport={"width": 1280, "height": 1600})
        page = ctx.new_page()
        page.on("response", on_response)
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=60000)
        except Exception:  # noqa: BLE001
            pass
        for _ in range(3):
            try:
                btn = page.get_by_role("button", name=COOKIE_BTN)
                if btn.count():
                    btn.first.click(timeout=3000)
                    break
            except Exception:  # noqa: BLE001
                pass
            page.wait_for_timeout(1000)
        page.wait_for_timeout(4000)            # let the creative render
        try:                                   # nudge videos to fetch media
            page.evaluate(
                "document.querySelectorAll('video').forEach(v=>{"
                "v.muted=true; v.play().catch(()=>{});})")
            page.wait_for_timeout(4000)
        except Exception:  # noqa: BLE001
            pass
        try:                                   # scrape visible perf signal
            body = page.inner_text("body")
            lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
            hits = [ln for ln in lines if len(ln) <= 60 and METRIC.search(ln)]
            if hits:
                meta = " | ".join(dict.fromkeys(hits[:8]))
        except Exception:  # noqa: BLE001
            pass
        try:
            page.screenshot(path=shot, full_page=False)
        except Exception:  # noqa: BLE001
            shot = None
        b.close()

    out = {"video": None, "image": None, "screenshot": shot, "meta": meta}
    if videos:
        try:
            out["video"] = _download(videos[0], ".mp4")
        except Exception:  # noqa: BLE001
            pass
    if not out["video"] and images:
        big = max(images, key=lambda t: t[0])[1]
        try:
            out["image"] = _download(big, _ext_of(big) or ".jpg")
        except Exception:  # noqa: BLE001
            pass
    return out


def fetch_guarded(url, timeout=None):
    """fetch(url), but run in an isolated child process with a HARD
    wall-clock timeout — see v4_fetch_guard.py. TrendTrack share links a
    user pastes route here; this can never be allowed to hang the caller."""
    import v4_fetch_guard
    t = timeout or v4_fetch_guard.DEFAULT_TIMEOUT
    here = os.path.dirname(os.path.abspath(__file__))
    return v4_fetch_guard.run_guarded(
        os.path.join(here, "v4_trendtrack.py"), [url], timeout=t)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--json":
        u = sys.argv[2] if len(sys.argv) > 2 else \
            "https://app.trendtrack.io/share/ads/the-ridge-kYINpP"
        import json
        print(json.dumps(fetch(u)))
        raise SystemExit(0)
    u = sys.argv[1] if len(sys.argv) > 1 else \
        "https://app.trendtrack.io/share/ads/the-ridge-kYINpP"
    print(f"fetching: {u}")
    r = fetch(u)
    for k, v in r.items():
        if k == "meta":
            print(f"  meta        {v}")
        else:
            sz = os.path.getsize(v) if v and os.path.exists(v) else 0
            print(f"  {k:11} {v}  ({sz:,} bytes)")
