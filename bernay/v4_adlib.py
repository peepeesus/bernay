"""
Meta Ad Library fetcher (headless Playwright) — makes a facebook.com/
ads/library/?id=... PAGE url work directly: load it in chromium, dismiss
the cookie wall, and capture the creative's real fbcdn media URL from
network traffic (the page is a JS app with signed CDN links, so we watch
responses rather than parse the DOM). Screenshot fallback when no clean
media URL is seen.

Returns local file paths for v4_media to transcribe (video) / describe
(image). Best-effort and defensive: Meta actively fights automation, so
every step is wrapped and degrades to a screenshot of the creative.
"""

import os
import re
import tempfile
import urllib.request
from urllib.parse import urlparse, parse_qs

HERE = os.path.dirname(os.path.abspath(__file__))
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
COOKIE_BTN = re.compile(
    r"allow all|accept all|only allow essential|decline optional|"
    r"allow essential", re.I)


# Persistent Chromium profile holding a logged-in Facebook session. Created
# by v4_fb_login.py (one-time visible login); when present, every fetch reuses
# it so Meta serves real ad creatives instead of a bot placeholder.
PROFILE_DIR = os.path.join(HERE, ".v4_fb_profile")


def _session(p, headless=True):
    """A browser context — logged in via the persistent FB profile if it
    exists (Meta then serves real creatives), else a fresh anonymous one.
    Returns (context, browser_or_None); pass both to _close."""
    vp = {"width": 1280, "height": 1600}
    if os.path.isdir(PROFILE_DIR):
        ctx = p.chromium.launch_persistent_context(
            PROFILE_DIR, headless=headless, user_agent=UA, locale="en-US",
            viewport=vp)
        return ctx, None
    b = p.chromium.launch(headless=headless)
    return b.new_context(user_agent=UA, locale="en-US", viewport=vp), b


def _close(ctx, br):
    try:
        br.close() if br is not None else ctx.close()
    except Exception:  # noqa: BLE001
        pass


def _stem(url):
    """A per-ad temp-file stem from the Ad Library URL's id, so two fetches
    (a live session + this one, or back-to-back ads) write to DIFFERENT files
    instead of clobbering a shared 'v4_adlib.mp4' — the cause of the
    'same data twice for different ads' bug. Falls back to a stable hash."""
    m = re.search(r"[?&]id=(\d{6,})", url)
    if m:
        return f"v4_adlib_{m.group(1)}"
    import hashlib
    return "v4_adlib_" + hashlib.md5(url.encode()).hexdigest()[:12]


def _download(url, ext, stem="v4_adlib"):
    out = os.path.join(tempfile.gettempdir(), f"{stem}{ext}")
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=120) as r, \
            open(out, "wb") as f:
        f.write(r.read())
    return out if os.path.getsize(out) > 1000 else None


# A real creative is a full-size image. Meta also serves, from the SAME
# scontent.*.fbcdn.net host, the advertiser's PAGE PROFILE PHOTO (an 80x80
# avatar) and assorted UI thumbnails — so 'came from scontent' is not enough
# to call something the ad. Measured on ?id=1431722525448271, an ad that is
# not in the Library at all: the only scontent asset on the page was the
# 6.9 KB 160x160 profile picture of the page "Resilia", and the whole
# pipeline analysed it as if it were the creative (no text to OCR, no face
# to detect -> gender/age/life_stage all "unclear", and a confident
# decomposition on top of nothing).
MIN_CREATIVE_PX = 200        # smaller side; Meta avatars top out at 160


def _pixels(path):
    """(w, h) of a downloaded image, or None if it isn't readable."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            return im.size
    except Exception:  # noqa: BLE001
        return None


def _is_real_creative(path):
    """Reject profile pictures / UI thumbnails masquerading as the ad."""
    if not path:
        return False
    wh = _pixels(path)
    if wh is None:              # not an image (video frame path etc.) — allow
        return True
    return min(wh) >= MIN_CREATIVE_PX


# Meta's OWN domains, including the ones it links from the Ad Library
# footer. metastatus.com is Meta's service-status site: it was being picked
# as the advertiser's landing page on any ad whose real outbound link was
# missing, which then named the product "Metastatus".
_META_HOSTS = ("facebook.", "fb.com", "fbcdn.", "instagram.", "whatsapp",
               "messenger.", "meta.com", "fb.me", "metastatus.",
               "metacareers.", "oculus.com", "threads.net", "workplace.com")

# Meta's own copy for "this ad ID resolves to nothing" — from the AD MODAL
# only.
#
# "No ads match your search criteria" was in this list and MUST NOT BE: the
# single-ad page renders the search-results panel BEHIND the modal, and that
# panel says so whenever the (empty) search matches nothing — on a perfectly
# valid ad. It caused a false refusal on ?id=1819239872392567, an ad with
# 5,595 characters of copy and a video creative. A phrase describing the
# page's search list is never evidence about the ad.
_UNAVAILABLE = re.compile(
    r"Ad isn't in the Ad Library"
    r"|hasn't received any impressions"
    r"|This ad is no longer available"
    r"|content isn't available right now",
    re.I)


def _pick_destination(hrefs):
    """Recover the ad's outbound DESTINATION url (the advertiser's landing
    page) from the Ad Library page's anchors. Meta wraps outbound links as
    l.facebook.com/l.php?u=<encoded>; unwrap those first, then accept any
    other external (non-Meta) http(s) anchor. Returns url | None."""
    wrapped, external = [], []
    for h in hrefs or []:
        if not isinstance(h, str) or not h.lower().startswith(
                ("http://", "https://")):
            continue
        try:
            pu = urlparse(h)
        except ValueError:
            continue
        host = (pu.netloc or "").lower()
        if "l.php" in pu.path or host.startswith("l."):
            u = parse_qs(pu.query).get("u", [None])[0]
            if u and u.lower().startswith(("http://", "https://")):
                wrapped.append(u)
        elif host and not any(m in host for m in _META_HOSTS):
            external.append(h)
    for cand in wrapped + external:
        host = urlparse(cand).netloc.lower()
        if host and not any(m in host for m in _META_HOSTS):
            return cand
    return None


def _extract_ad_copy(page_text):
    """Best-effort extraction of the ad's OWN displayed body copy from the
    single-ad Ad Library page's DOM text (page.inner_text('body')).

    fetch() used to capture ONLY the creative media (video/image/screenshot)
    and never the ad's own text — even when that text is sitting right there
    in the page, unrelated to OCR/CLIP/Gemini. Confirmed on a real ad: its
    full ~5,000-word body copy (with the exact 'beer belly'/'man boobs'
    phrases a video/image-led creative's OCR never recovers) was present
    verbatim in page.inner_text('body') the whole time. A video/image ad's
    'the model can't see the video' failures are often really 'the scraper
    never read the text Meta already gave us' failures.

    The copy sits between the 'Sponsored' UI label (right after the
    advertiser/page name) and the link-preview card, which echoes the
    destination domain twice in a row (a lowercase URL line immediately
    followed by the same domain UPPERCASED as Meta's card title). Defensive:
    returns '' rather than mis-extracting UI chrome as ad copy if the
    expected markers aren't found (Meta's layout can change)."""
    if not page_text:
        return ""
    idx = page_text.rfind("\nSponsored\n")
    if idx == -1:
        return ""
    body = page_text[idx + len("\nSponsored\n"):]
    m = re.search(r"\n([a-z0-9][a-z0-9.\-]*\.[a-z]{2,})\n\1\n", "\n" + body,
                  re.I)
    body = body[:m.start()] if m else body[:8000]
    body = body.strip()
    return body if len(body) >= 40 else ""   # UI-chrome slivers are short


def fetch(url):
    """-> {'video','image','screenshot','landing_page','body_text'}.
    landing_page is the ad's captured outbound destination (advertiser
    page); body_text is its own displayed ad copy (see _extract_ad_copy),
    or '' if the page layout didn't match."""
    from playwright.sync_api import sync_playwright

    images = []          # (size, url)
    videos = []          # url
    dest = None          # ad's outbound landing/destination url

    def on_response(resp):
        u = resp.url
        ct = (resp.headers or {}).get("content-type", "")
        # Real ad creatives load from scontent.*.fbcdn.net; Meta's UI sprites
        # / empty-state illustrations come from static.xx.fbcdn.net/rsrc.php.
        # Only the former is an actual creative — keying on 'scontent' avoids
        # capturing chrome (the 'MMeta 80 FB' sprite) as if it were the ad.
        is_creative = "scontent" in u
        if not is_creative and "video/mp4" not in ct:
            return
        try:
            size = int((resp.headers or {}).get("content-length", "0"))
        except ValueError:
            size = 0
        if ".mp4" in u or "video/mp4" in ct:
            videos.append(u)
        elif is_creative and (ct.startswith("image/") or
                              any(e in u for e in (".jpg", ".png", ".webp"))):
            images.append((size, u))

    stem = _stem(url)
    shot = os.path.join(tempfile.gettempdir(), f"{stem}_shot.png")
    with sync_playwright() as p:
        ctx, b = _session(p)
        best = None    # pre-init: read after this try/finally block
        body_text = ""
        unavailable = False
        is_placeholder = True   # until a real rendered creative is found
        try:
            page = ctx.new_page()
            page.on("response", on_response)
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=60000)
            except Exception:  # noqa: BLE001
                pass
            # dismiss cookie / login dialogs
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
            # nudge any video to load its media so the .mp4 hits the network
            try:
                page.evaluate(
                    "document.querySelectorAll('video').forEach(v=>{"
                    "v.muted=true; v.play().catch(()=>{});})")
                page.wait_for_timeout(4000)
            except Exception:  # noqa: BLE001
                pass
            # Locate the LARGEST rendered <img>/<video> — that is the actual ad
            # creative. A full-page screenshot here grabs Meta's header/logo
            # (OCR'd as 'MMeta FB ...'), so instead we crop to this element (or
            # download its fbcdn src), which is the real ad image.
            best = None
            try:
                best = page.evaluate("""() => {
                  const els=[...document.querySelectorAll('img,video')];
                  let b=null,area=0;
                  for(const e of els){const r=e.getBoundingClientRect();
                    const a=r.width*r.height;
                    if(a>area && r.width>200 && r.height>200){area=a;b=e;}}
                  if(!b) return null; const r=b.getBoundingClientRect();
                  return {src:(b.currentSrc||b.src||''),
                          x:r.x,y:r.y,w:r.width,h:r.height};
                }""")
            except Exception:  # noqa: BLE001
                best = None
            # Self-screenshot the creative element via Playwright (cropped) as a
            # fallback when network capture missed the media. This grabs real
            # image creatives AND VIDEO FRAMES — a video element's src is a blob:
            # URL, not scontent, so it's never caught as network media; a cropped
            # screenshot of it yields a frame to OCR / face-detect. We only skip
            # FB's own UI/empty-state placeholder graphics (never a real ad).
            best_src = (best or {}).get("src", "")
            is_placeholder = any(s in best_src for s in (
                "empty-state", "/images/ads/", "/rsrc.php/", "static.xx"))
            try:
                if best and not is_placeholder and best.get("w") and best.get("h"):
                    page.screenshot(path=shot, clip={
                        "x": max(best["x"], 0), "y": max(best["y"], 0),
                        "width": min(best["w"], 1280),
                        "height": min(best["h"], 1600)})
                else:
                    shot = None              # only the placeholder/chrome present
            except Exception:  # noqa: BLE001
                shot = None
            # The ad's outbound DESTINATION link (advertiser landing page): its
            # domain names the product — the signal a DR creative withholds. Meta
            # wraps outbound links in l.facebook.com/l.php; harvest the page's
            # anchors so _pick_destination can unwrap the real one.
            try:
                _hrefs = page.evaluate(
                    "() => [...document.querySelectorAll('a[href]')]"
                    ".map(a => a.href)")
            except Exception:  # noqa: BLE001
                _hrefs = []
            dest = _pick_destination(_hrefs)
            # The ad's OWN displayed body copy — see _extract_ad_copy. Best-
            # effort: a failed extraction degrades to '' (existing OCR/CLIP path
            # unaffected), never raises.
            try:
                page_text = page.inner_text("body")
            except Exception:  # noqa: BLE001
                page_text = ""
            unavailable = bool(_UNAVAILABLE.search(page_text or ""))
            try:
                body_text = _extract_ad_copy(page_text)
            except Exception:  # noqa: BLE001
                body_text = ""
            # SECOND GATE: a marker found in page chrome must never be able to
            # discard an ad that plainly HAS content. Refusing is only correct
            # when there is nothing to analyse — so recovered copy or a real
            # creative on the page overrides the marker. (Learned the hard
            # way: the first cut of this check keyed on a phrase belonging to
            # the search panel and threw away a live ad.)
            # NOT `images`: that list contains the advertiser's profile photo,
            # which is served from scontent on the empty-state page too, and
            # keying on it would undo the thumbnail fix above. Only real copy,
            # a video, or a non-placeholder rendered creative counts.
            if unavailable and (len(body_text) >= 200 or videos
                                or (best and not is_placeholder)):
                unavailable = False
        finally:
            _close(ctx, b)

    # Only real scontent creatives count. If none were seen, Meta served us a
    # degraded/placeholder page (login-walled headless, or an inactive ad) —
    # report failure honestly instead of returning chrome.
    out = {"video": None, "image": None, "screenshot": None,
           "landing_page": dest, "body_text": body_text, "empty": False,
           "unavailable": unavailable, "reason": ""}
    # Meta told us there is no ad here. Everything else on the page is chrome
    # (the empty-state illustration, the page's profile photo, Meta's own
    # footer links) — return the failure instead of dressing it up.
    if unavailable:
        out["empty"] = True
        out["reason"] = ("this ad is not in the Meta Ad Library "
                         "(inactive, or it never received impressions)")
        out["landing_page"] = None
        out["body_text"] = ""
        return out
    if videos:
        try:
            out["video"] = _download(videos[0], ".mp4", stem)
        except Exception:  # noqa: BLE001
            pass
    if not out["video"]:
        src = (best or {}).get("src", "")
        if "scontent" in src:                      # rendered creative element
            try:
                out["image"] = _download(src, ".jpg", stem)
            except Exception:  # noqa: BLE001
                pass
        if not out["image"] and images:            # largest network creative
            big = max(images, key=lambda t: t[0])[1]
            try:
                out["image"] = _download(big, ".jpg", stem)
            except Exception:  # noqa: BLE001
                pass
        # A profile photo / UI thumbnail is not the ad. Drop it rather than
        # hand a 160px avatar to OCR, face detection and CLIP.
        if out["image"] and not _is_real_creative(out["image"]):
            wh = _pixels(out["image"])
            out["image"] = None
            out["reason"] = (f"only a {wh[0]}x{wh[1]} thumbnail was served "
                             f"(page profile photo, not the creative)"
                             if wh else "no full-size creative was served")
    if not out["video"] and not out["image"]:
        out["screenshot"] = shot                   # cropped creative, if any
    if not (out["video"] or out["image"] or out["screenshot"]):
        out["empty"] = True
        if not out["reason"]:
            out["reason"] = "no creative media was served for this ad"
    return out


def _harvest_ids(url, max_ads, country):
    """Render an Ad Library results URL, lazy-scroll, and harvest the per-ad
    'Library ID: NNNN' numbers + any /ads/library/?id=NNNN links. Shared by
    list_page_ads (by page) and search_ads (by keyword). Returns a deduped
    id list, or [] if Meta login-walls / renders nothing (caller must treat
    [] as 'scrape blocked', never 'no ads')."""
    from playwright.sync_api import sync_playwright

    txt = ""
    with sync_playwright() as p:
        ctx, b = _session(p)
        page = ctx.new_page()
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
        page.wait_for_timeout(5000)
        for _ in range(4):                      # lazy-load a few ad cards
            try:
                page.mouse.wheel(0, 4000)
            except Exception:  # noqa: BLE001
                pass
            page.wait_for_timeout(1500)
        try:
            txt = page.content() + "\n" + page.inner_text("body")
        except Exception:  # noqa: BLE001
            txt = ""
        _close(ctx, b)

    ids = []
    for pat in (r"[Ll]ibrary ID:?\s*(\d{6,})",
                r"/ads/library/\?id=(\d{6,})"):
        for m in re.finditer(pat, txt):
            if m.group(1) not in ids:
                ids.append(m.group(1))
    return ids[:max_ads]


def search_ads(query, max_ads=10, country="US"):
    """Enumerate live ads by brand-name KEYWORD search (no page_id needed) —
    the Ad Library keyword results list the same Library IDs."""
    from urllib.parse import quote
    url = (f"https://www.facebook.com/ads/library/?active_status=all"
           f"&ad_type=all&country={country}&q={quote(query)}"
           f"&search_type=keyword_unordered&media_type=all")
    return _harvest_ids(url, max_ads, country)


def list_page_ads(page_id, max_ads=10, country="US"):
    """Enumerate a brand's live ads from its Ad Library 'view all' page.

    The Ad Library is a React app, so we render the page, scroll to load a
    few cards, and harvest the per-ad 'Library ID: NNNNN' numbers (also any
    `/ads/library/?id=NNNNN` links). Best-effort: returns a (deduped) list
    of ad ids, or [] if the page login-walls / renders nothing — the caller
    must treat [] as 'scrape blocked', never as 'no ads'."""
    # active only: inactive ads render an 'empty-state' placeholder (no
    # creative), which OCRs as Meta chrome — skip them at the source.
    url = (f"https://www.facebook.com/ads/library/?active_status=active"
           f"&ad_type=all&country={country}&view_all_page_id={page_id}"
           f"&media_type=all")
    return _harvest_ids(url, max_ads, country)


def fetch_guarded(url, timeout=None):
    """fetch(url), but run in an isolated child process with a HARD
    wall-clock timeout — see v4_fetch_guard.py for why fetch() alone can't
    be trusted not to hang the caller (a real user pasting an ad URL into
    Bernay). Same return shape as fetch(), plus 'timed_out' on failure."""
    import v4_fetch_guard
    t = timeout or v4_fetch_guard.DEFAULT_TIMEOUT
    here = os.path.dirname(os.path.abspath(__file__))
    return v4_fetch_guard.run_guarded(
        os.path.join(here, "v4_adlib.py"), [url], timeout=t)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--json":
        u = sys.argv[2] if len(sys.argv) > 2 else \
            "https://www.facebook.com/ads/library/?id=983733107492934"
        import json
        print(json.dumps(fetch(u)))
        raise SystemExit(0)
    if len(sys.argv) > 1 and sys.argv[1] == "--page":
        pid = sys.argv[2] if len(sys.argv) > 2 else "61577260490433"
        print(f"listing ads for page {pid} ...")
        for i in list_page_ads(pid):
            print(f"  ad id: {i}")
        raise SystemExit(0)
    u = sys.argv[1] if len(sys.argv) > 1 else \
        "https://www.facebook.com/ads/library/?id=983733107492934"
    print(f"fetching: {u}")
    r = fetch(u)
    for k, v in r.items():
        if k == "body_text":
            print(f"  {k:11} {len(v or '')} chars"
                  f"{'  ' + repr(v[:80]) + '...' if v else ''}")
            continue
        sz = os.path.getsize(v) if v and os.path.exists(v) else 0
        print(f"  {k:11} {v}  ({sz:,} bytes)")
