"""
UI smoke — drives the real SPA in headless Chromium against a live server.

Complements test_parity.py: that one asserts the API returns the locked
expectations, this one asserts the UI actually *renders* an analysis —
shell boots, health reaches ready, the Analyze gate behaves, and a real
result paints into the panels. Between them, an API-shape change that the
UI silently drops on the floor cannot pass.

Targets whatever server BERNAY_URL points at, so it is meant to run inside
a sandbox and never against the app you are using:

Run:  sandbox.cmd -- ..\\..\\.venv-vlm\\Scripts\\python.exe test_ui.py
      (or, against the real app on 8756:
       .venv-vlm\\Scripts\\python.exe bernay-app\\server\\test_ui.py)

Uses the Playwright + chromium-headless-shell already installed in
.venv-vlm — no extra browser download. Honors the BERNAY.md rail: this
only ever spawns chrome-headless-shell.exe.
"""
import os
import sys
import time

from playwright.sync_api import sync_playwright

BASE = (os.environ.get("BERNAY_URL")
        or os.environ.get("BERNAY_API_BASE")
        or "http://127.0.0.1:8756")

# Real ad copy — a synthetic string like "test" routes differently (the
# stack guards against junk input), so it would not exercise the render path.
AD = ("Tired of bloating after every meal? This 3-second morning ritual "
      "flattens your stomach without cutting carbs.")

READY_TIMEOUT_MS = 240_000     # model may still be loading when we arrive
ANALYZE_TIMEOUT_MS = 300_000   # CPU inference, no GPU on this box

TEXTAREA = "textarea[aria-label='Ad copy, URL, or file path to analyze']"


def test_shell_and_health(page):
    """The SPA boots and reports the engine ready."""
    page.goto(BASE + "/", wait_until="domcontentloaded")
    status = page.locator(".status")
    status.wait_for(timeout=30_000)
    page.wait_for_function(
        "() => document.querySelector('.status')"
        "?.textContent?.includes('ready')",
        timeout=READY_TIMEOUT_MS)
    txt = status.inner_text()
    assert "ready" in txt, f"engine never reported ready: {txt!r}"

    # the header should name the model it actually loaded
    badge = page.locator(".params-badge").inner_text()
    assert "k params" in badge, f"missing params badge: {badge!r}"


def test_analyze_gate(page):
    """Analyze stays disabled until there is input — the button drives a
    slow, expensive run, so an accidental empty submit is a real bug."""
    btn = page.get_by_role("button", name="Analyze")
    assert btn.is_disabled(), "Analyze enabled with empty input"

    page.fill(TEXTAREA, "   ")
    assert btn.is_disabled(), "Analyze enabled on whitespace-only input"

    page.fill(TEXTAREA, AD)
    btn.wait_for(timeout=5_000)
    assert not btn.is_disabled(), "Analyze still disabled with real input"


def test_analysis_renders(page):
    """A real analysis paints into the result panels."""
    page.get_by_role("button", name="Analyze").click()

    # result panels replace the empty state when the run lands
    page.wait_for_selector(".kv-v", timeout=ANALYZE_TIMEOUT_MS)
    assert page.locator(".empty").count() == 0, \
        "empty state still shown after a completed analysis"

    err = page.locator(".error")
    assert err.count() == 0, f"UI surfaced an error: {err.first.inner_text()!r}"

    # the model's actual output, not just chrome
    kvs = {k.inner_text().strip(): v.inner_text().strip()
           for k, v in zip(page.locator(".kv-k").all(),
                           page.locator(".kv-v").all())}
    # Only `problem` is asserted. `product` and `visual_category` are
    # conditionally rendered (App.tsx) and legitimately absent here: AD is
    # text-only copy that never names a brand, and product-ID is multi-source
    # (creative/VO/landing) by design. Do NOT "fix" a future failure by
    # requiring product — that would be asserting a hallucinated brand.
    assert kvs.get("problem"), f"problem missing or empty in results: {kvs}"

    chips = page.locator(".chip").count()
    assert chips > 0, "no desire/motif chips rendered"


def test_history_records_run(page):
    """The completed run is added to the history list."""
    items = page.locator(".hist-item")
    items.first.wait_for(timeout=15_000)
    assert items.count() >= 1, "completed analysis did not enter history"


def test_theme_toggle(page):
    """Dark mode applies — and the card canvas tokens do NOT follow it.

    That second half is the real guard: `tone="screen"` cards are a black
    canvas with white text in BOTH themes. If someone ever themes --paper /
    --ink directly instead of the semantic aliases, every card body turns
    dark-on-black and this fails.
    """
    def tokens():
        return page.evaluate("""() => {
          const s = getComputedStyle(document.documentElement)
          return {
            page: s.getPropertyValue('--surface-page').trim(),
            screen: s.getPropertyValue('--surface-screen').trim(),
            onScreen: s.getPropertyValue('--text-on-screen').trim(),
          }
        }""")

    before = tokens()
    page.get_by_role("button", name="Switch to dark mode").click()
    page.wait_for_timeout(200)
    after = tokens()

    assert page.evaluate(
        "() => document.documentElement.dataset.theme") == "dark", \
        "toggle did not set data-theme"
    assert after["page"] != before["page"], "page surface did not darken"
    assert after["screen"] == before["screen"], \
        "card canvas followed the theme — card text will go dark-on-black"
    assert after["onScreen"] == before["onScreen"], \
        "on-canvas text colour followed the theme"

    page.get_by_role("button", name="Switch to light mode").click()
    page.wait_for_timeout(200)


TESTS = [test_shell_and_health, test_analyze_gate,
         test_analysis_renders, test_history_records_run,
         test_theme_toggle]


def main():
    print(f"UI smoke against {BASE}")
    failures = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 860})
        console_errors = []
        page.on("console", lambda m: (m.type == "error"
                                      and console_errors.append(m.text)))
        try:
            # sequential and stateful on purpose: the tests walk one session
            # (boot -> type -> analyze -> history), which is the flow a user
            # actually performs; re-booting the SPA per test would cost
            # another model-ready wait for no added coverage.
            for t in TESTS:
                t0 = time.time()
                try:
                    t(page)
                    print(f"  PASS  {t.__name__} ({time.time() - t0:.1f}s)")
                except Exception as e:  # noqa: BLE001
                    failures.append((t.__name__, e))
                    print(f"  FAIL  {t.__name__} ({time.time() - t0:.1f}s)"
                          f"\n        {type(e).__name__}: {e}")
                    page.screenshot(
                        path=os.path.join(os.path.dirname(
                            os.path.abspath(__file__)),
                            f"ui_fail_{t.__name__}.png"))
        finally:
            browser.close()

    if console_errors:
        print(f"\nconsole errors ({len(console_errors)}):")
        for m in console_errors[:10]:
            print(f"  {m}")

    print(f"\n{len(TESTS) - len(failures)}/{len(TESTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
