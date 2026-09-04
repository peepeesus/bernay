"""v4_fetch_guard — a hard wall-clock timeout for the Playwright-based
fetchers (v4_adlib, v4_landing, v4_trendtrack).

WHY THIS EXISTS: every one of those fetchers is reachable from a real user
pasting a URL into Bernay (v4_media.is_media routes adlib/landing/trendtrack
kinds straight to them). Diagnosed 2026-07-18: a heavy scraping session left
this pattern exposed — a fetch can genuinely hang with NO exception and NO
per-step timeout firing (most likely cause: two Playwright launches racing
for the SAME persistent profile lock in `.v4_fb_profile`; Chromium's
SingletonLock on a userDataDir doesn't always fail fast). Per-step
try/except (already present in all three fetchers) only bounds INDIVIDUAL
page actions; it does nothing if `p.chromium.launch()` / a native call
itself blocks. There is no reliable in-process way to interrupt a stuck
Playwright/Chromium call from Python (no signal.alarm on Windows; a Python
thread can't preempt a blocked native call) — so the only actually-robust
fix is an OUTER process boundary: run the fetch in a child process and
KILL THE WHOLE PROCESS TREE if it overruns, no matter what's stuck inside.

Each fetcher gets a `--json <url>` CLI mode (prints `fetch(url)`'s return
dict as one JSON line) and a `fetch_guarded(url, timeout=...)` wrapper that
shells out to itself via this module. On timeout: kill the process tree
(not just the direct child — a plain .kill() would orphan exactly the
node-driver/chromium processes that caused the hang, defeating the point)
and return an honest {'timed_out': True, ...} result instead of blocking
the caller (and therefore the whole desktop-app request) forever.
"""
import json
import os
import subprocess
import sys

DEFAULT_TIMEOUT = 90     # seconds; generous for a slow real video ad,
                          # bounded enough that a user is never left hanging

# Superset of every key any of the 3 fetchers' success dict can carry
# (adlib: video/image/screenshot/landing_page/body_text/empty; landing:
# text/screenshot/title; trendtrack: video/image/screenshot/meta). Every
# failure return below is seeded with ALL of them defaulted to falsy, so a
# caller's got.get("whatever") is always safe no matter which fetcher it
# came from or which one failed.
_EMPTY_RESULT = {
    "video": None, "image": None, "screenshot": None, "landing_page": None,
    "body_text": "", "text": "", "title": "", "meta": None, "empty": True,
}


def _kill_tree(pid):
    """Force-kill a process and every descendant it spawned (the Playwright
    Node driver + any Chromium/chrome-headless-shell children) — the step a
    plain Popen.kill() skips, which is exactly what leaves orphaned
    chrome-headless-shell.exe processes behind on a hang/timeout."""
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                       capture_output=True)
    else:
        import signal
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except Exception:  # noqa: BLE001
            pass


def sweep_orphans():
    """Best-effort: kill any chrome-headless-shell.exe whose PARENT process
    no longer exists — the residue of a killed process tree (Windows'
    taskkill /T doesn't always catch every grandchild in one pass; measured:
    an artificially-forced timeout mid-Chromium-spawn left a few behind that
    then self-exited within ~30-60s on their own, but don't rely on that).
    Safe by established project policy (see Downloads/BERNAY.md's hard
    constraints): chrome-headless-shell.exe is ONLY ever the Playwright
    headless browser these fetchers spawn — never the user's real Chrome,
    never node.exe/the AgentsRoom harness. Windows-only; no-op elsewhere.
    Called opportunistically at the start of every guarded fetch so orphans
    from a prior timeout can't accumulate across a session."""
    if os.name != "nt":
        return
    try:
        import subprocess as sp
        ps = (
            "Get-CimInstance Win32_Process -Filter \"Name='chrome-headless-shell.exe'\" "
            "| ForEach-Object { "
            "if (-not (Get-Process -Id $_.ParentProcessId -ErrorAction SilentlyContinue)) "
            "{ Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue } }"
        )
        sp.run(["powershell", "-NoProfile", "-Command", ps],
              capture_output=True, timeout=10)
    except Exception:  # noqa: BLE001 — best-effort only, never block a fetch on this
        pass


def run_guarded(script_path, args, timeout=DEFAULT_TIMEOUT):
    """Run `python <script_path> --json <args...>` in a child process; parse
    its single JSON-line stdout. On timeout or a bad/empty result, returns a
    dict with 'timed_out'/'error' set rather than raising or blocking —
    callers should treat those as "couldn't fetch this one", the same as
    any other best-effort failure this codebase already handles."""
    sweep_orphans()
    py = sys.executable
    kwargs = {}
    if os.name == "nt":
        # own process group so taskkill /T can reach the whole tree; also
        # keeps Ctrl+C in an interactive parent from prematurely signalling
        # the child before we get a chance to time it out ourselves.
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    proc = subprocess.Popen(
        [py, script_path, "--json", *args],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, **kwargs)
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_tree(proc.pid)
        try:
            proc.communicate(timeout=5)   # reap, best-effort
        except Exception:  # noqa: BLE001
            pass
        return {**_EMPTY_RESULT, "timed_out": True,
                "error": f"fetch exceeded {timeout}s and was killed"}
    if proc.returncode != 0 or not out.strip():
        return {**_EMPTY_RESULT, "timed_out": False,
                "error": (err or out or "no output")[-800:]}
    try:
        return {**_EMPTY_RESULT, **json.loads(out.strip().splitlines()[-1])}
    except Exception as e:  # noqa: BLE001
        return {**_EMPTY_RESULT, "timed_out": False,
                "error": f"couldn't parse child output: {e}: {out[:300]!r}"}
