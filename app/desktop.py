"""
Bernay — desktop shell.

Opens the Bernay analysis app in a native window (pywebview / Edge
WebView2 — no Electron; this box has 7.5 GB RAM). Reuses an already
running Bernay API on 127.0.0.1:8756 (e.g. the "Bernay API Server"
console window); otherwise boots the server itself as a hidden child
process and shuts it down again when the window closes.

Launch: `bernay-desktop` (cmd shim in ~/.local/bin) or
        <your venv>/bin/python desktop.py

Env: BERNAY_APP_SMOKE=1 auto-closes the window after ~6 s (CI/smoke).
     BERNAY_PORT=8757   attach to a sandbox instance instead of the real
                        app (see server/sandbox.py) — a test loop that
                        drives the window then never touches port 8756.
     BERNAY_PYTHON      interpreter used to spawn the API server. Defaults to
                        the one running this file, so a normal
                        `python desktop.py` inside an activated venv needs no
                        configuration at all.
"""
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request

PORT = int(os.environ.get("BERNAY_PORT", "8756"))
BASE = f"http://127.0.0.1:{PORT}"
HERE = os.path.dirname(os.path.abspath(__file__))
SERVER_DIR = os.path.join(HERE, "server")
# sys.executable is the interpreter already running this file — inside an
# activated venv that IS the right one, so the app needs no configuration to
# start its own server. BERNAY_PYTHON overrides it for a split-venv setup.
VENV_PY = os.environ.get("BERNAY_PYTHON", sys.executable)


def server_up(timeout=2):
    try:
        with urllib.request.urlopen(BASE + "/api/health", timeout=timeout):
            return True
    except Exception:  # noqa: BLE001
        return False


def main():
    owned_proc = None
    if not server_up():
        owned_proc = subprocess.Popen(
            [VENV_PY, "-m", "uvicorn", "server:app",
             "--host", "127.0.0.1", "--port", str(PORT),
             "--app-dir", SERVER_DIR],
            cwd=SERVER_DIR,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        # wait for the HTTP bind (model keeps loading in the background;
        # the UI reads /api/health and shows its own loading state)
        for _ in range(90):
            if server_up(1):
                break
            if owned_proc.poll() is not None:
                sys.exit("Bernay server exited during startup "
                         f"(code {owned_proc.returncode})")
            time.sleep(1)
        else:
            owned_proc.terminate()
            sys.exit("Bernay server did not come up on port %d" % PORT)

    import webview  # deferred: import is slow, do it after the server check
    from webview.dom import DOMEventHandler

    class Api:
        """Bridged to the page as window.pywebview.api — lets the React
        shell trigger native window behavior (F11 fullscreen) that a
        pywebview window doesn't get for free the way a real browser tab
        does. Looks the window up via webview.windows rather than storing
        it on self: pywebview's bridge introspects the js_api object's own
        attributes, and a stashed live Window recurses into its WebView2
        COM handle (window.native.AccessibilityObject...) until
        RecursionError — confirmed by A/B testing with/without this."""

        def toggle_fullscreen(self):
            if webview.windows:
                webview.windows[0].toggle_fullscreen()

    window = webview.create_window(
        "",
        BASE + "/",
        width=1280, height=860, min_size=(960, 640),
        background_color="#0d0d12",
        js_api=Api(),
    )

    # --- local file drop -> real filesystem path --------------------------
    # WebView2 (like any browser) never exposes a dropped file's local path to
    # page JS, so the React `onDrop` handler can't get it on its own. pywebview
    # CAN read it — but only for drops that go through its own DOM listener,
    # and only then does the WebView2 backend bother to capture the native
    # CoreWebView2File paths (see webview/util.py: pywebviewFullPath is injected
    # into the *Python* event dict, never onto the browser File object). So the
    # drop has to be handled here, in Python, and the resolved path pushed into
    # the input field. Without this a dropped image/video silently does nothing.
    def _on_drop(event):
        try:
            files = (event.get("dataTransfer") or {}).get("files") or []
        except AttributeError:
            return
        path = next(
            (f.get("pywebviewFullPath") for f in files if f.get("pywebviewFullPath")),
            None,
        )
        if not path or not webview.windows:
            return
        # Prefer the app's own hook (clean, keeps React state authoritative);
        # fall back to driving the textarea's native setter + an input event so
        # this still works against an older ui/dist bundle that predates the
        # hook. One textarea in the app, so the selector is unambiguous.
        js = (
            "(function(p){"
            "if(window.__bernaySetInput){window.__bernaySetInput(p);return;}"
            "var ta=document.querySelector('textarea');if(!ta)return;"
            "var d=Object.getOwnPropertyDescriptor("
            "window.HTMLTextAreaElement.prototype,'value');"
            "d.set.call(ta,p);"
            "ta.dispatchEvent(new Event('input',{bubbles:true}));"
            "})(" + json.dumps(path) + ")"
        )
        try:
            webview.windows[0].evaluate_js(js)
        except Exception:  # noqa: BLE001 — a closed window mid-drop is harmless
            pass

    def _bind_dnd(window):
        # Registering a drop listener is what flips _dnd_state['num_listeners']
        # on and makes WebView2 collect the paths in the first place. The React
        # side already preventDefaults dragover, so the drop event fires and
        # bubbles to document, where this listener sees it.
        try:
            window.dom.document.on("drop", DOMEventHandler(_on_drop, prevent_default=True))
        except Exception:  # noqa: BLE001
            pass

    window.events.loaded += _bind_dnd

    if os.environ.get("BERNAY_APP_SMOKE"):
        threading.Timer(6.0, window.destroy).start()

    try:
        # Without an explicit icon, pywebview's Windows backend
        # (webview/platforms/winforms.py) falls back to ExtractIconW on
        # sys.executable — python.exe running from a venv Scripts/ dir
        # doesn't reliably carry an icon resource, and when extraction
        # fails the taskbar/title-bar icon renders as a blank black
        # square instead of falling back further to anything visible.
        webview.start(icon=os.path.join(HERE, "bernay.ico"))
    finally:
        if owned_proc is not None:
            owned_proc.terminate()
            try:
                owned_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                owned_proc.kill()


if __name__ == "__main__":
    main()
