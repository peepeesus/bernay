"""
Bernay — disposable sandbox instances for test loops.

Runs a throwaway Bernay server on its own port with its own written state,
so a test loop never touches the real app on 8756. Start it, hammer it,
throw it away; the hosted "Bernay API Server" window and any open desktop
window keep running untouched.

What is isolated (per instance):
  - the port                     BERNAY_PORT
  - the session log              logs/sandbox/<name>_sessions.log
What is shared (read-only, deliberately):
  - checkpoints/ and data/       — hundreds of MB; copying them per run
                                   would cost more than the isolation buys

Usage:
  sandbox.py -- <cmd>...       boot, run <cmd> against it, tear down, exit
                               with <cmd>'s code. BERNAY_PORT / BERNAY_URL /
                               BERNAY_SANDBOX are set in <cmd>'s environment.
  sandbox.py --keep            boot and leave it running; prints the port
  sandbox.py --list            show live sandboxes
  sandbox.py --stop <port>|all tear one (or all) down

  --name NAME    label for the log file (default: sandbox-<port>)
  --port N       pin the port instead of taking the next free one
  --timeout SEC  how long to wait for the model to load (default 240)

Examples:
  python sandbox.py -- pytest test_parity.py
  python sandbox.py --name uiloop --keep     # then drive it in a loop
  BERNAY_PORT=8757 python ..\\desktop.py     # window onto that sandbox
"""
import argparse
import atexit
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
# The interpreter already running this file is the right one inside an
# activated venv; BERNAY_PYTHON overrides for a split-venv setup.
VENV_PY = os.environ.get("BERNAY_PYTHON", sys.executable)

REAL_PORT = 8756           # the actual app — sandboxes never touch it
PORT_RANGE = range(8757, 8800)


def _health(port, timeout=1):
    """Return the health dict, or None if nothing is answering."""
    try:
        with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/api/health", timeout=timeout) as r:
            return json.load(r)
    except (urllib.error.URLError, OSError, ValueError):
        return None


def _free_port():
    for port in PORT_RANGE:
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port))
            except OSError:
                continue          # in use — next
        return port
    sys.exit(f"no free port in {PORT_RANGE.start}-{PORT_RANGE.stop - 1}; "
             f"`sandbox.py --stop all` if these are stale")


def _kill_on_port(port):
    """Kill the uvicorn serving `port`, and only that one."""
    if port == REAL_PORT:
        sys.exit(f"refusing to stop the real app on {REAL_PORT} — "
                 f"use restart_server.cmd if that is what you meant")
    subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "Get-CimInstance Win32_Process | Where-Object { "
         "$_.CommandLine -match 'uvicorn server:app' -and "
         f"$_.CommandLine -match '--port\\s+{port}\\b' "
         "} | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"],
        capture_output=True)


def boot(port, name, timeout):
    """Start a sandbox server and wait until the model is loaded."""
    env = dict(os.environ,
               BERNAY_PORT=str(port),
               BERNAY_SANDBOX=name,
               PYTHONIOENCODING="utf-8",
               PYTHONUTF8="1")
    proc = subprocess.Popen(
        [VENV_PY, "-m", "uvicorn", "server:app",
         "--host", "127.0.0.1", "--port", str(port), "--app-dir", HERE],
        cwd=HERE, env=env, creationflags=subprocess.CREATE_NO_WINDOW)

    # The socket binds in seconds; the Schwartz stack loads in a background
    # thread and /api/health flips to ready when it lands. Tests want ready,
    # not bound — /api/analyze 503s while loading.
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            sys.exit(f"sandbox server exited during startup "
                     f"(code {proc.returncode})")
        h = _health(port)
        if h and h.get("ok"):
            return proc
        if h and h.get("status") == "error":
            proc.terminate()
            sys.exit(f"sandbox stack failed to load: {h.get('error')}")
        time.sleep(1)

    proc.terminate()
    sys.exit(f"sandbox on {port} not ready within {timeout}s "
             f"(--timeout to allow longer)")


def teardown(proc, port):
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
    _kill_on_port(port)          # catch a detached uvicorn reload child


def cmd_list():
    live = [(p, _health(p)) for p in PORT_RANGE]
    live = [(p, h) for p, h in live if h]
    if not live:
        print("no sandboxes running")
        return
    for port, h in live:
        print(f"  {port}  {h.get('status'):8} {h.get('model') or ''}")


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--stop")
    ap.add_argument("--name")
    ap.add_argument("--port", type=int)
    ap.add_argument("--timeout", type=int, default=240)
    ap.add_argument("-h", "--help", action="store_true")
    args, rest = ap.parse_known_args()

    if args.help:
        print(__doc__)
        return 0
    if args.list:
        cmd_list()
        return 0
    if args.stop:
        if args.stop == "all":
            for port in PORT_RANGE:
                if _health(port):
                    _kill_on_port(port)
                    print(f"stopped {port}")
        else:
            _kill_on_port(int(args.stop))
            print(f"stopped {args.stop}")
        return 0

    command = rest[1:] if rest[:1] == ["--"] else rest
    if not command and not args.keep:
        print(__doc__)
        return 2

    port = args.port or _free_port()
    name = args.name or f"sandbox-{port}"

    print(f"[sandbox] booting {name} on 127.0.0.1:{port} "
          f"(real app on {REAL_PORT} untouched)", flush=True)
    proc = boot(port, name, args.timeout)
    print(f"[sandbox] ready: http://127.0.0.1:{port}", flush=True)

    if args.keep:
        print(f"[sandbox] left running — stop with: "
              f"python sandbox.py --stop {port}")
        return 0

    atexit.register(teardown, proc, port)
    env = dict(os.environ,
               BERNAY_PORT=str(port),
               BERNAY_URL=f"http://127.0.0.1:{port}",
               # test_parity.py already reads this (defaults to 8756), so
               # `sandbox.cmd -- python test_parity.py` needs no changes there
               BERNAY_API_BASE=f"http://127.0.0.1:{port}",
               BERNAY_SANDBOX=name,
               # same rail as run_server.cmd: a test printing the UI's ●/…
               # glyphs must not die on the cp1252 console default
               PYTHONIOENCODING="utf-8",
               PYTHONUTF8="1")
    try:
        return subprocess.call(command, env=env)
    except KeyboardInterrupt:
        return 130
    except FileNotFoundError:
        sys.exit(f"command not found: {command[0]}")


if __name__ == "__main__":
    sys.exit(main())
