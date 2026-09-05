@echo off
REM Bernay backend service (FastAPI on 127.0.0.1:8756).
REM Uses the interpreter on PATH (activate your venv first), or set
REM BERNAY_PYTHON. The model env (Schwartz-4.5 dims, checkpoints) is set
REM inside server.py.
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
if "%BERNAY_PYTHON%"=="" set BERNAY_PYTHON=python
"%BERNAY_PYTHON%" -m uvicorn server:app --host 127.0.0.1 --port 8756 --app-dir "%~dp0"
