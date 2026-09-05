@echo off
REM Bernay desktop shell. Uses the interpreter on PATH (activate your venv
REM first), or set BERNAY_PYTHON to point at a specific one.
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
if "%BERNAY_PYTHON%"=="" set BERNAY_PYTHON=pythonw
"%BERNAY_PYTHON%" "%~dp0desktop.py"
