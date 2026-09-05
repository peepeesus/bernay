@echo off
REM Bernay disposable sandbox — a throwaway instance on a free port so the real
REM app on 8756 survives a test loop.
REM   sandbox.cmd -- pytest test_parity.py
REM   sandbox.cmd --keep | --list | --stop all
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
if "%BERNAY_PYTHON%"=="" set BERNAY_PYTHON=python
"%BERNAY_PYTHON%" "%~dp0sandbox.py" %*
