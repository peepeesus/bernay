@echo off
REM Restart the hosted Bernay API console window (kills the uvicorn serving the
REM REAL app port, then relaunches run_server.cmd minimized, titled).
REM
REM Scoped to --port 8756 on purpose: sandbox instances run on other ports and
REM must survive a restart of the real app, and vice versa.
powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'uvicorn server:app' -and $_.CommandLine -match '--port\s+8756\b' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='cmd.exe'\" | Where-Object { $_.CommandLine -match 'Bernay API Server' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"
start "Bernay API Server" /min cmd /k "%~dp0run_server.cmd"
echo RESTARTED
