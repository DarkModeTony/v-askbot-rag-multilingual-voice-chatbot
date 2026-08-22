@echo off
title v-askbot - HackerHouse Goa 2026 Voice RAG
color 0b
echo ================================================================
echo    v-askbot: Voice-Enabled Multilingual RAG (HH Goa 2026)
echo ================================================================
echo.
cd /d "%~dp0"
echo Starting FastAPI Backend and UI on http://localhost:8000 ...
echo.
powershell -NoProfile -Command "$processes = Get-CimInstance Win32_Process -Filter 'Name = ''python.exe''' | Where-Object { $_.CommandLine -match 'uvicorn backend.app:app.*--port 8000' }; $processes | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"
start "v-askbot server" /b python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload

:wait_for_server
powershell -NoProfile -Command "try { Invoke-WebRequest -Uri 'http://127.0.0.1:8000/api/health' -UseBasicParsing -TimeoutSec 1 | Out-Null; exit 0 } catch { exit 1 }"
if errorlevel 1 (
	timeout /t 1 /nobreak >nul
	goto wait_for_server
)

start "" http://localhost:8000
echo Interface is ready. Press Ctrl+C to stop this launcher.
python -c "input()"
pause
