@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "PY=%LELAB_PY%"
if not defined PY set "PY=python"

echo.
echo ============================================================
echo   DEMO (one-shot) : say "qing li zhuo mian" / "清理桌面"
echo   Formal trial run uses grasp_retry\run_trial.bat
echo ============================================================
echo.
echo [1/4] Offline ASR model
if exist "models\faster-whisper-small\model.bin" (
  echo       OK   agent\models\faster-whisper-small ready, no network needed
) else (
  echo       WARN model missing - voice disabled, text command still works
)
echo.
echo [2/4] faster-whisper
"%PY%" -c "import faster_whisper" 2>nul && (echo       OK) || (echo       WARN not installed, pip install faster-whisper)
echo.
echo [3/4] LeLab on :8000
"%PY%" -c "import urllib.request as u;u.build_opener(u.ProxyHandler({})).open('http://127.0.0.1:8000/health',timeout=2)" >nul 2>nul && (echo       OK   LeLab is running) || echo       WARN LeLab not reachable - arm will not move
echo.
echo [4/4] Pre-flight reminder
echo       Run grasp_retry\check_ready.bat until all six checks are green.
echo       Verify F1/F2 not swapped, exactly 1 block on table, bin empty.
echo.
echo ------------------------------------------------------------
echo   Browser opens http://localhost:8100
echo   Flow: click VOICE -^> say "清理桌面" -^> check transcript
echo         -^> click CONFIRM -^> arm moves -^> result card
echo   Safety: require_confirm=true, arm never moves before CONFIRM.
echo   Data: --outdir demo_logs, never mixed into the formal trials.
echo   End demo: Ctrl+C here. No background / always-on listener.
echo ------------------------------------------------------------
echo.
"%PY%" agent_server.py --open --outdir demo_logs %*
if errorlevel 1 pause
