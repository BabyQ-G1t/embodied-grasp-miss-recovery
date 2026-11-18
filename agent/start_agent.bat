@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "PY=%LELAB_PY%"
if not defined PY set "PY=python"
echo.
echo Starting agent server ...  (Ctrl+C to quit)
echo.
"%PY%" agent_server.py --open %*
if errorlevel 1 pause
