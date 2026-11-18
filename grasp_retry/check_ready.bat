@echo off
chcp 936 >nul
cd /d "%~dp0"
set "PY=%LELAB_PY%"
if not defined PY set "PY=python"
"%PY%" check_ready.py %*
echo.
if "%~1"=="" pause
