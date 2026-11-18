@echo off
chcp 936 >nul
cd /d "%~dp0"
set "PY=%LELAB_PY%"
if not defined PY set "PY=python"
if not exist "bin_roi.json" (
  echo.
  echo [!] 还没标定收纳区，正式实验无法开始。
  echo     首次使用请先运行： calibrate_bin.bat
  echo     或直接执行： "%PY%" run_trial.py --calibrate-bin
  echo.
  pause
  exit /b 3
)
"%PY%" run_trial.py --manual %*
echo.
pause
