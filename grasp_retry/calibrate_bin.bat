@echo off
chcp 936 >nul
cd /d "%~dp0"
set "PY=%LELAB_PY%"
if not defined PY set "PY=python"
echo ============================================================
echo   收纳区标定（tkinter 版，不依赖 opencv 的 GUI）
echo   拖一个矩形框住【收纳区内部】，回车保存
echo   R 重取一帧 / Esc 取消
echo   框内不要含收纳区外沿；宁可比收纳区小，不要框进桌面
echo ============================================================
echo.
"%PY%" calibrate_gui.py %*
echo.
echo 已保存的话，请打开 bin_roi_preview.png 核对红框是否套准。
pause
