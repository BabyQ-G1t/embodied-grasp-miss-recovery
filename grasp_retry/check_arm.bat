@echo off
chcp 936 >nul
cd /d "%~dp0"
set "PY=%LELAB_PY%"
if not defined PY set "PY=python"
echo 正在自检串口与舵机（每个口会用各种波特率 ping 一遍，约 10-20 秒）。
echo 若提示端口被占用，先关掉 LeLab：lelab-zh.exe --stop
echo.
"%PY%" check_arm.py %*
echo.
pause
