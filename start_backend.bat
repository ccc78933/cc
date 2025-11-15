@echo off
echo 正在启动后端服务器...
echo.
cd /d %~dp0
python server/app.py
pause

