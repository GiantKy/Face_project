@echo off
chcp 65001 > nul
title Node.js Receiver Server (:3000)
echo ===================================================
echo   DANG KHOI DONG NODE.JS RECEIVER SERVER (:3000)
echo ===================================================
cd /d "%~dp0"

for /f "tokens=5" %%a in ('netstat -ano ^| findstr :3000 ^| findstr LISTENING') do (
    echo [THONG BAO] Phat hien cong 3000 dang bi chiem boi tien trinh cu [PID %%a]. Dang giai phong...
    taskkill /F /PID %%a >nul 2>nul
    ping 127.0.0.1 -n 2 >nul
)

if exist "E:\node,js\node.exe" (
    "E:\node,js\node.exe" server_module/nodejs_server_receiver.js
    pause
    exit /b
)

where node >nul 2>nul
if %ERRORLEVEL% equ 0 (
    node server_module/nodejs_server_receiver.js
    pause
    exit /b
)

if exist "C:\Program Files\nodejs\node.exe" (
    "C:\Program Files\nodejs\node.exe" server_module/nodejs_server_receiver.js
    pause
    exit /b
)

node server_module/nodejs_server_receiver.js
pause
