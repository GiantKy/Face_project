@echo off
chcp 65001 >nul
title eKYC System Launcher

echo ==============================================================================
echo           HỆ THỐNG eKYC KHUÔN MẶT SINH TRẮC HỌC VÀ CHỐNG GIẢ MẠO
echo ==============================================================================
echo.
echo [1/2] Đang khởi động Node.js Stream Relay Hub (Port 3000)...
start "eKYC - Node.js Stream Relay" cmd /k "cd server_module && node nodejs_server_receiver.js"

timeout /t 2 /nobreak >nul

echo [2/2] Đang khởi động AI Pipeline Server FastAPI (Port 8000)...
start "eKYC - AI Pipeline Server" cmd /k "uvicorn server_module.app:app --host 0.0.0.0 --port 8000 --reload"

timeout /t 2 /nobreak >nul

echo.
echo ==============================================================================
echo Khởi động hoàn tất!
echo - Web Dashboard:  http://localhost:3000/
echo - AI API Docs:    http://localhost:8000/docs
echo ==============================================================================
echo.
pause
