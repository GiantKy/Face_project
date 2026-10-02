@echo off
chcp 65001 > nul
title FastAPI AI Server (:8000)
echo ===================================================
echo   DANG KHOI DONG FASTAPI E-KYC AI SERVER (:8000)
echo ===================================================
cd /d "%~dp0"

for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8000 ^| findstr LISTENING') do (
    echo [THONG BAO] Phat hien cong 8000 dang bi chiem boi tien trinh cu [PID %%a]. Dang giai phong...
    taskkill /F /PID %%a >nul 2>nul
    ping 127.0.0.1 -n 2 >nul
)

if exist "C:\Users\HP\AppData\Local\Programs\Python\Python311\python.exe" (
    "C:\Users\HP\AppData\Local\Programs\Python\Python311\python.exe" server_module/app.py
    pause
    exit /b
)

where python >nul 2>nul
if %ERRORLEVEL% equ 0 (
    python server_module/app.py
    pause
    exit /b
)

where py >nul 2>nul
if %ERRORLEVEL% equ 0 (
    py -3.11 server_module/app.py
    pause
    exit /b
)

echo [LOI] Khong tim thay Python! Vui long kiem tra lai duong dan Python trong he thong.
pause
