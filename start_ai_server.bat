@echo off
chcp 65001 > nul
title FastAPI AI Server (:8000)
echo ===================================================
echo   DANG KHOI DONG FASTAPI E-KYC AI SERVER (:8000)
echo ===================================================
cd /d "%~dp0"

where python >nul 2>nul
if %ERRORLEVEL% equ 0 (
    python run_api_server.py
    pause
    exit /b
)

if exist "C:\Users\HP\AppData\Local\Programs\Python\Python311\python.exe" (
    "C:\Users\HP\AppData\Local\Programs\Python\Python311\python.exe" run_api_server.py
    pause
    exit /b
)

where py >nul 2>nul
if %ERRORLEVEL% equ 0 (
    py -3.11 run_api_server.py
    pause
    exit /b
)

echo [LOI] Khong tim thay Python! Vui long kiem tra lai duong dan Python trong he thong.
pause

