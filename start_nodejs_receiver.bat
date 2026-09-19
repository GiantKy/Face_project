@echo off
chcp 65001 > nul
title Node.js Receiver Server (:3000)
echo ===================================================
echo   DANG KHOI DONG NODE.JS RECEIVER SERVER (:3000)
echo ===================================================
cd /d "%~dp0"
node server_module/nodejs_server_receiver.js
pause
