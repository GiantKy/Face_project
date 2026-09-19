@echo off
chcp 65001 > nul
title Test ESP32 Pipeline Simulator
echo ===================================================
echo   CHAY KIEM THU GIA LAP ESP32-CAM -> AI -> NODE.JS
echo ===================================================
cd /d "%~dp0"
python tests/test_esp32_pipeline.py
pause
