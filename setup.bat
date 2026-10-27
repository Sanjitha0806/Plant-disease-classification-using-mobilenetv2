@echo off
title Plant Disease Recognition System - Setup
cd /d "%~dp0"

echo ========================================================
echo       Plant Disease Recognition System - Setup
echo ========================================================
echo.

:: 1. Check Python
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please download and install 64-bit Python from https://www.python.org/
    pause
    exit /b 1
)

:: 2. Check Architecture & Install Requirements
echo [1/2] Checking environment and installing dependencies...
python install_dependencies.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Setup encountered an issue. See instructions above.
    pause
    exit /b 1
)

:: 3. Download Model
echo.
echo [2/2] Verifying pre-trained model...
python download_model.py
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to obtain model weights.
    pause
    exit /b 1
)

echo.
echo ========================================================
echo  Setup Completed Successfully!
echo  Run "run.bat" to start the Streamlit web application.
echo ========================================================
pause
