@echo off
title SMS GUARD - AI-Powered SMS Spam Classifier
cls
echo ======================================================================
echo             SMS GUARD -- AI-POWERED SMS SPAM DETECTION
echo ======================================================================
echo.
echo [1/2] Checking Python environment...
cd /d "%~dp0"

echo [2/2] Starting Streamlit Web Application...
echo.
echo The app will automatically open in your default web browser!
echo URL: http://localhost:8501
echo.
echo (Keep this black window OPEN while using the app. Press Ctrl+C to stop.)
echo ======================================================================
echo.

"C:\Users\Saksham Singh\AppData\Local\Programs\Python\Python311\python.exe" -m streamlit run app.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Streamlit exited with an error. Trying standard python command...
    python -m streamlit run app.py
)

pause
