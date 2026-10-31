@echo off
title SMS GUARD - Master Backend Pipeline
cls
echo ======================================================================
echo           SMS GUARD -- MASTER BACKEND PIPELINE EXECUTION
echo ======================================================================
echo.
cd /d "%~dp0"

"C:\Users\Saksham Singh\AppData\Local\Programs\Python\Python311\python.exe" run_backend.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Trying standard python command...
    python run_backend.py
)

echo.
echo ======================================================================
echo Execution complete. Press any key to exit this window.
echo ======================================================================
pause
