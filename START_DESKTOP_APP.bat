@echo off
title SMS GUARD - Launching Desktop App
cls
echo ======================================================================
echo                SMS GUARD -- LAUNCHING DESKTOP APPLICATION
echo ======================================================================
echo.
cd /d "%~dp0"

echo [1/2] Starting local AI backend service...
start "" /B "C:\Users\Saksham Singh\AppData\Local\Programs\Python\Python311\python.exe" -m streamlit run app.py --server.headless true --server.port 8501

echo [2/2] Opening dedicated Desktop App window...
timeout /t 3 /nobreak >nul

where msedge >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    start msedge --app=http://localhost:8501 --window-size=1280,850
    goto done
)

where chrome >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    start chrome --app=http://localhost:8501 --window-size=1280,850
    goto done
)

start http://localhost:8501

:done
echo.
echo ======================================================================
echo App is now running in its own standalone desktop window!
echo ======================================================================
exit
