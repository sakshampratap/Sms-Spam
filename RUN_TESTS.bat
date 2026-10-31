@echo off
title SMS GUARD - Running Automated Unit Tests
cls
echo ======================================================================
echo              SMS GUARD -- RUNNING AUTOMATED UNIT TESTS
echo ======================================================================
echo.
cd /d "%~dp0"

"C:\Users\Saksham Singh\AppData\Local\Programs\Python\Python311\python.exe" -m pytest tests -v

echo.
echo ======================================================================
echo All 22 tests should show PASSED in green above.
echo Press any key to close this window.
echo ======================================================================
pause
