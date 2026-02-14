@echo off
REM Windows batch file to run the AdaptNav demo

echo ============================================================
echo AdaptNav Warehouse Navigation Demo
echo ============================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.7+ from https://python.org
    pause
    exit /b 1
)

REM Run the demo launcher
python run_demo.py

echo.
echo Demo finished. Press any key to exit.
pause >nul