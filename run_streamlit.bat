@echo off
REM Windows batch script to run the Streamlit demo

echo ============================================================
echo AdaptNav Streamlit Web Demo
echo ============================================================
echo.

REM Check if streamlit is installed
python -c "import streamlit" 2>nul
if errorlevel 1 (
    echo Streamlit is not installed.
    echo Installing Streamlit...
    pip install streamlit
    if errorlevel 1 (
        echo Failed to install Streamlit
        echo Please install manually: pip install streamlit
        pause
        exit /b 1
    )
)

echo Starting web demo...
echo The demo will open in your browser at http://localhost:8501
echo Press Ctrl+C to stop the server
echo.

streamlit run streamlit_app.py

pause
