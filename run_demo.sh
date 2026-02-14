#!/bin/bash
# Linux/Mac shell script to run the AdaptNav demo

echo "============================================================"
echo "AdaptNav Warehouse Navigation Demo"
echo "============================================================"
echo

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        echo "Error: Python is not installed or not in PATH"
        echo "Please install Python 3.7+ from your package manager"
        exit 1
    else
        PYTHON_CMD="python"
    fi
else
    PYTHON_CMD="python3"
fi

echo "Using Python: $PYTHON_CMD"
$PYTHON_CMD --version
echo

# Run the demo launcher
$PYTHON_CMD run_demo.py

echo
echo "Demo finished."