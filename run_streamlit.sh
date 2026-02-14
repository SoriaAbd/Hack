#!/bin/bash
# Linux/Mac script to run the Streamlit demo

echo "============================================================"
echo "AdaptNav Streamlit Web Demo"
echo "============================================================"
echo ""

# Check if streamlit is installed
if ! python3 -c "import streamlit" 2>/dev/null; then
    echo "Streamlit is not installed."
    read -p "Would you like to install it now? (y/n): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Installing Streamlit..."
        pip3 install streamlit
        if [ $? -ne 0 ]; then
            echo "Failed to install Streamlit"
            echo "Please install manually: pip3 install streamlit"
            exit 1
        fi
    else
        echo "Please install Streamlit manually: pip3 install streamlit"
        exit 1
    fi
fi

echo "Starting web demo..."
echo "The demo will open in your browser at http://localhost:8501"
echo "Press Ctrl+C to stop the server"
echo ""

streamlit run streamlit_app.py
