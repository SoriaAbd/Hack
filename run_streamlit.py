#!/usr/bin/env python3
"""
Quick launcher for the AdaptNav Streamlit web demo.
"""

import subprocess
import sys
import os

def check_streamlit():
    """Check if Streamlit is installed."""
    try:
        import streamlit
        return True
    except ImportError:
        return False

def main():
    print("=" * 60)
    print("AdaptNav Streamlit Web Demo Launcher")
    print("=" * 60)
    print()
    
    # Check if Streamlit is installed
    if not check_streamlit():
        print("Streamlit is not installed.")
        response = input("Would you like to install it now? (y/n): ")
        
        if response.lower() in ['y', 'yes']:
            print("Installing Streamlit...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])
                print("✓ Streamlit installed successfully!")
            except subprocess.CalledProcessError:
                print("✗ Failed to install Streamlit")
                print("Please install manually: pip install streamlit")
                return 1
        else:
            print("Please install Streamlit manually:")
            print("  pip install streamlit")
            return 1
    
    # Check if streamlit_app.py exists
    if not os.path.exists("streamlit_app.py"):
        print("Error: streamlit_app.py not found!")
        print("Make sure you're running this from the project root directory.")
        return 1
    
    print("Starting AdaptNav Web Demo...")
    print()
    print("The demo will open in your web browser at:")
    print("  http://localhost:8501")
    print()
    print("Controls:")
    print("  - Use the sidebar to control the simulation")
    print("  - Press Ctrl+C here to stop the server")
    print()
    
    # Launch Streamlit
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"])
    except KeyboardInterrupt:
        print("\nDemo stopped by user")
    except Exception as e:
        print(f"Failed to start demo: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
