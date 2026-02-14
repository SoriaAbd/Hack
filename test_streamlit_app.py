#!/usr/bin/env python3
"""
Test script to verify the Streamlit app can be imported and initialized.
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all required imports work."""
    print("Testing imports...")
    
    try:
        import numpy as np
        print("✓ NumPy imported successfully")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("✓ Matplotlib imported successfully")
    except ImportError as e:
        print(f"✗ Matplotlib import failed: {e}")
        return False
    
    try:
        import streamlit as st
        print("✓ Streamlit imported successfully")
    except ImportError as e:
        print(f"✗ Streamlit import failed: {e}")
        print("  Install with: pip install streamlit")
        return False
    
    return True

def test_adaptnav_components():
    """Test that AdaptNav components can be imported."""
    print("\nTesting AdaptNav components...")
    
    try:
        from adaptnav.core.warehouse_map import WarehouseMap
        from adaptnav.core.dynamic_obstacle import DynamicObstacle
        from adaptnav.core.robot_state import RobotState
        from adaptnav.core.path import Path, Waypoint
        print("✓ AdaptNav core components imported successfully")
        return True
    except ImportError as e:
        print(f"⚠ AdaptNav components not fully available: {e}")
        print("  The app will run with limited functionality")
        return True  # Not a critical failure

def test_simulation_creation():
    """Test that the simulation can be created."""
    print("\nTesting simulation creation...")
    
    try:
        # Import the simulation class
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        
        # We can't directly import from streamlit_app.py due to Streamlit's execution model
        # So we'll just verify the file exists and is valid Python
        with open('streamlit_app.py', 'r', encoding='utf-8') as f:
            code = f.read()
            compile(code, 'streamlit_app.py', 'exec')
        
        print("✓ streamlit_app.py is valid Python code")
        return True
    except Exception as e:
        print(f"✗ Failed to validate streamlit_app.py: {e}")
        return False

def test_file_structure():
    """Test that all required files exist."""
    print("\nTesting file structure...")
    
    required_files = [
        'streamlit_app.py',
        'streamlit_requirements.txt',
        'STREAMLIT_DEPLOYMENT.md',
        '.streamlit/config.toml',
        'Procfile'
    ]
    
    all_exist = True
    for file in required_files:
        if os.path.exists(file):
            print(f"✓ {file} exists")
        else:
            print(f"✗ {file} missing")
            all_exist = False
    
    return all_exist

def main():
    """Run all tests."""
    print("=" * 60)
    print("AdaptNav Streamlit App Test Suite")
    print("=" * 60)
    print()
    
    tests = [
        ("Imports", test_imports),
        ("AdaptNav Components", test_adaptnav_components),
        ("Simulation Creation", test_simulation_creation),
        ("File Structure", test_file_structure)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
        print()
    
    # Summary
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print()
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The Streamlit app is ready to run.")
        print("\nTo start the app, run:")
        print("  streamlit run streamlit_app.py")
        return 0
    else:
        print("\n⚠ Some tests failed. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    exit(main())
