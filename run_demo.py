#!/usr/bin/env python3
"""
Simple launcher for the AdaptNav demo.

This script checks dependencies and launches the demo with appropriate fallbacks.
"""

import sys
import subprocess
import importlib
import os

def check_package(package_name, install_name=None):
    """Check if a package is installed."""
    if install_name is None:
        install_name = package_name
    
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        print(f"Missing package: {install_name}")
        return False

def install_package(package_name):
    """Install a package using pip."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    """Main launcher function."""
    print("=" * 60)
    print("AdaptNav Demo Launcher")
    print("=" * 60)
    
    # Check required packages
    required_packages = [
        ("numpy", "numpy"),
        ("matplotlib", "matplotlib"),
    ]
    
    missing_packages = []
    for package, install_name in required_packages:
        if not check_package(package, install_name):
            missing_packages.append(install_name)
    
    # Install missing packages
    if missing_packages:
        print(f"\nMissing required packages: {', '.join(missing_packages)}")
        response = input("Would you like to install them automatically? (y/n): ")
        
        if response.lower() in ['y', 'yes']:
            print("Installing packages...")
            for package in missing_packages:
                print(f"Installing {package}...")
                if install_package(package):
                    print(f"✓ {package} installed successfully")
                else:
                    print(f"✗ Failed to install {package}")
                    print("Please install manually with: pip install " + package)
                    return 1
        else:
            print("Please install the required packages manually:")
            for package in missing_packages:
                print(f"  pip install {package}")
            return 1
    
    # Check if demo file exists
    demo_file = "demo_simulation.py"
    if not os.path.exists(demo_file):
        print(f"Error: {demo_file} not found!")
        print("Make sure you're running this from the project root directory.")
        return 1
    
    print("\n" + "=" * 60)
    print("Starting AdaptNav Demo...")
    print("=" * 60)
    print()
    print("Demo Features:")
    print("  ✓ Warehouse environment simulation")
    print("  ✓ Robot navigation with A* path planning")
    print("  ✓ Dynamic obstacle avoidance")
    print("  ✓ LiDAR sensor simulation")
    print("  ✓ Safety control system")
    print("  ✓ Real-time visualization")
    print()
    print("Controls:")
    print("  - Close the window to stop the demo")
    print("  - Press Ctrl+C here to force quit")
    print()
    print("Starting in 3 seconds...")
    
    import time
    for i in range(3, 0, -1):
        print(f"{i}...")
        time.sleep(1)
    
    print("Launching demo!")
    print()
    
    # Launch the demo
    try:
        import demo_simulation
        demo_simulation.main()
    except KeyboardInterrupt:
        print("\nDemo stopped by user")
    except Exception as e:
        print(f"Demo failed: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you have Python 3.7+ installed")
        print("2. Try installing packages manually: pip install numpy matplotlib")
        print("3. Check that you're in the correct directory")
        return 1
    
    print("\nDemo completed successfully!")
    return 0

if __name__ == "__main__":
    exit(main())