#!/usr/bin/env python3
"""
Test script to verify the demo can be imported and basic functionality works.
"""

import sys
import os

def test_imports():
    """Test that required packages can be imported."""
    print("Testing imports...")
    
    try:
        import numpy as np
        print("✓ numpy imported successfully")
    except ImportError:
        print("✗ numpy import failed")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("✓ matplotlib imported successfully")
    except ImportError:
        print("✗ matplotlib import failed")
        return False
    
    return True

def test_demo_import():
    """Test that the demo can be imported."""
    print("\nTesting demo import...")
    
    try:
        # Add current directory to path
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        
        import demo_simulation
        print("✓ demo_simulation imported successfully")
        
        # Test creating a demo instance (without starting it)
        demo = demo_simulation.DemoSimulation()
        print("✓ DemoSimulation instance created successfully")
        
        return True
    except Exception as e:
        print(f"✗ demo import failed: {e}")
        return False

def test_basic_functionality():
    """Test basic demo functionality."""
    print("\nTesting basic functionality...")
    
    try:
        import demo_simulation
        
        # Create demo instance
        demo = demo_simulation.DemoSimulation()
        
        # Test simulation step (without visualization)
        demo.current_detected_obstacles = []
        demo.current_velocities = (0.0, 0.0)
        demo.simulation_step()
        
        print("✓ Simulation step executed successfully")
        
        # Test sensor simulation
        lidar_scan, depth_image = demo.simulate_sensors()
        print(f"✓ Sensor simulation works (LiDAR: {len(lidar_scan)} points)")
        
        # Test obstacle detection
        detected = demo.detect_obstacles(lidar_scan)
        print(f"✓ Obstacle detection works ({len(detected)} obstacles detected)")
        
        # Test navigation control
        linear_vel, angular_vel = demo.simple_navigation_control()
        print(f"✓ Navigation control works (vel: {linear_vel:.2f}, ang: {angular_vel:.2f})")
        
        return True
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("=" * 60)
    print("AdaptNav Demo Test Suite")
    print("=" * 60)
    
    all_passed = True
    
    # Test imports
    if not test_imports():
        all_passed = False
    
    # Test demo import
    if not test_demo_import():
        all_passed = False
    
    # Test basic functionality
    if not test_basic_functionality():
        all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All tests passed! The demo should work correctly.")
        print("\nTo run the demo:")
        print("  python demo_simulation.py")
        print("  or")
        print("  python run_demo.py")
    else:
        print("✗ Some tests failed. Please check the errors above.")
        print("\nTroubleshooting:")
        print("1. Install required packages: pip install numpy matplotlib")
        print("2. Make sure you're in the correct directory")
        print("3. Check Python version (3.7+ required)")
    
    print("=" * 60)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit(main())