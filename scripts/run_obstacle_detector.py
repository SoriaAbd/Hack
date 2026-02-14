#!/usr/bin/env python3
"""
Launch script for the obstacle detector node.

This script initializes and runs the obstacle detector node.
"""

import sys
import os

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    from adaptnav.perception.obstacle_detector import main
    
    if __name__ == '__main__':
        main()
except ImportError as e:
    print(f"Error importing obstacle detector: {e}")
    print("Make sure ROS 2 is installed and sourced.")
    sys.exit(1)