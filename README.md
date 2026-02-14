# AdaptNav: Context-Aware Warehouse Navigation 🤖

[![Demo Status](https://img.shields.io/badge/Demo-Working%20✅-brightgreen)](demo_simulation.py)
[![Python](https://img.shields.io/badge/Python-3.8+-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

A hybrid autonomous navigation system combining traditional path planning (A*) with reinforcement learning (PPO) for safe, explainable navigation in dynamic warehouse environments.

## 🚀 **Working Demo Available!**

**The robot movement issue has been fixed!** The demo now features:
- ✅ **Proper waypoint following** - Robot follows planned path sequentially
- ✅ **Goal reaching** - Robot successfully navigates to target positions  
- ✅ **Real-time visualization** - See the robot move in real-time
- ✅ **Obstacle avoidance** - Dynamic safety controller prevents collisions
- ✅ **Debug tools** - Comprehensive troubleshooting and testing utilities

### 🎮 Quick Demo

```bash
# Run the working demo (robot will move from (2,2) to (17,17))
python demo_simulation.py

# Test robot movement programmatically  
python test_robot_movement.py

# Debug any issues
python debug_robot_movement.py
```

**Expected behavior**: Robot starts at bottom-left (2,2), plans a path to top-right (17,17), and successfully navigates while avoiding obstacles!

## Overview

AdaptNav demonstrates a realistic robotics stack designed for warehouse navigation with:
- **Safety-First Architecture**: Multi-layered safety system with hard constraints
- **Hybrid Navigation**: Global path planning + local RL-based obstacle avoidance
- **Explainability**: Real-time visualization of decision-making process
- **Sim-to-Real Ready**: Standard ROS 2 interfaces for hardware transfer

## ✨ Features

- **🗺️ Multi-Algorithm Path Planning**: A* algorithm with dynamic obstacle avoidance
- **👁️ Sensor Fusion**: LiDAR and depth camera integration for comprehensive environment perception  
- **🧠 Reinforcement Learning**: PPO-based adaptive navigation for complex scenarios
- **📊 Real-time Visualization**: Web dashboard and RViz integration for monitoring
- **🌐 Interactive Web Demo**: Streamlit-based web interface for easy demonstration
- **🛡️ Safety Systems**: Multi-layered safety controllers for collision avoidance
- **🔧 Modular Architecture**: Easy to extend and customize for different warehouse layouts
- **🐛 Debug Tools**: Comprehensive troubleshooting and movement verification

## 🌐 Web Demo

The AdaptNav system includes an interactive web demo built with Streamlit:

- **Live Visualization**: Watch the robot navigate in real-time
- **Interactive Controls**: Step through or auto-run the simulation
- **Metrics Dashboard**: Track position, velocity, and navigation progress
- **Browser-Based**: No installation needed for viewers
- **Easy Deployment**: Deploy to Streamlit Cloud in minutes

See [STREAMLIT_DEPLOYMENT.md](STREAMLIT_DEPLOYMENT.md) for deployment instructions.

## 🔧 Recent Fixes

### Robot Movement Issue Resolution ✅

**Problem**: Robot wasn't moving or reaching goals in the demo.

**Solution**: Completely rewrote the navigation control logic with:
- **Sequential waypoint following**: Robot now tracks and follows waypoints in order
- **Proper waypoint progression**: Advances to next waypoint when current one is reached
- **Lookahead logic**: Smoother path following with configurable lookahead distance
- **Improved velocity control**: Better speed and turning control based on distance and angle
- **Enhanced debugging**: Detailed progress tracking and status reporting

**Result**: Robot successfully navigates from start (2,2) to goal (17,17) in ~24 seconds!

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- NumPy, Matplotlib (for demo)
- Streamlit (for web demo)
- ROS 2 (optional, for full features)

### Installation

```bash
# Clone the repository
git clone https://github.com/SoriaAbd/Hack.git
cd Hack

# Install demo dependencies
pip install numpy matplotlib

# Install Streamlit for web demo
pip install streamlit

# For full features (optional)
pip install -r requirements.txt
```

### 🎯 Running the Demo

#### Web Demo (Recommended for Hackathons!)

```bash
# Run the interactive web demo
streamlit run streamlit_app.py

# Or use the launcher scripts
python run_streamlit.py        # Cross-platform
./run_streamlit.sh            # Linux/Mac
run_streamlit.bat             # Windows
```

The web demo will open in your browser at `http://localhost:8501`

**🌐 Live Demo URL**: [Deploy to Streamlit Cloud](STREAMLIT_DEPLOYMENT.md) for a public URL!

#### Desktop Demo

```bash
# 1. Run the main demo (visual simulation)
python demo_simulation.py

# 2. Test robot movement (programmatic test)
python test_robot_movement.py

# 3. Debug any issues
python debug_robot_movement.py

# 4. Use the launcher scripts
python run_demo.py        # Cross-platform
./run_demo.sh            # Linux/Mac
run_demo.bat             # Windows
```

## Project Structure

```
adaptnav/
├── adaptnav/              # Main Python package
│   ├── __init__.py
│   ├── core/              # Core data models and utilities
│   ├── simulation/        # Simulation environment
│   ├── sensors/           # Sensor simulation
│   ├── perception/        # Obstacle detection and tracking
│   ├── planning/          # Path planning algorithms
│   ├── control/           # Safety controller
│   ├── rl/                # Reinforcement learning components
│   ├── navigation/        # Navigation controller
│   └── visualization/     # Dashboard and visualization
├── launch/                # ROS 2 launch files
├── config/                # Configuration YAML files
├── tests/                 # Test suite
│   ├── unit/              # Unit tests
│   ├── properties/        # Property-based tests
│   └── integration/       # Integration tests
├── scripts/               # Training and evaluation scripts
├── package.xml            # ROS 2 package manifest
├── CMakeLists.txt         # Build configuration
├── setup.py               # Python package setup
└── requirements.txt       # Python dependencies
```

## Architecture

The system follows a layered architecture:

1. **Simulation Layer**: Physics-based warehouse environment (MuJoCo/Isaac Sim)
2. **Sensor Layer**: Simulated LiDAR and depth camera with realistic noise
3. **Perception Layer**: Obstacle detection and tracking with sensor fusion
4. **Planning Layer**: A* global planner + PPO local navigation
5. **Safety Layer**: Hard safety constraints and collision avoidance
6. **Control Layer**: Navigation controller orchestrating all components
7. **Visualization Layer**: Real-time dashboard for explainability

## 🧪 Testing

```bash
# Test robot movement
python test_robot_movement.py

# Run unit tests
python -m pytest tests/

# Test specific components
python test_navigation_components.py
```

## 📚 Documentation

- **[Demo Guide](DEMO_GUIDE.md)** - Complete guide to running demos
- **[Troubleshooting Guide](ROBOT_MOVEMENT_TROUBLESHOOTING.md)** - Fix common issues
- **[Demo Summary](DEMO_SUMMARY.md)** - Overview of demo capabilities
- **[Requirements Document](.kiro/specs/adaptnav-context-aware-warehouse-navigation/requirements.md)** - System requirements
- **[Design Document](.kiro/specs/adaptnav-context-aware-warehouse-navigation/design.md)** - Architecture details
- **[Implementation Tasks](.kiro/specs/adaptnav-context-aware-warehouse-navigation/tasks.md)** - Development roadmap

## 🤝 Contributing

We welcome contributions! The system is modular and easy to extend.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**🎯 Ready to see the robot in action? Run `python demo_simulation.py` and watch it navigate!**
