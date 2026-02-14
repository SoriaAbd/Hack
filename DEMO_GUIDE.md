# AdaptNav Demo Guide

This guide provides comprehensive instructions for running and understanding the AdaptNav warehouse navigation demo.

## Overview

The AdaptNav demo showcases an autonomous warehouse navigation system that combines:

- **Global Path Planning**: A* algorithm for optimal route planning
- **Local Navigation**: Real-time obstacle avoidance and control
- **Sensor Simulation**: LiDAR and depth camera simulation
- **Safety Systems**: Emergency stopping and collision avoidance
- **Dynamic Environment**: Moving obstacles (workers, forklifts)

## Quick Start (3 Steps)

### Step 1: Install Dependencies
```bash
pip install numpy matplotlib
```

### Step 2: Test the Setup
```bash
python test_demo.py
```

### Step 3: Run the Demo
```bash
python run_demo.py
```

## Detailed Setup

### System Requirements

- **Python**: 3.7 or higher
- **Operating System**: Windows, macOS, or Linux
- **Memory**: At least 2GB RAM
- **Display**: Graphics display for visualization

### Installation Options

#### Option A: Minimal Installation (Recommended)
```bash
# Install only what's needed for the demo
pip install numpy matplotlib

# Run the demo
python demo_simulation.py
```

#### Option B: Full Installation
```bash
# Install all dependencies for complete functionality
pip install -r demo_requirements.txt

# Run with full features
python demo_simulation.py
```

#### Option C: Development Installation
```bash
# For developers who want to modify the code
pip install -e .
pip install -r requirements.txt

# Run tests
python -m pytest tests/
```

## Running the Demo

### Method 1: Direct Execution
```bash
python demo_simulation.py
```

### Method 2: Using the Launcher
```bash
python run_demo.py
```

### Method 3: Platform-Specific Scripts

**Windows:**
```cmd
run_demo.bat
```

**Linux/macOS:**
```bash
./run_demo.sh
```

## Understanding the Demo

### What You'll See

The demo window shows a top-down view of a warehouse with:

1. **Blue Circle**: The robot (starts at bottom-left)
2. **Green Circle**: Goal position (top-right)
3. **Gray Rectangles**: Static obstacles (warehouse shelves)
4. **Orange/Red Circles**: Dynamic obstacles (workers/forklifts)
5. **Green Dashed Line**: Planned path from robot to goal
6. **Red Dashed Circle**: Safety zone around robot
7. **Status Panel**: Real-time information display

### Demo Sequence

1. **Initialization** (0-1 seconds)
   - Robot spawns at starting position
   - A* algorithm plans initial path to goal
   - Dynamic obstacles begin moving

2. **Navigation Phase** (1-30 seconds)
   - Robot follows planned path
   - LiDAR sensor scans environment
   - Obstacles are detected and tracked
   - Robot adjusts speed based on nearby obstacles

3. **Obstacle Avoidance** (Throughout)
   - Robot slows down when obstacles are nearby
   - Emergency stop when obstacles are too close
   - Resumes motion when path is clear

4. **Goal Reaching** (Variable time)
   - Robot stops when within 0.5m of goal
   - Mission complete message displayed

### Key Behaviors to Observe

- **Path Following**: Robot follows the green dashed line
- **Obstacle Detection**: Robot reacts to moving orange/red circles
- **Safety Zones**: Robot slows down when obstacles enter red dashed circle
- **Emergency Stops**: Robot stops completely when obstacles are very close
- **Adaptive Speed**: Robot speed varies based on obstacle proximity

## Customization

### Modifying the Demo

Edit `demo_simulation.py` to customize:

#### Warehouse Layout
```python
# In setup_warehouse() method
self.warehouse_map.add_rectangle(x, y, width, height)  # Add obstacles
```

#### Robot Behavior
```python
# In simple_navigation_control() method
linear_velocity = min(1.0, min_distance * 0.5)  # Adjust speed
angular_velocity = angle_error * 2.0  # Adjust turning
```

#### Safety Parameters
```python
# In __init__() method
self.safety_zone_radius = 1.0  # Safety zone size
self.emergency_stop_distance = 0.5  # Emergency stop distance
self.max_velocity = 1.0  # Maximum robot speed
```

#### Obstacle Patterns
```python
# In setup_obstacles() method
worker = DynamicObstacle(
    position=(x, y),
    velocity=(vx, vy),
    radius=0.4,
    classification="worker"
)
```

### Advanced Customization

#### Adding New Obstacle Types
```python
class CustomObstacle(DynamicObstacle):
    def __init__(self, position, velocity, radius):
        super().__init__(position, velocity, radius, "custom")
        self.special_behavior = True
```

#### Custom Navigation Algorithms
```python
def custom_navigation_control(self):
    # Implement your own navigation logic
    return linear_velocity, angular_velocity
```

#### Enhanced Visualization
```python
# Add new visualization elements
custom_patch = plt.Circle(position, radius, color='purple')
self.ax.add_patch(custom_patch)
```

## Troubleshooting

### Common Issues

#### 1. Import Errors
```
ImportError: No module named 'numpy'
```
**Solution**: Install required packages
```bash
pip install numpy matplotlib
```

#### 2. Display Issues
```
UserWarning: Matplotlib is currently using agg, which is a non-GUI backend
```
**Solution**: Install GUI backend
```bash
pip install PyQt5  # or tkinter
```

#### 3. Performance Issues
- **Slow animation**: Increase `self.dt` in demo script
- **High CPU usage**: Reduce number of obstacles or simplify visualization
- **Memory issues**: Close other applications

#### 4. Path Planning Issues
```
Warning: Could not plan initial path
```
**Solution**: Check warehouse layout for unreachable areas

### Platform-Specific Issues

#### Windows
- Use `python` instead of `python3`
- Install Visual C++ redistributables if needed
- Use Command Prompt or PowerShell

#### macOS
- Install Xcode command line tools: `xcode-select --install`
- Use `python3` command
- May need to install matplotlib GUI backend

#### Linux
- Install system packages: `sudo apt install python3-tk`
- Use `python3` command
- Check display environment variables

### Performance Optimization

#### For Slower Systems
```python
# Reduce update frequency
self.dt = 0.2  # Instead of 0.1

# Simplify visualization
# Comment out complex drawing operations

# Reduce obstacle count
# Remove some obstacles in setup_obstacles()
```

#### For Better Visualization
```python
# Increase resolution
self.warehouse_map = WarehouseMap(resolution=0.05)

# Add more detail
# Increase number of LiDAR points
# Add more visualization elements
```

## Understanding the Code

### Architecture Overview

```
demo_simulation.py
├── DemoSimulation (main class)
│   ├── setup_warehouse()      # Create environment
│   ├── setup_robot()          # Initialize robot
│   ├── setup_obstacles()      # Add dynamic obstacles
│   ├── setup_sensors()        # Configure sensors
│   ├── setup_planner()        # Initialize A* planner
│   ├── setup_safety()         # Configure safety system
│   ├── setup_visualization()  # Create matplotlib display
│   ├── simulation_step()      # Main simulation loop
│   └── update_visualization() # Update display
```

### Key Components

#### 1. Warehouse Environment
- Static obstacle map
- Boundary walls
- Shelf layouts

#### 2. Robot Model
- Position and orientation tracking
- Velocity control
- Goal-directed navigation

#### 3. Dynamic Obstacles
- Worker and forklift models
- Scripted movement patterns
- Collision detection

#### 4. Sensor Simulation
- 360° LiDAR scanning
- Depth camera simulation
- Noise modeling

#### 5. Path Planning
- A* algorithm implementation
- Obstacle avoidance
- Waypoint following

#### 6. Safety System
- Collision zone monitoring
- Emergency stopping
- Velocity limiting

### Code Structure

```python
class DemoSimulation:
    def __init__(self):
        # Initialize all components
        
    def simulation_step(self):
        # 1. Update obstacle positions
        # 2. Simulate sensor readings
        # 3. Detect obstacles
        # 4. Plan navigation
        # 5. Apply safety control
        # 6. Update robot state
        
    def update_visualization(self):
        # Update matplotlib display
```

## Next Steps

After running the demo, you can:

### 1. Explore the Full System
- Check out the complete AdaptNav codebase in `adaptnav/`
- Run the property-based tests in `tests/`
- Try the ROS 2 integration

### 2. Modify and Experiment
- Change warehouse layouts
- Add new obstacle types
- Implement different navigation algorithms
- Create custom visualizations

### 3. Learn More
- Study the A* path planning algorithm
- Understand sensor fusion techniques
- Explore reinforcement learning for navigation
- Learn about ROS 2 robotics middleware

### 4. Contribute
- Report bugs or suggest improvements
- Add new features to the demo
- Create additional test scenarios
- Improve documentation

## Resources

### Documentation
- [Main README](README.md) - Full system documentation
- [Requirements](requirements.md) - System requirements
- [Design Document](design.md) - Architecture details

### Code Examples
- `adaptnav/` - Full system implementation
- `tests/` - Test cases and examples
- `launch/` - ROS 2 launch files

### External Resources
- [A* Algorithm](https://en.wikipedia.org/wiki/A*_search_algorithm)
- [ROS 2 Documentation](https://docs.ros.org/en/humble/)
- [Matplotlib Animation](https://matplotlib.org/stable/api/animation_api.html)

## Support

If you encounter issues:

1. **Check this guide** for common solutions
2. **Run the test script**: `python test_demo.py`
3. **Check system requirements** and dependencies
4. **Try the minimal installation** first
5. **Look at error messages** for specific guidance

The demo is designed to work on most systems with minimal setup. If you're still having trouble, the issue is likely with package installation or system configuration.