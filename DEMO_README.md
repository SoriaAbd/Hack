# AdaptNav Demo Simulation

This demo showcases the AdaptNav warehouse navigation system with a simplified simulation that runs without requiring the full ROS 2 setup.

## What the Demo Shows

The demo demonstrates:

- **Warehouse Environment**: A 20x20 meter warehouse with static obstacles (shelves)
- **Robot Navigation**: A mobile robot navigating from start to goal position
- **Dynamic Obstacles**: Moving workers and forklifts that the robot must avoid
- **Sensor Simulation**: LiDAR and depth camera sensor simulation
- **Path Planning**: A* algorithm for global path planning
- **Obstacle Detection**: Real-time obstacle detection from sensor data
- **Safety Control**: Safety zones and emergency stopping
- **Real-time Visualization**: Live matplotlib visualization of the system

## Quick Start

### Option 1: Simple Demo (Recommended)

1. **Install minimal requirements:**
   ```bash
   pip install numpy matplotlib
   ```

2. **Run the demo:**
   ```bash
   python demo_simulation.py
   ```

### Option 2: Full System Demo

1. **Install all dependencies:**
   ```bash
   pip install -r demo_requirements.txt
   ```

2. **Set up ROS 2 (if desired):**
   ```bash
   # Ubuntu/Debian
   sudo apt install ros-humble-desktop
   source /opt/ros/humble/setup.bash
   
   # Build the workspace
   colcon build
   source install/setup.bash
   ```

3. **Run the full ROS 2 demo:**
   ```bash
   ros2 launch launch/navigation_demo.launch.py
   ```

## Demo Controls

- **Close the window** to stop the simulation
- **Press Ctrl+C** in the terminal to force stop
- The simulation runs in real-time with a 0.1 second time step

## What You'll See

The visualization shows:

- **Blue circle**: The robot with its current position
- **Green circle**: The goal position
- **Red/Orange circles**: Dynamic obstacles (forklifts/workers)
- **Gray rectangles**: Static warehouse obstacles (shelves)
- **Green dashed line**: Planned path from start to goal
- **Red dashed circle**: Safety zone around the robot
- **Status panel**: Real-time information about robot state

## Demo Behavior

1. **Initialization**: Robot starts at bottom-left, goal is at top-right
2. **Path Planning**: A* algorithm plans initial path avoiding static obstacles
3. **Navigation**: Robot follows the planned path using simple proportional control
4. **Obstacle Avoidance**: Robot detects dynamic obstacles and slows down/stops when they're too close
5. **Safety Control**: Emergency stopping when obstacles are within 0.5m
6. **Goal Reaching**: Robot stops when it reaches within 0.5m of the goal

## Customization

You can modify the demo by editing `demo_simulation.py`:

- **Warehouse size**: Change `self.warehouse_size`
- **Obstacle patterns**: Modify the `setup_obstacles()` method
- **Robot behavior**: Adjust the `simple_navigation_control()` method
- **Safety parameters**: Change `self.safety_zone_radius` and `self.emergency_stop_distance`
- **Visualization**: Modify the `setup_visualization()` method

## Architecture Overview

The demo implements a simplified version of the full AdaptNav system:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Simulation    │───▶│   Sensors        │───▶│ Obstacle        │
│   Environment   │    │ (LiDAR, Camera)  │    │ Detection       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Visualization  │◀───│ Safety Control   │◀───│ Path Planning   │
│   (Matplotlib)  │    │                  │    │ (A* Algorithm)  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure you have numpy and matplotlib installed
2. **Slow performance**: The demo runs in real-time; close other applications if needed
3. **Window doesn't appear**: Check your display settings and matplotlib backend

### Performance Tips

- The demo is designed to run smoothly on most systems
- If you experience lag, you can increase `self.dt` in the demo script
- For better performance, reduce the number of obstacles or simplify the visualization

## Next Steps

After running the demo, you can:

1. **Explore the full codebase** in the `adaptnav/` directory
2. **Run the property-based tests** to verify system correctness
3. **Set up the full ROS 2 system** for more advanced features
4. **Train your own PPO models** using the training scripts
5. **Customize the warehouse layout** and obstacle patterns

## Full System Features (Not in Demo)

The complete AdaptNav system includes additional features:

- **PPO Reinforcement Learning**: Learned local navigation policies
- **ROS 2 Integration**: Full robotics middleware support
- **MuJoCo Simulation**: High-fidelity physics simulation
- **Advanced Sensors**: Realistic sensor noise and failure models
- **Web Dashboard**: Browser-based visualization and control
- **Property-Based Testing**: Formal verification of system properties
- **Multi-Robot Support**: Coordination between multiple robots

To access these features, follow the full installation instructions in the main README.