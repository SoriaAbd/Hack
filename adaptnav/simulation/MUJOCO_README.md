# MuJoCo Simulation Backend

This directory contains the MuJoCo implementation of the AdaptNav warehouse simulation.

## Overview

The MuJoCo simulation backend provides:
- Realistic physics simulation of warehouse environment
- Differential drive robot with LiDAR and depth camera sensors
- Dynamic obstacles (workers and forklifts) with scripted motion
- Collision detection
- Ground truth data publishing via ROS 2

## Files

- `mujoco_simulation.py`: Main MuJoCo simulation class implementing BaseSimulation interface
- `models/warehouse.xml`: MJCF model file defining the warehouse environment
- `run_mujoco_sim.py`: ROS 2 node script to run the simulation

## Requirements

### Python Dependencies
```bash
pip install mujoco>=3.0.0
pip install numpy
```

### ROS 2 Dependencies
- ROS 2 Humble or later
- rclpy
- geometry_msgs
- nav_msgs
- sensor_msgs
- custom_msgs (from this package)

## Warehouse Environment

The warehouse model (`models/warehouse.xml`) includes:

### Static Obstacles
- **Walls**: 50m x 50m warehouse boundary
- **Shelving Units**: 12 shelves arranged in 4 rows of 3
- **Structural Columns**: 4 support columns

### Dynamic Obstacles
- **Worker 1**: Circular motion pattern, 0.5 m/s, 0.4m radius
- **Worker 2**: Circular motion pattern (opposite direction), 0.7 m/s, 0.4m radius
- **Forklift 1**: Back-and-forth motion, 1.5 m/s, 1.2m radius

### Robot
- **Type**: Differential drive mobile robot
- **Dimensions**: 0.3m radius cylinder, 0.15m height
- **Mass**: 20 kg
- **Sensors**:
  - LiDAR: 360° FOV, 1° resolution, 0.1-10m range
  - Depth Camera: 90° FOV, 640x480 resolution, 0.5-5m range

## Usage

### Running the Simulation

#### As a ROS 2 Node
```bash
# Source ROS 2 workspace
source /opt/ros/humble/setup.bash
source install/setup.bash

# Run simulation node
ros2 run adaptnav run_mujoco_sim.py
```

#### Using Launch File
```bash
# Headless mode (no visualization)
ros2 launch adaptnav mujoco_simulation.launch.py

# With viewer (requires display)
ros2 launch adaptnav mujoco_simulation.launch.py render_mode:=human

# With custom model
ros2 launch adaptnav mujoco_simulation.launch.py model_path:=/path/to/custom/warehouse.xml
```

### Programmatic Usage

```python
import rclpy
from adaptnav.simulation.mujoco_simulation import MuJoCoSimulation

# Initialize ROS 2
rclpy.init()

# Create simulation
sim = MuJoCoSimulation(render_mode=None)

# Reset to initial state
sim.reset()

# Simulation loop
dt = 0.01  # 100 Hz
while rclpy.ok():
    # Step simulation
    sim.step(dt)
    
    # Get observations
    obs = sim.get_observation()
    lidar_scan = obs['lidar_scan']
    odometry = obs['odometry']
    
    # Get ground truth (for debugging/training)
    robot_state = sim.get_ground_truth_robot_state()
    obstacles = sim.get_ground_truth_obstacles()
    
    # Check for collisions
    if sim.check_collision():
        print("Collision detected!")
    
    # Process ROS callbacks
    rclpy.spin_once(sim, timeout_sec=0)

# Cleanup
sim.destroy_node()
rclpy.shutdown()
```

## ROS 2 Topics

### Published Topics

#### Ground Truth Data
- `/ground_truth/robot_pose` (geometry_msgs/PoseStamped): True robot position and orientation
- `/ground_truth/obstacles` (custom_msgs/ObstacleArray): True obstacle positions and velocities

#### Sensor Data
- `/scan` (sensor_msgs/LaserScan): LiDAR scan data at 20 Hz
- `/odom` (nav_msgs/Odometry): Noisy odometry estimate at 20 Hz
- `/camera/depth/image_raw` (sensor_msgs/Image): Depth camera image at 15 Hz (future)
- `/camera/camera_info` (sensor_msgs/CameraInfo): Camera calibration (future)

### Subscribed Topics
- `/cmd_vel` (geometry_msgs/Twist): Velocity commands for robot control

## Simulation Features

### Differential Drive Control
The robot uses a simplified differential drive model:
- Linear velocity command (m/s) in robot frame
- Angular velocity command (rad/s) around vertical axis
- Forces applied to achieve desired velocities using proportional control

### Scripted Obstacle Motion
Dynamic obstacles follow predefined motion patterns:
- **Workers**: Circular paths around fixed centers
- **Forklift**: Linear back-and-forth motion along a path

Motion parameters can be customized by modifying the `MuJoCoSimulation` class.

### Sensor Simulation

#### LiDAR
- 360° coverage with 1° angular resolution (360 rays)
- Range: 0.1m to 10m
- Gaussian noise: σ = 0.01m
- Ray casting against obstacle bounding spheres
- Published at 20 Hz

#### Odometry
- Based on true robot state with added noise
- Position noise: σ = 0.01m (1cm)
- Orientation noise: σ = 0.01 rad (~0.5°)
- Simulates wheel encoder drift
- Published at 20 Hz

### Collision Detection
- Uses MuJoCo's built-in contact detection
- Checks for contacts between robot and dynamic obstacles
- Accessible via `check_collision()` method
- Does not automatically stop the robot (handled by safety controller)

## Customization

### Creating Custom Warehouse Models

You can create custom MJCF models with different layouts:

```xml
<mujoco model="custom_warehouse">
  <worldbody>
    <!-- Add your custom obstacles here -->
    <body name="custom_shelf" pos="10 5 0.75">
      <geom type="box" size="2 0.5 0.75" material="shelf"/>
    </body>
    
    <!-- Robot must be named "robot" -->
    <body name="robot" pos="0 0 0.15">
      <freejoint/>
      <geom name="robot_base" type="cylinder" size="0.3 0.15" mass="20"/>
    </body>
  </worldbody>
</mujoco>
```

Load custom model:
```python
sim = MuJoCoSimulation(model_path='/path/to/custom_warehouse.xml')
```

### Modifying Obstacle Motion

Edit the `_update_obstacle_motion()` method in `mujoco_simulation.py`:

```python
def _update_obstacle_motion(self, dt: float) -> None:
    # Custom motion pattern for worker 1
    # Example: Linear motion
    velocity = np.array([1.0, 0.0])  # 1 m/s in x direction
    current_pos = self._get_body_position(self._worker_1_body_id)
    new_pos = current_pos + velocity * dt
    self._set_body_pose(self._worker_1_body_id, new_pos[0], new_pos[1], 0.85, 0.0)
    self._set_body_velocity(self._worker_1_body_id, velocity[0], velocity[1], 0.0)
```

## Testing

Unit tests are located in `tests/unit/test_mujoco_simulation.py`.

Run tests:
```bash
pytest tests/unit/test_mujoco_simulation.py -v
```

Tests cover:
- Initialization
- Reset with default and custom positions
- Simulation stepping
- Ground truth data retrieval
- Observation retrieval
- Collision detection
- Obstacle motion
- Velocity command application
- Sensor properties

## Performance

Typical performance on modern hardware:
- Simulation rate: 100 Hz (real-time)
- Sensor publishing: 20 Hz (LiDAR, odometry)
- CPU usage: ~10-20% (single core)
- Memory usage: ~200 MB

## Troubleshooting

### MuJoCo Not Found
```
ImportError: MuJoCo is not installed
```
**Solution**: Install MuJoCo: `pip install mujoco>=3.0.0`

### Model File Not Found
```
FileNotFoundError: warehouse.xml not found
```
**Solution**: Ensure you're running from the workspace root, or provide full path to model file.

### Viewer Fails to Launch
```
Could not initialize viewer
```
**Solution**: Viewer requires a display. Use headless mode (`render_mode=None`) on servers.

### Simulation Runs Slowly
**Solution**: 
- Reduce simulation frequency (e.g., 50 Hz instead of 100 Hz)
- Disable viewer if not needed
- Check CPU usage and close other applications

## Requirements Validation

This implementation validates the following requirements:
- **Requirement 1.1**: Warehouse environment with static obstacles (walls, shelves, columns)
- **Requirement 1.2**: Dynamic obstacles (workers and forklifts)
- **Requirement 1.5**: Realistic physics with collision detection
- **Requirement 9.1**: Simulation platform integration (MuJoCo backend)
- **Requirement 9.3**: MJCF model files for warehouse environment

## Future Enhancements

Planned improvements:
- [ ] Depth camera image generation
- [ ] RGB camera simulation
- [ ] More complex obstacle motion patterns
- [ ] Configurable warehouse layouts via YAML
- [ ] Multi-robot support
- [ ] Improved differential drive dynamics
- [ ] GPU-accelerated rendering

## References

- [MuJoCo Documentation](https://mujoco.readthedocs.io/)
- [MJCF XML Reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html)
- [ROS 2 Documentation](https://docs.ros.org/)
