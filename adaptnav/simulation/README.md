# Simulation Module

This module provides the base simulation interface for AdaptNav warehouse environments.

## BaseSimulation Class

The `BaseSimulation` class is an abstract base class that defines the common interface for all simulation backends (MuJoCo, Isaac Sim, etc.).

### Key Features

- **Abstract Interface**: Defines required methods that all simulation backends must implement
- **ROS 2 Integration**: Built-in ROS 2 node with publishers and subscribers
- **Ground Truth Publishing**: Automatically publishes ground truth data for robot and obstacles
- **Velocity Command Handling**: Subscribes to `/cmd_vel` topic for robot control

### Abstract Methods

Subclasses must implement the following methods:

#### `step(dt: float) -> bool`
Advance the simulation by one time step.
- Apply velocity commands to robot
- Update physics simulation
- Update dynamic obstacle positions
- Detect collisions
- Publish ground truth data

#### `reset(robot_position, obstacle_configs) -> bool`
Reset the simulation to initial state.
- Reset robot to initial/specified position
- Reset obstacles to initial/specified configurations
- Clear accumulated state
- Reset physics simulation

#### `get_observation() -> Dict`
Get current observation from simulation.
- Returns sensor data (LiDAR, depth camera)
- Returns robot odometry
- Returns timestamp

#### `get_ground_truth_robot_state() -> RobotState`
Get the true robot state (position, orientation, velocities).

#### `get_ground_truth_obstacles() -> List[DynamicObstacle]`
Get the true positions and velocities of all dynamic obstacles.

#### `check_collision() -> bool`
Check if robot is currently in collision with any obstacle.

### ROS 2 Topics

#### Publishers
- `/ground_truth/robot_pose` (geometry_msgs/PoseStamped): True robot position
- `/ground_truth/obstacles` (custom_msgs/ObstacleArray): True obstacle positions/velocities

#### Subscribers
- `/cmd_vel` (geometry_msgs/Twist): Velocity commands for robot

### Usage Example

```python
from adaptnav.simulation.base_simulation import BaseSimulation
from adaptnav.core.robot_state import RobotState
from adaptnav.core.dynamic_obstacle import DynamicObstacle

class MySimulation(BaseSimulation):
    def __init__(self):
        super().__init__('my_simulation')
        # Initialize your simulation backend here
        self.set_initialized(True)
    
    def step(self, dt: float) -> bool:
        # Get current velocity command
        cmd_vel = self.get_current_cmd_vel()
        
        # Apply command to robot in your simulation
        # ... your simulation code ...
        
        # Publish ground truth data
        self.publish_ground_truth()
        
        # Increment step counter
        self.increment_step_count()
        
        return True
    
    def reset(self, robot_position=None, obstacle_configs=None) -> bool:
        # Reset your simulation
        # ... your reset code ...
        return True
    
    def get_observation(self) -> Dict:
        # Return sensor data from your simulation
        return {
            'lidar_scan': self._get_lidar_data(),
            'odometry': self._get_odometry(),
            'timestamp': self.get_clock().now().to_msg()
        }
    
    def get_ground_truth_robot_state(self) -> RobotState:
        # Return true robot state from your simulation
        return self._robot_state
    
    def get_ground_truth_obstacles(self) -> List[DynamicObstacle]:
        # Return true obstacle states from your simulation
        return self._obstacles
    
    def check_collision(self) -> bool:
        # Check for collisions in your simulation
        return self._collision_detected

# Usage
import rclpy
rclpy.init()

sim = MySimulation()

# Main simulation loop
while rclpy.ok():
    sim.step(dt=0.01)  # 100 Hz
    rclpy.spin_once(sim, timeout_sec=0)

sim.destroy_node()
rclpy.shutdown()
```

### Testing

Unit tests for the base simulation class are located in `tests/unit/test_base_simulation.py`.

Note: Tests require ROS 2 to be installed and will be skipped if ROS 2 is not available.

To run tests:
```bash
pytest tests/unit/test_base_simulation.py -v
```

### Requirements Validation

This implementation validates the following requirements:
- **Requirement 1.3**: Simulation initialization with predefined positions
- **Requirement 1.4**: Ground truth position data for all entities
- **Requirement 9.1**: Support for simulation platform integration
