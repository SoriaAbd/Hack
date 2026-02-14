# custom_msgs

Custom ROS 2 message definitions for the AdaptNav warehouse navigation system.

## Messages

### Obstacle.msg
Represents a detected dynamic obstacle (worker, forklift, etc.) in the warehouse environment.

**Fields:**
- `id` (uint32): Unique tracking ID
- `position` (geometry_msgs/Point): Position in map frame
- `velocity` (geometry_msgs/Vector3): Velocity in map frame
- `covariance` (float32[16]): Position/velocity covariance matrix
- `classification` (string): Type ("worker", "forklift", "unknown")
- `confidence` (float32): Detection confidence [0.0, 1.0]
- `last_seen` (builtin_interfaces/Time): Last detection timestamp

### ObstacleArray.msg
Array of detected obstacles with header.

**Fields:**
- `header` (std_msgs/Header): Timestamp and frame
- `obstacles` (Obstacle[]): Array of obstacles

### SafetyStatus.msg
Current safety state from the safety controller.

**Fields:**
- `header` (std_msgs/Header): Timestamp
- `state` (string): Safety state ("SAFE", "CAUTION", "EMERGENCY_STOP")
- `closest_obstacle_distance` (float32): Distance to nearest obstacle (meters)
- `velocity_scale` (float32): Velocity scaling factor [0.0, 1.0]
- `override_active` (bool): Whether safety override is active
- `time_until_clear` (float32): Time until collision zone clears (seconds)

### NavigationState.msg
Current navigation controller state for visualization.

**Fields:**
- `header` (std_msgs/Header): Timestamp
- `state` (string): Navigation state ("IDLE", "PLANNING", "FOLLOWING_PATH", etc.)
- `current_pose` (geometry_msgs/Pose): Current robot pose
- `goal_pose` (geometry_msgs/Pose): Target goal pose
- `distance_to_goal` (float32): Distance to goal (meters)
- `progress_percentage` (float32): Progress [0.0, 100.0]
- `reasoning` (string): Human-readable decision explanation

## Building

This package is built as part of the AdaptNav workspace:

```bash
cd <workspace_root>
colcon build --packages-select custom_msgs
source install/setup.bash
```

## Usage

Import in Python nodes:
```python
from custom_msgs.msg import Obstacle, ObstacleArray, SafetyStatus, NavigationState
```

Import in C++ nodes:
```cpp
#include "custom_msgs/msg/obstacle.hpp"
#include "custom_msgs/msg/obstacle_array.hpp"
#include "custom_msgs/msg/safety_status.hpp"
#include "custom_msgs/msg/navigation_state.hpp"
```

## Requirements Validation

This package validates the following requirements:
- **4.1**: Obstacle detection with position and velocity
- **4.2**: Obstacle velocity estimation
- **6.1**: Safety collision zone monitoring
- **6.2**: Safety velocity scaling
- **6.3**: Safety override mechanism
- **7.4**: Navigation state visualization
