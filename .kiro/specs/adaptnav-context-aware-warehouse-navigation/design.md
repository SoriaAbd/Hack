# Design Document: AdaptNav Context-Aware Warehouse Navigation

## Overview

AdaptNav is a context-aware autonomous navigation system for warehouse environments that combines traditional path planning with reinforcement learning for robust, safe navigation around dynamic obstacles. The system is designed for robotics hackathons and educational purposes, demonstrating a realistic robotics stack that bridges simulation and real-world deployment.

### Key Design Principles

1. **Safety-First Architecture**: Multi-layered safety system with hard constraints that override learned behaviors
2. **Hybrid Navigation**: Combines global path planning (A*) with local RL-based obstacle avoidance (PPO)
3. **Explainability**: Real-time visualization of decision-making process and internal state
4. **Sim-to-Real Ready**: Uses standard ROS 2 interfaces and realistic sensor simulation for hardware transfer
5. **Modular Design**: Loosely coupled components communicating via ROS 2 topics/services

### System Architecture

The system follows a layered architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                    Dashboard (Web/RViz)                      │
│              Visualization & Explainability Layer            │
└─────────────────────────────────────────────────────────────┘
                              ↑
                              │ (ROS 2 Topics)
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   Navigation Controller                      │
│         (Orchestrates planning, RL, and safety)              │
└─────────────────────────────────────────────────────────────┘
         ↑                    ↑                    ↑
         │                    │                    │
    ┌────┴────┐         ┌────┴────┐         ┌────┴────┐
    │ Global  │         │   PPO   │         │ Safety  │
    │ Planner │         │  Agent  │         │Controller│
    │  (A*)   │         │  (RL)   │         │         │
    └─────────┘         └─────────┘         └─────────┘
         ↑                    ↑                    ↑
         │                    │                    │
         └────────────────────┴────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │  Obstacle Detector │
                    │  (Sensor Fusion)   │
                    └─────────┬─────────┘
                              │
                    ┌─────────┴─────────┐
                    │   Sensor Layer    │
                    │ (LiDAR + Depth)   │
                    └─────────┬─────────┘
                              │
                    ┌─────────┴─────────┐
                    │ Simulation Engine │
                    │ (Isaac/MuJoCo)    │
                    └───────────────────┘
```

## Architecture

### Component Overview

The system consists of the following major components:

1. **Simulation Engine**: Physics-based warehouse environment (Isaac Sim or MuJoCo)
2. **Sensor Layer**: Simulated LiDAR and depth camera with realistic noise models
3. **Obstacle Detector**: Sensor fusion and dynamic obstacle tracking
4. **Global Planner**: A* path planning for static environment navigation
5. **PPO Agent**: Reinforcement learning policy for local navigation and obstacle avoidance
6. **Safety Controller**: Hard safety constraints and emergency stop logic
7. **Navigation Controller**: High-level orchestration and decision state machine
8. **Dashboard**: Real-time visualization and explainability interface

### Technology Stack

**Core Framework**:
- ROS 2 Humble (LTS release)
- Python 3.10+ for main navigation logic
- C++ for performance-critical components (optional)

**Simulation**:
- Primary: Isaac Sim 2023.1+ (NVIDIA Omniverse)
- Alternative: MuJoCo 3.0+ with dm_control
- ROS 2 bridge for simulator integration

**Machine Learning**:
- Stable Baselines3 3.0+ for PPO implementation
- PyTorch 2.0+ as ML backend
- Gymnasium (formerly OpenAI Gym) for RL environment interface

**Path Planning**:
- Python implementation of A* algorithm
- NumPy for grid-based representations
- SciPy for spatial operations

**Visualization**:
- RViz2 for ROS-native visualization
- Web dashboard: React + Three.js for 3D visualization
- WebSocket bridge for real-time data streaming

**Data Processing**:
- NumPy for numerical operations
- Open3D for point cloud processing
- OpenCV for image processing

### Component Interfaces

#### 1. Simulation Engine

**Responsibilities**:
- Simulate warehouse environment physics
- Provide ground truth state information
- Simulate sensor data with realistic noise
- Detect collisions and enforce physics constraints

**Interfaces**:
- **Outputs** (ROS 2 Topics):
  - `/ground_truth/robot_pose` (geometry_msgs/PoseStamped): True robot position
  - `/ground_truth/obstacles` (custom_msgs/ObstacleArray): True obstacle positions/velocities
  - `/scan` (sensor_msgs/LaserScan): Simulated LiDAR data
  - `/camera/depth/image_raw` (sensor_msgs/Image): Simulated depth image
  - `/camera/rgb/image_raw` (sensor_msgs/Image): Simulated RGB image
  - `/odom` (nav_msgs/Odometry): Noisy odometry estimate

- **Inputs** (ROS 2 Topics):
  - `/cmd_vel` (geometry_msgs/Twist): Velocity commands for robot

- **Configuration**:
  - Warehouse layout file (USD for Isaac, MJCF for MuJoCo)
  - Sensor noise parameters
  - Physics parameters (friction, mass, etc.)

#### 2. Sensor Layer

**Responsibilities**:
- Publish sensor data from simulation
- Apply realistic noise models
- Maintain sensor timing and synchronization

**Interfaces**:
- **Outputs** (ROS 2 Topics):
  - `/scan` (sensor_msgs/LaserScan): LiDAR point cloud at 20 Hz
  - `/camera/depth/image_raw` (sensor_msgs/Image): Depth image at 15 Hz
  - `/camera/rgb/image_raw` (sensor_msgs/Image): RGB image at 15 Hz
  - `/camera/camera_info` (sensor_msgs/CameraInfo): Camera calibration

- **Configuration**:
  - LiDAR: 360° FOV, 0.1-10m range, 1° angular resolution
  - Depth camera: 90° FOV, 0.5-5m range, 640x480 resolution

#### 3. Obstacle Detector

**Responsibilities**:
- Fuse LiDAR and depth camera data
- Detect and classify dynamic obstacles
- Track obstacle positions and velocities
- Estimate future obstacle trajectories

**Interfaces**:
- **Inputs** (ROS 2 Topics):
  - `/scan` (sensor_msgs/LaserScan)
  - `/camera/depth/image_raw` (sensor_msgs/Image)
  - `/odom` (nav_msgs/Odometry)

- **Outputs** (ROS 2 Topics):
  - `/obstacles/detected` (custom_msgs/ObstacleArray): Detected obstacles with positions, velocities, classifications
  - `/obstacles/visualization` (visualization_msgs/MarkerArray): Visualization markers

- **Algorithm**:
  - Clustering: DBSCAN on fused point cloud
  - Tracking: Kalman filter for position/velocity estimation
  - Classification: Simple heuristic based on size and velocity

**Data Structures**:
```python
class DetectedObstacle:
    id: int                    # Unique tracking ID
    position: np.ndarray       # [x, y] in world frame
    velocity: np.ndarray       # [vx, vy] in world frame
    covariance: np.ndarray     # 4x4 position/velocity covariance
    classification: str        # "worker", "forklift", "unknown"
    last_seen: float           # Timestamp
    confidence: float          # Detection confidence [0, 1]
```

#### 4. Global Planner

**Responsibilities**:
- Compute collision-free paths through static environment
- Replan when paths become blocked
- Provide waypoints for local navigation

**Interfaces**:
- **Inputs** (ROS 2 Services):
  - `/plan_path` (nav2_msgs/ComputePathToPose): Request path from start to goal

- **Inputs** (ROS 2 Topics):
  - `/map` (nav_msgs/OccupancyGrid): Static obstacle map

- **Outputs** (ROS 2 Topics):
  - `/global_plan` (nav_msgs/Path): Planned path as sequence of waypoints

- **Algorithm**:
  - A* search on occupancy grid
  - Grid resolution: 0.1m
  - Path smoothing with cubic splines
  - Timeout: 2 seconds

**Data Structures**:
```python
class GlobalPath:
    waypoints: List[Waypoint]  # Sequence of waypoints
    total_length: float        # Path length in meters
    computation_time: float    # Planning time in seconds
    valid: bool                # Whether path is collision-free
```

#### 5. PPO Agent

**Responsibilities**:
- Learn local navigation policy through reinforcement learning
- Output velocity commands based on observations
- Handle dynamic obstacle avoidance

**Interfaces**:
- **Inputs** (Observation Space):
  - LiDAR scan: 360 values (1° resolution, normalized to [0, 1])
  - Goal direction: 2 values (relative x, y to next waypoint)
  - Current velocity: 2 values (linear, angular)
  - Obstacle proximity: 8 values (distance to nearest obstacle in 8 sectors)
  - Total: 372-dimensional observation vector

- **Outputs** (Action Space):
  - Linear velocity: continuous [-1.0, 1.0] m/s
  - Angular velocity: continuous [-0.5, 0.5] rad/s

- **Reward Function**:
  ```python
  reward = (
      + 1.0 * progress_toward_goal      # Encourage forward progress
      - 10.0 * collision_penalty        # Strong penalty for collisions
      - 0.1 * distance_to_goal          # Encourage reaching goal
      - 0.01 * action_magnitude         # Encourage smooth control
      + 100.0 * goal_reached_bonus      # Large bonus for success
  )
  ```

- **Network Architecture**:
  - Actor: [372] → [256] → [128] → [2] (ReLU activations)
  - Critic: [372] → [256] → [128] → [1] (ReLU activations)
  - Shared feature extractor for efficiency

**Training Configuration**:
- Algorithm: PPO with clipped objective
- Learning rate: 3e-4 with linear decay
- Batch size: 2048 steps
- Epochs per update: 10
- Discount factor (γ): 0.99
- GAE lambda (λ): 0.95
- Clip range: 0.2
- Total timesteps: 1M-5M depending on complexity

#### 6. Safety Controller

**Responsibilities**:
- Enforce hard safety constraints
- Override unsafe commands from PPO agent
- Implement emergency stop logic
- Maintain collision-free buffer zone

**Interfaces**:
- **Inputs** (ROS 2 Topics):
  - `/obstacles/detected` (custom_msgs/ObstacleArray)
  - `/cmd_vel_raw` (geometry_msgs/Twist): Unfiltered commands from PPO
  - `/odom` (nav_msgs/Odometry)

- **Outputs** (ROS 2 Topics):
  - `/cmd_vel` (geometry_msgs/Twist): Safety-filtered velocity commands
  - `/safety_status` (custom_msgs/SafetyStatus): Current safety state

**Safety Rules**:
1. **Collision Zone**: Maintain 0.5m buffer around robot
2. **Velocity Limits**: Max 1.0 m/s linear, 0.5 rad/s angular
3. **Emergency Stop**: Stop if obstacle within 0.3m
4. **Clearance Time**: Wait 1 second after obstacle clears before resuming
5. **Deceleration**: Smooth deceleration when approaching obstacles

**Data Structures**:
```python
class SafetyStatus:
    state: str                 # "SAFE", "CAUTION", "EMERGENCY_STOP"
    closest_obstacle_dist: float
    velocity_scale: float      # [0, 1] multiplier for commanded velocity
    override_active: bool      # Whether safety override is active
    time_until_clear: float    # Estimated time until safe to proceed
```

#### 7. Navigation Controller

**Responsibilities**:
- Orchestrate global planner, PPO agent, and safety controller
- Implement high-level decision state machine
- Handle goal management and replanning
- Provide status updates for dashboard

**Interfaces**:
- **Inputs** (ROS 2 Actions):
  - `/navigate_to_pose` (nav2_msgs/NavigateToPose): Goal pose action server

- **Inputs** (ROS 2 Topics):
  - `/global_plan` (nav_msgs/Path)
  - `/obstacles/detected` (custom_msgs/ObstacleArray)
  - `/safety_status` (custom_msgs/SafetyStatus)
  - `/odom` (nav_msgs/Odometry)

- **Outputs** (ROS 2 Topics):
  - `/cmd_vel_raw` (geometry_msgs/Twist): Commands to safety controller
  - `/navigation_state` (custom_msgs/NavigationState): Current state for dashboard

**State Machine**:
```
IDLE → PLANNING → FOLLOWING_PATH → AVOIDING_OBSTACLE → GOAL_REACHED
  ↑                    ↓                    ↓
  └────────────────────┴────────────────────┘
           (replan or abort)
```

**Decision Logic**:
- **PLANNING**: Request path from global planner
- **FOLLOWING_PATH**: Use PPO agent to follow waypoints
- **AVOIDING_OBSTACLE**: PPO agent handles local avoidance
- **EMERGENCY_STOP**: Safety controller has taken over
- **GOAL_REACHED**: Within 0.2m of goal with velocity < 0.1 m/s

#### 8. Dashboard

**Responsibilities**:
- Visualize robot state and environment
- Display decision-making reasoning
- Provide controls for starting/stopping navigation
- Show performance metrics

**Interfaces**:
- **Inputs** (ROS 2 Topics):
  - `/odom` (nav_msgs/Odometry)
  - `/global_plan` (nav_msgs/Path)
  - `/obstacles/detected` (custom_msgs/ObstacleArray)
  - `/navigation_state` (custom_msgs/NavigationState)
  - `/safety_status` (custom_msgs/SafetyStatus)
  - `/scan` (sensor_msgs/LaserScan)
  - `/camera/rgb/image_raw` (sensor_msgs/Image)

- **Outputs** (ROS 2 Services):
  - `/start_navigation` (std_srvs/Trigger)
  - `/stop_navigation` (std_srvs/Trigger)
  - `/reset_simulation` (std_srvs/Trigger)

**Visualization Elements**:
- 3D warehouse environment with robot and obstacles
- LiDAR point cloud overlay
- Planned path visualization
- Obstacle velocity vectors
- Safety zone visualization (color-coded by status)
- Decision state indicator
- Performance metrics (success rate, avg time to goal)
- "Reasoning panel" showing current PPO observation values

## Components and Interfaces

### ROS 2 Message Definitions

**custom_msgs/Obstacle.msg**:
```
uint32 id
geometry_msgs/Point position
geometry_msgs/Vector3 velocity
float32[16] covariance
string classification
float32 confidence
builtin_interfaces/Time last_seen
```

**custom_msgs/ObstacleArray.msg**:
```
std_msgs/Header header
custom_msgs/Obstacle[] obstacles
```

**custom_msgs/SafetyStatus.msg**:
```
std_msgs/Header header
string state
float32 closest_obstacle_distance
float32 velocity_scale
bool override_active
float32 time_until_clear
```

**custom_msgs/NavigationState.msg**:
```
std_msgs/Header header
string state
geometry_msgs/Pose current_pose
geometry_msgs/Pose goal_pose
float32 distance_to_goal
float32 progress_percentage
string reasoning
```

### Coordinate Frames (tf2)

- **map**: Fixed world frame, origin at warehouse corner
- **odom**: Odometry frame, drifts over time
- **base_link**: Robot center frame
- **laser**: LiDAR sensor frame
- **camera_link**: Depth camera frame

Transforms published by:
- `map → odom`: Localization node (or ground truth in simulation)
- `odom → base_link`: Odometry from simulation
- `base_link → laser`: Static transform
- `base_link → camera_link`: Static transform

## Data Models

### Warehouse Environment Model

**Static Map Representation**:
```python
class WarehouseMap:
    width: float               # Width in meters
    height: float              # Height in meters
    resolution: float          # Grid cell size (0.1m)
    occupancy_grid: np.ndarray # 2D array: 0=free, 100=occupied, -1=unknown
    origin: Tuple[float, float] # Map origin in world coordinates
    
    def is_collision_free(self, x: float, y: float, radius: float) -> bool:
        """Check if circular robot at (x,y) collides with static obstacles"""
        pass
    
    def get_neighbors(self, x: int, y: int) -> List[Tuple[int, int]]:
        """Get valid neighboring grid cells for path planning"""
        pass
```

**Dynamic Obstacle Model**:
```python
class DynamicObstacle:
    id: int
    position: np.ndarray       # [x, y] in map frame
    velocity: np.ndarray       # [vx, vy] in map frame
    radius: float              # Bounding circle radius
    classification: str        # "worker", "forklift", "unknown"
    
    def predict_position(self, dt: float) -> np.ndarray:
        """Predict position after dt seconds using constant velocity model"""
        return self.position + self.velocity * dt
    
    def distance_to(self, point: np.ndarray) -> float:
        """Compute distance from obstacle center to point"""
        return np.linalg.norm(self.position - point)
```

### Robot State Model

```python
class RobotState:
    position: np.ndarray       # [x, y] in map frame
    orientation: float         # Theta in radians
    linear_velocity: float     # m/s
    angular_velocity: float    # rad/s
    timestamp: float           # ROS time
    
    def to_pose_stamped(self) -> PoseStamped:
        """Convert to ROS PoseStamped message"""
        pass
    
    def distance_to_goal(self, goal: np.ndarray) -> float:
        """Compute Euclidean distance to goal position"""
        return np.linalg.norm(self.position - goal)
```

### PPO Observation Model

```python
class PPOObservation:
    lidar_scan: np.ndarray     # Shape: (360,), normalized distances
    goal_direction: np.ndarray # Shape: (2,), relative [x, y] to goal
    current_velocity: np.ndarray # Shape: (2,), [linear, angular]
    obstacle_proximity: np.ndarray # Shape: (8,), min distance in 8 sectors
    
    def to_vector(self) -> np.ndarray:
        """Flatten to 372-dimensional vector for neural network"""
        return np.concatenate([
            self.lidar_scan,
            self.goal_direction,
            self.current_velocity,
            self.obstacle_proximity
        ])
    
    @classmethod
    def from_ros_messages(cls, scan: LaserScan, odom: Odometry, 
                          obstacles: ObstacleArray, goal: Pose) -> 'PPOObservation':
        """Construct observation from ROS messages"""
        pass
```

### Path Representation

```python
class Waypoint:
    x: float
    y: float
    theta: float               # Desired orientation at waypoint
    
class Path:
    waypoints: List[Waypoint]
    total_length: float
    timestamp: float
    
    def get_closest_waypoint(self, position: np.ndarray) -> Tuple[int, Waypoint]:
        """Find closest waypoint to current position"""
        pass
    
    def get_lookahead_point(self, position: np.ndarray, 
                           lookahead_dist: float) -> Waypoint:
        """Get point on path at lookahead distance ahead"""
        pass
```

### Training Episode Data

```python
class TrainingEpisode:
    episode_id: int
    start_position: np.ndarray
    goal_position: np.ndarray
    obstacle_configuration: List[DynamicObstacle]
    success: bool
    collision: bool
    steps: int
    total_reward: float
    path_length: float
    time_to_goal: float
    
    def compute_metrics(self) -> Dict[str, float]:
        """Compute performance metrics for episode"""
        return {
            "success_rate": float(self.success),
            "collision_rate": float(self.collision),
            "path_efficiency": self.path_length / self.optimal_path_length,
            "time_efficiency": self.time_to_goal / self.optimal_time
        }
```


## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Environment and Simulation Properties

**Property 1: Initialization Consistency**
*For any* valid warehouse configuration, when the simulation starts, all entities (robot and obstacles) should be positioned at their specified initial positions within a tolerance of 0.01m.
**Validates: Requirements 1.3**

**Property 2: Ground Truth Availability**
*For any* entity in the simulation at any time, querying its ground truth position should return a valid position within the warehouse bounds.
**Validates: Requirements 1.4**

**Property 3: Collision Detection**
*For any* two objects with overlapping bounding volumes, the physics engine should detect and report a collision.
**Validates: Requirements 1.5**

**Property 4: Sensor Publishing Frequency**
*For any* sensor data stream over a 10-second window, the average publishing frequency should be within the specified range (10-30 Hz for LiDAR, 10-20 Hz for depth camera).
**Validates: Requirements 2.3**

**Property 5: LiDAR Range Limitations**
*For any* LiDAR scan, all range readings should be either within the valid range [0.1m, 10m] or marked as invalid/infinite.
**Validates: Requirements 2.4**

**Property 6: Depth Camera Field of View**
*For any* object outside the depth camera's field of view (90° horizontal), that object should not appear in the depth image.
**Validates: Requirements 2.5**

### Path Planning Properties

**Property 7: Collision-Free Paths**
*For any* valid start and goal positions in the warehouse, if the path planner returns a path, then every waypoint on that path should be collision-free with respect to static obstacles (minimum 0.3m clearance).
**Validates: Requirements 3.1**

**Property 8: Planning Failure Handling**
*For any* planning request where the goal is inside a static obstacle or completely unreachable, the path planner should return a failure status (not a path).
**Validates: Requirements 3.3**

**Property 9: Replanning Trigger**
*For any* current path that becomes blocked by a dynamic obstacle (obstacle within 0.5m of any waypoint), the navigation system should trigger replanning within 1 second.
**Validates: Requirements 3.4**

**Property 10: Planning Performance**
*For any* valid planning request in a warehouse up to 50m x 50m, the path planner should return a result (success or failure) within 2 seconds.
**Validates: Requirements 3.5**

### Obstacle Detection Properties

**Property 11: Detection Completeness**
*For any* detected dynamic obstacle, the obstacle detector output should include both position estimate and velocity estimate (not null/missing values).
**Validates: Requirements 4.1, 4.2**

**Property 12: Sensor Fusion Accuracy**
*For any* obstacle visible to both LiDAR and depth camera, the fused position estimate should have lower error than either single-sensor estimate alone (when compared to ground truth).
**Validates: Requirements 4.3**

**Property 13: Detection Latency**
*For any* obstacle entering the sensor range, the obstacle detector should publish a detection within 0.5 seconds of the obstacle becoming visible.
**Validates: Requirements 4.4**

**Property 14: Tracking Consistency**
*For any* obstacle detected in consecutive frames (frame N and N+1) where the position change is less than 2 meters, the tracking ID should remain the same.
**Validates: Requirements 4.5**

### PPO Agent Properties

**Property 15: Observation Structure**
*For any* observation passed to the PPO agent, it should contain all required fields: lidar_scan (360 values), goal_direction (2 values), current_velocity (2 values), and obstacle_proximity (8 values).
**Validates: Requirements 5.1**

**Property 16: Action Output Structure**
*For any* observation input to the PPO agent, the output action should contain exactly two values: linear velocity and angular velocity.
**Validates: Requirements 5.2**

**Property 17: Training Improvement**
*For any* training run of at least 500k timesteps, the success rate in the final 100 evaluation episodes should be at least 20 percentage points higher than the success rate in the first 100 evaluation episodes.
**Validates: Requirements 5.3**

**Property 18: Hybrid Navigation Integration**
*For any* navigation episode, the system should use both path planner waypoints (for global direction) and PPO agent outputs (for local control), not exclusively one or the other.
**Validates: Requirements 5.5**

### Safety Properties

**Property 19: Collision Zone Enforcement**
*For any* robot position and any obstacle, if the obstacle is within 0.5m of the robot center, the safety controller should set the safety state to at least "CAUTION" (not "SAFE").
**Validates: Requirements 6.1**

**Property 20: Emergency Stop Response**
*For any* obstacle entering the collision zone (within 0.5m), the safety controller should reduce the commanded velocity to at most 50% of the original command within one control cycle (0.1s).
**Validates: Requirements 6.2**

**Property 21: Safety Override**
*For any* PPO agent command that would result in collision within 0.5 seconds (based on current velocity and obstacle positions), the safety controller should override the command with a safer alternative.
**Validates: Requirements 6.3**

**Property 22: Clearance Time Enforcement**
*For any* safety stop event, the robot should remain at zero velocity until the collision zone has been clear of obstacles for at least 1 second.
**Validates: Requirements 6.4**

**Property 23: Velocity Limit Enforcement**
*For any* velocity command (from PPO or any source), the safety controller output should be clamped to maximum limits: 1.0 m/s linear and 0.5 rad/s angular.
**Validates: Requirements 6.5**

### Dashboard and Visualization Properties

**Property 24: Obstacle Visualization Completeness**
*For any* detected obstacle in the obstacle array, the dashboard should render a corresponding visualization marker with position and velocity vector.
**Validates: Requirements 7.2**

**Property 25: State Display**
*For any* navigation state update, the dashboard should display the current state string (e.g., "FOLLOWING_PATH", "AVOIDING_OBSTACLE", "EMERGENCY_STOP").
**Validates: Requirements 7.4**

**Property 26: Dashboard Update Frequency**
*For any* 10-second observation window, the dashboard should update its visualizations at least 50 times (5 Hz minimum).
**Validates: Requirements 7.6**

**Property 27: Explainability Display**
*For any* PPO agent decision, the dashboard should display the observation values that were input to the agent (goal direction, obstacle proximity, etc.).
**Validates: Requirements 7.7**

### Performance Properties

**Property 28: Control Loop Frequency**
*For any* 10-second navigation period, the navigation system should publish velocity commands at least 100 times (10 Hz minimum).
**Validates: Requirements 11.1**

**Property 29: Obstacle Detection Latency**
*For any* sensor data frame, the obstacle detector should publish detection results within 100ms of receiving the sensor data.
**Validates: Requirements 11.2**

**Property 30: PPO Inference Latency**
*For any* observation input, the PPO agent should compute and return an action within 50ms.
**Validates: Requirements 11.3**

**Property 31: Safety Evaluation Latency**
*For any* velocity command input, the safety controller should evaluate constraints and output a filtered command within 20ms.
**Validates: Requirements 11.4**

**Property 32: Real-Time Performance**
*For any* 60-second navigation run, the system should maintain the target control frequency (10 Hz) with less than 5% frame drops.
**Validates: Requirements 11.5**

### Training and Evaluation Properties

**Property 33: Scenario Diversity**
*For any* training batch of 100 episodes, there should be at least 20 distinct obstacle configurations (measured by obstacle positions and velocities).
**Validates: Requirements 10.2**

**Property 34: Training Metrics Logging**
*For any* training episode, the system should log at minimum: success (bool), collision (bool), and path_length (float).
**Validates: Requirements 10.3**

**Property 35: Model Persistence Round-Trip**
*For any* trained PPO agent model, saving the model to disk and then loading it should result in identical action outputs for the same observation inputs (within floating-point tolerance of 1e-6).
**Validates: Requirements 10.5**

### System Integration Properties

**Property 36: Simulation Real-Time Performance**
*For any* 60-second simulation run, the simulation time should not exceed 90 seconds of wall-clock time (1.5x real-time maximum).
**Validates: Requirements 9.4**

**Property 37: Demonstration Success Rate**
*For any* demonstration scenario run 20 times, the navigation system should successfully reach the goal in at least 16 runs (80% success rate).
**Validates: Requirements 12.2**

## Error Handling

### Error Categories and Responses

**1. Planning Failures**
- **Cause**: No valid path exists (goal unreachable, start/goal in obstacle)
- **Detection**: Path planner returns failure status
- **Response**: 
  - Publish failure status to navigation state topic
  - Log diagnostic information (reason for failure)
  - Notify user via dashboard
  - Wait for new goal or environment change
- **Recovery**: User provides new goal or obstacles are moved

**2. Sensor Failures**
- **Cause**: Sensor data not received, invalid data, or excessive noise
- **Detection**: Timeout on sensor topics (>1 second), data validation checks
- **Response**:
  - Log warning message
  - Use last known valid sensor data for up to 2 seconds
  - If timeout exceeds 2 seconds, trigger emergency stop
  - Notify user via dashboard
- **Recovery**: Restart sensor nodes or simulation

**3. Obstacle Detection Failures**
- **Cause**: Detector crashes, excessive processing time, or invalid detections
- **Detection**: Timeout on detection topic (>200ms), invalid obstacle data
- **Response**:
  - Log error message
  - Assume worst-case: obstacles everywhere in sensor range
  - Trigger safety stop
  - Attempt to restart detector node
- **Recovery**: Manual restart or system reset

**4. PPO Agent Failures**
- **Cause**: Model not loaded, inference timeout, or invalid actions
- **Detection**: Action computation timeout (>100ms), NaN/Inf in actions
- **Response**:
  - Log error message
  - Fall back to pure path-following behavior (no RL)
  - Clamp any invalid actions to zero velocity
  - Continue operation in degraded mode
- **Recovery**: Reload model or restart agent node

**5. Safety Violations**
- **Cause**: Collision detected, obstacle too close, or velocity limits exceeded
- **Detection**: Safety controller detects constraint violation
- **Response**:
  - Immediate emergency stop (zero velocity)
  - Log safety event with details
  - Update dashboard with emergency status
  - Wait for clearance before resuming
- **Recovery**: Automatic resume after clearance time (1 second)

**6. Communication Failures**
- **Cause**: ROS 2 node crashes, network issues, or message queue overflow
- **Detection**: Topic timeout, service call failures
- **Response**:
  - Log communication error
  - Attempt to reconnect for up to 5 seconds
  - If reconnection fails, trigger emergency stop
  - Notify user via dashboard (if dashboard still connected)
- **Recovery**: Restart affected nodes or entire system

**7. Simulation Failures**
- **Cause**: Physics engine crash, GPU memory exhaustion, or simulation divergence
- **Detection**: Simulation process exit, invalid physics state
- **Response**:
  - Log critical error
  - Attempt to save current state
  - Gracefully shut down all nodes
  - Notify user
- **Recovery**: Manual restart of simulation

### Error Handling Principles

1. **Fail-Safe**: Always default to stopping the robot when in doubt
2. **Graceful Degradation**: Continue operation in reduced capacity when possible
3. **Transparency**: Log all errors and notify user via dashboard
4. **Automatic Recovery**: Attempt automatic recovery for transient failures
5. **Manual Override**: Provide user controls to reset or restart system

### Logging Strategy

All errors are logged with:
- Timestamp (ROS time and wall time)
- Error category and severity (INFO, WARN, ERROR, FATAL)
- Component that detected the error
- Detailed diagnostic information
- Current system state snapshot

Logs are written to:
- ROS 2 logging system (rosout)
- Local log files (rotating, max 100MB per file)
- Dashboard error panel (for user visibility)

## Testing Strategy

### Dual Testing Approach

The AdaptNav system requires both unit testing and property-based testing for comprehensive validation:

**Unit Tests**: Verify specific examples, edge cases, and error conditions
- Specific scenario tests (e.g., "robot navigates around single worker")
- Edge case tests (e.g., "goal exactly on obstacle boundary")
- Integration tests between components
- Error condition tests (e.g., "sensor timeout triggers safety stop")

**Property Tests**: Verify universal properties across all inputs
- Test properties with randomized inputs (positions, velocities, configurations)
- Minimum 100 iterations per property test
- Each property test validates one correctness property from this design
- Properties catch bugs that specific examples might miss

Together, these approaches provide comprehensive coverage: unit tests catch concrete bugs in specific scenarios, while property tests verify general correctness across the input space.

### Property-Based Testing Configuration

**Framework**: Hypothesis (Python) for property-based testing

**Test Configuration**:
- Minimum 100 iterations per property test (due to randomization)
- Timeout: 60 seconds per property test
- Shrinking enabled to find minimal failing examples
- Random seed logged for reproducibility

**Test Tagging**:
Each property-based test must include a comment tag referencing the design property:
```python
# Feature: adaptnav-context-aware-warehouse-navigation, Property 7: Collision-Free Paths
@given(start_position=valid_positions(), goal_position=valid_positions())
def test_collision_free_paths(start_position, goal_position):
    # Test implementation
    pass
```

**Generators for Property Tests**:
- `valid_positions()`: Random (x, y) positions within warehouse bounds, not in obstacles
- `valid_velocities()`: Random velocities within physical limits
- `obstacle_configurations()`: Random sets of dynamic obstacles
- `sensor_data()`: Random but realistic sensor readings
- `warehouse_layouts()`: Random warehouse configurations with varying obstacle placements

### Unit Testing Strategy

**Test Organization**:
- One test file per component (e.g., `test_path_planner.py`, `test_safety_controller.py`)
- Group related tests using test classes
- Use descriptive test names: `test_<component>_<scenario>_<expected_outcome>`

**Coverage Goals**:
- Minimum 80% code coverage for core navigation logic
- 100% coverage for safety-critical components (Safety Controller)
- All error handling paths must be tested

**Test Categories**:

1. **Component Unit Tests**:
   - Path Planner: Test A* algorithm, path smoothing, failure cases
   - Obstacle Detector: Test clustering, tracking, sensor fusion
   - Safety Controller: Test collision zone, velocity limits, override logic
   - PPO Agent: Test observation/action structure, inference

2. **Integration Tests**:
   - Navigation Controller: Test state machine transitions
   - Sensor → Detector → Safety pipeline
   - Planner → PPO → Safety integration
   - ROS 2 topic/service communication

3. **End-to-End Tests**:
   - Complete navigation scenarios (start to goal)
   - Multi-obstacle avoidance scenarios
   - Safety stop and recovery scenarios
   - Replanning scenarios

4. **Performance Tests**:
   - Measure and assert latency requirements
   - Measure and assert frequency requirements
   - Memory usage tests (no leaks over 1000 episodes)

**Mocking Strategy**:
- Mock simulation for faster unit tests
- Mock sensors for detector tests
- Mock ROS 2 communication for isolated component tests
- Use real integration for end-to-end tests

### Test Execution

**Continuous Testing**:
```bash
# Run all unit tests
pytest tests/unit/ -v

# Run all property tests
pytest tests/properties/ -v --hypothesis-show-statistics

# Run integration tests
pytest tests/integration/ -v

# Run with coverage
pytest tests/ --cov=adaptnav --cov-report=html
```

**Pre-commit Testing**:
- Fast unit tests (<30 seconds total)
- Critical property tests (10 iterations for speed)

**CI/CD Testing**:
- Full unit test suite
- Full property test suite (100+ iterations)
- Integration tests
- Performance benchmarks

**Simulation Testing**:
- Test with both Isaac Sim and MuJoCo backends
- Test on different hardware configurations
- Test with varying simulation speeds

### Test Data and Fixtures

**Warehouse Configurations**:
- `simple_warehouse.yaml`: 10m x 10m, minimal obstacles
- `medium_warehouse.yaml`: 30m x 30m, moderate obstacles
- `complex_warehouse.yaml`: 50m x 50m, dense obstacles

**Obstacle Scenarios**:
- `static_only.yaml`: No dynamic obstacles
- `single_worker.yaml`: One moving worker
- `multi_obstacle.yaml`: Multiple workers and forklifts
- `crowded.yaml`: High-density dynamic obstacles

**Benchmark Scenarios**:
- `benchmark_easy.yaml`: Short path, few obstacles
- `benchmark_medium.yaml`: Medium path, moderate obstacles
- `benchmark_hard.yaml`: Long path, many obstacles, narrow passages

### Success Criteria

A test suite passes if:
1. All unit tests pass (100%)
2. All property tests pass (100%)
3. Code coverage meets minimum thresholds (80% overall, 100% safety)
4. No memory leaks detected
5. Performance benchmarks meet requirements
6. Integration tests pass on both simulation backends

### Debugging Failed Property Tests

When a property test fails:
1. Hypothesis will shrink to minimal failing example
2. Log the failing input (seed, generated values)
3. Re-run with `--hypothesis-seed=<seed>` to reproduce
4. Add the failing case as a unit test for regression
5. Fix the bug and verify all tests pass
6. Consider if the property needs refinement

