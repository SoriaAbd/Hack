# Implementation Plan: AdaptNav Context-Aware Warehouse Navigation

## Overview

This implementation plan breaks down the AdaptNav system into incremental coding tasks. The system combines traditional path planning (A*) with reinforcement learning (PPO) for autonomous warehouse navigation with dynamic obstacle avoidance. The implementation follows a bottom-up approach: starting with core data structures and utilities, building up individual components, then integrating them into the complete navigation system.

The implementation uses Python 3.10+ with ROS 2 Humble, Stable Baselines3 for RL, and either Isaac Sim or MuJoCo for simulation.

## Tasks

- [x] 1. Set up project structure and ROS 2 workspace
  - Create ROS 2 package structure with proper dependencies
  - Set up Python package with src/adaptnav directory
  - Create package.xml with all required dependencies (rclpy, sensor_msgs, nav_msgs, geometry_msgs, etc.)
  - Create CMakeLists.txt for ROS 2 build system
  - Set up requirements.txt for Python dependencies (numpy, stable-baselines3, hypothesis, pytest, etc.)
  - Create launch directory for ROS 2 launch files
  - Create config directory for YAML configuration files
  - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 9.5_

- [x] 2. Define custom ROS 2 message types
  - [x] 2.1 Create custom_msgs package for message definitions
    - Create Obstacle.msg with fields: id, position, velocity, covariance, classification, confidence, last_seen
    - Create ObstacleArray.msg with header and obstacles array
    - Create SafetyStatus.msg with state, closest_obstacle_distance, velocity_scale, override_active, time_until_clear
    - Create NavigationState.msg with state, current_pose, goal_pose, distance_to_goal, progress_percentage, reasoning
    - _Requirements: 4.1, 4.2, 6.1, 6.2, 6.3, 7.4_


- [x] 3. Implement core data models and utilities
  - [x] 3.1 Create WarehouseMap class for static environment representation
    - Implement occupancy grid storage and access methods
    - Implement is_collision_free() method for collision checking
    - Implement get_neighbors() method for path planning
    - Add methods to load map from file (YAML format)
    - _Requirements: 1.1, 3.1, 3.2_

  - [x] 3.2 Write property test for WarehouseMap collision checking
    - **Property 7: Collision-Free Paths**
    - **Validates: Requirements 3.1**

  - [x] 3.3 Create DynamicObstacle class for moving obstacle representation
    - Implement position, velocity, radius, and classification fields
    - Implement predict_position() method using constant velocity model
    - Implement distance_to() method for distance calculations
    - _Requirements: 1.2, 4.1, 4.2_

  - [x] 3.4 Create RobotState class for robot state representation
    - Implement position, orientation, velocity fields
    - Implement to_pose_stamped() conversion method
    - Implement distance_to_goal() method
    - _Requirements: 8.1_

  - [x] 3.5 Create Waypoint and Path classes for path representation
    - Implement Waypoint with x, y, theta fields
    - Implement Path with waypoints list and total_length
    - Implement get_closest_waypoint() method
    - Implement get_lookahead_point() method for path following
    - _Requirements: 3.1_

- [x] 4. Implement warehouse environment simulation
  - [x] 4.1 Create simulation base class with common interface
    - Define abstract methods for step(), reset(), get_observation()
    - Define interface for ground truth data access
    - Implement ROS 2 topic publishers for ground truth data
    - _Requirements: 1.3, 1.4, 9.1_

  - [x] 4.2 Implement MuJoCo simulation backend
    - Create MJCF model file for warehouse environment with walls, shelves, columns
    - Implement robot model with differential drive dynamics
    - Implement worker and forklift obstacle models with scripted motion
    - Integrate with ROS 2 bridge for sensor data publishing
    - _Requirements: 1.1, 1.2, 1.5, 9.1, 9.3_

  - [x] 4.3 Write property test for simulation initialization
    - **Property 1: Initialization Consistency**
    - **Validates: Requirements 1.3**

  - [x] 4.4 Write property test for ground truth availability
    - **Property 2: Ground Truth Availability**
    - **Validates: Requirements 1.4**

  - [x] 4.5 Write property test for collision detection
    - **Property 3: Collision Detection**
    - **Validates: Requirements 1.5**

- [x] 5. Implement sensor simulation layer
  - [x] 5.1 Create LiDAR sensor simulator
    - Implement ray casting for 360° scan with 1° resolution
    - Add realistic noise model (Gaussian noise on range measurements)
    - Implement range limitations (0.1-10m) and occlusion effects
    - Publish sensor_msgs/LaserScan at 20 Hz
    - _Requirements: 2.1, 2.3, 2.4_

  - [x] 5.2 Create depth camera simulator
    - Implement depth image generation with 90° FOV, 640x480 resolution
    - Add realistic noise model (depth accuracy degradation with distance)
    - Implement range limitations (0.5-5m)
    - Publish sensor_msgs/Image and camera_info at 15 Hz
    - _Requirements: 2.2, 2.3, 2.5_

  - [x] 5.3 Write property test for sensor publishing frequency
    - **Property 4: Sensor Publishing Frequency**
    - **Validates: Requirements 2.3**

  - [x] 5.4 Write property test for LiDAR range limitations
    - **Property 5: LiDAR Range Limitations**
    - **Validates: Requirements 2.4**

  - [x] 5.5 Write property test for depth camera field of view
    - **Property 6: Depth Camera Field of View**
    
    - **Validates: Requirements 2.5**

- [x] 6. Checkpoint - Ensure simulation and sensors work
  - Ensure all tests pass, ask the user if questions arise.


- [ ] 7. Implement obstacle detection and tracking
  - [x] 7.1 Create obstacle detector node
    - Subscribe to /scan and /camera/depth/image_raw topics
    - Implement sensor data fusion (combine LiDAR and depth camera point clouds)
    - Implement DBSCAN clustering for obstacle detection
    - Publish custom_msgs/ObstacleArray with detected obstacles
    - _Requirements: 4.1, 4.3_

  - [x] 7.2 Implement Kalman filter for obstacle tracking
    - Create KalmanFilter class for position and velocity estimation
    - Implement data association (match detections to existing tracks)
    - Maintain tracking IDs across frames
    - Estimate obstacle velocities from position history
    - _Requirements: 4.2, 4.5_

  - [x] 7.3 Add obstacle classification heuristics
    - Classify based on size (workers ~0.5m radius, forklifts ~1.0m radius)
    - Classify based on velocity (workers <2 m/s, forklifts <5 m/s)
    - Add confidence scoring based on detection consistency
    - _Requirements: 4.1_

  - [x] 7.4 Write property test for detection completeness
    - **Property 11: Detection Completeness**
    - **Validates: Requirements 4.1, 4.2**

  - [x] 7.5 Write property test for sensor fusion accuracy
    - **Property 12: Sensor Fusion Accuracy**
    - **Validates: Requirements 4.3**

  - [x] 7.6 Write property test for detection latency
    - **Property 13: Detection Latency**
    - **Validates: Requirements 4.4**

  - [x] 7.7 Write property test for tracking consistency
    - **Property 14: Tracking Consistency**
    - **Validates: Requirements 4.5**

  - [x] 7.8 Write unit tests for obstacle detector
    - Test clustering with known point clouds
    - Test tracking with simulated obstacle trajectories
    - Test edge cases (no obstacles, single obstacle, many obstacles)
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

- [ ] 8. Implement global path planner
  - [x] 8.1 Create A* path planning algorithm
    - Implement A* search on occupancy grid
    - Use Euclidean distance heuristic
    - Implement path reconstruction from search result
    - Add timeout mechanism (2 seconds max)
    - _Requirements: 3.1, 3.2, 3.5_

  - [x] 8.2 Add path smoothing with cubic splines
    - Implement spline interpolation for waypoint smoothing
    - Ensure smoothed path remains collision-free
    - _Requirements: 3.1_

  - [x] 8.3 Create path planner ROS 2 service
    - Implement service server for nav2_msgs/ComputePathToPose
    - Subscribe to /map topic for occupancy grid
    - Publish planned path to /global_plan topic
    - Handle planning failures with diagnostic messages
    - _Requirements: 3.1, 3.3, 8.2_

  - [x] 8.4 Write property test for collision-free paths
    - **Property 7: Collision-Free Paths**
    - **Validates: Requirements 3.1**

  - [x] 8.5 Write property test for planning failure handling
    - **Property 8: Planning Failure Handling**
    - **Validates: Requirements 3.3**

  - [x] 8.6 Write property test for planning performance
    - **Property 10: Planning Performance**
    - **Validates: Requirements 3.5**

  - [x] 8.7 Write unit tests for A* planner
    - Test simple paths (straight line, L-shape, U-turn)
    - Test edge cases (start equals goal, unreachable goal)
    - Test performance on various map sizes
    - _Requirements: 3.1, 3.2, 3.3, 3.5_

- [ ] 9. Implement PPO agent for local navigation
  - [x] 9.1 Create PPOObservation class
    - Implement observation structure with lidar_scan, goal_direction, current_velocity, obstacle_proximity
    - Implement to_vector() method to flatten to 372-dimensional vector
    - Implement from_ros_messages() class method to construct from ROS topics
    - _Requirements: 5.1_

  - [x] 9.2 Create Gymnasium environment for training
    - Implement reset() method to initialize random scenarios
    - Implement step() method to execute actions and compute rewards
    - Implement reward function (progress toward goal, collision penalty, goal bonus)
    - Define observation space (372-dim) and action space (2-dim continuous)
    - _Requirements: 5.1, 5.2, 5.3_

  - [x] 9.3 Implement PPO training script
    - Use Stable Baselines3 PPO implementation
    - Configure hyperparameters (learning rate 3e-4, batch size 2048, etc.)
    - Implement training loop with periodic evaluation
    - Log training metrics (success rate, collision rate, path efficiency)
    - Save model checkpoints periodically
    - _Requirements: 5.3, 5.4, 10.1, 10.2, 10.3_

  - [x] 9.4 Create PPO agent ROS 2 node
    - Load trained model from file
    - Subscribe to sensor topics and construct observations
    - Compute actions using loaded model
    - Publish velocity commands to /cmd_vel_raw topic
    - _Requirements: 5.2, 5.5_

  - [~] 9.5 Write property test for observation structure
    - **Property 15: Observation Structure**
    - **Validates: Requirements 5.1**

  - [~] 9.6 Write property test for action output structure
    - **Property 16: Action Output Structure**
    - **Validates: Requirements 5.2**

  - [~] 9.7 Write property test for PPO inference latency
    - **Property 30: PPO Inference Latency**
    - **Validates: Requirements 11.3**

  - [~] 9.8 Write unit tests for PPO observation construction
    - Test observation vector dimensions
    - Test normalization of sensor data
    - Test goal direction computation
    - _Requirements: 5.1_

- [~] 10. Checkpoint - Ensure planning and RL components work independently
  - Ensure all tests pass, ask the user if questions arise.


- [-] 11. Implement safety controller
  - [x] 11.1 Create safety controller node
    - Subscribe to /obstacles/detected, /cmd_vel_raw, and /odom topics
    - Implement collision zone checking (0.5m buffer around robot)
    - Implement velocity limit enforcement (1.0 m/s linear, 0.5 rad/s angular)
    - Implement emergency stop logic when obstacles within 0.3m
    - Publish filtered commands to /cmd_vel topic
    - Publish safety status to /safety_status topic
    - _Requirements: 6.1, 6.2, 6.3, 6.5_

  - [x] 11.2 Implement clearance time enforcement
    - Track time since collision zone became clear
    - Require 1 second clearance before resuming motion
    - Implement smooth deceleration when approaching obstacles
    - _Requirements: 6.4_

  - [x] 11.3 Add safety override logic
    - Predict future robot position based on current velocity
    - Check if predicted trajectory would collide within 0.5 seconds
    - Override unsafe commands with safer alternatives (reduce speed or stop)
    - _Requirements: 6.3_

  - [~] 11.4 Write property test for collision zone enforcement
    - **Property 19: Collision Zone Enforcement**
    - **Validates: Requirements 6.1**

  - [~] 11.5 Write property test for emergency stop response
    - **Property 20: Emergency Stop Response**
    - **Validates: Requirements 6.2**

  - [~] 11.6 Write property test for safety override
    - **Property 21: Safety Override**
    - **Validates: Requirements 6.3**

  - [~] 11.7 Write property test for clearance time enforcement
    - **Property 22: Clearance Time Enforcement**
    - **Validates: Requirements 6.4**

  - [~] 11.8 Write property test for velocity limit enforcement
    - **Property 23: Velocity Limit Enforcement**
    - **Validates: Requirements 6.5**

  - [~] 11.9 Write property test for safety evaluation latency
    - **Property 31: Safety Evaluation Latency**
    - **Validates: Requirements 11.4**

  - [~] 11.10 Write unit tests for safety controller
    - Test collision zone detection with various obstacle positions
    - Test velocity clamping with out-of-range commands
    - Test emergency stop triggering and recovery
    - Test clearance time enforcement
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

- [-] 12. Implement navigation controller
  - [x] 12.1 Create navigation state machine
    - Implement states: IDLE, PLANNING, FOLLOWING_PATH, AVOIDING_OBSTACLE, EMERGENCY_STOP, GOAL_REACHED
    - Implement state transition logic based on events
    - Track current state and publish to /navigation_state topic
    - _Requirements: 7.4_

  - [x] 12.2 Create navigation controller node
    - Implement action server for nav2_msgs/NavigateToPose
    - Subscribe to /global_plan, /obstacles/detected, /safety_status, /odom topics
    - Orchestrate global planner, PPO agent, and safety controller
    - Implement replanning logic when path becomes blocked
    - Publish navigation state updates
    - _Requirements: 3.4, 5.5, 8.2_

  - [x] 12.3 Implement hybrid navigation logic
    - Extract next waypoint from global plan
    - Provide waypoint to PPO agent as goal direction
    - Switch between path following and obstacle avoidance based on context
    - Detect goal reached condition (within 0.2m, velocity < 0.1 m/s)
    - _Requirements: 5.5_

  - [~] 12.4 Add replanning trigger logic
    - Monitor obstacles relative to planned path
    - Trigger replanning when obstacle within 0.5m of any waypoint
    - Implement replanning cooldown (don't replan more than once per second)
    - _Requirements: 3.4_

  - [~] 12.5 Write property test for replanning trigger
    - **Property 9: Replanning Trigger**
    - **Validates: Requirements 3.4**

  - [~] 12.6 Write property test for hybrid navigation integration
    - **Property 18: Hybrid Navigation Integration**
    - **Validates: Requirements 5.5**

  - [~] 12.7 Write property test for control loop frequency
    - **Property 28: Control Loop Frequency**
    - **Validates: Requirements 11.1**

  - [~] 12.8 Write unit tests for navigation controller
    - Test state machine transitions
    - Test goal reached detection
    - Test replanning trigger conditions
    - Test integration of planner and PPO outputs
    - _Requirements: 3.4, 5.5, 8.2_

- [-] 13. Implement visualization dashboard
  - [x] 13.1 Create RViz2 configuration
    - Add robot model visualization
    - Add LiDAR point cloud display
    - Add planned path visualization
    - Add obstacle markers with velocity vectors
    - Add safety zone visualization (color-coded by status)
    - Configure camera views and display panels
    - _Requirements: 7.1, 7.2, 7.3, 7.5_

  - [x] 13.2 Create web dashboard backend
    - Implement WebSocket server for real-time data streaming
    - Subscribe to all relevant ROS 2 topics
    - Convert ROS messages to JSON for web frontend
    - Implement service proxies for start/stop/reset commands
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7_

  - [x] 13.3 Create web dashboard frontend
    - Implement 3D visualization using Three.js
    - Display robot, obstacles, and planned path
    - Add control panel for start/stop/reset
    - Add decision state indicator panel
    - Add reasoning panel showing PPO observation values
    - Add performance metrics display
    - Implement real-time updates via WebSocket
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7_

  - [~] 13.4 Write property test for obstacle visualization completeness
    - **Property 24: Obstacle Visualization Completeness**
    - **Validates: Requirements 7.2**

  - [~] 13.5 Write property test for state display
    - **Property 25: State Display**
    - **Validates: Requirements 7.4**

  - [~] 13.6 Write property test for dashboard update frequency
    - **Property 26: Dashboard Update Frequency**
    - **Validates: Requirements 7.6**

  - [~] 13.7 Write property test for explainability display
    - **Property 27: Explainability Display**
    - **Validates: Requirements 7.7**

- [~] 14. Checkpoint - Ensure complete system integration
  - Ensure all tests pass, ask the user if questions arise.


- [x] 15. Create launch files and configuration
  - [x] 15.1 Create main launch file
    - Launch simulation node (MuJoCo or Isaac Sim)
    - Launch all sensor nodes
    - Launch obstacle detector node
    - Launch global planner node
    - Launch PPO agent node
    - Launch safety controller node
    - Launch navigation controller node
    - Launch dashboard nodes (RViz2 and/or web)
    - Set up all required parameters and remappings
    - _Requirements: 9.5_

  - [x] 15.2 Create configuration files
    - Create warehouse layout YAML files (simple, medium, complex)
    - Create obstacle scenario YAML files (static_only, single_worker, multi_obstacle, crowded)
    - Create sensor configuration YAML (noise parameters, frequencies)
    - Create planner configuration YAML (grid resolution, timeout)
    - Create safety configuration YAML (collision zone, velocity limits)
    - _Requirements: 1.1, 1.2, 2.4, 2.5, 3.5, 6.1, 6.5_

  - [x] 15.3 Create demonstration scenario configurations
    - Create benchmark_easy.yaml (short path, few obstacles)
    - Create benchmark_medium.yaml (medium path, moderate obstacles)
    - Create benchmark_hard.yaml (long path, many obstacles, narrow passages)
    - _Requirements: 12.1_

- [ ] 16. Implement training and evaluation scripts
  - [~] 16.1 Create PPO training script
    - Set up training environment with diverse scenarios
    - Implement training loop with Stable Baselines3
    - Log training metrics to TensorBoard
    - Save model checkpoints every 100k steps
    - Implement early stopping based on success rate
    - _Requirements: 10.1, 10.2, 10.3, 10.5_

  - [~] 16.2 Create evaluation script
    - Load trained model
    - Run evaluation on benchmark scenarios
    - Compute success rate, collision rate, path efficiency
    - Generate evaluation report with statistics
    - _Requirements: 10.4_

  - [~] 16.3 Add scenario generation for training
    - Implement random warehouse layout generation
    - Implement random obstacle configuration generation
    - Implement random start/goal position generation
    - Ensure diversity in training scenarios
    - _Requirements: 10.2_

  - [~] 16.4 Write property test for training improvement
    - **Property 17: Training Improvement**
    - **Validates: Requirements 5.3**

  - [~] 16.5 Write property test for scenario diversity
    - **Property 33: Scenario Diversity**
    - **Validates: Requirements 10.2**

  - [~] 16.6 Write property test for training metrics logging
    - **Property 34: Training Metrics Logging**
    - **Validates: Requirements 10.3**

  - [~] 16.7 Write property test for model persistence round-trip
    - **Property 35: Model Persistence Round-Trip**
    - **Validates: Requirements 10.5**

- [ ] 17. Implement error handling and recovery
  - [~] 17.1 Add planning failure handling
    - Detect planning failures and log diagnostics
    - Publish failure status to navigation state
    - Notify user via dashboard
    - Implement retry logic with timeout
    - _Requirements: 3.3_

  - [~] 17.2 Add sensor failure handling
    - Detect sensor timeouts (>1 second)
    - Use last known valid data for up to 2 seconds
    - Trigger emergency stop if timeout exceeds 2 seconds
    - Log warnings and notify dashboard
    - _Requirements: 2.3_

  - [~] 17.3 Add obstacle detector failure handling
    - Detect detector timeouts (>200ms)
    - Assume worst-case (obstacles everywhere) on failure
    - Trigger safety stop
    - Attempt automatic restart of detector node
    - _Requirements: 4.4_

  - [~] 17.4 Add PPO agent failure handling
    - Detect inference timeouts (>100ms) or NaN/Inf outputs
    - Fall back to pure path-following behavior
    - Clamp invalid actions to zero velocity
    - Continue in degraded mode and log error
    - _Requirements: 5.2, 11.3_

  - [~] 17.5 Add communication failure handling
    - Detect ROS 2 topic timeouts
    - Attempt reconnection for up to 5 seconds
    - Trigger emergency stop if reconnection fails
    - Log communication errors
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

  - [~] 17.6 Write unit tests for error handling
    - Test planning failure scenarios
    - Test sensor timeout handling
    - Test detector failure recovery
    - Test PPO fallback behavior
    - Test communication failure handling
    - _Requirements: 3.3, 2.3, 4.4, 5.2, 8.1_

- [ ] 18. Add performance monitoring and optimization
  - [~] 18.1 Implement latency monitoring
    - Add timing instrumentation to all components
    - Log processing latencies for detector, planner, PPO, safety
    - Publish latency metrics to dashboard
    - Alert if latencies exceed requirements
    - _Requirements: 11.1, 11.2, 11.3, 11.4_

  - [~] 18.2 Implement frequency monitoring
    - Monitor publishing frequencies for all topics
    - Log frequency metrics
    - Alert if frequencies drop below requirements
    - _Requirements: 2.3, 11.1_

  - [~] 18.3 Write property test for obstacle detection latency
    - **Property 29: Obstacle Detection Latency**
    - **Validates: Requirements 11.2**

  - [~] 18.4 Write property test for real-time performance
    - **Property 32: Real-Time Performance**
    - **Validates: Requirements 11.5**

- [ ] 19. Create documentation and demo setup
  - [~] 19.1 Write setup documentation
    - Document installation steps (ROS 2, dependencies, simulation)
    - Document build instructions
    - Document configuration options
    - _Requirements: 12.4_

  - [~] 19.2 Write training documentation
    - Document training process and hyperparameters
    - Document how to customize scenarios
    - Document evaluation process
    - _Requirements: 12.4_

  - [~] 19.3 Write demonstration documentation
    - Document how to run demo scenarios
    - Document dashboard usage
    - Document troubleshooting common issues
    - _Requirements: 12.4_

  - [~] 19.4 Create demo mode for dashboard
    - Add enhanced visualizations for presentation
    - Add preset camera angles
    - Add performance statistics overlay
    - Add "replay" functionality for recorded runs
    - _Requirements: 12.3_

  - [~] 19.5 Create quick-start launch script
    - Single command to launch complete system
    - Auto-load default configuration
    - Auto-start demo scenario
    - _Requirements: 12.5_


- [ ] 20. Integration testing and validation
  - [~] 20.1 Write end-to-end integration tests
    - Test complete navigation from start to goal in simple scenario
    - Test multi-obstacle avoidance scenario
    - Test safety stop and recovery scenario
    - Test replanning scenario
    - _Requirements: 3.1, 3.4, 4.1, 5.5, 6.2, 6.4_

  - [~] 20.2 Write property test for simulation real-time performance
    - **Property 36: Simulation Real-Time Performance**
    - **Validates: Requirements 9.4**

  - [~] 20.3 Write property test for demonstration success rate
    - **Property 37: Demonstration Success Rate**
    - **Validates: Requirements 12.2**

  - [~] 20.4 Run performance benchmarks
    - Measure latencies for all components
    - Measure publishing frequencies
    - Measure memory usage over 1000 episodes
    - Verify all performance requirements are met
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5_

  - [~] 20.5 Run demonstration scenarios
    - Run benchmark_easy 20 times and verify 80%+ success rate
    - Run benchmark_medium 20 times and verify 80%+ success rate
    - Run benchmark_hard 20 times and verify 80%+ success rate
    - _Requirements: 12.1, 12.2_

- [~] 21. Final checkpoint and system validation
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation at key milestones
- Property tests validate universal correctness properties across randomized inputs
- Unit tests validate specific examples, edge cases, and error conditions
- The implementation follows a bottom-up approach: data structures → components → integration
- Python 3.10+ is used for all implementation with ROS 2 Humble
- Stable Baselines3 is used for PPO implementation
- MuJoCo is the primary simulation backend (Isaac Sim as alternative)
- Hypothesis framework is used for property-based testing with minimum 100 iterations per test
