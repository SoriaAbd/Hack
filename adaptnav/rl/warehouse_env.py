"""
Gymnasium environment for AdaptNav warehouse navigation training.

This module implements a Gymnasium environment that wraps the MuJoCo simulation
for training PPO agents. The environment provides:
- 372-dimensional observation space (LiDAR + goal direction + velocity + obstacle proximity)
- 2-dimensional continuous action space (linear and angular velocity)
- Reward function encouraging goal-reaching while avoiding collisions
- Random scenario generation for diverse training experiences
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Optional, Tuple, Any
import random
import math

# Handle ROS 2 availability
try:
    from adaptnav.simulation.mujoco_simulation import MuJoCoSimulation
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    # Create mock class for testing
    class MuJoCoSimulation:
        def __init__(self, *args, **kwargs):
            pass

from adaptnav.rl.ppo_observation import PPOObservation
from adaptnav.core.robot_state import RobotState
from adaptnav.core.dynamic_obstacle import DynamicObstacle


class WarehouseNavigationEnv(gym.Env):
    """
    Gymnasium environment for warehouse navigation training.
    
    This environment wraps the MuJoCo simulation and provides the interface
    needed for training PPO agents with Stable Baselines3.
    
    Observation Space (372-dim):
    - LiDAR scan: 360 values (normalized distances)
    - Goal direction: 2 values (relative x, y to goal)
    - Current velocity: 2 values (linear, angular)
    - Obstacle proximity: 8 values (min distance in 8 sectors)
    
    Action Space (2-dim continuous):
    - Linear velocity: [-1.0, 1.0] m/s
    - Angular velocity: [-0.5, 0.5] rad/s
    
    Reward Function:
    - +1.0 * progress_toward_goal (normalized)
    - -10.0 * collision_penalty (binary)
    - -0.1 * distance_to_goal (normalized)
    - -0.01 * action_magnitude (L2 norm)
    - +100.0 * goal_reached_bonus (binary)
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 30}
    
    def __init__(self, 
                 warehouse_config_path: Optional[str] = None,
                 max_episode_steps: int = 1000,
                 goal_tolerance: float = 0.5,
                 collision_penalty: float = 10.0,
                 goal_bonus: float = 100.0,
                 progress_weight: float = 1.0,
                 distance_weight: float = 0.1,
                 action_weight: float = 0.01,
                 render_mode: Optional[str] = None):
        """
        Initialize the warehouse navigation environment.
        
        Args:
            warehouse_config_path: Path to warehouse configuration file
            max_episode_steps: Maximum steps per episode
            goal_tolerance: Distance threshold for goal reached (meters)
            collision_penalty: Penalty for collisions
            goal_bonus: Bonus for reaching goal
            progress_weight: Weight for progress toward goal reward
            distance_weight: Weight for distance to goal penalty
            action_weight: Weight for action magnitude penalty
            render_mode: Rendering mode ("human" or None)
        """
        super().__init__()
        
        # Environment parameters
        self.max_episode_steps = max_episode_steps
        self.goal_tolerance = goal_tolerance
        self.collision_penalty = collision_penalty
        self.goal_bonus = goal_bonus
        self.progress_weight = progress_weight
        self.distance_weight = distance_weight
        self.action_weight = action_weight
        self.render_mode = render_mode
        
        # Define observation space (372-dimensional)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(372,),
            dtype=np.float32
        )
        
        # Define action space (2-dimensional continuous)
        self.action_space = spaces.Box(
            low=np.array([-1.0, -0.5], dtype=np.float32),  # [linear_vel, angular_vel]
            high=np.array([1.0, 0.5], dtype=np.float32),
            dtype=np.float32
        )
        
        # Initialize simulation
        if not ROS_AVAILABLE:
            raise ValueError("ROS 2 not available. Cannot create MuJoCoSimulation.")
        
        self.simulation = MuJoCoSimulation(
            model_path=warehouse_config_path or "adaptnav/simulation/models/warehouse.xml",
            node_name="warehouse_env_simulation"
        )
        
        # Episode state
        self.current_step = 0
        self.goal_position = np.array([0.0, 0.0])
        self.initial_distance_to_goal = 0.0
        self.previous_distance_to_goal = 0.0
        self.episode_reward = 0.0
        self.collision_occurred = False
        self.goal_reached = False
        
        # Scenario generation parameters
        self.warehouse_bounds = {
            'x_min': -10.0, 'x_max': 10.0,
            'y_min': -10.0, 'y_max': 10.0
        }
        self.min_goal_distance = 3.0  # Minimum distance between start and goal
        self.max_goal_distance = 15.0  # Maximum distance between start and goal
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to initial state with random scenario.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options (unused)
            
        Returns:
            Tuple of (observation, info_dict)
        """
        super().reset(seed=seed)
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Generate random scenario
        robot_position, goal_position, obstacle_configs = self._generate_random_scenario()
        
        # Reset simulation with new scenario
        success = self.simulation.reset(
            robot_position=robot_position,
            obstacle_configs=obstacle_configs
        )
        
        if not success:
            raise RuntimeError("Failed to reset simulation")
        
        # Set goal and initialize episode state
        self.goal_position = np.array(goal_position[:2])  # Only x, y components
        self.current_step = 0
        self.episode_reward = 0.0
        self.collision_occurred = False
        self.goal_reached = False
        
        # Calculate initial distance for progress tracking
        robot_state = self.simulation.get_ground_truth_robot_state()
        self.initial_distance_to_goal = np.linalg.norm(robot_state.position - self.goal_position)
        self.previous_distance_to_goal = self.initial_distance_to_goal
        
        # Get initial observation
        observation = self._get_observation()
        
        info = {
            'robot_position': robot_state.position.tolist(),
            'goal_position': self.goal_position.tolist(),
            'initial_distance': self.initial_distance_to_goal,
            'obstacle_count': len(obstacle_configs) if obstacle_configs else 0
        }
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action array [linear_velocity, angular_velocity]
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Validate action
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Apply action to simulation
        if ROS_AVAILABLE:
            from geometry_msgs.msg import Twist
            cmd_vel = Twist()
            cmd_vel.linear.x = float(action[0])
            cmd_vel.angular.z = float(action[1])
            
            # Store command for simulation to use
            self.simulation._current_cmd_vel = cmd_vel
        else:
            # Mock command for testing
            pass
        
        # Step simulation
        dt = 0.1  # 10 Hz control frequency
        success = self.simulation.step(dt)
        
        if not success:
            # Simulation error - terminate episode
            observation = self._get_observation()
            return observation, -100.0, True, False, {'error': 'simulation_failed'}
        
        # Get current state
        robot_state = self.simulation.get_ground_truth_robot_state()
        collision = self.simulation.check_collision()
        
        # Calculate reward
        reward = self._compute_reward(robot_state, action, collision)
        
        # Check termination conditions
        distance_to_goal = np.linalg.norm(robot_state.position - self.goal_position)
        goal_reached = distance_to_goal < self.goal_tolerance
        
        terminated = collision or goal_reached
        truncated = self.current_step >= self.max_episode_steps
        
        # Update episode state
        self.current_step += 1
        self.episode_reward += reward
        self.collision_occurred = self.collision_occurred or collision
        self.goal_reached = goal_reached
        self.previous_distance_to_goal = distance_to_goal
        
        # Get observation
        observation = self._get_observation()
        
        # Prepare info dict
        info = {
            'robot_position': robot_state.position.tolist(),
            'goal_position': self.goal_position.tolist(),
            'distance_to_goal': distance_to_goal,
            'collision': collision,
            'goal_reached': goal_reached,
            'episode_reward': self.episode_reward,
            'step': self.current_step
        }
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """
        Get the current observation from the simulation.
        
        Returns:
            372-dimensional observation vector
        """
        # Get simulation observation
        sim_obs = self.simulation.get_observation()
        
        # Create mock ROS messages for PPOObservation.from_ros_messages
        # In a real implementation, these would come from ROS topics
        if ROS_AVAILABLE:
            from sensor_msgs.msg import LaserScan
            from nav_msgs.msg import Odometry
            from geometry_msgs.msg import Pose
            from custom_msgs.msg import ObstacleArray
            
            # Convert simulation data to ROS message format
            lidar_scan = sim_obs.get('lidar_scan')
            odometry = sim_obs.get('odometry')
            
            # Create goal pose message
            goal_pose = Pose()
            goal_pose.position.x = float(self.goal_position[0])
            goal_pose.position.y = float(self.goal_position[1])
            goal_pose.position.z = 0.0
            
            # Get obstacles from simulation
            obstacles = self.simulation.get_ground_truth_obstacles()
            obstacle_array = ObstacleArray()
            # Note: In a real implementation, this would be populated from detected obstacles
            
            try:
                # Use PPOObservation to construct observation
                ppo_obs = PPOObservation.from_ros_messages(
                    lidar_scan, odometry, obstacle_array, goal_pose
                )
                return ppo_obs.to_vector()
            except Exception as e:
                # Fallback: create observation manually
                return self._create_fallback_observation()
        else:
            # When ROS is not available, always use fallback
            return self._create_fallback_observation()
    
    def _create_fallback_observation(self) -> np.ndarray:
        """
        Create a fallback observation when ROS message conversion fails.
        
        Returns:
            372-dimensional observation vector
        """
        # Get basic state information
        robot_state = self.simulation.get_ground_truth_robot_state()
        obstacles = self.simulation.get_ground_truth_obstacles()
        
        # Create LiDAR scan (360 values)
        lidar_scan = np.ones(360, dtype=np.float32) * 10.0  # Max range
        
        # Simulate basic obstacle detection in LiDAR
        for obstacle in obstacles:
            rel_pos = obstacle.position - robot_state.position
            distance = np.linalg.norm(rel_pos)
            if distance < 10.0:  # Within LiDAR range
                angle = math.atan2(rel_pos[1], rel_pos[0])
                angle_deg = math.degrees(angle)
                if angle_deg < 0:
                    angle_deg += 360
                lidar_idx = int(angle_deg) % 360
                lidar_scan[lidar_idx] = min(lidar_scan[lidar_idx], distance)
        
        # Normalize LiDAR scan
        lidar_scan = lidar_scan / 10.0  # Normalize to [0, 1]
        
        # Goal direction (2 values)
        goal_direction = self.goal_position - robot_state.position
        goal_distance = np.linalg.norm(goal_direction)
        if goal_distance > 0:
            goal_direction = goal_direction / goal_distance
        else:
            goal_direction = np.array([0.0, 0.0])
        
        # Current velocity (2 values)
        current_velocity = np.array([
            robot_state.linear_velocity,
            robot_state.angular_velocity
        ], dtype=np.float32)
        
        # Obstacle proximity in 8 sectors (8 values)
        obstacle_proximity = np.full(8, 10.0, dtype=np.float32)  # Max distance
        
        for obstacle in obstacles:
            rel_pos = obstacle.position - robot_state.position
            distance = np.linalg.norm(rel_pos)
            if distance < 10.0:
                angle = math.atan2(rel_pos[1], rel_pos[0])
                # Convert to sector (0-7)
                sector = int((angle + math.pi) / (2 * math.pi / 8)) % 8
                obstacle_proximity[sector] = min(obstacle_proximity[sector], distance)
        
        # Normalize obstacle proximity
        obstacle_proximity = obstacle_proximity / 10.0
        
        # Combine all components (360 + 2 + 2 + 8 = 372)
        observation = np.concatenate([
            lidar_scan,
            goal_direction.astype(np.float32),
            current_velocity,
            obstacle_proximity
        ])
        
        return observation
    
    def _compute_reward(self, robot_state: RobotState, action: np.ndarray, collision: bool) -> float:
        """
        Compute the reward for the current step.
        
        Args:
            robot_state: Current robot state
            action: Action taken this step
            collision: Whether collision occurred
            
        Returns:
            Reward value
        """
        reward = 0.0
        
        # Distance to goal
        current_distance = np.linalg.norm(robot_state.position - self.goal_position)
        
        # Progress toward goal (positive reward for getting closer)
        if self.initial_distance_to_goal > 0:
            progress = (self.previous_distance_to_goal - current_distance) / self.initial_distance_to_goal
            reward += self.progress_weight * progress
        
        # Distance penalty (encourage reaching goal)
        if self.initial_distance_to_goal > 0:
            normalized_distance = current_distance / self.initial_distance_to_goal
            reward -= self.distance_weight * normalized_distance
        
        # Action magnitude penalty (encourage smooth control)
        action_magnitude = np.linalg.norm(action)
        reward -= self.action_weight * action_magnitude
        
        # Collision penalty
        if collision:
            reward -= self.collision_penalty
        
        # Goal reached bonus
        if current_distance < self.goal_tolerance:
            reward += self.goal_bonus
        
        return reward
    
    def _generate_random_scenario(self) -> Tuple[Tuple[float, float, float], Tuple[float, float], List[Dict]]:
        """
        Generate a random training scenario.
        
        Returns:
            Tuple of (robot_position, goal_position, obstacle_configs)
        """
        # Generate random robot start position
        robot_x = random.uniform(self.warehouse_bounds['x_min'], self.warehouse_bounds['x_max'])
        robot_y = random.uniform(self.warehouse_bounds['y_min'], self.warehouse_bounds['y_max'])
        robot_theta = random.uniform(-math.pi, math.pi)
        robot_position = (robot_x, robot_y, robot_theta)
        
        # Generate random goal position (ensuring minimum distance from start)
        attempts = 0
        max_attempts = 100
        
        while attempts < max_attempts:
            goal_x = random.uniform(self.warehouse_bounds['x_min'], self.warehouse_bounds['x_max'])
            goal_y = random.uniform(self.warehouse_bounds['y_min'], self.warehouse_bounds['y_max'])
            
            distance = math.sqrt((goal_x - robot_x)**2 + (goal_y - robot_y)**2)
            
            if self.min_goal_distance <= distance <= self.max_goal_distance:
                break
            
            attempts += 1
        
        if attempts >= max_attempts:
            # Fallback: place goal at fixed distance
            angle = random.uniform(-math.pi, math.pi)
            distance = (self.min_goal_distance + self.max_goal_distance) / 2
            goal_x = robot_x + distance * math.cos(angle)
            goal_y = robot_y + distance * math.sin(angle)
            
            # Clamp to bounds
            goal_x = np.clip(goal_x, self.warehouse_bounds['x_min'], self.warehouse_bounds['x_max'])
            goal_y = np.clip(goal_y, self.warehouse_bounds['y_min'], self.warehouse_bounds['y_max'])
        
        goal_position = (goal_x, goal_y)
        
        # Generate random obstacle configuration
        obstacle_configs = self._generate_random_obstacles(robot_position, goal_position)
        
        return robot_position, goal_position, obstacle_configs
    
    def _generate_random_obstacles(self, robot_pos: Tuple[float, float, float], 
                                 goal_pos: Tuple[float, float]) -> List[Dict]:
        """
        Generate random obstacle configurations.
        
        Args:
            robot_pos: Robot starting position (x, y, theta)
            goal_pos: Goal position (x, y)
            
        Returns:
            List of obstacle configuration dictionaries
        """
        obstacle_configs = []
        
        # Random number of obstacles (0-5)
        num_obstacles = random.randint(0, 5)
        
        for i in range(num_obstacles):
            # Random obstacle type
            obstacle_type = random.choice(['worker', 'forklift'])
            
            # Random position (avoiding robot and goal)
            attempts = 0
            max_attempts = 50
            
            while attempts < max_attempts:
                obs_x = random.uniform(self.warehouse_bounds['x_min'], self.warehouse_bounds['x_max'])
                obs_y = random.uniform(self.warehouse_bounds['y_min'], self.warehouse_bounds['y_max'])
                
                # Check distance from robot and goal
                robot_dist = math.sqrt((obs_x - robot_pos[0])**2 + (obs_y - robot_pos[1])**2)
                goal_dist = math.sqrt((obs_x - goal_pos[0])**2 + (obs_y - goal_pos[1])**2)
                
                if robot_dist > 2.0 and goal_dist > 2.0:  # Minimum clearance
                    break
                
                attempts += 1
            
            if attempts >= max_attempts:
                continue  # Skip this obstacle
            
            # Random velocity
            if obstacle_type == 'worker':
                max_speed = 1.5  # m/s
                radius = 0.5
            else:  # forklift
                max_speed = 3.0  # m/s
                radius = 1.0
            
            speed = random.uniform(0.0, max_speed)
            direction = random.uniform(-math.pi, math.pi)
            
            velocity = (speed * math.cos(direction), speed * math.sin(direction))
            
            obstacle_config = {
                'type': obstacle_type,
                'position': (obs_x, obs_y, 0.0),
                'velocity': velocity,
                'radius': radius
            }
            
            obstacle_configs.append(obstacle_config)
        
        return obstacle_configs
    
    def render(self):
        """
        Render the environment (placeholder).
        
        In a full implementation, this would visualize the warehouse,
        robot, obstacles, and goal position.
        """
        if self.render_mode == "human":
            # Placeholder for rendering
            # In a real implementation, this could use matplotlib or pygame
            # to visualize the environment state
            pass
    
    def close(self):
        """Clean up resources."""
        if hasattr(self, 'simulation'):
            # Clean up simulation resources
            pass