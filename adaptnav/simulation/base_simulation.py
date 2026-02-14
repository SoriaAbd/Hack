"""
Base simulation class for AdaptNav warehouse environment.

This module provides an abstract base class that defines the common interface
for simulation backends (MuJoCo, Isaac Sim, etc.). It handles ground truth
data publishing via ROS 2 topics and defines the core simulation lifecycle.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan, Image
from custom_msgs.msg import ObstacleArray, Obstacle

from adaptnav.core.robot_state import RobotState
from adaptnav.core.dynamic_obstacle import DynamicObstacle


class BaseSimulation(ABC, Node):
    """
    Abstract base class for warehouse simulation environments.
    
    This class defines the common interface that all simulation backends
    (MuJoCo, Isaac Sim) must implement. It provides:
    - Abstract methods for simulation lifecycle (step, reset, get_observation)
    - Ground truth data access interface
    - ROS 2 topic publishers for ground truth data
    - Velocity command subscription
    
    Subclasses must implement the abstract methods to integrate with
    specific physics engines.
    """
    
    def __init__(self, node_name: str = 'base_simulation'):
        """
        Initialize the base simulation node.
        
        Args:
            node_name: Name for the ROS 2 node
        """
        super().__init__(node_name)
        
        # ROS 2 publishers for ground truth data
        self._ground_truth_robot_pose_pub = self.create_publisher(
            PoseStamped,
            '/ground_truth/robot_pose',
            10
        )
        
        self._ground_truth_obstacles_pub = self.create_publisher(
            ObstacleArray,
            '/ground_truth/obstacles',
            10
        )
        
        # ROS 2 subscriber for velocity commands
        self._cmd_vel_sub = self.create_subscription(
            Twist,
            '/cmd_vel',
            self._cmd_vel_callback,
            10
        )
        
        # Store latest velocity command
        self._current_cmd_vel: Optional[Twist] = None
        
        # Simulation state
        self._is_initialized = False
        self._step_count = 0
        
        self.get_logger().info(f'{node_name} initialized')
    
    @abstractmethod
    def step(self, dt: float) -> bool:
        """
        Advance the simulation by one time step.
        
        This method should:
        1. Apply the current velocity command to the robot
        2. Update physics simulation
        3. Update dynamic obstacle positions
        4. Detect collisions
        5. Publish ground truth data
        
        Args:
            dt: Time step in seconds
            
        Returns:
            True if step was successful, False if simulation error occurred
        """
        pass
    
    @abstractmethod
    def reset(self, 
              robot_position: Optional[Tuple[float, float, float]] = None,
              obstacle_configs: Optional[List[Dict]] = None) -> bool:
        """
        Reset the simulation to initial state.
        
        This method should:
        1. Reset robot to initial/specified position
        2. Reset obstacles to initial/specified configurations
        3. Clear any accumulated state
        4. Reset physics simulation
        
        Args:
            robot_position: Optional (x, y, theta) for robot spawn position.
                          If None, use default spawn position.
            obstacle_configs: Optional list of obstacle configurations.
                            Each config is a dict with keys: 'type', 'position', 'velocity'
                            If None, use default obstacle configuration.
                            
        Returns:
            True if reset was successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_observation(self) -> Dict:
        """
        Get the current observation from the simulation.
        
        This method should return sensor data and state information needed
        for the navigation system. The observation should include:
        - LiDAR scan data
        - Depth camera data (if available)
        - Robot odometry
        - Any other relevant sensor data
        
        Returns:
            Dictionary containing observation data with keys:
            - 'lidar_scan': LaserScan message or numpy array
            - 'depth_image': Image message or numpy array (optional)
            - 'odometry': Odometry message or RobotState
            - 'timestamp': Current simulation time
        """
        pass
    
    @abstractmethod
    def get_ground_truth_robot_state(self) -> RobotState:
        """
        Get the ground truth robot state.
        
        Returns:
            RobotState object with true position, orientation, and velocities
        """
        pass
    
    @abstractmethod
    def get_ground_truth_obstacles(self) -> List[DynamicObstacle]:
        """
        Get the ground truth positions and velocities of all dynamic obstacles.
        
        Returns:
            List of DynamicObstacle objects with true positions and velocities
        """
        pass
    
    @abstractmethod
    def check_collision(self) -> bool:
        """
        Check if the robot is currently in collision with any obstacle.
        
        Returns:
            True if collision detected, False otherwise
        """
        pass
    
    def publish_ground_truth(self) -> None:
        """
        Publish ground truth data to ROS 2 topics.
        
        This method retrieves ground truth data from the simulation and
        publishes it to the appropriate topics for visualization and debugging.
        """
        # Publish robot pose
        robot_state = self.get_ground_truth_robot_state()
        pose_msg = robot_state.to_pose_stamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'map'
        self._ground_truth_robot_pose_pub.publish(pose_msg)
        
        # Publish obstacles
        obstacles = self.get_ground_truth_obstacles()
        obstacle_array_msg = ObstacleArray()
        obstacle_array_msg.header.stamp = self.get_clock().now().to_msg()
        obstacle_array_msg.header.frame_id = 'map'
        
        for obs in obstacles:
            obs_msg = Obstacle()
            obs_msg.id = obs.id
            obs_msg.position.x = float(obs.position[0])
            obs_msg.position.y = float(obs.position[1])
            obs_msg.position.z = 0.0
            obs_msg.velocity.x = float(obs.velocity[0])
            obs_msg.velocity.y = float(obs.velocity[1])
            obs_msg.velocity.z = 0.0
            obs_msg.classification = obs.classification
            obs_msg.confidence = 1.0  # Ground truth has perfect confidence
            
            # Covariance is zero for ground truth (perfect knowledge)
            obs_msg.covariance = [0.0] * 16
            
            obstacle_array_msg.obstacles.append(obs_msg)
        
        self._ground_truth_obstacles_pub.publish(obstacle_array_msg)
    
    def _cmd_vel_callback(self, msg: Twist) -> None:
        """
        Callback for velocity command messages.
        
        Args:
            msg: Twist message with linear and angular velocity commands
        """
        self._current_cmd_vel = msg
    
    def get_current_cmd_vel(self) -> Optional[Twist]:
        """
        Get the most recent velocity command.
        
        Returns:
            Latest Twist message, or None if no command received yet
        """
        return self._current_cmd_vel
    
    def is_initialized(self) -> bool:
        """
        Check if the simulation has been initialized.
        
        Returns:
            True if simulation is ready, False otherwise
        """
        return self._is_initialized
    
    def get_step_count(self) -> int:
        """
        Get the number of simulation steps executed since last reset.
        
        Returns:
            Step count
        """
        return self._step_count
    
    def increment_step_count(self) -> None:
        """Increment the step counter."""
        self._step_count += 1
    
    def set_initialized(self, value: bool) -> None:
        """
        Set the initialization state.
        
        Args:
            value: True if initialized, False otherwise
        """
        self._is_initialized = value
