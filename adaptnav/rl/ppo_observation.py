"""
PPO Observation class for reinforcement learning agent.

This module defines the observation structure used by the PPO agent for navigation
decisions. The observation includes LiDAR data, goal direction, current velocity,
and obstacle proximity information.
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass

# ROS 2 message imports with fallback for testing
try:
    from sensor_msgs.msg import LaserScan
    from nav_msgs.msg import Odometry
    from geometry_msgs.msg import Pose
    from custom_msgs.msg import ObstacleArray
    ROS_AVAILABLE = True
except ImportError:
    # Fallback for testing without ROS
    LaserScan = None
    Odometry = None
    Pose = None
    ObstacleArray = None
    ROS_AVAILABLE = False


@dataclass
class PPOObservation:
    """
    Observation structure for PPO agent containing all necessary information
    for navigation decisions.
    
    The observation is structured to provide:
    - 360-degree LiDAR scan data (normalized distances)
    - Relative direction to goal position
    - Current robot velocity (linear and angular)
    - Obstacle proximity in 8 directional sectors
    
    Total observation dimension: 372 (360 + 2 + 2 + 8)
    """
    
    lidar_scan: np.ndarray          # Shape: (360,), normalized distances [0, 1]
    goal_direction: np.ndarray      # Shape: (2,), relative [x, y] to goal
    current_velocity: np.ndarray    # Shape: (2,), [linear, angular] velocity
    obstacle_proximity: np.ndarray  # Shape: (8,), min distance in 8 sectors
    
    def __post_init__(self):
        """Validate observation dimensions after initialization."""
        if self.lidar_scan.shape != (360,):
            raise ValueError(f"lidar_scan must have shape (360,), got {self.lidar_scan.shape}")
        if self.goal_direction.shape != (2,):
            raise ValueError(f"goal_direction must have shape (2,), got {self.goal_direction.shape}")
        if self.current_velocity.shape != (2,):
            raise ValueError(f"current_velocity must have shape (2,), got {self.current_velocity.shape}")
        if self.obstacle_proximity.shape != (8,):
            raise ValueError(f"obstacle_proximity must have shape (8,), got {self.obstacle_proximity.shape}")
    
    def to_vector(self) -> np.ndarray:
        """
        Flatten observation to 372-dimensional vector for neural network input.
        
        Returns:
            np.ndarray: Flattened observation vector of shape (372,)
                       [lidar_scan(360) + goal_direction(2) + current_velocity(2) + obstacle_proximity(8)]
        """
        return np.concatenate([
            self.lidar_scan,
            self.goal_direction,
            self.current_velocity,
            self.obstacle_proximity
        ])
    
    @classmethod
    def from_ros_messages(cls, scan: LaserScan, odom: Odometry, 
                          obstacles: ObstacleArray, goal: Pose) -> 'PPOObservation':
        """
        Construct PPOObservation from ROS messages.
        
        Args:
            scan: LiDAR scan data (sensor_msgs/LaserScan)
            odom: Robot odometry (nav_msgs/Odometry)
            obstacles: Detected obstacles (custom_msgs/ObstacleArray)
            goal: Goal pose (geometry_msgs/Pose)
            
        Returns:
            PPOObservation: Constructed observation object
            
        Raises:
            ValueError: If ROS messages are not available or invalid
        """
        if not ROS_AVAILABLE:
            raise ValueError("ROS messages not available. Cannot construct from ROS messages.")
        
        # Process LiDAR scan
        lidar_scan = cls._process_lidar_scan(scan)
        
        # Compute goal direction relative to robot
        goal_direction = cls._compute_goal_direction(odom, goal)
        
        # Extract current velocity
        current_velocity = cls._extract_velocity(odom)
        
        # Compute obstacle proximity in 8 sectors
        obstacle_proximity = cls._compute_obstacle_proximity(obstacles, odom)
        
        return cls(
            lidar_scan=lidar_scan,
            goal_direction=goal_direction,
            current_velocity=current_velocity,
            obstacle_proximity=obstacle_proximity
        )
    
    @staticmethod
    def _process_lidar_scan(scan: LaserScan) -> np.ndarray:
        """
        Process LiDAR scan data into normalized 360-degree array.
        
        Args:
            scan: ROS LaserScan message
            
        Returns:
            np.ndarray: Normalized distances of shape (360,)
        """
        # Convert ranges to numpy array
        ranges = np.array(scan.ranges)
        
        # Handle infinite and NaN values
        ranges = np.where(np.isfinite(ranges), ranges, scan.range_max)
        
        # Clamp to valid range
        ranges = np.clip(ranges, scan.range_min, scan.range_max)
        
        # Normalize to [0, 1] range
        normalized_ranges = (ranges - scan.range_min) / (scan.range_max - scan.range_min)
        
        # Ensure exactly 360 readings (interpolate if necessary)
        if len(normalized_ranges) != 360:
            # Interpolate to 360 points
            original_angles = np.linspace(0, 2*np.pi, len(normalized_ranges), endpoint=False)
            target_angles = np.linspace(0, 2*np.pi, 360, endpoint=False)
            normalized_ranges = np.interp(target_angles, original_angles, normalized_ranges)
        
        return normalized_ranges.astype(np.float32)
    
    @staticmethod
    def _compute_goal_direction(odom: Odometry, goal: Pose) -> np.ndarray:
        """
        Compute relative direction from robot to goal.
        
        Args:
            odom: Robot odometry
            goal: Goal pose
            
        Returns:
            np.ndarray: Relative [x, y] direction to goal
        """
        # Extract robot position and orientation
        robot_x = odom.pose.pose.position.x
        robot_y = odom.pose.pose.position.y
        
        # Extract robot orientation (yaw from quaternion)
        quat = odom.pose.pose.orientation
        robot_yaw = np.arctan2(
            2.0 * (quat.w * quat.z + quat.x * quat.y),
            1.0 - 2.0 * (quat.y * quat.y + quat.z * quat.z)
        )
        
        # Extract goal position
        goal_x = goal.position.x
        goal_y = goal.position.y
        
        # Compute relative position in world frame
        rel_x = goal_x - robot_x
        rel_y = goal_y - robot_y
        
        # Transform to robot frame
        cos_yaw = np.cos(robot_yaw)
        sin_yaw = np.sin(robot_yaw)
        
        rel_x_robot = cos_yaw * rel_x + sin_yaw * rel_y
        rel_y_robot = -sin_yaw * rel_x + cos_yaw * rel_y
        
        # Normalize to unit vector (or zero if at goal)
        distance = np.sqrt(rel_x_robot**2 + rel_y_robot**2)
        if distance > 1e-6:
            rel_x_robot /= distance
            rel_y_robot /= distance
        
        return np.array([rel_x_robot, rel_y_robot], dtype=np.float32)
    
    @staticmethod
    def _extract_velocity(odom: Odometry) -> np.ndarray:
        """
        Extract current velocity from odometry.
        
        Args:
            odom: Robot odometry
            
        Returns:
            np.ndarray: [linear_velocity, angular_velocity]
        """
        linear_vel = odom.twist.twist.linear.x
        angular_vel = odom.twist.twist.angular.z
        
        return np.array([linear_vel, angular_vel], dtype=np.float32)
    
    @staticmethod
    def _compute_obstacle_proximity(obstacles: ObstacleArray, odom: Odometry) -> np.ndarray:
        """
        Compute minimum distance to obstacles in 8 directional sectors.
        
        The 8 sectors are:
        0: Forward (0°)
        1: Forward-right (45°)
        2: Right (90°)
        3: Backward-right (135°)
        4: Backward (180°)
        5: Backward-left (225°)
        6: Left (270°)
        7: Forward-left (315°)
        
        Args:
            obstacles: Detected obstacles
            odom: Robot odometry
            
        Returns:
            np.ndarray: Minimum distances in 8 sectors, shape (8,)
        """
        # Initialize with maximum distance (10m)
        sector_distances = np.full(8, 10.0, dtype=np.float32)
        
        if len(obstacles.obstacles) == 0:
            return sector_distances
        
        # Extract robot position and orientation
        robot_x = odom.pose.pose.position.x
        robot_y = odom.pose.pose.position.y
        
        # Extract robot orientation (yaw from quaternion)
        quat = odom.pose.pose.orientation
        robot_yaw = np.arctan2(
            2.0 * (quat.w * quat.z + quat.x * quat.y),
            1.0 - 2.0 * (quat.y * quat.y + quat.z * quat.z)
        )
        
        # Process each obstacle
        for obstacle in obstacles.obstacles:
            # Compute relative position
            rel_x = obstacle.position.x - robot_x
            rel_y = obstacle.position.y - robot_y
            
            # Transform to robot frame
            cos_yaw = np.cos(robot_yaw)
            sin_yaw = np.sin(robot_yaw)
            
            rel_x_robot = cos_yaw * rel_x + sin_yaw * rel_y
            rel_y_robot = -sin_yaw * rel_x + cos_yaw * rel_y
            
            # Compute distance and angle
            distance = np.sqrt(rel_x_robot**2 + rel_y_robot**2)
            angle = np.arctan2(rel_y_robot, rel_x_robot)
            
            # Convert angle to sector (0-7)
            # Normalize angle to [0, 2π]
            angle = (angle + 2*np.pi) % (2*np.pi)
            
            # Map to sector (each sector is 45° = π/4 radians)
            sector = int((angle + np.pi/8) / (np.pi/4)) % 8
            
            # Update minimum distance for this sector
            sector_distances[sector] = min(sector_distances[sector], distance)
        
        return sector_distances
    
    @classmethod
    def create_empty(cls) -> 'PPOObservation':
        """
        Create an empty observation with default values for testing.
        
        Returns:
            PPOObservation: Empty observation with default values
        """
        return cls(
            lidar_scan=np.ones(360, dtype=np.float32),  # All max range
            goal_direction=np.array([1.0, 0.0], dtype=np.float32),  # Forward
            current_velocity=np.array([0.0, 0.0], dtype=np.float32),  # Stationary
            obstacle_proximity=np.full(8, 10.0, dtype=np.float32)  # No obstacles
        )
    
    def __eq__(self, other) -> bool:
        """Check equality with another PPOObservation."""
        if not isinstance(other, PPOObservation):
            return False
        
        return (
            np.allclose(self.lidar_scan, other.lidar_scan, rtol=1e-6) and
            np.allclose(self.goal_direction, other.goal_direction, rtol=1e-6) and
            np.allclose(self.current_velocity, other.current_velocity, rtol=1e-6) and
            np.allclose(self.obstacle_proximity, other.obstacle_proximity, rtol=1e-6)
        )
    
    def __repr__(self) -> str:
        """String representation of the observation."""
        return (
            f"PPOObservation(\n"
            f"  lidar_scan: shape={self.lidar_scan.shape}, "
            f"min={self.lidar_scan.min():.3f}, max={self.lidar_scan.max():.3f}\n"
            f"  goal_direction: {self.goal_direction}\n"
            f"  current_velocity: {self.current_velocity}\n"
            f"  obstacle_proximity: {self.obstacle_proximity}\n"
            f")"
        )