"""
RobotState class for robot state representation.

This module provides the RobotState class which represents the current state
of the robot including position, orientation, and velocity. It supports
conversion to ROS messages and distance calculations to goal positions.
"""

import numpy as np
from typing import Optional

try:
    from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
    from std_msgs.msg import Header
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False


class RobotState:
    """
    Represents the current state of the robot.
    
    The robot state includes position, orientation, and velocity information
    in the map frame. This class provides methods for converting to ROS messages
    and computing distances to goal positions.
    
    Attributes:
        position: [x, y] position in map frame (meters)
        orientation: Theta (yaw angle) in radians
        linear_velocity: Linear velocity in meters/second
        angular_velocity: Angular velocity in radians/second
        timestamp: ROS time in seconds (optional)
    """
    
    def __init__(
        self,
        position: np.ndarray,
        orientation: float,
        linear_velocity: float = 0.0,
        angular_velocity: float = 0.0,
        timestamp: Optional[float] = None
    ):
        """
        Initialize a RobotState.
        
        Args:
            position: [x, y] position in map frame (meters)
            orientation: Theta (yaw angle) in radians
            linear_velocity: Linear velocity in meters/second (default: 0.0)
            angular_velocity: Angular velocity in radians/second (default: 0.0)
            timestamp: ROS time in seconds (optional)
        """
        self.position = np.asarray(position, dtype=np.float64)
        self.orientation = float(orientation)
        self.linear_velocity = float(linear_velocity)
        self.angular_velocity = float(angular_velocity)
        self.timestamp = float(timestamp) if timestamp is not None else None
        
        # Validate inputs
        if self.position.shape != (2,):
            raise ValueError(f"Position must be a 2D array, got shape {self.position.shape}")
    
    def to_pose_stamped(self, frame_id: str = "map") -> 'PoseStamped':
        """
        Convert robot state to ROS PoseStamped message.
        
        This method converts the robot's position and orientation to a ROS
        PoseStamped message, which is commonly used in ROS navigation systems.
        The orientation is converted from a yaw angle to a quaternion.
        
        Args:
            frame_id: Reference frame for the pose (default: "map")
            
        Returns:
            PoseStamped message with robot position and orientation
            
        Raises:
            ImportError: If ROS messages are not available
        """
        if not ROS_AVAILABLE:
            raise ImportError("ROS messages not available. Install rclpy and geometry_msgs.")
        
        pose_stamped = PoseStamped()
        
        # Set header
        pose_stamped.header = Header()
        pose_stamped.header.frame_id = frame_id
        if self.timestamp is not None:
            # Convert timestamp to ROS time (seconds and nanoseconds)
            sec = int(self.timestamp)
            nanosec = int((self.timestamp - sec) * 1e9)
            pose_stamped.header.stamp.sec = sec
            pose_stamped.header.stamp.nanosec = nanosec
        
        # Set position
        pose_stamped.pose = Pose()
        pose_stamped.pose.position = Point()
        pose_stamped.pose.position.x = float(self.position[0])
        pose_stamped.pose.position.y = float(self.position[1])
        pose_stamped.pose.position.z = 0.0
        
        # Convert yaw to quaternion (rotation around z-axis)
        # q = [qx, qy, qz, qw] where qw = cos(theta/2), qz = sin(theta/2)
        half_yaw = self.orientation / 2.0
        pose_stamped.pose.orientation = Quaternion()
        pose_stamped.pose.orientation.x = 0.0
        pose_stamped.pose.orientation.y = 0.0
        pose_stamped.pose.orientation.z = float(np.sin(half_yaw))
        pose_stamped.pose.orientation.w = float(np.cos(half_yaw))
        
        return pose_stamped
    
    def distance_to_goal(self, goal: np.ndarray) -> float:
        """
        Compute Euclidean distance to goal position.
        
        This method calculates the straight-line distance from the robot's
        current position to a goal position in the map frame.
        
        Args:
            goal: [x, y] goal position in map frame (meters)
            
        Returns:
            Euclidean distance in meters
            
        Raises:
            ValueError: If goal is not a 2D array
        """
        goal = np.asarray(goal, dtype=np.float64)
        if goal.shape != (2,):
            raise ValueError(f"Goal must be a 2D array, got shape {goal.shape}")
        
        return np.linalg.norm(self.position - goal)
    
    def __repr__(self) -> str:
        """Return string representation of the robot state."""
        return (
            f"RobotState(position={self.position}, "
            f"orientation={self.orientation:.3f}, "
            f"linear_velocity={self.linear_velocity:.3f}, "
            f"angular_velocity={self.angular_velocity:.3f}, "
            f"timestamp={self.timestamp})"
        )
    
    def __eq__(self, other) -> bool:
        """Check equality based on position, orientation, and velocities."""
        if not isinstance(other, RobotState):
            return False
        return (
            np.allclose(self.position, other.position) and
            np.isclose(self.orientation, other.orientation) and
            np.isclose(self.linear_velocity, other.linear_velocity) and
            np.isclose(self.angular_velocity, other.angular_velocity)
        )
