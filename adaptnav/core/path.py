"""
Path representation for navigation planning.

This module provides classes for representing waypoints and paths in the warehouse
navigation system. Paths consist of sequences of waypoints that the robot follows
to reach its goal.
"""

from typing import List, Tuple, Optional
import numpy as np


class Waypoint:
    """
    Represents a single waypoint in the warehouse coordinate frame.
    
    A waypoint specifies a target position (x, y) and desired orientation (theta)
    for the robot to reach during navigation.
    
    Attributes:
        x: X-coordinate in meters (map frame)
        y: Y-coordinate in meters (map frame)
        theta: Desired orientation in radians (map frame)
    """
    
    def __init__(self, x: float, y: float, theta: float = 0.0):
        """
        Initialize a waypoint.
        
        Args:
            x: X-coordinate in meters
            y: Y-coordinate in meters
            theta: Desired orientation in radians (default: 0.0)
        """
        self.x = x
        self.y = y
        self.theta = theta
    
    def position(self) -> np.ndarray:
        """
        Get the position as a numpy array.
        
        Returns:
            numpy array [x, y]
        """
        return np.array([self.x, self.y])
    
    def distance_to(self, point: np.ndarray) -> float:
        """
        Compute Euclidean distance from this waypoint to a point.
        
        Args:
            point: Target point as [x, y] array
            
        Returns:
            Distance in meters
        """
        return np.linalg.norm(self.position() - point)
    
    def __repr__(self) -> str:
        return f"Waypoint(x={self.x:.2f}, y={self.y:.2f}, theta={self.theta:.2f})"
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, Waypoint):
            return False
        return (np.isclose(self.x, other.x) and 
                np.isclose(self.y, other.y) and 
                np.isclose(self.theta, other.theta))


class Path:
    """
    Represents a path as a sequence of waypoints.
    
    A path is computed by the global planner and provides waypoints for the robot
    to follow. The path includes methods for finding the closest waypoint and
    computing lookahead points for path following algorithms.
    
    Attributes:
        waypoints: List of waypoints defining the path
        total_length: Total path length in meters
        timestamp: Time when path was computed (optional)
    """
    
    def __init__(self, waypoints: List[Waypoint], timestamp: Optional[float] = None):
        """
        Initialize a path from a list of waypoints.
        
        Args:
            waypoints: List of waypoints defining the path
            timestamp: Optional timestamp when path was computed
        """
        if not waypoints:
            raise ValueError("Path must contain at least one waypoint")
        
        self.waypoints = waypoints
        self.timestamp = timestamp
        self.total_length = self._compute_length()
    
    def _compute_length(self) -> float:
        """
        Compute the total length of the path.
        
        Returns:
            Total path length in meters
        """
        if len(self.waypoints) < 2:
            return 0.0
        
        length = 0.0
        for i in range(len(self.waypoints) - 1):
            p1 = self.waypoints[i].position()
            p2 = self.waypoints[i + 1].position()
            length += np.linalg.norm(p2 - p1)
        
        return length
    
    def get_closest_waypoint(self, position: np.ndarray) -> Tuple[int, Waypoint]:
        """
        Find the closest waypoint to the current position.
        
        This method searches through all waypoints and returns the one with
        minimum Euclidean distance to the given position.
        
        Args:
            position: Current position as [x, y] array
            
        Returns:
            Tuple of (index, waypoint) where index is the position in the
            waypoints list and waypoint is the closest Waypoint object
        """
        if not self.waypoints:
            raise ValueError("Path has no waypoints")
        
        min_distance = float('inf')
        closest_idx = 0
        
        for i, waypoint in enumerate(self.waypoints):
            distance = waypoint.distance_to(position)
            if distance < min_distance:
                min_distance = distance
                closest_idx = i
        
        return closest_idx, self.waypoints[closest_idx]
    
    def get_lookahead_point(self, position: np.ndarray, 
                           lookahead_dist: float) -> Waypoint:
        """
        Get a point on the path at a lookahead distance ahead of current position.
        
        This method is used for path following algorithms (e.g., pure pursuit).
        It finds the point on the path that is approximately lookahead_dist meters
        ahead of the current position along the path.
        
        Algorithm:
        1. Find the closest waypoint to current position
        2. Starting from that waypoint, traverse forward along the path
        3. Return the first waypoint that is at least lookahead_dist away
        4. If no waypoint is far enough, return the last waypoint (goal)
        
        Args:
            position: Current position as [x, y] array
            lookahead_dist: Desired lookahead distance in meters
            
        Returns:
            Waypoint at approximately lookahead_dist ahead on the path
        """
        if not self.waypoints:
            raise ValueError("Path has no waypoints")
        
        if lookahead_dist <= 0:
            raise ValueError("Lookahead distance must be positive")
        
        # Find closest waypoint
        closest_idx, _ = self.get_closest_waypoint(position)
        
        # Search forward from closest waypoint for lookahead point
        for i in range(closest_idx, len(self.waypoints)):
            waypoint = self.waypoints[i]
            distance = waypoint.distance_to(position)
            
            if distance >= lookahead_dist:
                return waypoint
        
        # If no waypoint is far enough, return the last waypoint (goal)
        return self.waypoints[-1]
    
    def __len__(self) -> int:
        """Return the number of waypoints in the path."""
        return len(self.waypoints)
    
    def __repr__(self) -> str:
        return f"Path(waypoints={len(self.waypoints)}, length={self.total_length:.2f}m)"
