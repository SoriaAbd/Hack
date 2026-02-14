"""
Path smoothing using cubic splines.

This module provides path smoothing functionality to convert discrete waypoint
paths from A* into smooth, continuous trajectories suitable for robot following.
The smoother uses cubic spline interpolation while ensuring the smoothed path
remains collision-free.
"""

from typing import List, Optional
import numpy as np
from scipy.interpolate import CubicSpline

from ..core.warehouse_map import WarehouseMap
from ..core.path import Path, Waypoint


class PathSmoother:
    """
    Path smoothing using cubic spline interpolation.
    
    This class takes a discrete path from A* planning and generates a smooth
    trajectory using cubic splines. The smoother ensures that the resulting
    path remains collision-free by checking intermediate points along the
    smoothed trajectory.
    """
    
    def __init__(self, warehouse_map: WarehouseMap, robot_radius: float = 0.3,
                 smoothing_resolution: float = 0.2):
        """
        Initialize the path smoother.
        
        Args:
            warehouse_map: WarehouseMap for collision checking
            robot_radius: Robot radius for collision checking (meters)
            smoothing_resolution: Distance between collision check points (meters)
        """
        self.warehouse_map = warehouse_map
        self.robot_radius = robot_radius
        self.smoothing_resolution = smoothing_resolution
    
    def smooth_path(self, path: Path, num_points: Optional[int] = None) -> Optional[Path]:
        """
        Smooth a path using cubic spline interpolation.
        
        This method takes a discrete path and generates a smooth trajectory
        by fitting cubic splines to the waypoint positions. The smoothed path
        is validated for collision-free status.
        
        Args:
            path: Input path to smooth
            num_points: Number of points in smoothed path (default: auto-calculate)
            
        Returns:
            Smoothed Path object if successful, None if smoothing fails
        """
        if len(path.waypoints) < 2:
            return path  # Cannot smooth single waypoint
        
        # Extract positions from waypoints
        positions = np.array([[wp.x, wp.y] for wp in path.waypoints])
        
        # If path has only 2 points, return original (already smooth)
        if len(positions) == 2:
            return path
        
        # Create parameter array (cumulative distance along path)
        distances = np.zeros(len(positions))
        for i in range(1, len(positions)):
            distances[i] = distances[i-1] + np.linalg.norm(positions[i] - positions[i-1])
        
        # Handle case where all points are the same (shouldn't happen in practice)
        if distances[-1] == 0:
            return path
        
        # Calculate number of points for smoothed path
        if num_points is None:
            # Use one point per smoothing_resolution meters
            num_points = max(10, int(distances[-1] / self.smoothing_resolution))
        
        try:
            # Create cubic splines for x and y coordinates
            cs_x = CubicSpline(distances, positions[:, 0])
            cs_y = CubicSpline(distances, positions[:, 1])
            
            # Generate smoothed path points
            smooth_distances = np.linspace(0, distances[-1], num_points)
            smooth_x = cs_x(smooth_distances)
            smooth_y = cs_y(smooth_distances)
            
            # Create smoothed waypoints
            smoothed_waypoints = []
            for i in range(len(smooth_x)):
                # Calculate orientation (tangent to path)
                if i < len(smooth_x) - 1:
                    dx = smooth_x[i+1] - smooth_x[i]
                    dy = smooth_y[i+1] - smooth_y[i]
                    theta = np.arctan2(dy, dx)
                else:
                    # Use previous orientation for last waypoint
                    theta = smoothed_waypoints[-1].theta if smoothed_waypoints else 0.0
                
                smoothed_waypoints.append(Waypoint(smooth_x[i], smooth_y[i], theta))
            
            # Validate that smoothed path is collision-free
            if not self._is_path_collision_free(smoothed_waypoints):
                print("Smoothed path contains collisions, returning original path")
                return path
            
            # Create and return smoothed path
            smoothed_path = Path(smoothed_waypoints, path.timestamp)
            print(f"Path smoothed: {len(path.waypoints)} -> {len(smoothed_waypoints)} waypoints")
            return smoothed_path
            
        except Exception as e:
            print(f"Path smoothing failed: {e}")
            return path  # Return original path if smoothing fails
    
    def _is_path_collision_free(self, waypoints: List[Waypoint]) -> bool:
        """
        Check if a path is collision-free by sampling points along segments.
        
        Args:
            waypoints: List of waypoints defining the path
            
        Returns:
            True if path is collision-free, False otherwise
        """
        if len(waypoints) < 2:
            return True
        
        for i in range(len(waypoints) - 1):
            start = waypoints[i]
            end = waypoints[i + 1]
            
            # Calculate segment length
            segment_length = np.linalg.norm([end.x - start.x, end.y - start.y])
            
            # Number of collision check points along segment
            num_checks = max(2, int(segment_length / self.smoothing_resolution))
            
            # Check collision at intermediate points
            for j in range(num_checks + 1):
                t = j / num_checks
                x = start.x + t * (end.x - start.x)
                y = start.y + t * (end.y - start.y)
                
                if not self.warehouse_map.is_collision_free(x, y, self.robot_radius):
                    return False
        
        return True
    
    def adaptive_smooth_path(self, path: Path, max_deviation: float = 0.5) -> Optional[Path]:
        """
        Smooth a path with adaptive resolution based on curvature.
        
        This method uses higher resolution in areas with high curvature and
        lower resolution in straight sections.
        
        Args:
            path: Input path to smooth
            max_deviation: Maximum allowed deviation from original path (meters)
            
        Returns:
            Smoothed Path object if successful, None if smoothing fails
        """
        if len(path.waypoints) < 3:
            return self.smooth_path(path)  # Use regular smoothing for simple paths
        
        # Extract positions
        positions = np.array([[wp.x, wp.y] for wp in path.waypoints])
        
        # Calculate curvature at each point
        curvatures = self._calculate_curvature(positions)
        
        # Determine adaptive resolution based on curvature
        resolutions = []
        for curvature in curvatures:
            # Higher curvature -> higher resolution (more points)
            if curvature > 2.0:  # High curvature
                resolution = self.smoothing_resolution * 0.5
            elif curvature > 1.0:  # Medium curvature
                resolution = self.smoothing_resolution * 0.75
            else:  # Low curvature
                resolution = self.smoothing_resolution * 1.5
            resolutions.append(resolution)
        
        # Calculate total path length and adaptive number of points
        total_length = 0.0
        for i in range(len(positions) - 1):
            total_length += np.linalg.norm(positions[i+1] - positions[i])
        
        avg_resolution = np.mean(resolutions)
        num_points = max(10, int(total_length / avg_resolution))
        
        # Use regular smoothing with adaptive point count
        smoothed_path = self.smooth_path(path, num_points)
        
        # Validate deviation constraint
        if smoothed_path and self._check_deviation(path, smoothed_path, max_deviation):
            return smoothed_path
        else:
            print(f"Adaptive smoothing exceeded max deviation {max_deviation}m")
            return path
    
    def _calculate_curvature(self, positions: np.ndarray) -> List[float]:
        """
        Calculate curvature at each point along the path.
        
        Args:
            positions: Array of [x, y] positions
            
        Returns:
            List of curvature values (1/radius)
        """
        curvatures = []
        
        for i in range(len(positions)):
            if i == 0 or i == len(positions) - 1:
                curvatures.append(0.0)  # No curvature at endpoints
            else:
                # Use three consecutive points to estimate curvature
                p1 = positions[i-1]
                p2 = positions[i]
                p3 = positions[i+1]
                
                # Calculate vectors
                v1 = p2 - p1
                v2 = p3 - p2
                
                # Calculate angle change
                angle1 = np.arctan2(v1[1], v1[0])
                angle2 = np.arctan2(v2[1], v2[0])
                angle_diff = abs(angle2 - angle1)
                
                # Normalize angle difference to [0, pi]
                if angle_diff > np.pi:
                    angle_diff = 2 * np.pi - angle_diff
                
                # Calculate curvature (angle change per unit distance)
                distance = (np.linalg.norm(v1) + np.linalg.norm(v2)) / 2
                curvature = angle_diff / max(distance, 0.01)  # Avoid division by zero
                curvatures.append(curvature)
        
        return curvatures
    
    def _check_deviation(self, original_path: Path, smoothed_path: Path, 
                        max_deviation: float) -> bool:
        """
        Check if smoothed path deviates too much from original path.
        
        Args:
            original_path: Original discrete path
            smoothed_path: Smoothed path to validate
            max_deviation: Maximum allowed deviation (meters)
            
        Returns:
            True if deviation is within limits, False otherwise
        """
        # Sample points along original path
        original_positions = np.array([[wp.x, wp.y] for wp in original_path.waypoints])
        
        # Check deviation for each smoothed waypoint
        for smooth_wp in smoothed_path.waypoints:
            smooth_pos = np.array([smooth_wp.x, smooth_wp.y])
            
            # Find minimum distance to original path
            min_distance = float('inf')
            for orig_pos in original_positions:
                distance = np.linalg.norm(smooth_pos - orig_pos)
                min_distance = min(min_distance, distance)
            
            if min_distance > max_deviation:
                return False
        
        return True