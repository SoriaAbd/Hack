"""
Hybrid navigation logic combining global planning with local PPO-based navigation.

This module implements the hybrid navigation approach that uses global path planning
for strategic route planning and PPO-based reinforcement learning for local
obstacle avoidance and path following.
"""

import numpy as np
from typing import Optional, Tuple, List
import time

from geometry_msgs.msg import Twist
from nav_msgs.msg import Path as NavPath
from custom_msgs.msg import ObstacleArray, SafetyStatus

from ..core.path import Path, Waypoint
from .navigation_state_machine import NavigationState


class HybridNavigator:
    """
    Hybrid navigation system combining global planning with local RL-based control.
    
    This class orchestrates between:
    1. Global path planner (A*) for strategic route planning
    2. PPO agent for local navigation and obstacle avoidance
    3. Safety controller for hard constraints
    
    The hybrid approach allows the system to follow global plans while
    adapting locally to dynamic obstacles using learned behaviors.
    """
    
    def __init__(self, lookahead_distance: float = 2.0, 
                 waypoint_tolerance: float = 0.5,
                 max_linear_velocity: float = 1.0,
                 max_angular_velocity: float = 0.5):
        """
        Initialize the hybrid navigator.
        
        Args:
            lookahead_distance: Distance ahead to look for waypoints (meters)
            waypoint_tolerance: Distance tolerance for waypoint reaching (meters)
            max_linear_velocity: Maximum linear velocity (m/s)
            max_angular_velocity: Maximum angular velocity (rad/s)
        """
        self.lookahead_distance = lookahead_distance
        self.waypoint_tolerance = waypoint_tolerance
        self.max_linear_velocity = max_linear_velocity
        self.max_angular_velocity = max_angular_velocity
        
        # Navigation state
        self.current_path: Optional[Path] = None
        self.current_waypoint_index = 0
        self.robot_position = np.array([0.0, 0.0])
        self.robot_orientation = 0.0
        self.robot_velocity = np.array([0.0, 0.0])  # [linear, angular]
        
        # PPO integration (placeholder - would be actual PPO agent in full implementation)
        self.ppo_agent = None  # Would be loaded PPO model
        self.use_ppo = False   # Enable when PPO agent is available
        
        # Performance tracking
        self.last_waypoint_time = time.time()
        self.waypoints_reached = 0
        self.total_distance_traveled = 0.0
        self.last_position = np.array([0.0, 0.0])
    
    def update_robot_state(self, position: np.ndarray, orientation: float, 
                          velocity: np.ndarray):
        """
        Update robot state information.
        
        Args:
            position: Robot position [x, y]
            orientation: Robot orientation (radians)
            velocity: Robot velocity [linear, angular]
        """
        # Update distance traveled
        if hasattr(self, 'last_position'):
            distance_delta = np.linalg.norm(position - self.last_position)
            self.total_distance_traveled += distance_delta
        
        self.robot_position = position
        self.robot_orientation = orientation
        self.robot_velocity = velocity
        self.last_position = position.copy()
    
    def set_path(self, path: NavPath):
        """
        Set a new global path to follow.
        
        Args:
            path: ROS nav_msgs/Path message
        """
        if len(path.poses) == 0:
            self.current_path = None
            return
        
        # Convert ROS path to internal format
        waypoints = []
        for pose_stamped in path.poses:
            pose = pose_stamped.pose
            x = pose.position.x
            y = pose.position.y
            
            # Extract orientation from quaternion (simplified)
            qz = pose.orientation.z
            qw = pose.orientation.w
            theta = 2.0 * np.arctan2(qz, qw)
            
            waypoints.append(Waypoint(x, y, theta))
        
        self.current_path = Path(waypoints, time.time())
        self.current_waypoint_index = 0
        self.waypoints_reached = 0
        
        print(f"New path set with {len(waypoints)} waypoints, total length: {self.current_path.total_length:.2f}m")
    
    def compute_navigation_command(self, navigation_state: NavigationState,
                                  obstacles: ObstacleArray,
                                  safety_status: SafetyStatus) -> Twist:
        """
        Compute navigation command using hybrid approach.
        
        Args:
            navigation_state: Current navigation state
            obstacles: Detected obstacles
            safety_status: Safety system status
            
        Returns:
            Twist command for robot control
        """
        cmd = Twist()
        
        # Handle different navigation states
        if navigation_state in [NavigationState.IDLE, NavigationState.PLANNING,
                               NavigationState.EMERGENCY_STOP, NavigationState.GOAL_REACHED,
                               NavigationState.PLANNING_FAILED]:
            return cmd  # Zero velocity
        
        if self.current_path is None:
            return cmd  # No path to follow
        
        # Determine navigation mode based on obstacles and state
        if navigation_state == NavigationState.AVOIDING_OBSTACLE and self.use_ppo:
            # Use PPO agent for obstacle avoidance
            return self._compute_ppo_command(obstacles, safety_status)
        else:
            # Use path following with obstacle awareness
            return self._compute_path_following_command(obstacles, safety_status)
    
    def _compute_path_following_command(self, obstacles: ObstacleArray,
                                       safety_status: SafetyStatus) -> Twist:
        """
        Compute path following command with obstacle awareness.
        
        This method implements pure pursuit path following with dynamic
        obstacle avoidance using potential fields.
        
        Args:
            obstacles: Detected obstacles
            safety_status: Safety system status
            
        Returns:
            Twist command for path following
        """
        cmd = Twist()
        
        # Get target waypoint using lookahead
        target_waypoint = self._get_lookahead_waypoint()
        if target_waypoint is None:
            return cmd
        
        # Calculate base command for path following
        base_linear, base_angular = self._pure_pursuit_control(target_waypoint)
        
        # Apply obstacle avoidance using potential fields
        if obstacles and len(obstacles.obstacles) > 0:
            avoidance_linear, avoidance_angular = self._compute_obstacle_avoidance(
                obstacles, base_linear, base_angular
            )
            
            # Blend path following with obstacle avoidance
            blend_factor = self._compute_blend_factor(obstacles)
            
            final_linear = (1.0 - blend_factor) * base_linear + blend_factor * avoidance_linear
            final_angular = (1.0 - blend_factor) * base_angular + blend_factor * avoidance_angular
        else:
            final_linear = base_linear
            final_angular = base_angular
        
        # Apply velocity limits
        cmd.linear.x = np.clip(final_linear, -self.max_linear_velocity, self.max_linear_velocity)
        cmd.angular.z = np.clip(final_angular, -self.max_angular_velocity, self.max_angular_velocity)
        
        return cmd
    
    def _compute_ppo_command(self, obstacles: ObstacleArray,
                            safety_status: SafetyStatus) -> Twist:
        """
        Compute navigation command using PPO agent.
        
        This is a placeholder for PPO integration. In the full implementation,
        this would construct observations and query the trained PPO model.
        
        Args:
            obstacles: Detected obstacles
            safety_status: Safety system status
            
        Returns:
            Twist command from PPO agent
        """
        cmd = Twist()
        
        if not self.use_ppo or self.ppo_agent is None:
            # Fallback to path following
            return self._compute_path_following_command(obstacles, safety_status)
        
        # TODO: Implement PPO observation construction and inference
        # observation = self._construct_ppo_observation(obstacles)
        # action = self.ppo_agent.predict(observation)
        # cmd.linear.x = action[0]
        # cmd.angular.z = action[1]
        
        # For now, use path following as placeholder
        return self._compute_path_following_command(obstacles, safety_status)
    
    def _get_lookahead_waypoint(self) -> Optional[Waypoint]:
        """
        Get the lookahead waypoint for path following.
        
        Returns:
            Target waypoint at lookahead distance, or None if path complete
        """
        if self.current_path is None:
            return None
        
        # Update current waypoint index based on robot position
        self._update_current_waypoint()
        
        # Use lookahead to find target waypoint
        try:
            target_waypoint = self.current_path.get_lookahead_point(
                self.robot_position, self.lookahead_distance
            )
            return target_waypoint
        except (ValueError, IndexError):
            # Path complete or error
            return None
    
    def _update_current_waypoint(self):
        """Update the current waypoint index based on robot position."""
        if self.current_path is None:
            return
        
        # Check if we've reached the current waypoint
        while self.current_waypoint_index < len(self.current_path.waypoints):
            waypoint = self.current_path.waypoints[self.current_waypoint_index]
            distance = waypoint.distance_to(self.robot_position)
            
            if distance <= self.waypoint_tolerance:
                self.current_waypoint_index += 1
                self.waypoints_reached += 1
                self.last_waypoint_time = time.time()
                print(f"Reached waypoint {self.current_waypoint_index}/{len(self.current_path.waypoints)}")
            else:
                break
    
    def _pure_pursuit_control(self, target_waypoint: Waypoint) -> Tuple[float, float]:
        """
        Compute pure pursuit control commands.
        
        Args:
            target_waypoint: Target waypoint to pursue
            
        Returns:
            Tuple of (linear_velocity, angular_velocity)
        """
        # Calculate distance and angle to target
        target_pos = target_waypoint.position()
        distance = np.linalg.norm(target_pos - self.robot_position)
        
        # Calculate angle to target
        direction = target_pos - self.robot_position
        target_angle = np.arctan2(direction[1], direction[0])
        
        # Calculate angle error
        angle_error = target_angle - self.robot_orientation
        
        # Normalize angle error to [-pi, pi]
        while angle_error > np.pi:
            angle_error -= 2 * np.pi
        while angle_error < -np.pi:
            angle_error += 2 * np.pi
        
        # Pure pursuit control law
        linear_velocity = min(self.max_linear_velocity, distance * 0.5)  # Proportional to distance
        
        # Angular velocity based on pure pursuit geometry
        if distance > 0.1:  # Avoid division by zero
            curvature = 2.0 * np.sin(angle_error) / distance
            angular_velocity = linear_velocity * curvature
        else:
            angular_velocity = 0.0
        
        return linear_velocity, angular_velocity
    
    def _compute_obstacle_avoidance(self, obstacles: ObstacleArray,
                                   base_linear: float, base_angular: float) -> Tuple[float, float]:
        """
        Compute obstacle avoidance using potential fields.
        
        Args:
            obstacles: Detected obstacles
            base_linear: Base linear velocity from path following
            base_angular: Base angular velocity from path following
            
        Returns:
            Tuple of (avoidance_linear, avoidance_angular)
        """
        # Repulsive force from obstacles
        repulsive_force = np.array([0.0, 0.0])
        
        for obstacle in obstacles.obstacles:
            obstacle_pos = np.array([obstacle.position.x, obstacle.position.y])
            distance = np.linalg.norm(self.robot_position - obstacle_pos)
            
            if distance < 2.0:  # Only consider nearby obstacles
                # Repulsive force inversely proportional to distance squared
                direction = self.robot_position - obstacle_pos
                if distance > 0.1:  # Avoid division by zero
                    direction = direction / distance  # Normalize
                    force_magnitude = min(1.0, 1.0 / (distance * distance))
                    repulsive_force += force_magnitude * direction
        
        # Convert repulsive force to velocity commands
        if np.linalg.norm(repulsive_force) > 0.1:
            # Reduce forward velocity when avoiding obstacles
            avoidance_linear = base_linear * 0.5
            
            # Add angular velocity to steer away from obstacles
            avoidance_angle = np.arctan2(repulsive_force[1], repulsive_force[0])
            angle_diff = avoidance_angle - self.robot_orientation
            
            # Normalize angle
            while angle_diff > np.pi:
                angle_diff -= 2 * np.pi
            while angle_diff < -np.pi:
                angle_diff += 2 * np.pi
            
            avoidance_angular = base_angular + np.clip(angle_diff * 0.5, -0.3, 0.3)
        else:
            avoidance_linear = base_linear
            avoidance_angular = base_angular
        
        return avoidance_linear, avoidance_angular
    
    def _compute_blend_factor(self, obstacles: ObstacleArray) -> float:
        """
        Compute blending factor between path following and obstacle avoidance.
        
        Args:
            obstacles: Detected obstacles
            
        Returns:
            Blend factor [0, 1] where 0=pure path following, 1=pure avoidance
        """
        if not obstacles or len(obstacles.obstacles) == 0:
            return 0.0
        
        # Find closest obstacle
        min_distance = float('inf')
        for obstacle in obstacles.obstacles:
            obstacle_pos = np.array([obstacle.position.x, obstacle.position.y])
            distance = np.linalg.norm(self.robot_position - obstacle_pos)
            min_distance = min(min_distance, distance)
        
        # Blend factor based on closest obstacle distance
        if min_distance > 2.0:
            return 0.0  # Pure path following
        elif min_distance < 0.5:
            return 1.0  # Pure obstacle avoidance
        else:
            # Linear interpolation
            return (2.0 - min_distance) / 1.5
    
    def is_path_complete(self) -> bool:
        """
        Check if the current path has been completed.
        
        Returns:
            True if path is complete, False otherwise
        """
        if self.current_path is None:
            return True
        
        return self.current_waypoint_index >= len(self.current_path.waypoints)
    
    def get_progress_info(self) -> dict:
        """
        Get navigation progress information.
        
        Returns:
            Dictionary with progress metrics
        """
        if self.current_path is None:
            return {
                'progress_percentage': 0.0,
                'waypoints_reached': 0,
                'total_waypoints': 0,
                'distance_remaining': 0.0,
                'distance_traveled': self.total_distance_traveled
            }
        
        progress_percentage = (self.waypoints_reached / len(self.current_path.waypoints)) * 100.0
        
        # Calculate remaining distance (approximate)
        remaining_distance = 0.0
        if self.current_waypoint_index < len(self.current_path.waypoints):
            # Distance to next waypoint
            next_waypoint = self.current_path.waypoints[self.current_waypoint_index]
            remaining_distance = next_waypoint.distance_to(self.robot_position)
            
            # Add distances between remaining waypoints
            for i in range(self.current_waypoint_index, len(self.current_path.waypoints) - 1):
                wp1 = self.current_path.waypoints[i]
                wp2 = self.current_path.waypoints[i + 1]
                remaining_distance += np.linalg.norm(wp2.position() - wp1.position())
        
        return {
            'progress_percentage': progress_percentage,
            'waypoints_reached': self.waypoints_reached,
            'total_waypoints': len(self.current_path.waypoints),
            'distance_remaining': remaining_distance,
            'distance_traveled': self.total_distance_traveled,
            'current_waypoint_index': self.current_waypoint_index
        }