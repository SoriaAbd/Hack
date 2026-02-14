"""
Navigation state machine for autonomous warehouse navigation.

This module implements the state machine that orchestrates the navigation
process, managing transitions between different navigation states based
on environmental conditions and system status.
"""

from enum import Enum
from typing import Optional, Dict, Any
import time
import numpy as np

from geometry_msgs.msg import Pose
from custom_msgs.msg import SafetyStatus, ObstacleArray
from nav_msgs.msg import Path as NavPath


class NavigationState(Enum):
    """
    Navigation states for the autonomous navigation system.
    
    States:
    - IDLE: System is inactive, waiting for goal
    - PLANNING: Computing path to goal
    - FOLLOWING_PATH: Following planned path to goal
    - AVOIDING_OBSTACLE: Actively avoiding dynamic obstacles
    - EMERGENCY_STOP: Safety system has taken control
    - GOAL_REACHED: Successfully reached the goal
    - PLANNING_FAILED: Path planning failed
    """
    IDLE = "IDLE"
    PLANNING = "PLANNING"
    FOLLOWING_PATH = "FOLLOWING_PATH"
    AVOIDING_OBSTACLE = "AVOIDING_OBSTACLE"
    EMERGENCY_STOP = "EMERGENCY_STOP"
    GOAL_REACHED = "GOAL_REACHED"
    PLANNING_FAILED = "PLANNING_FAILED"


class NavigationStateMachine:
    """
    State machine for navigation control.
    
    This class manages the navigation state transitions and provides
    reasoning for state changes. It coordinates between path planning,
    obstacle avoidance, and safety systems.
    """
    
    def __init__(self, goal_tolerance: float = 0.2, velocity_tolerance: float = 0.1):
        """
        Initialize the navigation state machine.
        
        Args:
            goal_tolerance: Distance tolerance for goal reached (meters)
            velocity_tolerance: Velocity tolerance for goal reached (m/s)
        """
        self.current_state = NavigationState.IDLE
        self.previous_state = NavigationState.IDLE
        self.state_entry_time = time.time()
        self.goal_tolerance = goal_tolerance
        self.velocity_tolerance = velocity_tolerance
        
        # State data
        self.current_goal: Optional[Pose] = None
        self.current_path: Optional[NavPath] = None
        self.robot_position = np.array([0.0, 0.0])
        self.robot_velocity = 0.0
        self.safety_status: Optional[SafetyStatus] = None
        self.obstacles: Optional[ObstacleArray] = None
        self.planning_attempts = 0
        self.max_planning_attempts = 3
        self.last_replan_time = 0.0
        self.replan_cooldown = 1.0  # Minimum time between replanning attempts
        
        # State transition history for debugging
        self.state_history = []
        self.reasoning = "System initialized"
    
    def update(self, robot_position: np.ndarray, robot_velocity: float,
               safety_status: SafetyStatus, obstacles: ObstacleArray,
               current_path: Optional[NavPath] = None) -> NavigationState:
        """
        Update the state machine with current system status.
        
        Args:
            robot_position: Current robot position [x, y]
            robot_velocity: Current robot linear velocity
            safety_status: Current safety system status
            obstacles: Current obstacle detections
            current_path: Current planned path (if available)
            
        Returns:
            Updated navigation state
        """
        # Update internal state
        self.robot_position = robot_position
        self.robot_velocity = robot_velocity
        self.safety_status = safety_status
        self.obstacles = obstacles
        if current_path is not None:
            self.current_path = current_path
        
        # Determine new state
        new_state = self._determine_next_state()
        
        # Handle state transition
        if new_state != self.current_state:
            self._transition_to_state(new_state)
        
        return self.current_state
    
    def set_goal(self, goal: Pose):
        """
        Set a new navigation goal.
        
        Args:
            goal: Target pose to navigate to
        """
        self.current_goal = goal
        self.planning_attempts = 0
        
        if self.current_state == NavigationState.IDLE:
            self._transition_to_state(NavigationState.PLANNING)
        elif self.current_state in [NavigationState.FOLLOWING_PATH, 
                                   NavigationState.AVOIDING_OBSTACLE,
                                   NavigationState.GOAL_REACHED]:
            # New goal while navigating - start replanning
            self._transition_to_state(NavigationState.PLANNING)
    
    def cancel_goal(self):
        """Cancel the current navigation goal."""
        self.current_goal = None
        self.current_path = None
        self._transition_to_state(NavigationState.IDLE)
    
    def path_planning_succeeded(self, path: NavPath):
        """
        Notify that path planning succeeded.
        
        Args:
            path: The computed path
        """
        if self.current_state == NavigationState.PLANNING:
            self.current_path = path
            self.planning_attempts = 0
            self._transition_to_state(NavigationState.FOLLOWING_PATH)
    
    def path_planning_failed(self):
        """Notify that path planning failed."""
        if self.current_state == NavigationState.PLANNING:
            self.planning_attempts += 1
            if self.planning_attempts >= self.max_planning_attempts:
                self._transition_to_state(NavigationState.PLANNING_FAILED)
            else:
                # Stay in planning state for retry
                self.reasoning = f"Planning failed, attempt {self.planning_attempts}/{self.max_planning_attempts}"
    
    def should_replan(self) -> bool:
        """
        Check if replanning should be triggered.
        
        Returns:
            True if replanning is needed, False otherwise
        """
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_replan_time < self.replan_cooldown:
            return False
        
        # Only replan when following path or avoiding obstacles
        if self.current_state not in [NavigationState.FOLLOWING_PATH, 
                                     NavigationState.AVOIDING_OBSTACLE]:
            return False
        
        # Check if path is blocked by obstacles
        if self._is_path_blocked():
            self.last_replan_time = current_time
            return True
        
        return False
    
    def get_state_info(self) -> Dict[str, Any]:
        """
        Get current state information for debugging/visualization.
        
        Returns:
            Dictionary with state information
        """
        return {
            'current_state': self.current_state.value,
            'previous_state': self.previous_state.value,
            'time_in_state': time.time() - self.state_entry_time,
            'reasoning': self.reasoning,
            'goal_distance': self._calculate_goal_distance(),
            'planning_attempts': self.planning_attempts,
            'has_path': self.current_path is not None,
            'safety_state': self.safety_status.state if self.safety_status else "UNKNOWN"
        }
    
    def _determine_next_state(self) -> NavigationState:
        """
        Determine the next state based on current conditions.
        
        Returns:
            Next navigation state
        """
        # Safety override - emergency stop takes precedence
        if (self.safety_status and 
            self.safety_status.state == "EMERGENCY_STOP"):
            self.reasoning = "Safety system emergency stop active"
            return NavigationState.EMERGENCY_STOP
        
        # Handle current state transitions
        if self.current_state == NavigationState.IDLE:
            if self.current_goal is not None:
                self.reasoning = "Goal received, starting planning"
                return NavigationState.PLANNING
        
        elif self.current_state == NavigationState.PLANNING:
            # Stay in planning until explicitly transitioned by planning result
            return NavigationState.PLANNING
        
        elif self.current_state == NavigationState.PLANNING_FAILED:
            if self.current_goal is None:
                self.reasoning = "Goal cancelled after planning failure"
                return NavigationState.IDLE
            # Stay in failed state until new goal or manual reset
            return NavigationState.PLANNING_FAILED
        
        elif self.current_state == NavigationState.FOLLOWING_PATH:
            # Check if goal reached
            if self._is_goal_reached():
                self.reasoning = "Goal reached successfully"
                return NavigationState.GOAL_REACHED
            
            # Check if obstacles require avoidance
            if self._should_avoid_obstacles():
                self.reasoning = "Obstacles detected, switching to avoidance mode"
                return NavigationState.AVOIDING_OBSTACLE
            
            # Continue following path
            self.reasoning = "Following planned path"
            return NavigationState.FOLLOWING_PATH
        
        elif self.current_state == NavigationState.AVOIDING_OBSTACLE:
            # Check if goal reached
            if self._is_goal_reached():
                self.reasoning = "Goal reached during obstacle avoidance"
                return NavigationState.GOAL_REACHED
            
            # Check if obstacles cleared and can return to path following
            if not self._should_avoid_obstacles():
                self.reasoning = "Obstacles cleared, returning to path following"
                return NavigationState.FOLLOWING_PATH
            
            # Continue avoiding obstacles
            self.reasoning = "Actively avoiding obstacles"
            return NavigationState.AVOIDING_OBSTACLE
        
        elif self.current_state == NavigationState.EMERGENCY_STOP:
            # Check if safety system allows resuming
            if (self.safety_status and 
                self.safety_status.state in ["SAFE", "CAUTION"]):
                if self.current_path is not None:
                    self.reasoning = "Safety cleared, resuming navigation"
                    return NavigationState.FOLLOWING_PATH
                else:
                    self.reasoning = "Safety cleared, but no path available"
                    return NavigationState.PLANNING
            
            # Stay in emergency stop
            self.reasoning = "Waiting for safety system clearance"
            return NavigationState.EMERGENCY_STOP
        
        elif self.current_state == NavigationState.GOAL_REACHED:
            if self.current_goal is None:
                self.reasoning = "Goal completed and cleared"
                return NavigationState.IDLE
            
            # Check if robot moved away from goal (new goal set)
            if not self._is_goal_reached():
                self.reasoning = "New goal set, starting planning"
                return NavigationState.PLANNING
            
            # Stay at goal
            self.reasoning = "Remaining at goal position"
            return NavigationState.GOAL_REACHED
        
        # Default: stay in current state
        return self.current_state
    
    def _transition_to_state(self, new_state: NavigationState):
        """
        Handle state transition.
        
        Args:
            new_state: State to transition to
        """
        if new_state != self.current_state:
            # Record transition
            self.state_history.append({
                'from_state': self.current_state.value,
                'to_state': new_state.value,
                'timestamp': time.time(),
                'reasoning': self.reasoning
            })
            
            # Update state
            self.previous_state = self.current_state
            self.current_state = new_state
            self.state_entry_time = time.time()
    
    def _is_goal_reached(self) -> bool:
        """
        Check if the robot has reached the current goal.
        
        Returns:
            True if goal is reached, False otherwise
        """
        if self.current_goal is None:
            return False
        
        # Calculate distance to goal
        goal_pos = np.array([
            self.current_goal.position.x,
            self.current_goal.position.y
        ])
        distance = np.linalg.norm(self.robot_position - goal_pos)
        
        # Check position and velocity tolerances
        position_ok = distance <= self.goal_tolerance
        velocity_ok = abs(self.robot_velocity) <= self.velocity_tolerance
        
        return position_ok and velocity_ok
    
    def _calculate_goal_distance(self) -> float:
        """
        Calculate distance to current goal.
        
        Returns:
            Distance to goal in meters, or -1 if no goal
        """
        if self.current_goal is None:
            return -1.0
        
        goal_pos = np.array([
            self.current_goal.position.x,
            self.current_goal.position.y
        ])
        return np.linalg.norm(self.robot_position - goal_pos)
    
    def _should_avoid_obstacles(self) -> bool:
        """
        Check if obstacle avoidance mode should be activated.
        
        Returns:
            True if obstacles require active avoidance, False otherwise
        """
        if not self.obstacles or not self.obstacles.obstacles:
            return False
        
        # Check if any obstacles are close to the robot or planned path
        for obstacle in self.obstacles.obstacles:
            obstacle_pos = np.array([obstacle.position.x, obstacle.position.y])
            distance = np.linalg.norm(self.robot_position - obstacle_pos)
            
            # If obstacle is very close, activate avoidance
            if distance <= 1.0:  # 1 meter threshold
                return True
        
        # Check if obstacles are blocking the planned path
        return self._is_path_blocked()
    
    def _is_path_blocked(self) -> bool:
        """
        Check if the current path is blocked by obstacles.
        
        Returns:
            True if path is blocked, False otherwise
        """
        if not self.current_path or not self.obstacles:
            return False
        
        # Check if any obstacles are close to path waypoints
        for pose in self.current_path.poses:
            waypoint_pos = np.array([pose.pose.position.x, pose.pose.position.y])
            
            for obstacle in self.obstacles.obstacles:
                obstacle_pos = np.array([obstacle.position.x, obstacle.position.y])
                distance = np.linalg.norm(waypoint_pos - obstacle_pos)
                
                # Consider path blocked if obstacle is within 0.5m of any waypoint
                if distance <= 0.5:
                    return True
        
        return False