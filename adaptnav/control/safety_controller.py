#!/usr/bin/env python3
"""
Safety controller for autonomous navigation.

This module implements the safety controller that enforces hard safety constraints
and collision avoidance. It monitors obstacles and overrides unsafe velocity commands
to ensure the robot maintains safe distances from dynamic obstacles.
"""

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

import numpy as np
import time
from typing import List, Optional, Tuple

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from custom_msgs.msg import ObstacleArray, Obstacle, SafetyStatus
from std_msgs.msg import Header


class SafetyController(Node):
    """
    Safety controller node for collision avoidance and velocity limiting.
    
    This node implements hard safety constraints that override any unsafe
    velocity commands from the navigation system. It maintains a collision
    zone around the robot and enforces velocity limits.
    
    Safety Rules:
    1. Collision Zone: Maintain 0.5m buffer around robot
    2. Velocity Limits: Max 1.0 m/s linear, 0.5 rad/s angular
    3. Emergency Stop: Stop if obstacle within 0.3m
    4. Clearance Time: Wait 1 second after obstacle clears before resuming
    5. Deceleration: Smooth deceleration when approaching obstacles
    """
    
    def __init__(self):
        super().__init__('safety_controller')
        
        # Parameters
        self.declare_parameter('collision_zone_radius', 0.5)
        self.declare_parameter('emergency_stop_radius', 0.3)
        self.declare_parameter('max_linear_velocity', 1.0)
        self.declare_parameter('max_angular_velocity', 0.5)
        self.declare_parameter('clearance_time', 1.0)
        self.declare_parameter('deceleration_factor', 0.5)
        self.declare_parameter('robot_radius', 0.3)
        
        self.collision_zone_radius = self.get_parameter('collision_zone_radius').value
        self.emergency_stop_radius = self.get_parameter('emergency_stop_radius').value
        self.max_linear_velocity = self.get_parameter('max_linear_velocity').value
        self.max_angular_velocity = self.get_parameter('max_angular_velocity').value
        self.clearance_time = self.get_parameter('clearance_time').value
        self.deceleration_factor = self.get_parameter('deceleration_factor').value
        self.robot_radius = self.get_parameter('robot_radius').value
        
        # State
        self.robot_position = np.array([0.0, 0.0])
        self.robot_velocity = np.array([0.0, 0.0])  # [linear, angular]
        self.obstacles: List[Obstacle] = []
        self.last_obstacle_time = 0.0
        self.collision_zone_clear_time = 0.0  # Time when collision zone became clear
        self.safety_state = "SAFE"
        self.previous_safety_state = "SAFE"
        self.velocity_scale = 1.0
        self.override_active = False
        self.closest_obstacle_distance = float('inf')
        self.clearance_timer_active = False
        
        # Callback group for concurrent execution
        self.callback_group = ReentrantCallbackGroup()
        
        # Subscribers
        self.cmd_vel_subscriber = self.create_subscription(
            Twist,
            '/cmd_vel_raw',
            self.cmd_vel_callback,
            10,
            callback_group=self.callback_group
        )
        
        self.odom_subscriber = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10,
            callback_group=self.callback_group
        )
        
        self.obstacles_subscriber = self.create_subscription(
            ObstacleArray,
            '/obstacles/detected',
            self.obstacles_callback,
            10,
            callback_group=self.callback_group
        )
        
        # Publishers
        self.cmd_vel_publisher = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )
        
        self.safety_status_publisher = self.create_publisher(
            SafetyStatus,
            '/safety_status',
            10
        )
        
        # Timer for safety monitoring
        self.safety_timer = self.create_timer(
            0.05,  # 20 Hz
            self.safety_monitor_callback,
            callback_group=self.callback_group
        )
        
        self.get_logger().info('Safety controller initialized')
        self.get_logger().info(f'Collision zone radius: {self.collision_zone_radius}m')
        self.get_logger().info(f'Emergency stop radius: {self.emergency_stop_radius}m')
        self.get_logger().info(f'Max velocities: {self.max_linear_velocity}m/s linear, {self.max_angular_velocity}rad/s angular')
    
    def odom_callback(self, msg: Odometry):
        """
        Handle odometry updates.
        
        Args:
            msg: Odometry message with robot pose and velocity
        """
        # Update robot position
        self.robot_position[0] = msg.pose.pose.position.x
        self.robot_position[1] = msg.pose.pose.position.y
        
        # Update robot velocity
        self.robot_velocity[0] = msg.twist.twist.linear.x
        self.robot_velocity[1] = msg.twist.twist.angular.z
    
    def obstacles_callback(self, msg: ObstacleArray):
        """
        Handle obstacle detection updates.
        
        Args:
            msg: ObstacleArray message with detected obstacles
        """
        self.obstacles = msg.obstacles
        
        # Update closest obstacle distance
        self.closest_obstacle_distance = self._get_closest_obstacle_distance()
        
        # Update last obstacle time if obstacles are present
        if self.obstacles:
            self.last_obstacle_time = time.time()
    
    def cmd_vel_callback(self, msg: Twist):
        """
        Handle incoming velocity commands and apply safety filtering.
        
        Args:
            msg: Twist message with commanded velocities
        """
        # Apply safety filtering
        safe_cmd = self._apply_safety_filter(msg)
        
        # Publish filtered command
        self.cmd_vel_publisher.publish(safe_cmd)
    
    def safety_monitor_callback(self):
        """
        Periodic safety monitoring and status publishing.
        """
        # Update safety state
        self._update_safety_state()
        
        # Publish safety status
        self._publish_safety_status()
    
    def _apply_safety_filter(self, cmd: Twist) -> Twist:
        """
        Apply safety filtering to velocity commands.
        
        Args:
            cmd: Input velocity command
            
        Returns:
            Safety-filtered velocity command
        """
        safe_cmd = Twist()
        
        # Start with original command
        linear_vel = cmd.linear.x
        angular_vel = cmd.angular.z
        
        # Apply velocity limits
        linear_vel = np.clip(linear_vel, -self.max_linear_velocity, self.max_linear_velocity)
        angular_vel = np.clip(angular_vel, -self.max_angular_velocity, self.max_angular_velocity)
        
        # Apply safety constraints based on obstacles
        if self.safety_state == "EMERGENCY_STOP":
            # Full stop
            linear_vel = 0.0
            angular_vel = 0.0
            self.velocity_scale = 0.0
            self.override_active = True
            
        elif self.safety_state == "CAUTION":
            # Scale down velocity based on obstacle proximity
            scale_factor = self._calculate_velocity_scale()
            linear_vel *= scale_factor
            angular_vel *= scale_factor
            self.velocity_scale = scale_factor
            self.override_active = scale_factor < 1.0
            
        else:  # SAFE
            self.velocity_scale = 1.0
            self.override_active = False
        
        # Check for collision prediction
        if self._predict_collision(linear_vel, angular_vel):
            # Override with safer command
            safe_linear, safe_angular = self._compute_safe_velocity(linear_vel, angular_vel)
            linear_vel = safe_linear
            angular_vel = safe_angular
            self.override_active = True
        
        # Set filtered velocities
        safe_cmd.linear.x = linear_vel
        safe_cmd.angular.z = angular_vel
        
        return safe_cmd
    
    def _compute_safe_velocity(self, commanded_linear: float, commanded_angular: float) -> Tuple[float, float]:
        """
        Compute safe velocity when collision is predicted.
        
        This method tries different velocity combinations to find the safest
        option that still makes progress toward the goal.
        
        Args:
            commanded_linear: Original commanded linear velocity
            commanded_angular: Original commanded angular velocity
            
        Returns:
            Tuple of (safe_linear_vel, safe_angular_vel)
        """
        # Try reducing linear velocity while keeping angular velocity
        for scale in [0.5, 0.25, 0.1, 0.0]:
            test_linear = commanded_linear * scale
            if not self._predict_collision(test_linear, commanded_angular):
                return test_linear, commanded_angular
        
        # Try reducing both velocities
        for scale in [0.5, 0.25, 0.1]:
            test_linear = commanded_linear * scale
            test_angular = commanded_angular * scale
            if not self._predict_collision(test_linear, test_angular):
                return test_linear, test_angular
        
        # Try pure rotation (might help avoid obstacle)
        if abs(commanded_angular) > 0.1:
            for scale in [0.5, 0.25]:
                test_angular = commanded_angular * scale
                if not self._predict_collision(0.0, test_angular):
                    return 0.0, test_angular
        
        # Last resort: full stop
        return 0.0, 0.0
    
    def _update_safety_state(self):
        """
        Update the current safety state based on obstacle proximity.
        """
        current_time = time.time()
        previous_state = self.safety_state
        
        # Check for emergency stop condition
        if self.closest_obstacle_distance <= self.emergency_stop_radius:
            self.safety_state = "EMERGENCY_STOP"
            self.clearance_timer_active = False
            return
        
        # Check for caution condition
        if self.closest_obstacle_distance <= self.collision_zone_radius:
            self.safety_state = "CAUTION"
            self.clearance_timer_active = False
            return
        
        # Collision zone is clear - handle clearance time enforcement
        if previous_state in ["EMERGENCY_STOP", "CAUTION"]:
            if not self.clearance_timer_active:
                # Start clearance timer
                self.collision_zone_clear_time = current_time
                self.clearance_timer_active = True
                self.get_logger().info(f"Collision zone clear, starting {self.clearance_time}s clearance timer")
            
            # Check if clearance time has passed
            time_since_clear = current_time - self.collision_zone_clear_time
            if time_since_clear >= self.clearance_time:
                self.safety_state = "SAFE"
                self.clearance_timer_active = False
                self.get_logger().info("Clearance time satisfied, transitioning to SAFE state")
            else:
                # Stay in previous state until clearance time passes
                self.safety_state = previous_state
                remaining_time = self.clearance_time - time_since_clear
                if int(remaining_time * 10) % 10 == 0:  # Log every 0.1 seconds
                    self.get_logger().debug(f"Waiting for clearance: {remaining_time:.1f}s remaining")
        else:
            # Already safe or transitioning from safe state
            self.safety_state = "SAFE"
            self.clearance_timer_active = False
    
    def _get_closest_obstacle_distance(self) -> float:
        """
        Calculate distance to closest obstacle.
        
        Returns:
            Distance to closest obstacle in meters
        """
        if not self.obstacles:
            return float('inf')
        
        min_distance = float('inf')
        
        for obstacle in self.obstacles:
            obstacle_pos = np.array([obstacle.position.x, obstacle.position.y])
            distance = np.linalg.norm(self.robot_position - obstacle_pos)
            
            # Account for robot and obstacle radii
            # Assume obstacle radius based on classification
            if obstacle.classification == "worker":
                obstacle_radius = 0.5
            elif obstacle.classification == "forklift":
                obstacle_radius = 1.0
            else:
                obstacle_radius = 0.5  # Default
            
            # Distance between surfaces (not centers)
            surface_distance = distance - self.robot_radius - obstacle_radius
            min_distance = min(min_distance, max(0.0, surface_distance))
        
        return min_distance
    
    def _calculate_velocity_scale(self) -> float:
        """
        Calculate velocity scaling factor based on obstacle proximity.
        
        Returns:
            Velocity scale factor [0.0, 1.0]
        """
        if self.closest_obstacle_distance >= self.collision_zone_radius:
            return 1.0
        
        if self.closest_obstacle_distance <= self.emergency_stop_radius:
            return 0.0
        
        # Linear scaling between emergency stop and collision zone
        scale_range = self.collision_zone_radius - self.emergency_stop_radius
        distance_above_emergency = self.closest_obstacle_distance - self.emergency_stop_radius
        
        return distance_above_emergency / scale_range
    
    def _predict_collision(self, linear_vel: float, angular_vel: float, 
                          prediction_time: float = 0.5) -> bool:
        """
        Predict if current velocity would lead to collision.
        
        Uses more sophisticated trajectory prediction considering both
        linear and angular velocity to predict robot path.
        
        Args:
            linear_vel: Commanded linear velocity
            angular_vel: Commanded angular velocity
            prediction_time: Time horizon for prediction (seconds)
            
        Returns:
            True if collision predicted, False otherwise
        """
        if not self.obstacles:
            return False
        
        # Predict robot trajectory
        robot_trajectory = self._predict_robot_trajectory(
            linear_vel, angular_vel, prediction_time
        )
        
        # Check collision with each obstacle
        for obstacle in self.obstacles:
            if self._check_trajectory_collision(obstacle, robot_trajectory, prediction_time):
                self.get_logger().warn(
                    f"Collision predicted with {obstacle.classification} "
                    f"at distance {self.closest_obstacle_distance:.2f}m"
                )
                return True
        
        return False
    
    def _predict_robot_trajectory(self, linear_vel: float, angular_vel: float, 
                                 prediction_time: float, dt: float = 0.1) -> List[np.ndarray]:
        """
        Predict robot trajectory considering differential drive kinematics.
        
        Args:
            linear_vel: Linear velocity (m/s)
            angular_vel: Angular velocity (rad/s)
            prediction_time: Time horizon (seconds)
            dt: Time step for trajectory sampling (seconds)
            
        Returns:
            List of predicted positions [x, y]
        """
        trajectory = []
        
        # Current robot state (assume facing along x-axis for simplicity)
        x, y = self.robot_position
        theta = 0.0  # Simplified - in real implementation, get from odometry
        
        # Predict trajectory using differential drive kinematics
        num_steps = int(prediction_time / dt)
        
        for i in range(num_steps + 1):
            trajectory.append(np.array([x, y]))
            
            if i < num_steps:  # Don't update on last iteration
                if abs(angular_vel) < 1e-6:
                    # Straight line motion
                    x += linear_vel * dt * np.cos(theta)
                    y += linear_vel * dt * np.sin(theta)
                else:
                    # Curved motion
                    radius = linear_vel / angular_vel
                    
                    # Update position using circular arc
                    x += radius * (np.sin(theta + angular_vel * dt) - np.sin(theta))
                    y += radius * (np.cos(theta) - np.cos(theta + angular_vel * dt))
                    theta += angular_vel * dt
        
        return trajectory
    
    def _check_trajectory_collision(self, obstacle: Obstacle, 
                                   robot_trajectory: List[np.ndarray], 
                                   prediction_time: float) -> bool:
        """
        Check if robot trajectory collides with obstacle trajectory.
        
        Args:
            obstacle: Obstacle to check collision with
            robot_trajectory: Predicted robot positions
            prediction_time: Time horizon for prediction
            
        Returns:
            True if collision detected, False otherwise
        """
        obstacle_pos = np.array([obstacle.position.x, obstacle.position.y])
        obstacle_vel = np.array([obstacle.velocity.x, obstacle.velocity.y])
        
        # Obstacle radius based on classification
        if obstacle.classification == "worker":
            obstacle_radius = 0.5
        elif obstacle.classification == "forklift":
            obstacle_radius = 1.0
        else:
            obstacle_radius = 0.5  # Default
        
        # Check collision at each trajectory point
        dt = prediction_time / (len(robot_trajectory) - 1) if len(robot_trajectory) > 1 else 0.1
        
        for i, robot_pos in enumerate(robot_trajectory):
            t = i * dt
            
            # Predict obstacle position at time t
            predicted_obstacle_pos = obstacle_pos + obstacle_vel * t
            
            # Calculate distance between robot and obstacle
            distance = np.linalg.norm(robot_pos - predicted_obstacle_pos)
            
            # Check collision (including safety margin)
            total_radius = self.robot_radius + obstacle_radius + 0.1  # 10cm safety margin
            
            if distance <= total_radius:
                return True
        
        return False
    
    def _calculate_time_until_clear(self) -> float:
        """
        Calculate estimated time until collision zone is clear.
        
        Returns:
            Time in seconds, 0.0 if already clear, -1.0 if unknown
        """
        current_time = time.time()
        
        if self.safety_state == "SAFE":
            return 0.0
        
        # If clearance timer is active, return remaining clearance time
        if self.clearance_timer_active:
            remaining_clearance = self.clearance_time - (current_time - self.collision_zone_clear_time)
            return max(0.0, remaining_clearance)
        
        if not self.obstacles:
            return 0.0
        
        # Simple estimation: assume obstacles continue at current velocity
        min_clear_time = float('inf')
        
        for obstacle in self.obstacles:
            obstacle_pos = np.array([obstacle.position.x, obstacle.position.y])
            obstacle_vel = np.array([obstacle.velocity.x, obstacle.velocity.y])
            
            # Distance from robot to obstacle
            distance = np.linalg.norm(self.robot_position - obstacle_pos)
            
            # If obstacle is moving away, calculate when it clears collision zone
            if np.dot(obstacle_pos - self.robot_position, obstacle_vel) > 0:
                # Obstacle moving away
                vel_magnitude = np.linalg.norm(obstacle_vel)
                if vel_magnitude > 0.1:  # Avoid division by very small numbers
                    clear_distance = self.collision_zone_radius + 0.1  # Small margin
                    time_to_clear = max(0.0, (clear_distance - distance) / vel_magnitude)
                    # Add clearance time requirement
                    total_time = time_to_clear + self.clearance_time
                    min_clear_time = min(min_clear_time, total_time)
            
        return min_clear_time if min_clear_time != float('inf') else -1.0
    
    def _publish_safety_status(self):
        """
        Publish current safety status.
        """
        status = SafetyStatus()
        status.header.stamp = self.get_clock().now().to_msg()
        status.header.frame_id = "base_link"
        
        status.state = self.safety_state
        status.closest_obstacle_distance = self.closest_obstacle_distance
        status.velocity_scale = self.velocity_scale
        status.override_active = self.override_active
        status.time_until_clear = self._calculate_time_until_clear()
        
        self.safety_status_publisher.publish(status)


def main(args=None):
    """Main entry point for the safety controller node."""
    rclpy.init(args=args)
    
    try:
        node = SafetyController()
        executor = MultiThreadedExecutor()
        executor.add_node(node)
        
        try:
            executor.spin()
        except KeyboardInterrupt:
            pass
        finally:
            executor.shutdown()
            node.destroy_node()
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()