#!/usr/bin/env python3
"""
Navigation controller node for autonomous warehouse navigation.

This node orchestrates the complete navigation system, coordinating between
global path planning, local obstacle avoidance, and safety systems. It
implements the high-level navigation logic and state management.
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

import numpy as np
import time
from typing import Optional

# ROS 2 messages and actions
from geometry_msgs.msg import Twist, Pose, PoseStamped
from nav_msgs.msg import Odometry, Path as NavPath
from nav2_msgs.action import NavigateToPose
from nav2_msgs.srv import ComputePathToPose
from custom_msgs.msg import ObstacleArray, SafetyStatus, NavigationState as NavStateMsg
from std_msgs.msg import Header

# Internal modules
from .navigation_state_machine import NavigationStateMachine, NavigationState
from .hybrid_navigator import HybridNavigator


class NavigationController(Node):
    """
    Main navigation controller node.
    
    This node provides:
    - Action server for nav2_msgs/NavigateToPose
    - Orchestration of global planner, PPO agent, and safety controller
    - State machine management for navigation states
    - Replanning logic when paths become blocked
    - Status publishing for dashboard and monitoring
    """
    
    def __init__(self):
        super().__init__('navigation_controller')
        
        # Parameters
        self.declare_parameter('goal_tolerance', 0.2)
        self.declare_parameter('velocity_tolerance', 0.1)
        self.declare_parameter('control_frequency', 10.0)
        self.declare_parameter('planning_timeout', 5.0)
        self.declare_parameter('max_replanning_attempts', 3)
        
        self.goal_tolerance = self.get_parameter('goal_tolerance').value
        self.velocity_tolerance = self.get_parameter('velocity_tolerance').value
        self.control_frequency = self.get_parameter('control_frequency').value
        self.planning_timeout = self.get_parameter('planning_timeout').value
        self.max_replanning_attempts = self.get_parameter('max_replanning_attempts').value
        
        # State
        self.robot_position = np.array([0.0, 0.0])
        self.robot_velocity = 0.0
        self.current_goal: Optional[Pose] = None
        self.current_path: Optional[NavPath] = None
        self.safety_status: Optional[SafetyStatus] = None
        self.obstacles: Optional[ObstacleArray] = None
        self.replanning_attempts = 0
        
        # State machine
        self.state_machine = NavigationStateMachine(
            goal_tolerance=self.goal_tolerance,
            velocity_tolerance=self.velocity_tolerance
        )
        
        # Hybrid navigator
        self.hybrid_navigator = HybridNavigator(
            lookahead_distance=2.0,
            waypoint_tolerance=0.5,
            max_linear_velocity=1.0,
            max_angular_velocity=0.5
        )
        
        # Callback groups
        self.callback_group = ReentrantCallbackGroup()
        
        # Action server for navigation goals
        self.navigate_action_server = ActionServer(
            self,
            NavigateToPose,
            '/navigate_to_pose',
            self.navigate_callback,
            callback_group=self.callback_group
        )
        
        # Action client for path planning
        self.path_planning_client = ActionClient(
            self,
            ComputePathToPose,
            '/compute_path_to_pose'
        )
        
        # Subscribers
        self.odom_subscriber = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10,
            callback_group=self.callback_group
        )
        
        self.global_plan_subscriber = self.create_subscription(
            NavPath,
            '/global_plan',
            self.global_plan_callback,
            10,
            callback_group=self.callback_group
        )
        
        self.safety_status_subscriber = self.create_subscription(
            SafetyStatus,
            '/safety_status',
            self.safety_status_callback,
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
            '/cmd_vel_raw',
            10
        )
        
        self.navigation_state_publisher = self.create_publisher(
            NavStateMsg,
            '/navigation_state',
            10
        )
        
        # Service client for path planning
        self.path_planning_service = self.create_client(
            ComputePathToPose,
            '/compute_path_to_pose'
        )
        
        # Control timer
        self.control_timer = self.create_timer(
            1.0 / self.control_frequency,
            self.control_loop_callback,
            callback_group=self.callback_group
        )
        
        # Current navigation goal handle
        self.current_goal_handle = None
        self.planning_future = None
        
        self.get_logger().info('Navigation controller initialized')
        self.get_logger().info(f'Control frequency: {self.control_frequency} Hz')
        self.get_logger().info(f'Goal tolerance: {self.goal_tolerance}m')
    
    def odom_callback(self, msg: Odometry):
        """Handle odometry updates."""
        self.robot_position[0] = msg.pose.pose.position.x
        self.robot_position[1] = msg.pose.pose.position.y
        self.robot_velocity = msg.twist.twist.linear.x
        
        # Update hybrid navigator with robot state
        # Extract orientation from quaternion (simplified)
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        orientation = 2.0 * np.arctan2(qz, qw)
        
        velocity = np.array([msg.twist.twist.linear.x, msg.twist.twist.angular.z])
        self.hybrid_navigator.update_robot_state(self.robot_position, orientation, velocity)
    
    def global_plan_callback(self, msg: NavPath):
        """Handle global plan updates."""
        self.current_path = msg
        
        # Update hybrid navigator with new path
        self.hybrid_navigator.set_path(msg)
        
        # Notify state machine of successful planning
        if (self.state_machine.current_state == NavigationState.PLANNING and
            len(msg.poses) > 0):
            self.state_machine.path_planning_succeeded(msg)
            self.replanning_attempts = 0
            self.get_logger().info(f'Path received with {len(msg.poses)} waypoints')
    
    def safety_status_callback(self, msg: SafetyStatus):
        """Handle safety status updates."""
        self.safety_status = msg
    
    def obstacles_callback(self, msg: ObstacleArray):
        """Handle obstacle detection updates."""
        self.obstacles = msg
    
    def navigate_callback(self, goal_handle):
        """
        Handle navigation action requests.
        
        Args:
            goal_handle: Action goal handle for NavigateToPose
        """
        self.get_logger().info('Navigation goal received')
        
        # Cancel previous goal if active
        if self.current_goal_handle is not None:
            self.current_goal_handle.abort()
        
        # Accept new goal
        goal_handle.accept()
        self.current_goal_handle = goal_handle
        
        # Extract goal pose
        goal_pose = goal_handle.request.pose.pose
        self.current_goal = goal_pose
        
        # Set goal in state machine
        self.state_machine.set_goal(goal_pose)
        
        self.get_logger().info(
            f'Navigating to goal: ({goal_pose.position.x:.2f}, {goal_pose.position.y:.2f})'
        )
        
        # Start navigation execution
        return self.execute_navigation(goal_handle)
    
    def execute_navigation(self, goal_handle):
        """
        Execute navigation to goal.
        
        Args:
            goal_handle: Action goal handle
        """
        feedback_msg = NavigateToPose.Feedback()
        
        while rclpy.ok():
            # Check if goal was cancelled
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.state_machine.cancel_goal()
                self.get_logger().info('Navigation goal cancelled')
                return NavigateToPose.Result()
            
            # Update state machine
            current_state = self.state_machine.update(
                self.robot_position,
                self.robot_velocity,
                self.safety_status or SafetyStatus(),
                self.obstacles or ObstacleArray(),
                self.current_path
            )
            
            # Handle state-specific actions
            if current_state == NavigationState.PLANNING:
                self._handle_planning_state()
            elif current_state == NavigationState.GOAL_REACHED:
                # Success!
                result = NavigateToPose.Result()
                goal_handle.succeed()
                self.current_goal_handle = None
                self.get_logger().info('Navigation goal reached successfully')
                return result
            elif current_state == NavigationState.PLANNING_FAILED:
                # Planning failed
                result = NavigateToPose.Result()
                goal_handle.abort()
                self.current_goal_handle = None
                self.get_logger().error('Navigation failed: planning failed')
                return result
            
            # Check for replanning
            if self.state_machine.should_replan():
                self._trigger_replanning()
            
            # Publish feedback
            feedback_msg.current_pose.header.stamp = self.get_clock().now().to_msg()
            feedback_msg.current_pose.header.frame_id = 'map'
            feedback_msg.current_pose.pose.position.x = self.robot_position[0]
            feedback_msg.current_pose.pose.position.y = self.robot_position[1]
            feedback_msg.distance_remaining = self.state_machine._calculate_goal_distance()
            
            goal_handle.publish_feedback(feedback_msg)
            
            # Sleep for control loop timing
            time.sleep(1.0 / self.control_frequency)
        
        return NavigateToPose.Result()
    
    def control_loop_callback(self):
        """Main control loop for navigation."""
        if self.current_goal is None:
            return
        
        # Update state machine
        current_state = self.state_machine.update(
            self.robot_position,
            self.robot_velocity,
            self.safety_status or SafetyStatus(),
            self.obstacles or ObstacleArray(),
            self.current_path
        )
        
        # Generate velocity commands based on current state
        cmd_vel = self._generate_velocity_command(current_state)
        
        # Publish velocity command
        if cmd_vel is not None:
            self.cmd_vel_publisher.publish(cmd_vel)
        
        # Publish navigation state
        self._publish_navigation_state()
    
    def _handle_planning_state(self):
        """Handle actions when in planning state."""
        if self.planning_future is not None and not self.planning_future.done():
            return  # Planning already in progress
        
        # Request path planning
        self._request_path_planning()
    
    def _request_path_planning(self):
        """Request path planning from global planner."""
        if not self.path_planning_service.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('Path planning service not available')
            return
        
        # Create planning request
        request = ComputePathToPose.Request()
        
        # Start pose (current robot position)
        request.start.header.stamp = self.get_clock().now().to_msg()
        request.start.header.frame_id = 'map'
        request.start.pose.position.x = self.robot_position[0]
        request.start.pose.position.y = self.robot_position[1]
        request.start.pose.orientation.w = 1.0
        
        # Goal pose
        request.goal.header.stamp = self.get_clock().now().to_msg()
        request.goal.header.frame_id = 'map'
        request.goal.pose = self.current_goal
        
        # Send request
        self.planning_future = self.path_planning_service.call_async(request)
        self.planning_future.add_done_callback(self._planning_response_callback)
        
        self.get_logger().info('Requesting path planning...')
    
    def _planning_response_callback(self, future):
        """Handle path planning response."""
        try:
            response = future.result()
            if len(response.path.poses) > 0:
                self.current_path = response.path
                self.state_machine.path_planning_succeeded(response.path)
                self.replanning_attempts = 0
                self.get_logger().info(f'Path planning succeeded: {len(response.path.poses)} waypoints')
            else:
                self.state_machine.path_planning_failed()
                self.get_logger().warn('Path planning failed: empty path returned')
        except Exception as e:
            self.state_machine.path_planning_failed()
            self.get_logger().error(f'Path planning service call failed: {e}')
    
    def _trigger_replanning(self):
        """Trigger replanning when path becomes blocked."""
        if self.replanning_attempts >= self.max_replanning_attempts:
            self.get_logger().warn('Maximum replanning attempts reached')
            return
        
        self.replanning_attempts += 1
        self.state_machine._transition_to_state(NavigationState.PLANNING)
        self.get_logger().info(f'Triggering replanning (attempt {self.replanning_attempts})')
    
    def _generate_velocity_command(self, state: NavigationState) -> Optional[Twist]:
        """
        Generate velocity commands using hybrid navigation approach.
        
        This method uses the hybrid navigator to combine global path planning
        with local obstacle avoidance and PPO-based navigation.
        
        Args:
            state: Current navigation state
            
        Returns:
            Twist command or None
        """
        if state in [NavigationState.IDLE, NavigationState.PLANNING, 
                    NavigationState.EMERGENCY_STOP, NavigationState.GOAL_REACHED,
                    NavigationState.PLANNING_FAILED]:
            # Stop robot
            cmd = Twist()
            return cmd
        
        if state in [NavigationState.FOLLOWING_PATH, NavigationState.AVOIDING_OBSTACLE]:
            # Use hybrid navigator for intelligent navigation
            return self.hybrid_navigator.compute_navigation_command(
                state,
                self.obstacles or ObstacleArray(),
                self.safety_status or SafetyStatus()
            )
        
        return None

    
    def _publish_navigation_state(self):
        """Publish current navigation state for monitoring."""
        state_info = self.state_machine.get_state_info()
        progress_info = self.hybrid_navigator.get_progress_info()
        
        msg = NavStateMsg()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        
        msg.state = state_info['current_state']
        
        # Current pose
        msg.current_pose.position.x = self.robot_position[0]
        msg.current_pose.position.y = self.robot_position[1]
        msg.current_pose.orientation.w = 1.0
        
        # Goal pose
        if self.current_goal is not None:
            msg.goal_pose = self.current_goal
        
        msg.distance_to_goal = state_info['goal_distance']
        msg.progress_percentage = progress_info['progress_percentage']
        msg.reasoning = state_info['reasoning']
        
        self.navigation_state_publisher.publish(msg)


def main(args=None):
    """Main entry point for the navigation controller node."""
    rclpy.init(args=args)
    
    try:
        node = NavigationController()
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