#!/usr/bin/env python3
"""
PPO Agent ROS 2 Node for AdaptNav warehouse navigation.

This node implements the PPO agent as a ROS 2 node that:
1. Loads a trained PPO model from file
2. Subscribes to sensor topics to construct observations
3. Computes actions using the loaded model
4. Publishes velocity commands to /cmd_vel_raw topic

The node integrates with the navigation controller and safety controller
to provide learned local navigation behaviors.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
import numpy as np
import os
import threading
from typing import Optional, Dict, Any
import time

# ROS 2 message imports
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, Twist
from custom_msgs.msg import ObstacleArray, NavigationState

# AdaptNav imports
from adaptnav.rl.ppo_observation import PPOObservation

# Stable Baselines3 imports
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.policies import BasePolicy
    SB3_AVAILABLE = True
except ImportError:
    PPO = None
    BasePolicy = None
    SB3_AVAILABLE = False


class PPOAgentNode(Node):
    """
    ROS 2 node that runs a trained PPO agent for local navigation.
    
    This node subscribes to sensor data, constructs observations,
    and publishes velocity commands based on PPO policy decisions.
    """
    
    def __init__(self):
        """Initialize the PPO agent node."""
        super().__init__('ppo_agent_node')
        
        # Declare parameters
        self.declare_parameter('model_path', '')
        self.declare_parameter('control_frequency', 10.0)
        self.declare_parameter('observation_timeout', 1.0)
        self.declare_parameter('max_linear_velocity', 1.0)
        self.declare_parameter('max_angular_velocity', 0.5)
        self.declare_parameter('enable_agent', True)
        
        # Get parameters
        self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.control_frequency = self.get_parameter('control_frequency').get_parameter_value().double_value
        self.observation_timeout = self.get_parameter('observation_timeout').get_parameter_value().double_value
        self.max_linear_velocity = self.get_parameter('max_linear_velocity').get_parameter_value().double_value
        self.max_angular_velocity = self.get_parameter('max_angular_velocity').get_parameter_value().double_value
        self.enable_agent = self.get_parameter('enable_agent').get_parameter_value().bool_value
        
        # QoS profiles
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=1
        )
        
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )
        
        # Subscribers
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            sensor_qos
        )
        
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            reliable_qos
        )
        
        self.obstacles_sub = self.create_subscription(
            ObstacleArray,
            '/obstacles/detected',
            self.obstacles_callback,
            reliable_qos
        )
        
        self.nav_state_sub = self.create_subscription(
            NavigationState,
            '/navigation_state',
            self.nav_state_callback,
            reliable_qos
        )
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            '/cmd_vel_raw',
            reliable_qos
        )
        
        # Data storage with thread safety
        self.data_lock = threading.Lock()
        self.latest_scan: Optional[LaserScan] = None
        self.latest_odom: Optional[Odometry] = None
        self.latest_obstacles: Optional[ObstacleArray] = None
        self.latest_nav_state: Optional[NavigationState] = None
        self.goal_pose: Optional[Pose] = None
        
        # Timing tracking
        self.last_scan_time = 0.0
        self.last_odom_time = 0.0
        self.last_obstacles_time = 0.0
        self.last_inference_time = 0.0
        
        # PPO model
        self.model: Optional[BasePolicy] = None
        self.model_loaded = False
        
        # Control state
        self.agent_active = False
        self.last_action = np.array([0.0, 0.0])
        
        # Statistics
        self.inference_count = 0
        self.total_inference_time = 0.0
        self.max_inference_time = 0.0
        
        # Load model
        self.load_model()
        
        # Control timer
        self.control_timer = self.create_timer(
            1.0 / self.control_frequency,
            self.control_callback
        )
        
        # Status timer
        self.status_timer = self.create_timer(
            5.0,  # 5 second status updates
            self.status_callback
        )
        
        self.get_logger().info(f"PPO Agent Node initialized")
        self.get_logger().info(f"Model path: {self.model_path}")
        self.get_logger().info(f"Control frequency: {self.control_frequency} Hz")
        self.get_logger().info(f"Agent enabled: {self.enable_agent}")
    
    def load_model(self) -> bool:
        """
        Load the trained PPO model from file.
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        if not SB3_AVAILABLE:
            self.get_logger().error("Stable Baselines3 not available. Cannot load PPO model.")
            return False
        
        if not self.model_path:
            self.get_logger().warn("No model path specified. Agent will not be active.")
            return False
        
        if not os.path.exists(self.model_path):
            self.get_logger().error(f"Model file not found: {self.model_path}")
            return False
        
        try:
            self.get_logger().info(f"Loading PPO model from: {self.model_path}")
            self.model = PPO.load(self.model_path)
            self.model_loaded = True
            self.get_logger().info("PPO model loaded successfully")
            return True
            
        except Exception as e:
            self.get_logger().error(f"Failed to load PPO model: {str(e)}")
            self.model_loaded = False
            return False
    
    def scan_callback(self, msg: LaserScan) -> None:
        """Handle LiDAR scan messages."""
        with self.data_lock:
            self.latest_scan = msg
            self.last_scan_time = time.time()
    
    def odom_callback(self, msg: Odometry) -> None:
        """Handle odometry messages."""
        with self.data_lock:
            self.latest_odom = msg
            self.last_odom_time = time.time()
    
    def obstacles_callback(self, msg: ObstacleArray) -> None:
        """Handle obstacle detection messages."""
        with self.data_lock:
            self.latest_obstacles = msg
            self.last_obstacles_time = time.time()
    
    def nav_state_callback(self, msg: NavigationState) -> None:
        """Handle navigation state messages."""
        with self.data_lock:
            self.latest_nav_state = msg
            
            # Extract goal pose from navigation state
            if msg.goal_pose:
                self.goal_pose = msg.goal_pose
            
            # Update agent active state based on navigation state
            if msg.state in ['FOLLOWING_PATH', 'AVOIDING_OBSTACLE']:
                self.agent_active = True
            elif msg.state in ['IDLE', 'PLANNING', 'EMERGENCY_STOP', 'GOAL_REACHED']:
                self.agent_active = False
    
    def control_callback(self) -> None:
        """Main control loop callback."""
        if not self.enable_agent or not self.model_loaded or not self.agent_active:
            # Publish zero velocity when agent is not active
            self.publish_zero_velocity()
            return
        
        # Check data freshness
        current_time = time.time()
        
        with self.data_lock:
            # Check if we have all required data
            if (self.latest_scan is None or 
                self.latest_odom is None or 
                self.latest_obstacles is None or 
                self.goal_pose is None):
                self.get_logger().warn("Missing required data for PPO inference")
                self.publish_zero_velocity()
                return
            
            # Check data timeout
            if (current_time - self.last_scan_time > self.observation_timeout or
                current_time - self.last_odom_time > self.observation_timeout):
                self.get_logger().warn("Sensor data timeout - stopping robot")
                self.publish_zero_velocity()
                return
            
            # Copy data for processing (release lock quickly)
            scan = self.latest_scan
            odom = self.latest_odom
            obstacles = self.latest_obstacles
            goal = self.goal_pose
        
        # Construct observation and compute action
        try:
            start_time = time.time()
            
            # Create PPO observation
            observation = PPOObservation.from_ros_messages(scan, odom, obstacles, goal)
            obs_vector = observation.to_vector()
            
            # Predict action using PPO model
            action, _states = self.model.predict(obs_vector, deterministic=True)
            
            # Track inference timing
            inference_time = time.time() - start_time
            self.update_inference_stats(inference_time)
            
            # Validate and clamp action
            action = self.validate_action(action)
            
            # Publish velocity command
            self.publish_velocity_command(action)
            
            # Store last action
            self.last_action = action.copy()
            
        except Exception as e:
            self.get_logger().error(f"PPO inference failed: {str(e)}")
            self.publish_zero_velocity()
    
    def validate_action(self, action: np.ndarray) -> np.ndarray:
        """
        Validate and clamp action to safe limits.
        
        Args:
            action: Raw action from PPO model [linear_vel, angular_vel]
            
        Returns:
            np.ndarray: Validated and clamped action
        """
        # Check for NaN or infinite values
        if not np.isfinite(action).all():
            self.get_logger().warn("PPO output contains NaN/Inf values - using zero action")
            return np.array([0.0, 0.0])
        
        # Clamp to velocity limits
        action[0] = np.clip(action[0], -self.max_linear_velocity, self.max_linear_velocity)
        action[1] = np.clip(action[1], -self.max_angular_velocity, self.max_angular_velocity)
        
        return action
    
    def publish_velocity_command(self, action: np.ndarray) -> None:
        """
        Publish velocity command based on PPO action.
        
        Args:
            action: Action array [linear_velocity, angular_velocity]
        """
        cmd_vel = Twist()
        cmd_vel.linear.x = float(action[0])
        cmd_vel.linear.y = 0.0
        cmd_vel.linear.z = 0.0
        cmd_vel.angular.x = 0.0
        cmd_vel.angular.y = 0.0
        cmd_vel.angular.z = float(action[1])
        
        self.cmd_vel_pub.publish(cmd_vel)
    
    def publish_zero_velocity(self) -> None:
        """Publish zero velocity command."""
        cmd_vel = Twist()  # All fields default to 0.0
        self.cmd_vel_pub.publish(cmd_vel)
    
    def update_inference_stats(self, inference_time: float) -> None:
        """
        Update inference timing statistics.
        
        Args:
            inference_time: Time taken for inference in seconds
        """
        self.inference_count += 1
        self.total_inference_time += inference_time
        self.max_inference_time = max(self.max_inference_time, inference_time)
        self.last_inference_time = inference_time
        
        # Log warning if inference is too slow
        if inference_time > 0.05:  # 50ms threshold
            self.get_logger().warn(f"Slow PPO inference: {inference_time*1000:.1f}ms")
    
    def status_callback(self) -> None:
        """Periodic status logging."""
        if self.inference_count > 0:
            avg_inference_time = self.total_inference_time / self.inference_count
            
            self.get_logger().info(
                f"PPO Agent Status - "
                f"Active: {self.agent_active}, "
                f"Model loaded: {self.model_loaded}, "
                f"Inferences: {self.inference_count}, "
                f"Avg inference time: {avg_inference_time*1000:.1f}ms, "
                f"Max inference time: {self.max_inference_time*1000:.1f}ms, "
                f"Last action: [{self.last_action[0]:.3f}, {self.last_action[1]:.3f}]"
            )
        else:
            self.get_logger().info(
                f"PPO Agent Status - "
                f"Active: {self.agent_active}, "
                f"Model loaded: {self.model_loaded}, "
                f"No inferences yet"
            )
    
    def destroy_node(self) -> None:
        """Clean up resources when node is destroyed."""
        self.get_logger().info("Shutting down PPO Agent Node")
        
        # Publish final zero velocity
        self.publish_zero_velocity()
        
        # Clean up model
        if self.model is not None:
            del self.model
            self.model = None
        
        super().destroy_node()


def main(args=None):
    """Main entry point for the PPO agent node."""
    rclpy.init(args=args)
    
    try:
        node = PPOAgentNode()
        
        # Use MultiThreadedExecutor for better performance
        from rclpy.executors import MultiThreadedExecutor
        executor = MultiThreadedExecutor()
        executor.add_node(node)
        
        try:
            executor.spin()
        except KeyboardInterrupt:
            pass
        finally:
            node.destroy_node()
            executor.shutdown()
            
    except Exception as e:
        print(f"Failed to start PPO agent node: {e}")
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()