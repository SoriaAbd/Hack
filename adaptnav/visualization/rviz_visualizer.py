#!/usr/bin/env python3
"""
RViz visualization node for AdaptNav system.

This module creates visualization markers for obstacles, safety zones, and other
navigation-related information to be displayed in RViz2.
"""

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup

import numpy as np
from typing import List

from geometry_msgs.msg import Point, Vector3
from std_msgs.msg import ColorRGBA, Header
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Odometry
from custom_msgs.msg import ObstacleArray, Obstacle, SafetyStatus


class RVizVisualizer(Node):
    """
    RViz visualization node for AdaptNav navigation system.
    
    This node subscribes to navigation-related topics and publishes
    visualization markers for display in RViz2.
    """
    
    def __init__(self):
        super().__init__('rviz_visualizer')
        
        # Parameters
        self.declare_parameter('robot_radius', 0.3)
        self.declare_parameter('collision_zone_radius', 0.5)
        self.declare_parameter('emergency_stop_radius', 0.3)
        
        self.robot_radius = self.get_parameter('robot_radius').value
        self.collision_zone_radius = self.get_parameter('collision_zone_radius').value
        self.emergency_stop_radius = self.get_parameter('emergency_stop_radius').value
        
        # State
        self.robot_position = np.array([0.0, 0.0])
        self.safety_status = None
        self.obstacles: List[Obstacle] = []
        
        # Callback group for concurrent execution
        self.callback_group = ReentrantCallbackGroup()
        
        # Subscribers
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
        
        self.safety_status_subscriber = self.create_subscription(
            SafetyStatus,
            '/safety_status',
            self.safety_status_callback,
            10,
            callback_group=self.callback_group
        )
        
        # Publishers
        self.obstacle_markers_publisher = self.create_publisher(
            MarkerArray,
            '/obstacles/visualization',
            10
        )
        
        self.safety_zone_publisher = self.create_publisher(
            MarkerArray,
            '/safety_zone/visualization',
            10
        )
        
        # Timer for periodic visualization updates
        self.visualization_timer = self.create_timer(
            0.1,  # 10 Hz
            self.visualization_callback,
            callback_group=self.callback_group
        )
        
        self.get_logger().info('RViz visualizer initialized')
    
    def odom_callback(self, msg: Odometry):
        """Handle odometry updates."""
        self.robot_position[0] = msg.pose.pose.position.x
        self.robot_position[1] = msg.pose.pose.position.y
    
    def obstacles_callback(self, msg: ObstacleArray):
        """Handle obstacle detection updates."""
        self.obstacles = msg.obstacles
    
    def safety_status_callback(self, msg: SafetyStatus):
        """Handle safety status updates."""
        self.safety_status = msg
    
    def visualization_callback(self):
        """Periodic visualization update."""
        # Publish obstacle markers
        self._publish_obstacle_markers()
        
        # Publish safety zone markers
        self._publish_safety_zone_markers()
    
    def _publish_obstacle_markers(self):
        """Publish visualization markers for detected obstacles."""
        marker_array = MarkerArray()
        
        # Clear previous markers
        delete_marker = Marker()
        delete_marker.header.frame_id = "map"
        delete_marker.header.stamp = self.get_clock().now().to_msg()
        delete_marker.ns = "obstacles"
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)
        
        delete_velocity_marker = Marker()
        delete_velocity_marker.header.frame_id = "map"
        delete_velocity_marker.header.stamp = self.get_clock().now().to_msg()
        delete_velocity_marker.ns = "obstacle_velocities"
        delete_velocity_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_velocity_marker)
        
        # Create markers for each obstacle
        for i, obstacle in enumerate(self.obstacles):
            # Obstacle position marker
            obstacle_marker = self._create_obstacle_marker(obstacle, i)
            marker_array.markers.append(obstacle_marker)
            
            # Velocity vector marker
            velocity_marker = self._create_velocity_marker(obstacle, i)
            if velocity_marker:
                marker_array.markers.append(velocity_marker)
        
        self.obstacle_markers_publisher.publish(marker_array)
    
    def _create_obstacle_marker(self, obstacle: Obstacle, marker_id: int) -> Marker:
        """Create a visualization marker for an obstacle."""
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "obstacles"
        marker.id = marker_id
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        
        # Position
        marker.pose.position.x = obstacle.position.x
        marker.pose.position.y = obstacle.position.y
        marker.pose.position.z = 0.5  # Half height for visualization
        marker.pose.orientation.w = 1.0
        
        # Size based on classification
        if obstacle.classification == "worker":
            marker.scale.x = 1.0  # Diameter
            marker.scale.y = 1.0
            marker.scale.z = 1.8  # Human height
            # Blue for workers
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 1.0
            marker.color.a = 0.7
        elif obstacle.classification == "forklift":
            marker.scale.x = 2.0  # Diameter
            marker.scale.y = 2.0
            marker.scale.z = 2.0  # Forklift height
            # Orange for forklifts
            marker.color.r = 1.0
            marker.color.g = 0.5
            marker.color.b = 0.0
            marker.color.a = 0.7
        else:
            # Unknown obstacle
            marker.scale.x = 1.0
            marker.scale.y = 1.0
            marker.scale.z = 1.0
            # Gray for unknown
            marker.color.r = 0.5
            marker.color.g = 0.5
            marker.color.b = 0.5
            marker.color.a = 0.7
        
        # Add text label
        marker.text = f"{obstacle.classification}_{obstacle.id}"
        
        return marker
    
    def _create_velocity_marker(self, obstacle: Obstacle, marker_id: int) -> Marker:
        """Create a velocity vector marker for an obstacle."""
        # Only create velocity marker if obstacle is moving
        velocity_magnitude = np.sqrt(obstacle.velocity.x**2 + obstacle.velocity.y**2)
        if velocity_magnitude < 0.1:  # Threshold for stationary
            return None
        
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "obstacle_velocities"
        marker.id = marker_id
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        
        # Start point (obstacle position)
        start_point = Point()
        start_point.x = obstacle.position.x
        start_point.y = obstacle.position.y
        start_point.z = 1.0
        
        # End point (position + velocity vector scaled for visibility)
        scale_factor = 2.0  # Scale velocity for better visibility
        end_point = Point()
        end_point.x = obstacle.position.x + obstacle.velocity.x * scale_factor
        end_point.y = obstacle.position.y + obstacle.velocity.y * scale_factor
        end_point.z = 1.0
        
        marker.points = [start_point, end_point]
        
        # Arrow properties
        marker.scale.x = 0.1  # Shaft diameter
        marker.scale.y = 0.2  # Head diameter
        marker.scale.z = 0.0  # Not used for arrows
        
        # Color based on velocity magnitude
        if velocity_magnitude > 2.0:
            # Fast - red
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
        elif velocity_magnitude > 1.0:
            # Medium - yellow
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 0.0
        else:
            # Slow - green
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
        
        marker.color.a = 0.8
        
        return marker
    
    def _publish_safety_zone_markers(self):
        """Publish visualization markers for safety zones."""
        marker_array = MarkerArray()
        
        # Clear previous markers
        delete_marker = Marker()
        delete_marker.header.frame_id = "map"
        delete_marker.header.stamp = self.get_clock().now().to_msg()
        delete_marker.ns = "safety_zone"
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)
        
        # Collision zone marker
        collision_zone_marker = self._create_safety_zone_marker(
            "collision_zone", 0, self.collision_zone_radius
        )
        marker_array.markers.append(collision_zone_marker)
        
        # Emergency stop zone marker
        emergency_zone_marker = self._create_safety_zone_marker(
            "emergency_zone", 1, self.emergency_stop_radius
        )
        marker_array.markers.append(emergency_zone_marker)
        
        self.safety_zone_publisher.publish(marker_array)
    
    def _create_safety_zone_marker(self, zone_name: str, marker_id: int, radius: float) -> Marker:
        """Create a safety zone visualization marker."""
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "safety_zone"
        marker.id = marker_id
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        
        # Position at robot location
        marker.pose.position.x = self.robot_position[0]
        marker.pose.position.y = self.robot_position[1]
        marker.pose.position.z = 0.01  # Just above ground
        marker.pose.orientation.w = 1.0
        
        # Size
        marker.scale.x = radius * 2  # Diameter
        marker.scale.y = radius * 2
        marker.scale.z = 0.02  # Thin disk
        
        # Color based on safety status and zone type
        if zone_name == "emergency_zone":
            # Emergency zone - red
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 0.3
            
            # Make more opaque if in emergency state
            if self.safety_status and self.safety_status.state == "EMERGENCY_STOP":
                marker.color.a = 0.6
                
        else:  # collision_zone
            # Collision zone - yellow/orange
            marker.color.r = 1.0
            marker.color.g = 0.8
            marker.color.b = 0.0
            marker.color.a = 0.2
            
            # Color based on safety status
            if self.safety_status:
                if self.safety_status.state == "EMERGENCY_STOP":
                    marker.color.r = 1.0
                    marker.color.g = 0.0
                    marker.color.b = 0.0
                    marker.color.a = 0.4
                elif self.safety_status.state == "CAUTION":
                    marker.color.r = 1.0
                    marker.color.g = 0.5
                    marker.color.b = 0.0
                    marker.color.a = 0.3
                else:  # SAFE
                    marker.color.r = 0.0
                    marker.color.g = 1.0
                    marker.color.b = 0.0
                    marker.color.a = 0.2
        
        return marker


def main(args=None):
    """Main entry point for the RViz visualizer node."""
    rclpy.init(args=args)
    
    try:
        node = RVizVisualizer()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()