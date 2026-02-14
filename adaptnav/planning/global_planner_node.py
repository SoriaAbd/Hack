#!/usr/bin/env python3
"""
Global path planner ROS 2 node.

This node provides a ROS 2 service interface for the A* path planner.
It subscribes to occupancy grid maps and provides path planning services
for the navigation system.
"""

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from nav2_msgs.srv import ComputePathToPose
from nav_msgs.msg import OccupancyGrid, Path as NavPath
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from std_msgs.msg import Header
import numpy as np

from .astar_planner import AStarPlanner
from .path_smoother import PathSmoother
from ..core.warehouse_map import WarehouseMap
from ..core.path import Path, Waypoint


class GlobalPlannerNode(Node):
    """
    ROS 2 node for global path planning using A* algorithm.
    
    This node provides:
    - Service server for nav2_msgs/ComputePathToPose
    - Subscription to /map topic for occupancy grid
    - Publisher for /global_plan topic
    - Integration with A* planner and path smoother
    """
    
    def __init__(self):
        super().__init__('global_planner')
        
        # Parameters
        self.declare_parameter('robot_radius', 0.3)
        self.declare_parameter('planning_timeout', 2.0)
        self.declare_parameter('enable_smoothing', True)
        self.declare_parameter('smoothing_resolution', 0.2)
        self.declare_parameter('publish_plan', True)
        
        self.robot_radius = self.get_parameter('robot_radius').value
        self.planning_timeout = self.get_parameter('planning_timeout').value
        self.enable_smoothing = self.get_parameter('enable_smoothing').value
        self.smoothing_resolution = self.get_parameter('smoothing_resolution').value
        self.publish_plan = self.get_parameter('publish_plan').value
        
        # State
        self.warehouse_map = None
        self.planner = None
        self.smoother = None
        self.map_frame = 'map'
        
        # Callback group for concurrent execution
        self.callback_group = ReentrantCallbackGroup()
        
        # Subscribers
        self.map_subscriber = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10,
            callback_group=self.callback_group
        )
        
        # Publishers
        if self.publish_plan:
            self.plan_publisher = self.create_publisher(
                NavPath,
                '/global_plan',
                10
            )
        
        # Service server
        self.plan_service = self.create_service(
            ComputePathToPose,
            '/compute_path_to_pose',
            self.compute_path_callback,
            callback_group=self.callback_group
        )
        
        self.get_logger().info('Global planner node initialized')
        self.get_logger().info(f'Robot radius: {self.robot_radius}m')
        self.get_logger().info(f'Planning timeout: {self.planning_timeout}s')
        self.get_logger().info(f'Smoothing enabled: {self.enable_smoothing}')
    
    def map_callback(self, msg: OccupancyGrid):
        """
        Handle incoming occupancy grid map.
        
        Args:
            msg: OccupancyGrid message containing the static map
        """
        try:
            # Extract map parameters
            width_m = msg.info.width * msg.info.resolution
            height_m = msg.info.height * msg.info.resolution
            resolution = msg.info.resolution
            origin = (msg.info.origin.position.x, msg.info.origin.position.y)
            
            # Create warehouse map
            self.warehouse_map = WarehouseMap(
                width=width_m,
                height=height_m,
                resolution=resolution,
                origin=origin
            )
            
            # Convert ROS occupancy grid to our format
            # ROS: -1=unknown, 0=free, 100=occupied
            # Our format: -1=unknown, 0=free, 100=occupied (same)
            occupancy_data = np.array(msg.data, dtype=np.int8)
            occupancy_grid = occupancy_data.reshape((msg.info.height, msg.info.width))
            
            # Flip vertically to match our coordinate system
            self.warehouse_map.occupancy_grid = np.flipud(occupancy_grid)
            
            # Update map frame
            self.map_frame = msg.header.frame_id
            
            # Create planner and smoother
            self.planner = AStarPlanner(
                self.warehouse_map,
                robot_radius=self.robot_radius,
                timeout_seconds=self.planning_timeout
            )
            
            if self.enable_smoothing:
                self.smoother = PathSmoother(
                    self.warehouse_map,
                    robot_radius=self.robot_radius,
                    smoothing_resolution=self.smoothing_resolution
                )
            
            self.get_logger().info(
                f'Map updated: {msg.info.width}x{msg.info.height} '
                f'({width_m:.1f}x{height_m:.1f}m) at {resolution:.2f}m/cell'
            )
            
        except Exception as e:
            self.get_logger().error(f'Failed to process map: {e}')
    
    def compute_path_callback(self, request, response):
        """
        Handle path planning service requests.
        
        Args:
            request: ComputePathToPose request
            response: ComputePathToPose response
            
        Returns:
            Response with computed path or error status
        """
        try:
            # Check if planner is ready
            if self.planner is None:
                self.get_logger().warn('No map available for planning')
                response.path.header.stamp = self.get_clock().now().to_msg()
                response.path.header.frame_id = self.map_frame
                return response
            
            # Extract start and goal positions
            start_pose = request.start
            goal_pose = request.goal.pose
            
            start_x = start_pose.pose.position.x
            start_y = start_pose.pose.position.y
            goal_x = goal_pose.position.x
            goal_y = goal_pose.position.y
            
            self.get_logger().info(
                f'Planning path from ({start_x:.2f}, {start_y:.2f}) '
                f'to ({goal_x:.2f}, {goal_y:.2f})'
            )
            
            # Plan path using A*
            path = self.planner.plan_path(start_x, start_y, goal_x, goal_y)
            
            if path is None:
                self.get_logger().warn('Path planning failed')
                response.path.header.stamp = self.get_clock().now().to_msg()
                response.path.header.frame_id = self.map_frame
                return response
            
            # Apply smoothing if enabled
            if self.enable_smoothing and self.smoother is not None:
                smoothed_path = self.smoother.smooth_path(path)
                if smoothed_path is not None:
                    path = smoothed_path
            
            # Convert to ROS path message
            ros_path = self.convert_to_ros_path(path)
            response.path = ros_path
            
            # Publish path if enabled
            if self.publish_plan:
                self.plan_publisher.publish(ros_path)
            
            self.get_logger().info(
                f'Path computed successfully: {len(path.waypoints)} waypoints, '
                f'{path.total_length:.2f}m length'
            )
            
        except Exception as e:
            self.get_logger().error(f'Path planning service error: {e}')
            response.path.header.stamp = self.get_clock().now().to_msg()
            response.path.header.frame_id = self.map_frame
        
        return response
    
    def convert_to_ros_path(self, path: Path) -> NavPath:
        """
        Convert internal Path object to ROS nav_msgs/Path.
        
        Args:
            path: Internal Path object
            
        Returns:
            ROS nav_msgs/Path message
        """
        ros_path = NavPath()
        ros_path.header.stamp = self.get_clock().now().to_msg()
        ros_path.header.frame_id = self.map_frame
        
        for waypoint in path.waypoints:
            pose_stamped = PoseStamped()
            pose_stamped.header = ros_path.header
            
            # Position
            pose_stamped.pose.position.x = waypoint.x
            pose_stamped.pose.position.y = waypoint.y
            pose_stamped.pose.position.z = 0.0
            
            # Orientation (quaternion from theta)
            pose_stamped.pose.orientation.x = 0.0
            pose_stamped.pose.orientation.y = 0.0
            pose_stamped.pose.orientation.z = np.sin(waypoint.theta / 2.0)
            pose_stamped.pose.orientation.w = np.cos(waypoint.theta / 2.0)
            
            ros_path.poses.append(pose_stamped)
        
        return ros_path


def main(args=None):
    """Main entry point for the global planner node."""
    rclpy.init(args=args)
    
    try:
        node = GlobalPlannerNode()
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