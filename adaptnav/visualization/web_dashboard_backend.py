#!/usr/bin/env python3
"""
Web dashboard backend for AdaptNav system.

This module implements a WebSocket server that streams real-time navigation
data to a web-based dashboard frontend.
"""

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup

import json
import asyncio
import websockets
import threading
from typing import Set, Dict, Any, Optional
import numpy as np

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, Path as NavPath
from sensor_msgs.msg import LaserScan, Image
from std_srvs.srv import Trigger
from custom_msgs.msg import ObstacleArray, Obstacle, SafetyStatus, NavigationState


class WebDashboardBackend(Node):
    """
    WebSocket server backend for the AdaptNav web dashboard.
    
    This node subscribes to all relevant ROS 2 topics and streams
    the data to connected web clients via WebSocket.
    """
    
    def __init__(self):
        super().__init__('web_dashboard_backend')
        
        # Parameters
        self.declare_parameter('websocket_port', 8765)
        self.declare_parameter('websocket_host', 'localhost')
        
        self.websocket_port = self.get_parameter('websocket_port').value
        self.websocket_host = self.get_parameter('websocket_host').value
        
        # Connected WebSocket clients
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        
        # Latest data cache
        self.latest_data = {
            'robot_pose': None,
            'robot_velocity': None,
            'global_plan': None,
            'obstacles': [],
            'navigation_state': None,
            'safety_status': None,
            'lidar_scan': None,
            'performance_metrics': {
                'control_frequency': 0.0,
                'detection_latency': 0.0,
                'planning_time': 0.0
            }
        }
        
        # Performance tracking
        self.last_control_time = 0.0
        self.control_frequency = 0.0
        
        # Callback group for concurrent execution
        self.callback_group = ReentrantCallbackGroup()
        
        # Subscribers
        self._setup_subscribers()
        
        # Service clients
        self._setup_service_clients()
        
        # WebSocket server
        self.websocket_server = None
        self.websocket_thread = None
        
        # Start WebSocket server
        self._start_websocket_server()
        
        # Timer for periodic data broadcasting
        self.broadcast_timer = self.create_timer(
            0.1,  # 10 Hz
            self.broadcast_data,
            callback_group=self.callback_group
        )
        
        self.get_logger().info(f'Web dashboard backend initialized on {self.websocket_host}:{self.websocket_port}')
    
    def _setup_subscribers(self):
        """Set up ROS 2 subscribers for all relevant topics."""
        
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
        
        self.obstacles_subscriber = self.create_subscription(
            ObstacleArray,
            '/obstacles/detected',
            self.obstacles_callback,
            10,
            callback_group=self.callback_group
        )
        
        self.navigation_state_subscriber = self.create_subscription(
            NavigationState,
            '/navigation_state',
            self.navigation_state_callback,
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
        
        self.lidar_subscriber = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            10,
            callback_group=self.callback_group
        )
        
        self.cmd_vel_subscriber = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            10,
            callback_group=self.callback_group
        )
    
    def _setup_service_clients(self):
        """Set up ROS 2 service clients for system control."""
        
        self.start_navigation_client = self.create_client(
            Trigger,
            '/start_navigation'
        )
        
        self.stop_navigation_client = self.create_client(
            Trigger,
            '/stop_navigation'
        )
        
        self.reset_simulation_client = self.create_client(
            Trigger,
            '/reset_simulation'
        )
    
    def _start_websocket_server(self):
        """Start the WebSocket server in a separate thread."""
        
        def run_server():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            async def server_handler():
                self.websocket_server = await websockets.serve(
                    self.handle_websocket_connection,
                    self.websocket_host,
                    self.websocket_port
                )
                await self.websocket_server.wait_closed()
            
            loop.run_until_complete(server_handler())
        
        self.websocket_thread = threading.Thread(target=run_server, daemon=True)
        self.websocket_thread.start()
    
    async def handle_websocket_connection(self, websocket, path):
        """Handle new WebSocket connections."""
        self.clients.add(websocket)
        self.get_logger().info(f'New WebSocket client connected: {websocket.remote_address}')
        
        try:
            # Send initial data to new client
            await self.send_to_client(websocket, {
                'type': 'initial_data',
                'data': self.latest_data
            })
            
            # Handle incoming messages from client
            async for message in websocket:
                await self.handle_client_message(websocket, message)
                
        except websockets.exceptions.ConnectionClosed:
            self.get_logger().info(f'WebSocket client disconnected: {websocket.remote_address}')
        except Exception as e:
            self.get_logger().error(f'WebSocket error: {e}')
        finally:
            self.clients.discard(websocket)
    
    async def handle_client_message(self, websocket, message):
        """Handle messages from WebSocket clients."""
        try:
            data = json.loads(message)
            command = data.get('command')
            
            if command == 'start_navigation':
                await self.handle_start_navigation(websocket)
            elif command == 'stop_navigation':
                await self.handle_stop_navigation(websocket)
            elif command == 'reset_simulation':
                await self.handle_reset_simulation(websocket)
            else:
                await self.send_to_client(websocket, {
                    'type': 'error',
                    'message': f'Unknown command: {command}'
                })
                
        except json.JSONDecodeError:
            await self.send_to_client(websocket, {
                'type': 'error',
                'message': 'Invalid JSON message'
            })
        except Exception as e:
            self.get_logger().error(f'Error handling client message: {e}')
            await self.send_to_client(websocket, {
                'type': 'error',
                'message': str(e)
            })
    
    async def handle_start_navigation(self, websocket):
        """Handle start navigation command."""
        if self.start_navigation_client.service_is_ready():
            request = Trigger.Request()
            future = self.start_navigation_client.call_async(request)
            
            # Note: In a real implementation, you'd want to handle this asynchronously
            # For now, we'll just send a response
            await self.send_to_client(websocket, {
                'type': 'command_response',
                'command': 'start_navigation',
                'success': True,
                'message': 'Navigation start requested'
            })
        else:
            await self.send_to_client(websocket, {
                'type': 'command_response',
                'command': 'start_navigation',
                'success': False,
                'message': 'Start navigation service not available'
            })
    
    async def handle_stop_navigation(self, websocket):
        """Handle stop navigation command."""
        if self.stop_navigation_client.service_is_ready():
            request = Trigger.Request()
            future = self.stop_navigation_client.call_async(request)
            
            await self.send_to_client(websocket, {
                'type': 'command_response',
                'command': 'stop_navigation',
                'success': True,
                'message': 'Navigation stop requested'
            })
        else:
            await self.send_to_client(websocket, {
                'type': 'command_response',
                'command': 'stop_navigation',
                'success': False,
                'message': 'Stop navigation service not available'
            })
    
    async def handle_reset_simulation(self, websocket):
        """Handle reset simulation command."""
        if self.reset_simulation_client.service_is_ready():
            request = Trigger.Request()
            future = self.reset_simulation_client.call_async(request)
            
            await self.send_to_client(websocket, {
                'type': 'command_response',
                'command': 'reset_simulation',
                'success': True,
                'message': 'Simulation reset requested'
            })
        else:
            await self.send_to_client(websocket, {
                'type': 'command_response',
                'command': 'reset_simulation',
                'success': False,
                'message': 'Reset simulation service not available'
            })
    
    async def send_to_client(self, websocket, data):
        """Send data to a specific WebSocket client."""
        try:
            message = json.dumps(data, default=self.json_serializer)
            await websocket.send(message)
        except Exception as e:
            self.get_logger().error(f'Error sending to client: {e}')
    
    async def broadcast_to_all_clients(self, data):
        """Broadcast data to all connected WebSocket clients."""
        if not self.clients:
            return
        
        message = json.dumps(data, default=self.json_serializer)
        disconnected_clients = set()
        
        for client in self.clients:
            try:
                await client.send(message)
            except websockets.exceptions.ConnectionClosed:
                disconnected_clients.add(client)
            except Exception as e:
                self.get_logger().error(f'Error broadcasting to client: {e}')
                disconnected_clients.add(client)
        
        # Remove disconnected clients
        self.clients -= disconnected_clients
    
    def json_serializer(self, obj):
        """Custom JSON serializer for numpy arrays and other objects."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, np.int32):
            return int(obj)
        elif hasattr(obj, 'to_dict'):
            return obj.to_dict()
        else:
            return str(obj)
    
    def broadcast_data(self):
        """Periodic data broadcasting to WebSocket clients."""
        if not self.clients:
            return
        
        # Create data package
        data_package = {
            'type': 'navigation_update',
            'timestamp': self.get_clock().now().nanoseconds / 1e9,
            'data': self.latest_data
        }
        
        # Broadcast asynchronously
        asyncio.run_coroutine_threadsafe(
            self.broadcast_to_all_clients(data_package),
            asyncio.get_event_loop()
        )
    
    # ROS 2 Callback methods
    
    def odom_callback(self, msg: Odometry):
        """Handle odometry updates."""
        self.latest_data['robot_pose'] = {
            'position': {
                'x': msg.pose.pose.position.x,
                'y': msg.pose.pose.position.y,
                'z': msg.pose.pose.position.z
            },
            'orientation': {
                'x': msg.pose.pose.orientation.x,
                'y': msg.pose.pose.orientation.y,
                'z': msg.pose.pose.orientation.z,
                'w': msg.pose.pose.orientation.w
            }
        }
        
        self.latest_data['robot_velocity'] = {
            'linear': {
                'x': msg.twist.twist.linear.x,
                'y': msg.twist.twist.linear.y,
                'z': msg.twist.twist.linear.z
            },
            'angular': {
                'x': msg.twist.twist.angular.x,
                'y': msg.twist.twist.angular.y,
                'z': msg.twist.twist.angular.z
            }
        }
    
    def global_plan_callback(self, msg: NavPath):
        """Handle global plan updates."""
        waypoints = []
        for pose_stamped in msg.poses:
            waypoints.append({
                'position': {
                    'x': pose_stamped.pose.position.x,
                    'y': pose_stamped.pose.position.y,
                    'z': pose_stamped.pose.position.z
                },
                'orientation': {
                    'x': pose_stamped.pose.orientation.x,
                    'y': pose_stamped.pose.orientation.y,
                    'z': pose_stamped.pose.orientation.z,
                    'w': pose_stamped.pose.orientation.w
                }
            })
        
        self.latest_data['global_plan'] = {
            'header': {
                'frame_id': msg.header.frame_id,
                'stamp': msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
            },
            'waypoints': waypoints
        }
    
    def obstacles_callback(self, msg: ObstacleArray):
        """Handle obstacle detection updates."""
        obstacles = []
        for obstacle in msg.obstacles:
            obstacles.append({
                'id': obstacle.id,
                'position': {
                    'x': obstacle.position.x,
                    'y': obstacle.position.y,
                    'z': obstacle.position.z
                },
                'velocity': {
                    'x': obstacle.velocity.x,
                    'y': obstacle.velocity.y,
                    'z': obstacle.velocity.z
                },
                'classification': obstacle.classification,
                'confidence': obstacle.confidence,
                'last_seen': obstacle.last_seen.sec + obstacle.last_seen.nanosec / 1e9
            })
        
        self.latest_data['obstacles'] = obstacles
    
    def navigation_state_callback(self, msg: NavigationState):
        """Handle navigation state updates."""
        self.latest_data['navigation_state'] = {
            'state': msg.state,
            'current_pose': {
                'position': {
                    'x': msg.current_pose.position.x,
                    'y': msg.current_pose.position.y,
                    'z': msg.current_pose.position.z
                },
                'orientation': {
                    'x': msg.current_pose.orientation.x,
                    'y': msg.current_pose.orientation.y,
                    'z': msg.current_pose.orientation.z,
                    'w': msg.current_pose.orientation.w
                }
            },
            'goal_pose': {
                'position': {
                    'x': msg.goal_pose.position.x,
                    'y': msg.goal_pose.position.y,
                    'z': msg.goal_pose.position.z
                },
                'orientation': {
                    'x': msg.goal_pose.orientation.x,
                    'y': msg.goal_pose.orientation.y,
                    'z': msg.goal_pose.orientation.z,
                    'w': msg.goal_pose.orientation.w
                }
            },
            'distance_to_goal': msg.distance_to_goal,
            'progress_percentage': msg.progress_percentage,
            'reasoning': msg.reasoning
        }
    
    def safety_status_callback(self, msg: SafetyStatus):
        """Handle safety status updates."""
        self.latest_data['safety_status'] = {
            'state': msg.state,
            'closest_obstacle_distance': msg.closest_obstacle_distance,
            'velocity_scale': msg.velocity_scale,
            'override_active': msg.override_active,
            'time_until_clear': msg.time_until_clear
        }
    
    def lidar_callback(self, msg: LaserScan):
        """Handle LiDAR scan updates."""
        # Downsample for web transmission (every 5th point)
        ranges = msg.ranges[::5]
        angles = []
        for i in range(0, len(msg.ranges), 5):
            angle = msg.angle_min + i * msg.angle_increment
            angles.append(angle)
        
        self.latest_data['lidar_scan'] = {
            'header': {
                'frame_id': msg.header.frame_id,
                'stamp': msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
            },
            'angle_min': msg.angle_min,
            'angle_max': msg.angle_max,
            'angle_increment': msg.angle_increment * 5,  # Adjusted for downsampling
            'range_min': msg.range_min,
            'range_max': msg.range_max,
            'ranges': ranges,
            'angles': angles
        }
    
    def cmd_vel_callback(self, msg: Twist):
        """Handle velocity command updates for performance tracking."""
        current_time = self.get_clock().now().nanoseconds / 1e9
        
        if self.last_control_time > 0:
            dt = current_time - self.last_control_time
            if dt > 0:
                # Exponential moving average for frequency
                alpha = 0.1
                new_frequency = 1.0 / dt
                self.control_frequency = (alpha * new_frequency + 
                                        (1 - alpha) * self.control_frequency)
        
        self.last_control_time = current_time
        self.latest_data['performance_metrics']['control_frequency'] = self.control_frequency


def main(args=None):
    """Main entry point for the web dashboard backend."""
    rclpy.init(args=args)
    
    try:
        node = WebDashboardBackend()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()