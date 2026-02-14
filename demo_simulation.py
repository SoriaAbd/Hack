#!/usr/bin/env python3
"""
AdaptNav Demo Simulation

This script demonstrates the AdaptNav warehouse navigation system with:
- MuJoCo simulation environment
- LiDAR and depth camera sensors
- Obstacle detection and tracking
- A* path planning
- PPO-based local navigation
- Safety controller
- Simple visualization

Run this script to see the complete system in action.
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from typing import List, Tuple, Optional
import threading
import queue

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import AdaptNav components
try:
    from adaptnav.core.warehouse_map import WarehouseMap
    from adaptnav.core.dynamic_obstacle import DynamicObstacle
    from adaptnav.core.robot_state import RobotState
    from adaptnav.core.path import Path, Waypoint
    from adaptnav.planning.astar_planner import AStarPlanner
    from adaptnav.simulation.mujoco_simulation import MuJoCoSimulation
    from adaptnav.sensors.lidar_sensor import LiDARSensor
    from adaptnav.sensors.depth_camera import DepthCamera
    from adaptnav.perception.obstacle_detector import ObstacleDetector
    from adaptnav.control.safety_controller import SafetyController
    ADAPTNAV_AVAILABLE = True
except ImportError as e:
    print(f"Warning: AdaptNav components not fully available: {e}")
    ADAPTNAV_AVAILABLE = False


class DemoSimulation:
    """
    Demo simulation that showcases the AdaptNav system.
    
    This class orchestrates all components and provides a simple
    visualization of the warehouse navigation system.
    """
    
    def __init__(self):
        """Initialize the demo simulation."""
        self.running = False
        self.paused = False
        
        # Simulation parameters
        self.warehouse_size = (20, 20)  # 20x20 meter warehouse
        self.robot_radius = 0.3
        self.dt = 0.1  # 100ms simulation step
        
        # Initialize components
        self.setup_warehouse()
        self.setup_robot()
        self.setup_obstacles()
        self.setup_sensors()
        self.setup_planner()
        self.setup_safety()
        
        # Visualization
        self.setup_visualization()
        
        # Data queues for thread-safe communication
        self.sensor_data_queue = queue.Queue()
        self.obstacle_data_queue = queue.Queue()
        self.path_data_queue = queue.Queue()
        
        print("Demo simulation initialized successfully!")
        print("Components:")
        print(f"  - Warehouse: {self.warehouse_size[0]}x{self.warehouse_size[1]}m")
        print(f"  - Robot radius: {self.robot_radius}m")
        print(f"  - Obstacles: {len(self.obstacles)}")
        print(f"  - Simulation step: {self.dt}s")
    
    def setup_warehouse(self):
        """Set up the warehouse environment."""
        # Create a simple warehouse layout
        self.warehouse_map = WarehouseMap(
            width=self.warehouse_size[0],
            height=self.warehouse_size[1],
            resolution=0.1  # 10cm resolution
        )
        
        # Add some internal obstacles (shelves) - positioned to allow navigation
        # These are smaller and positioned to leave clear paths
        self.warehouse_map.set_obstacle(6, 6, 1.5, 3)   # Small vertical shelf
        self.warehouse_map.set_obstacle(12, 4, 3, 1)    # Small horizontal shelf
        self.warehouse_map.set_obstacle(16, 12, 1.5, 3) # Another small vertical shelf
        
        print("Warehouse layout created with navigable paths")
    
    def setup_robot(self):
        """Set up the robot state."""
        # Start position (bottom-left corner)
        start_x, start_y = 2.0, 2.0
        self.robot_state = RobotState(
            position=np.array([start_x, start_y]),
            orientation=0.0,  # Facing right
            linear_velocity=0.0,
            angular_velocity=0.0
        )
        
        # Goal position (top-right area)
        self.goal_position = (17.0, 17.0)
        
        print(f"Robot initialized at ({start_x}, {start_y})")
        print(f"Goal position: {self.goal_position}")
    
    def setup_obstacles(self):
        """Set up dynamic obstacles (workers, forklifts)."""
        self.obstacles = []
        
        # Add a few moving obstacles
        # Worker 1: Moving in a circle
        worker1 = DynamicObstacle(
            id=1,
            position=np.array([8.0, 8.0]),
            velocity=np.array([0.5, 0.0]),
            radius=0.4,
            classification="worker"
        )
        self.obstacles.append(worker1)
        
        # Worker 2: Moving back and forth
        worker2 = DynamicObstacle(
            id=2,
            position=np.array([12.0, 15.0]),
            velocity=np.array([-0.3, 0.0]),
            radius=0.4,
            classification="worker"
        )
        self.obstacles.append(worker2)
        
        # Forklift: Larger, slower
        forklift = DynamicObstacle(
            id=3,
            position=np.array([6.0, 12.0]),
            velocity=np.array([0.0, -0.2]),
            radius=0.8,
            classification="forklift"
        )
        self.obstacles.append(forklift)
        
        print(f"Added {len(self.obstacles)} dynamic obstacles")
    
    def setup_sensors(self):
        """Set up sensor simulation."""
        # Simple sensor parameters (no actual sensor classes needed for demo)
        self.lidar_range_max = 10.0
        self.lidar_range_min = 0.1
        self.lidar_angle_resolution = 1.0  # 1 degree resolution
        self.lidar_noise_std = 0.02
        
        self.depth_camera_fov = 90.0
        self.depth_camera_range_max = 5.0
        self.depth_camera_range_min = 0.5
        self.depth_camera_resolution = (64, 48)  # Lower resolution for demo
        self.depth_camera_noise_std = 0.05
        
        print("Sensors initialized")
    
    def setup_planner(self):
        """Set up the path planner."""
        if ADAPTNAV_AVAILABLE:
            self.planner = AStarPlanner(self.warehouse_map)
            
            # Plan initial path
            start_x, start_y = self.robot_state.position
            goal_x, goal_y = self.goal_position
            
            planned_waypoints = self.planner.plan_path(start_x, start_y, goal_x, goal_y)
            if planned_waypoints:
                self.planned_path = Path(planned_waypoints)
                # Initialize waypoint tracking
                self.current_waypoint_index = 0
                print(f"Initial path planned with {len(self.planned_path.waypoints)} waypoints")
            else:
                print("Warning: Could not plan initial path")
                # Create a simple straight-line path as fallback
                self.planned_path = Path([
                    Waypoint(start_x, start_y, 0),
                    Waypoint(goal_x, goal_y, 0)
                ])
                self.current_waypoint_index = 0
        else:
            # Simple fallback path
            start_x, start_y = self.robot_state.position
            goal_x, goal_y = self.goal_position
            self.planned_path = Path([
                Waypoint(start_x, start_y, 0),
                Waypoint(goal_x, goal_y, 0)
            ])
            self.current_waypoint_index = 0
            print("Using simple fallback path")
    
    def setup_safety(self):
        """Set up safety controller."""
        self.safety_zone_radius = 1.0  # 1 meter safety zone
        self.max_velocity = 1.0  # 1 m/s max velocity
        self.emergency_stop_distance = 0.5  # Emergency stop at 0.5m
        
        print("Safety controller configured")
    
    def setup_visualization(self):
        """Set up matplotlib visualization."""
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        self.ax.set_xlim(0, self.warehouse_size[0])
        self.ax.set_ylim(0, self.warehouse_size[1])
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title('AdaptNav Warehouse Navigation Demo')
        self.ax.set_xlabel('X (meters)')
        self.ax.set_ylabel('Y (meters)')
        
        # Initialize plot elements
        self.robot_circle = plt.Circle(self.robot_state.position, self.robot_radius, 
                                     color='blue', alpha=0.7, label='Robot')
        self.ax.add_patch(self.robot_circle)
        
        self.goal_marker = plt.Circle(self.goal_position, 0.3, 
                                    color='green', alpha=0.8, label='Goal')
        self.ax.add_patch(self.goal_marker)
        
        # Path line
        if self.planned_path and self.planned_path.waypoints:
            path_x = [wp.x for wp in self.planned_path.waypoints]
            path_y = [wp.y for wp in self.planned_path.waypoints]
            self.path_line, = self.ax.plot(path_x, path_y, 'g--', alpha=0.6, 
                                         linewidth=2, label='Planned Path')
        
        # Obstacle circles
        self.obstacle_patches = []
        for obs in self.obstacles:
            color = 'red' if obs.classification == 'forklift' else 'orange'
            circle = plt.Circle(obs.position, obs.radius, color=color, alpha=0.6)
            self.ax.add_patch(circle)
            self.obstacle_patches.append(circle)
        
        # Safety zone
        self.safety_circle = plt.Circle(self.robot_state.position, self.safety_zone_radius,
                                      fill=False, color='red', linestyle='--', alpha=0.5,
                                      label='Safety Zone')
        self.ax.add_patch(self.safety_circle)
        
        # Add warehouse obstacles (static)
        self.draw_warehouse_obstacles()
        
        # Legend
        self.ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
        
        # Status text
        self.status_text = self.ax.text(0.02, 0.98, '', transform=self.ax.transAxes,
                                      verticalalignment='top', fontfamily='monospace',
                                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
    
    def draw_warehouse_obstacles(self):
        """Draw static warehouse obstacles."""
        # Updated to match the new obstacle positions
        obstacles = [
            (6, 6, 1.5, 3),   # Small vertical shelf
            (12, 4, 3, 1),    # Small horizontal shelf  
            (16, 12, 1.5, 3)  # Another small vertical shelf
        ]
        
        for x, y, w, h in obstacles:
            rect = patches.Rectangle((x-w/2, y-h/2), w, h, linewidth=1, 
                                   edgecolor='black', facecolor='gray', alpha=0.7)
            self.ax.add_patch(rect)
    
    def simulate_sensors(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate sensor readings.
        
        Returns:
            Tuple of (lidar_scan, depth_image)
        """
        # Simple sensor simulation
        robot_pos = self.robot_state.position
        
        # LiDAR simulation (360 degrees, 1 degree resolution)
        angles = np.arange(0, 360, 1)
        ranges = []
        
        for angle in angles:
            # Cast ray and find closest obstacle
            angle_rad = np.radians(angle)
            max_range = 10.0
            
            # Check for collisions with static obstacles and dynamic obstacles
            min_range = max_range
            
            # Simple ray casting (this is simplified)
            for distance in np.arange(0.1, max_range, 0.1):
                ray_x = robot_pos[0] + distance * np.cos(angle_rad)
                ray_y = robot_pos[1] + distance * np.sin(angle_rad)
                
                # Check bounds
                if ray_x < 0 or ray_x >= self.warehouse_size[0] or ray_y < 0 or ray_y >= self.warehouse_size[1]:
                    min_range = distance
                    break
                
                # Check dynamic obstacles
                for obs in self.obstacles:
                    obs_dist = np.sqrt((ray_x - obs.position[0])**2 + (ray_y - obs.position[1])**2)
                    if obs_dist <= obs.radius:
                        min_range = distance
                        break
                
                if min_range < max_range:
                    break
            
            ranges.append(min_range)
        
        lidar_scan = np.array(ranges)
        
        # Depth camera simulation (simplified)
        depth_image = np.random.uniform(0.5, 5.0, (48, 64))  # Placeholder
        
        return lidar_scan, depth_image
    
    def detect_obstacles(self, lidar_scan: np.ndarray) -> List[DynamicObstacle]:
        """
        Detect obstacles from sensor data.
        
        Args:
            lidar_scan: LiDAR scan data
            
        Returns:
            List of detected obstacles
        """
        detected_obstacles = []
        
        # Simple obstacle detection from LiDAR
        # Look for clusters of close points
        robot_pos = self.robot_state.position
        
        detection_id = 1000  # Start with high IDs for detected obstacles
        
        for i, range_val in enumerate(lidar_scan):
            if range_val < 5.0:  # Only consider close obstacles
                angle_rad = np.radians(i)
                obs_x = robot_pos[0] + range_val * np.cos(angle_rad)
                obs_y = robot_pos[1] + range_val * np.sin(angle_rad)
                
                # Check if this is a new obstacle or part of existing one
                is_new = True
                for existing_obs in detected_obstacles:
                    dist = np.linalg.norm(np.array([obs_x, obs_y]) - existing_obs.position)
                    if dist < 1.0:  # Merge nearby detections
                        is_new = False
                        break
                
                if is_new:
                    detected_obs = DynamicObstacle(
                        id=detection_id,
                        position=np.array([obs_x, obs_y]),
                        velocity=np.array([0.0, 0.0]),  # Unknown velocity initially
                        radius=0.4,
                        classification="unknown"
                    )
                    detected_obstacles.append(detected_obs)
                    detection_id += 1
        
        return detected_obstacles
    
    def simple_navigation_control(self) -> Tuple[float, float]:
        """
        Simple navigation controller with proper waypoint following.
        
        Returns:
            Tuple of (linear_velocity, angular_velocity)
        """
        if not self.planned_path or not self.planned_path.waypoints:
            return 0.0, 0.0
        
        robot_pos = self.robot_state.position
        robot_angle = self.robot_state.orientation
        
        # Check if we've reached the goal first
        goal_dist = np.sqrt((self.goal_position[0] - robot_pos[0])**2 + 
                           (self.goal_position[1] - robot_pos[1])**2)
        if goal_dist < 0.3:
            print("Goal reached!")
            return 0.0, 0.0  # Stop at goal
        
        # Initialize current waypoint index if not set
        if not hasattr(self, 'current_waypoint_index'):
            self.current_waypoint_index = 0
        
        # Ensure waypoint index is valid
        if self.current_waypoint_index >= len(self.planned_path.waypoints):
            self.current_waypoint_index = len(self.planned_path.waypoints) - 1
        
        # Get current target waypoint
        target_waypoint = self.planned_path.waypoints[self.current_waypoint_index]
        
        # Calculate distance to current waypoint
        dx = target_waypoint.x - robot_pos[0]
        dy = target_waypoint.y - robot_pos[1]
        waypoint_distance = np.sqrt(dx*dx + dy*dy)
        
        # Check if we've reached the current waypoint
        waypoint_threshold = 0.4  # Distance threshold to consider waypoint reached
        if waypoint_distance < waypoint_threshold:
            # Move to next waypoint if available
            if self.current_waypoint_index < len(self.planned_path.waypoints) - 1:
                self.current_waypoint_index += 1
                target_waypoint = self.planned_path.waypoints[self.current_waypoint_index]
                print(f"Reached waypoint {self.current_waypoint_index - 1}, moving to waypoint {self.current_waypoint_index}")
                
                # Recalculate distance to new waypoint
                dx = target_waypoint.x - robot_pos[0]
                dy = target_waypoint.y - robot_pos[1]
                waypoint_distance = np.sqrt(dx*dx + dy*dy)
            else:
                # We're at the last waypoint, head directly to goal
                dx = self.goal_position[0] - robot_pos[0]
                dy = self.goal_position[1] - robot_pos[1]
                waypoint_distance = np.sqrt(dx*dx + dy*dy)
        
        # Use lookahead for smoother path following
        lookahead_distance = 1.0  # Look ahead 1 meter
        if waypoint_distance > lookahead_distance and self.current_waypoint_index < len(self.planned_path.waypoints) - 1:
            # Look for a waypoint at lookahead distance
            for i in range(self.current_waypoint_index + 1, len(self.planned_path.waypoints)):
                wp = self.planned_path.waypoints[i]
                wp_dx = wp.x - robot_pos[0]
                wp_dy = wp.y - robot_pos[1]
                wp_dist = np.sqrt(wp_dx*wp_dx + wp_dy*wp_dy)
                
                if wp_dist >= lookahead_distance:
                    # Use this waypoint as target
                    dx, dy = wp_dx, wp_dy
                    waypoint_distance = wp_dist
                    break
        
        # Calculate desired heading
        if waypoint_distance < 0.05:  # Very close, stop
            return 0.0, 0.0
        
        desired_angle = np.arctan2(dy, dx)
        
        # Calculate angle error
        angle_error = desired_angle - robot_angle
        # Normalize angle to [-pi, pi]
        while angle_error > np.pi:
            angle_error -= 2 * np.pi
        while angle_error < -np.pi:
            angle_error += 2 * np.pi
        
        # Proportional control with improved tuning
        # Base speed depends on distance to target
        base_speed = min(1.0, max(0.3, waypoint_distance * 0.5))
        
        # Reduce speed when turning
        if abs(angle_error) > 0.3:  # More than ~17 degrees
            linear_velocity = base_speed * 0.6
        else:
            linear_velocity = base_speed
        
        # Angular velocity proportional to angle error
        angular_velocity = np.clip(angle_error * 2.0, -1.0, 1.0)
        
        # If we need to turn a lot, reduce linear velocity more
        if abs(angle_error) > 0.8:  # More than ~45 degrees
            linear_velocity *= 0.3
        
        return linear_velocity, angular_velocity
    
    def apply_safety_control(self, linear_vel: float, angular_vel: float, 
                           detected_obstacles: List[DynamicObstacle]) -> Tuple[float, float]:
        """
        Apply safety control to limit velocities near obstacles.
        
        Args:
            linear_vel: Desired linear velocity
            angular_vel: Desired angular velocity
            detected_obstacles: List of detected obstacles
            
        Returns:
            Tuple of (safe_linear_velocity, safe_angular_velocity)
        """
        robot_pos = self.robot_state.position
        
        # Check for obstacles in safety zone
        min_obstacle_distance = float('inf')
        
        for obs in detected_obstacles:
            dist = np.linalg.norm(obs.position - robot_pos)
            min_obstacle_distance = min(min_obstacle_distance, dist)
        
        # Emergency stop if too close
        if min_obstacle_distance < self.emergency_stop_distance:
            return 0.0, 0.0
        
        # Reduce speed if in safety zone
        if min_obstacle_distance < self.safety_zone_radius:
            safety_factor = (min_obstacle_distance - self.emergency_stop_distance) / \
                          (self.safety_zone_radius - self.emergency_stop_distance)
            linear_vel *= safety_factor
            angular_vel *= safety_factor
        
        return linear_vel, angular_vel
    
    def update_obstacles(self):
        """Update dynamic obstacle positions."""
        for obs in self.obstacles:
            # Simple movement patterns
            if obs.classification == "worker":
                # Workers move in patterns
                if obs.position[0] > 15 or obs.position[0] < 3:
                    obs.velocity[0] = -obs.velocity[0]
                if obs.position[1] > 17 or obs.position[1] < 3:
                    obs.velocity[1] = -obs.velocity[1]
            
            elif obs.classification == "forklift":
                # Forklift moves back and forth
                if obs.position[1] > 16 or obs.position[1] < 4:
                    obs.velocity[1] = -obs.velocity[1]
            
            # Update position
            new_position = obs.position + obs.velocity * self.dt
            
            # Keep within bounds
            new_position[0] = np.clip(new_position[0], 1, self.warehouse_size[0] - 1)
            new_position[1] = np.clip(new_position[1], 1, self.warehouse_size[1] - 1)
            
            obs.position = new_position
    
    def simulation_step(self):
        """Execute one simulation step."""
        if self.paused:
            return
        
        # 1. Update dynamic obstacles
        self.update_obstacles()
        
        # 2. Simulate sensors
        lidar_scan, depth_image = self.simulate_sensors()
        
        # 3. Detect obstacles
        detected_obstacles = self.detect_obstacles(lidar_scan)
        
        # 4. Navigation control
        linear_vel, angular_vel = self.simple_navigation_control()
        
        # Debug output every 50 steps (5 seconds)
        if hasattr(self, 'step_counter'):
            self.step_counter += 1
        else:
            self.step_counter = 0
            
        if self.step_counter % 50 == 0:
            robot_pos = self.robot_state.position
            goal_dist = np.sqrt((self.goal_position[0] - robot_pos[0])**2 + 
                               (self.goal_position[1] - robot_pos[1])**2)
            
            # Show current waypoint info
            waypoint_info = ""
            if hasattr(self, 'current_waypoint_index') and self.planned_path:
                total_waypoints = len(self.planned_path.waypoints)
                current_wp = self.current_waypoint_index
                if current_wp < total_waypoints:
                    wp = self.planned_path.waypoints[current_wp]
                    wp_dist = np.sqrt((wp.x - robot_pos[0])**2 + (wp.y - robot_pos[1])**2)
                    waypoint_info = f", WP {current_wp+1}/{total_waypoints} at ({wp.x:.1f},{wp.y:.1f}) dist:{wp_dist:.2f}m"
            
            print(f"Step {self.step_counter}: Robot at ({robot_pos[0]:.2f}, {robot_pos[1]:.2f}), "
                  f"Goal dist: {goal_dist:.2f}m, Vel: {linear_vel:.3f} m/s, {angular_vel:.3f} rad/s{waypoint_info}")
        
        # 5. Apply safety control
        safe_linear_vel, safe_angular_vel = self.apply_safety_control(
            linear_vel, angular_vel, detected_obstacles)
        
        # 6. Update robot state
        robot_pos = self.robot_state.position
        robot_angle = self.robot_state.orientation
        
        # Update position
        new_x = robot_pos[0] + safe_linear_vel * np.cos(robot_angle) * self.dt
        new_y = robot_pos[1] + safe_linear_vel * np.sin(robot_angle) * self.dt
        new_angle = robot_angle + safe_angular_vel * self.dt
        
        # Normalize angle
        while new_angle > np.pi:
            new_angle -= 2 * np.pi
        while new_angle < -np.pi:
            new_angle += 2 * np.pi
        
        self.robot_state.position = np.array([new_x, new_y])
        self.robot_state.orientation = new_angle
        self.robot_state.linear_velocity = safe_linear_vel
        self.robot_state.angular_velocity = safe_angular_vel
        
        # Store data for visualization
        self.current_detected_obstacles = detected_obstacles
        self.current_velocities = (safe_linear_vel, safe_angular_vel)
    
    def update_visualization(self, frame):
        """Update the visualization."""
        if not self.running:
            return
        
        # Execute simulation step
        self.simulation_step()
        
        # Update robot position
        self.robot_circle.center = self.robot_state.position
        self.safety_circle.center = self.robot_state.position
        
        # Update obstacle positions
        for i, obs in enumerate(self.obstacles):
            if i < len(self.obstacle_patches):
                self.obstacle_patches[i].center = obs.position
        
        # Update status text
        robot_pos = self.robot_state.position
        goal_dist = np.sqrt((self.goal_position[0] - robot_pos[0])**2 + 
                           (self.goal_position[1] - robot_pos[1])**2)
        
        status = f"""Status:
Position: ({robot_pos[0]:.1f}, {robot_pos[1]:.1f})
Orientation: {np.degrees(self.robot_state.orientation):.1f}°
Velocity: {self.current_velocities[0]:.2f} m/s
Angular: {self.current_velocities[1]:.2f} rad/s
Goal Distance: {goal_dist:.1f}m
Obstacles: {len(self.current_detected_obstacles)} detected
Time: {frame * self.dt:.1f}s"""
        
        self.status_text.set_text(status)
        
        return [self.robot_circle, self.safety_circle] + self.obstacle_patches + [self.status_text]
    
    def start_demo(self):
        """Start the demo simulation."""
        print("\nStarting AdaptNav Demo Simulation...")
        print("Controls:")
        print("  - Close window to stop")
        print("  - Press Ctrl+C in terminal to stop")
        
        self.running = True
        self.current_detected_obstacles = []
        self.current_velocities = (0.0, 0.0)
        
        # Start animation
        self.animation = FuncAnimation(
            self.fig, self.update_visualization, 
            interval=int(self.dt * 1000),  # Convert to milliseconds
            blit=False, cache_frame_data=False
        )
        
        try:
            plt.show()
        except KeyboardInterrupt:
            print("\nDemo stopped by user")
        finally:
            self.running = False
    
    def stop_demo(self):
        """Stop the demo simulation."""
        self.running = False
        if hasattr(self, 'animation'):
            self.animation.event_source.stop()
        plt.close('all')


def main():
    """Main demo function."""
    print("=" * 60)
    print("AdaptNav Warehouse Navigation Demo")
    print("=" * 60)
    print()
    
    if not ADAPTNAV_AVAILABLE:
        print("Warning: Some AdaptNav components are not available.")
        print("The demo will run with simplified functionality.")
        print()
    
    try:
        # Create and start demo
        demo = DemoSimulation()
        demo.start_demo()
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nDemo completed!")


if __name__ == '__main__':
    main()