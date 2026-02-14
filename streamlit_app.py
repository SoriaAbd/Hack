#!/usr/bin/env python3
"""
AdaptNav Streamlit Web Demo

A web-based interactive demo of the AdaptNav warehouse navigation system.
Runs in the browser with real-time visualization.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import time
import sys
import os
from typing import List, Tuple, Optional

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import AdaptNav components
try:
    from adaptnav.core.warehouse_map import WarehouseMap
    from adaptnav.core.dynamic_obstacle import DynamicObstacle
    from adaptnav.core.robot_state import RobotState
    from adaptnav.core.path import Path, Waypoint
    from adaptnav.planning.astar_planner import AStarPlanner
    ADAPTNAV_AVAILABLE = True
except ImportError as e:
    ADAPTNAV_AVAILABLE = False


# Page configuration
st.set_page_config(
    page_title="AdaptNav - Warehouse Navigation Demo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .status-success {
        color: #28a745;
        font-weight: bold;
    }
    .status-warning {
        color: #ffc107;
        font-weight: bold;
    }
    .status-danger {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


class StreamlitDemoSimulation:
    """Streamlit-compatible demo simulation."""
    
    def __init__(self):
        """Initialize the demo simulation."""
        self.warehouse_size = (20, 20)
        self.robot_radius = 0.3
        self.dt = 0.1
        
        self.setup_warehouse()
        self.setup_robot()
        self.setup_obstacles()
        self.setup_planner()
        
        self.step_counter = 0
        self.current_detected_obstacles = []
        self.current_velocities = (0.0, 0.0)
    
    def setup_warehouse(self):
        """Set up the warehouse environment."""
        self.warehouse_map = WarehouseMap(
            width=self.warehouse_size[0],
            height=self.warehouse_size[1],
            resolution=0.1
        )
        
        self.warehouse_map.set_obstacle(6, 6, 1.5, 3)
        self.warehouse_map.set_obstacle(12, 4, 3, 1)
        self.warehouse_map.set_obstacle(16, 12, 1.5, 3)
    
    def setup_robot(self):
        """Set up the robot state."""
        start_x, start_y = 2.0, 2.0
        self.robot_state = RobotState(
            position=np.array([start_x, start_y]),
            orientation=0.0,
            linear_velocity=0.0,
            angular_velocity=0.0
        )
        self.goal_position = (17.0, 17.0)
    
    def setup_obstacles(self):
        """Set up dynamic obstacles."""
        self.obstacles = []
        
        worker1 = DynamicObstacle(
            id=1,
            position=np.array([8.0, 8.0]),
            velocity=np.array([0.5, 0.0]),
            radius=0.4,
            classification="worker"
        )
        self.obstacles.append(worker1)
        
        worker2 = DynamicObstacle(
            id=2,
            position=np.array([12.0, 15.0]),
            velocity=np.array([-0.3, 0.0]),
            radius=0.4,
            classification="worker"
        )
        self.obstacles.append(worker2)
        
        forklift = DynamicObstacle(
            id=3,
            position=np.array([6.0, 12.0]),
            velocity=np.array([0.0, -0.2]),
            radius=0.8,
            classification="forklift"
        )
        self.obstacles.append(forklift)
    
    def setup_planner(self):
        """Set up the path planner."""
        if ADAPTNAV_AVAILABLE:
            self.planner = AStarPlanner(self.warehouse_map)
            start_x, start_y = self.robot_state.position
            goal_x, goal_y = self.goal_position
            
            planned_waypoints = self.planner.plan_path(start_x, start_y, goal_x, goal_y)
            if planned_waypoints:
                self.planned_path = Path(planned_waypoints)
                self.current_waypoint_index = 0
            else:
                self.planned_path = Path([
                    Waypoint(start_x, start_y, 0),
                    Waypoint(goal_x, goal_y, 0)
                ])
                self.current_waypoint_index = 0
        else:
            start_x, start_y = self.robot_state.position
            goal_x, goal_y = self.goal_position
            self.planned_path = Path([
                Waypoint(start_x, start_y, 0),
                Waypoint(goal_x, goal_y, 0)
            ])
            self.current_waypoint_index = 0
    
    def simple_navigation_control(self) -> Tuple[float, float]:
        """Simple navigation controller."""
        if not self.planned_path or not self.planned_path.waypoints:
            return 0.0, 0.0
        
        robot_pos = self.robot_state.position
        robot_angle = self.robot_state.orientation
        
        goal_dist = np.sqrt((self.goal_position[0] - robot_pos[0])**2 + 
                           (self.goal_position[1] - robot_pos[1])**2)
        if goal_dist < 0.3:
            return 0.0, 0.0
        
        if not hasattr(self, 'current_waypoint_index'):
            self.current_waypoint_index = 0
        
        if self.current_waypoint_index >= len(self.planned_path.waypoints):
            self.current_waypoint_index = len(self.planned_path.waypoints) - 1
        
        target_waypoint = self.planned_path.waypoints[self.current_waypoint_index]
        
        dx = target_waypoint.x - robot_pos[0]
        dy = target_waypoint.y - robot_pos[1]
        waypoint_distance = np.sqrt(dx*dx + dy*dy)
        
        waypoint_threshold = 0.4
        if waypoint_distance < waypoint_threshold:
            if self.current_waypoint_index < len(self.planned_path.waypoints) - 1:
                self.current_waypoint_index += 1
                target_waypoint = self.planned_path.waypoints[self.current_waypoint_index]
                dx = target_waypoint.x - robot_pos[0]
                dy = target_waypoint.y - robot_pos[1]
                waypoint_distance = np.sqrt(dx*dx + dy*dy)
            else:
                dx = self.goal_position[0] - robot_pos[0]
                dy = self.goal_position[1] - robot_pos[1]
                waypoint_distance = np.sqrt(dx*dx + dy*dy)
        
        if waypoint_distance < 0.05:
            return 0.0, 0.0
        
        desired_angle = np.arctan2(dy, dx)
        angle_error = desired_angle - robot_angle
        
        while angle_error > np.pi:
            angle_error -= 2 * np.pi
        while angle_error < -np.pi:
            angle_error += 2 * np.pi
        
        base_speed = min(1.0, max(0.3, waypoint_distance * 0.5))
        
        if abs(angle_error) > 0.3:
            linear_velocity = base_speed * 0.6
        else:
            linear_velocity = base_speed
        
        angular_velocity = np.clip(angle_error * 2.0, -1.0, 1.0)
        
        if abs(angle_error) > 0.8:
            linear_velocity *= 0.3
        
        return linear_velocity, angular_velocity
    
    def update_obstacles(self):
        """Update dynamic obstacle positions."""
        for obs in self.obstacles:
            if obs.classification == "worker":
                if obs.position[0] > 15 or obs.position[0] < 3:
                    obs.velocity[0] = -obs.velocity[0]
                if obs.position[1] > 17 or obs.position[1] < 3:
                    obs.velocity[1] = -obs.velocity[1]
            elif obs.classification == "forklift":
                if obs.position[1] > 16 or obs.position[1] < 4:
                    obs.velocity[1] = -obs.velocity[1]
            
            new_position = obs.position + obs.velocity * self.dt
            new_position[0] = np.clip(new_position[0], 1, self.warehouse_size[0] - 1)
            new_position[1] = np.clip(new_position[1], 1, self.warehouse_size[1] - 1)
            obs.position = new_position
    
    def simulation_step(self):
        """Execute one simulation step."""
        self.update_obstacles()
        linear_vel, angular_vel = self.simple_navigation_control()
        
        robot_pos = self.robot_state.position
        robot_angle = self.robot_state.orientation
        
        new_x = robot_pos[0] + linear_vel * np.cos(robot_angle) * self.dt
        new_y = robot_pos[1] + linear_vel * np.sin(robot_angle) * self.dt
        new_angle = robot_angle + angular_vel * self.dt
        
        while new_angle > np.pi:
            new_angle -= 2 * np.pi
        while new_angle < -np.pi:
            new_angle += 2 * np.pi
        
        self.robot_state.position = np.array([new_x, new_y])
        self.robot_state.orientation = new_angle
        self.robot_state.linear_velocity = linear_vel
        self.robot_state.angular_velocity = angular_vel
        
        self.current_velocities = (linear_vel, angular_vel)
        self.step_counter += 1
    
    def create_visualization(self):
        """Create matplotlib visualization."""
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlim(0, self.warehouse_size[0])
        ax.set_ylim(0, self.warehouse_size[1])
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title('AdaptNav Warehouse Navigation', fontsize=16, fontweight='bold')
        ax.set_xlabel('X (meters)', fontsize=12)
        ax.set_ylabel('Y (meters)', fontsize=12)
        
        # Draw warehouse obstacles
        obstacles_static = [
            (6, 6, 1.5, 3),
            (12, 4, 3, 1),
            (16, 12, 1.5, 3)
        ]
        for x, y, w, h in obstacles_static:
            rect = patches.Rectangle((x-w/2, y-h/2), w, h, linewidth=1,
                                   edgecolor='black', facecolor='gray', alpha=0.7)
            ax.add_patch(rect)
        
        # Draw planned path
        if self.planned_path and self.planned_path.waypoints:
            path_x = [wp.x for wp in self.planned_path.waypoints]
            path_y = [wp.y for wp in self.planned_path.waypoints]
            ax.plot(path_x, path_y, 'g--', alpha=0.6, linewidth=2, label='Planned Path')
        
        # Draw robot
        robot_circle = plt.Circle(self.robot_state.position, self.robot_radius,
                                 color='blue', alpha=0.7, label='Robot', zorder=5)
        ax.add_patch(robot_circle)
        
        # Draw robot orientation
        robot_pos = self.robot_state.position
        robot_angle = self.robot_state.orientation
        arrow_length = 0.5
        ax.arrow(robot_pos[0], robot_pos[1],
                arrow_length * np.cos(robot_angle),
                arrow_length * np.sin(robot_angle),
                head_width=0.2, head_length=0.2, fc='darkblue', ec='darkblue', zorder=6)
        
        # Draw goal
        goal_marker = plt.Circle(self.goal_position, 0.3,
                                color='green', alpha=0.8, label='Goal', zorder=5)
        ax.add_patch(goal_marker)
        
        # Draw dynamic obstacles
        for obs in self.obstacles:
            color = 'red' if obs.classification == 'forklift' else 'orange'
            circle = plt.Circle(obs.position, obs.radius, color=color, alpha=0.6, zorder=4)
            ax.add_patch(circle)
            ax.text(obs.position[0], obs.position[1] + obs.radius + 0.3,
                   obs.classification.capitalize(), ha='center', fontsize=8)
        
        # Draw safety zone
        safety_circle = plt.Circle(self.robot_state.position, 1.0,
                                  fill=False, color='red', linestyle='--',
                                  alpha=0.5, label='Safety Zone', zorder=3)
        ax.add_patch(safety_circle)
        
        ax.legend(loc='upper left', fontsize=10)
        
        return fig


# Initialize session state
if 'simulation' not in st.session_state:
    st.session_state.simulation = None
if 'running' not in st.session_state:
    st.session_state.running = False
if 'auto_run' not in st.session_state:
    st.session_state.auto_run = False


def main():
    """Main Streamlit app."""
    
    # Header
    st.markdown('<div class="main-header">🤖 AdaptNav</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Context-Aware Warehouse Navigation System</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Controls")
        
        if st.button("🔄 Initialize Simulation", use_container_width=True):
            st.session_state.simulation = StreamlitDemoSimulation()
            st.session_state.running = False
            st.success("Simulation initialized!")
        
        st.divider()
        
        if st.session_state.simulation is not None:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("▶️ Step", use_container_width=True):
                    st.session_state.simulation.simulation_step()
                    st.rerun()
            
            with col2:
                if st.button("⏩ Run 10", use_container_width=True):
                    for _ in range(10):
                        st.session_state.simulation.simulation_step()
                    st.rerun()
            
            st.divider()
            
            auto_run = st.checkbox("🔁 Auto-run", value=st.session_state.auto_run)
            st.session_state.auto_run = auto_run
            
            if auto_run:
                st.info("Auto-running... The simulation will update automatically.")
        
        st.divider()
        
        st.header("📊 About")
        st.markdown("""
        **AdaptNav** is a hybrid autonomous navigation system combining:
        
        - 🗺️ A* Path Planning
        - 🤖 Reinforcement Learning
        - 👁️ Sensor Fusion
        - 🛡️ Safety Control
        - 📈 Real-time Visualization
        
        Perfect for dynamic warehouse environments!
        """)
        
        st.divider()
        
        with st.expander("ℹ️ System Status"):
            st.write(f"**AdaptNav Components:** {'✅ Available' if ADAPTNAV_AVAILABLE else '⚠️ Limited'}")
            st.write(f"**Simulation:** {'✅ Ready' if st.session_state.simulation else '⏸️ Not initialized'}")
    
    # Main content
    if st.session_state.simulation is None:
        st.info("👈 Click 'Initialize Simulation' in the sidebar to start!")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Warehouse Size", "20x20m")
        with col2:
            st.metric("Robot Radius", "0.3m")
        with col3:
            st.metric("Dynamic Obstacles", "3")
        
        st.markdown("---")
        
        st.subheader("✨ Features")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            - **Global Path Planning** with A* algorithm
            - **Dynamic Obstacle Avoidance**
            - **Real-time Sensor Simulation**
            - **Safety Control System**
            """)
        
        with col2:
            st.markdown("""
            - **Waypoint Following**
            - **Collision Prevention**
            - **Live Visualization**
            - **Performance Metrics**
            """)
    
    else:
        sim = st.session_state.simulation
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        robot_pos = sim.robot_state.position
        goal_dist = np.sqrt((sim.goal_position[0] - robot_pos[0])**2 + 
                           (sim.goal_position[1] - robot_pos[1])**2)
        
        with col1:
            st.metric("Position", f"({robot_pos[0]:.1f}, {robot_pos[1]:.1f})")
        with col2:
            st.metric("Goal Distance", f"{goal_dist:.2f}m")
        with col3:
            st.metric("Velocity", f"{sim.current_velocities[0]:.2f} m/s")
        with col4:
            st.metric("Simulation Time", f"{sim.step_counter * sim.dt:.1f}s")
        
        # Status indicator
        if goal_dist < 0.3:
            st.success("🎯 Goal Reached!")
        elif goal_dist < 2.0:
            st.info("🎯 Approaching goal...")
        else:
            st.info("🚀 Navigating...")
        
        # Visualization
        st.subheader("📍 Live Warehouse View")
        fig = sim.create_visualization()
        st.pyplot(fig)
        plt.close(fig)
        
        # Detailed metrics
        with st.expander("📈 Detailed Metrics"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Robot State:**")
                st.write(f"- Orientation: {np.degrees(sim.robot_state.orientation):.1f}°")
                st.write(f"- Linear Velocity: {sim.current_velocities[0]:.3f} m/s")
                st.write(f"- Angular Velocity: {sim.current_velocities[1]:.3f} rad/s")
            
            with col2:
                st.write("**Navigation:**")
                if hasattr(sim, 'current_waypoint_index') and sim.planned_path:
                    total_wp = len(sim.planned_path.waypoints)
                    current_wp = sim.current_waypoint_index
                    st.write(f"- Waypoint: {current_wp + 1}/{total_wp}")
                    if current_wp < total_wp:
                        wp = sim.planned_path.waypoints[current_wp]
                        st.write(f"- Target: ({wp.x:.1f}, {wp.y:.1f})")
                st.write(f"- Steps: {sim.step_counter}")
        
        # Auto-run logic
        if st.session_state.auto_run and goal_dist >= 0.3:
            time.sleep(0.1)
            sim.simulation_step()
            st.rerun()


if __name__ == "__main__":
    main()
