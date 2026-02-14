#!/usr/bin/env python3
"""
Debug script to identify why the robot isn't moving in the demo.

This script will check various components and provide diagnostic information
to help identify the root cause of movement issues.
"""

import sys
import os
import numpy as np

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from adaptnav.core.warehouse_map import WarehouseMap
    from adaptnav.core.robot_state import RobotState
    from adaptnav.core.path import Path, Waypoint
    from adaptnav.planning.astar_planner import AStarPlanner
    ADAPTNAV_AVAILABLE = True
except ImportError as e:
    print(f"AdaptNav components not available: {e}")
    ADAPTNAV_AVAILABLE = False


def test_warehouse_map():
    """Test warehouse map creation and obstacle placement."""
    print("=" * 50)
    print("TESTING WAREHOUSE MAP")
    print("=" * 50)
    
    # Create warehouse map (same as demo)
    warehouse_size = (20, 20)
    warehouse_map = WarehouseMap(
        width=warehouse_size[0],
        height=warehouse_size[1],
        resolution=0.1
    )
    
    print(f"Warehouse size: {warehouse_size[0]}x{warehouse_size[1]}m")
    print(f"Grid size: {warehouse_map.grid_width}x{warehouse_map.grid_height}")
    print(f"Resolution: {warehouse_map.resolution}m")
    
    # Add obstacles (same as demo)
    obstacles = [
        (5, 5, 2, 8),   # Vertical shelf
        (10, 3, 6, 2),  # Horizontal shelf
        (15, 10, 2, 6)  # Another vertical shelf
    ]
    
    print(f"\nAdding {len(obstacles)} obstacles:")
    for i, (x, y, w, h) in enumerate(obstacles):
        print(f"  Obstacle {i+1}: center=({x}, {y}), size={w}x{h}m")
        warehouse_map.set_obstacle(x, y, w, h)
    
    # Test robot positions
    robot_start = (2.0, 2.0)
    goal_position = (17.0, 17.0)
    robot_radius = 0.3
    
    print(f"\nTesting positions with robot radius {robot_radius}m:")
    
    start_valid = warehouse_map.is_collision_free(robot_start[0], robot_start[1], robot_radius)
    print(f"  Start position ({robot_start[0]}, {robot_start[1]}): {'VALID' if start_valid else 'INVALID'}")
    
    goal_valid = warehouse_map.is_collision_free(goal_position[0], goal_position[1], robot_radius)
    print(f"  Goal position ({goal_position[0]}, {goal_position[1]}): {'VALID' if goal_valid else 'INVALID'}")
    
    # Test some intermediate positions
    test_positions = [
        (5.0, 5.0), (10.0, 10.0), (15.0, 15.0),
        (1.0, 1.0), (19.0, 19.0), (10.0, 5.0)
    ]
    
    print(f"\nTesting intermediate positions:")
    for pos in test_positions:
        valid = warehouse_map.is_collision_free(pos[0], pos[1], robot_radius)
        print(f"  Position ({pos[0]}, {pos[1]}): {'VALID' if valid else 'INVALID'}")
    
    return warehouse_map, start_valid, goal_valid


def test_path_planning(warehouse_map):
    """Test A* path planning."""
    print("\n" + "=" * 50)
    print("TESTING PATH PLANNING")
    print("=" * 50)
    
    if not ADAPTNAV_AVAILABLE:
        print("AdaptNav not available - skipping path planning test")
        return None
    
    # Create planner
    planner = AStarPlanner(warehouse_map, robot_radius=0.3, timeout_seconds=5.0)
    
    # Test path planning
    start_x, start_y = 2.0, 2.0
    goal_x, goal_y = 17.0, 17.0
    
    print(f"Planning path from ({start_x}, {start_y}) to ({goal_x}, {goal_y})")
    
    try:
        path = planner.plan_path(start_x, start_y, goal_x, goal_y)
        
        if path is None:
            print("❌ PATH PLANNING FAILED - No path found!")
            
            # Try alternative goals
            alternative_goals = [
                (18.0, 18.0), (16.0, 16.0), (15.0, 15.0),
                (10.0, 18.0), (18.0, 10.0), (8.0, 8.0)
            ]
            
            print("\nTrying alternative goals:")
            for alt_goal in alternative_goals:
                alt_path = planner.plan_path(start_x, start_y, alt_goal[0], alt_goal[1])
                status = "✅ SUCCESS" if alt_path else "❌ FAILED"
                print(f"  To ({alt_goal[0]}, {alt_goal[1]}): {status}")
                if alt_path:
                    print(f"    Path length: {len(alt_path.waypoints)} waypoints")
                    break
            
            return None
        else:
            print(f"✅ PATH PLANNING SUCCESS!")
            print(f"  Path length: {len(path.waypoints)} waypoints")
            print(f"  First few waypoints:")
            for i, wp in enumerate(path.waypoints[:5]):
                print(f"    {i+1}: ({wp.x:.2f}, {wp.y:.2f})")
            if len(path.waypoints) > 5:
                print(f"    ... and {len(path.waypoints) - 5} more")
            
            return path
            
    except Exception as e:
        print(f"❌ PATH PLANNING ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_robot_movement_logic():
    """Test the robot movement control logic."""
    print("\n" + "=" * 50)
    print("TESTING ROBOT MOVEMENT LOGIC")
    print("=" * 50)
    
    # Simulate robot state
    robot_state = RobotState(
        position=np.array([2.0, 2.0]),
        orientation=0.0,
        linear_velocity=0.0,
        angular_velocity=0.0
    )
    
    # Create a simple test path
    test_waypoints = [
        Waypoint(2.0, 2.0, 0.0),
        Waypoint(5.0, 5.0, 0.0),
        Waypoint(10.0, 10.0, 0.0),
        Waypoint(17.0, 17.0, 0.0)
    ]
    test_path = Path(test_waypoints)
    
    print(f"Robot position: ({robot_state.position[0]:.2f}, {robot_state.position[1]:.2f})")
    print(f"Robot orientation: {np.degrees(robot_state.orientation):.1f}°")
    print(f"Test path waypoints: {len(test_path.waypoints)}")
    
    # Test navigation control logic (simplified version from demo)
    def simple_navigation_control(robot_pos, robot_angle, path, goal_pos):
        if not path or not path.waypoints:
            return 0.0, 0.0, "No path available"
        
        # Find closest waypoint ahead
        target_waypoint = None
        min_distance = float('inf')
        
        for waypoint in path.waypoints:
            dist = np.sqrt((waypoint.x - robot_pos[0])**2 + (waypoint.y - robot_pos[1])**2)
            if dist < min_distance:
                min_distance = dist
                target_waypoint = waypoint
        
        if target_waypoint is None:
            return 0.0, 0.0, "No target waypoint found"
        
        # Check if we've reached the goal
        goal_dist = np.sqrt((goal_pos[0] - robot_pos[0])**2 + (goal_pos[1] - robot_pos[1])**2)
        if goal_dist < 0.5:
            return 0.0, 0.0, "Goal reached"
        
        # Calculate desired heading
        dx = target_waypoint.x - robot_pos[0]
        dy = target_waypoint.y - robot_pos[1]
        desired_angle = np.arctan2(dy, dx)
        
        # Calculate angle error
        angle_error = desired_angle - robot_angle
        # Normalize angle to [-pi, pi]
        while angle_error > np.pi:
            angle_error -= 2 * np.pi
        while angle_error < -np.pi:
            angle_error += 2 * np.pi
        
        # Simple proportional control
        linear_velocity = min(1.0, min_distance * 0.5)
        angular_velocity = angle_error * 2.0
        
        # Limit velocities
        linear_velocity = np.clip(linear_velocity, 0.0, 1.0)
        angular_velocity = np.clip(angular_velocity, -1.0, 1.0)
        
        return linear_velocity, angular_velocity, f"Moving to waypoint ({target_waypoint.x:.2f}, {target_waypoint.y:.2f})"
    
    # Test movement control
    goal_position = (17.0, 17.0)
    linear_vel, angular_vel, status = simple_navigation_control(
        robot_state.position, robot_state.orientation, test_path, goal_position
    )
    
    print(f"\nMovement control test:")
    print(f"  Status: {status}")
    print(f"  Commanded linear velocity: {linear_vel:.3f} m/s")
    print(f"  Commanded angular velocity: {angular_vel:.3f} rad/s")
    
    if linear_vel == 0.0 and angular_vel == 0.0:
        print("  ❌ ROBOT WOULD NOT MOVE!")
        return False
    else:
        print("  ✅ Robot would move")
        return True


def test_safety_controller():
    """Test safety controller logic."""
    print("\n" + "=" * 50)
    print("TESTING SAFETY CONTROLLER")
    print("=" * 50)
    
    # Simulate safety controller logic
    robot_pos = np.array([2.0, 2.0])
    safety_zone_radius = 1.0
    emergency_stop_distance = 0.5
    max_velocity = 1.0
    
    # Test with no obstacles
    print("Test 1: No obstacles")
    detected_obstacles = []
    
    def apply_safety_control(linear_vel, angular_vel, robot_pos, obstacles):
        min_obstacle_distance = float('inf')
        
        for obs_pos in obstacles:
            dist = np.linalg.norm(obs_pos - robot_pos)
            min_obstacle_distance = min(min_obstacle_distance, dist)
        
        # Emergency stop if too close
        if min_obstacle_distance < emergency_stop_distance:
            return 0.0, 0.0, "EMERGENCY_STOP"
        
        # Reduce speed if in safety zone
        if min_obstacle_distance < safety_zone_radius:
            safety_factor = (min_obstacle_distance - emergency_stop_distance) / \
                          (safety_zone_radius - emergency_stop_distance)
            return linear_vel * safety_factor, angular_vel * safety_factor, "CAUTION"
        
        return linear_vel, angular_vel, "SAFE"
    
    # Test with no obstacles
    safe_linear, safe_angular, safety_status = apply_safety_control(
        0.5, 0.2, robot_pos, []
    )
    print(f"  Input: linear=0.5, angular=0.2")
    print(f"  Output: linear={safe_linear:.3f}, angular={safe_angular:.3f}")
    print(f"  Status: {safety_status}")
    
    # Test with distant obstacle
    print("\nTest 2: Distant obstacle")
    distant_obstacle = [np.array([8.0, 8.0])]
    safe_linear, safe_angular, safety_status = apply_safety_control(
        0.5, 0.2, robot_pos, distant_obstacle
    )
    print(f"  Obstacle at (8.0, 8.0) - distance: {np.linalg.norm(distant_obstacle[0] - robot_pos):.2f}m")
    print(f"  Output: linear={safe_linear:.3f}, angular={safe_angular:.3f}")
    print(f"  Status: {safety_status}")
    
    # Test with close obstacle
    print("\nTest 3: Close obstacle (in safety zone)")
    close_obstacle = [np.array([2.8, 2.8])]
    safe_linear, safe_angular, safety_status = apply_safety_control(
        0.5, 0.2, robot_pos, close_obstacle
    )
    print(f"  Obstacle at (2.8, 2.8) - distance: {np.linalg.norm(close_obstacle[0] - robot_pos):.2f}m")
    print(f"  Output: linear={safe_linear:.3f}, angular={safe_angular:.3f}")
    print(f"  Status: {safety_status}")
    
    # Test with very close obstacle
    print("\nTest 4: Very close obstacle (emergency stop)")
    very_close_obstacle = [np.array([2.3, 2.3])]
    safe_linear, safe_angular, safety_status = apply_safety_control(
        0.5, 0.2, robot_pos, very_close_obstacle
    )
    print(f"  Obstacle at (2.3, 2.3) - distance: {np.linalg.norm(very_close_obstacle[0] - robot_pos):.2f}m")
    print(f"  Output: linear={safe_linear:.3f}, angular={safe_angular:.3f}")
    print(f"  Status: {safety_status}")
    
    return True


def main():
    """Main diagnostic function."""
    print("🔍 AdaptNav Robot Movement Diagnostic Tool")
    print("=" * 60)
    
    # Test 1: Warehouse Map
    warehouse_map, start_valid, goal_valid = test_warehouse_map()
    
    if not start_valid:
        print("\n❌ PROBLEM FOUND: Robot start position is invalid!")
        print("   The robot is starting inside an obstacle or outside bounds.")
        return
    
    if not goal_valid:
        print("\n❌ PROBLEM FOUND: Goal position is invalid!")
        print("   The goal is inside an obstacle or outside bounds.")
        return
    
    # Test 2: Path Planning
    path = test_path_planning(warehouse_map)
    
    if path is None:
        print("\n❌ PROBLEM FOUND: Path planning failed!")
        print("   The robot cannot find a path to the goal.")
        print("   This is likely why the robot isn't moving.")
        return
    
    # Test 3: Robot Movement Logic
    movement_ok = test_robot_movement_logic()
    
    if not movement_ok:
        print("\n❌ PROBLEM FOUND: Robot movement logic issue!")
        print("   The movement controller is not generating velocity commands.")
        return
    
    # Test 4: Safety Controller
    test_safety_controller()
    
    print("\n" + "=" * 60)
    print("🎉 DIAGNOSTIC COMPLETE")
    print("=" * 60)
    print("\nAll major components appear to be working correctly.")
    print("If the robot still isn't moving in the demo, check:")
    print("1. Make sure the demo animation is running (window should update)")
    print("2. Check if the robot is very close to the goal already")
    print("3. Look for error messages in the demo console output")
    print("4. Try clicking on the demo window to ensure it has focus")
    
    print(f"\nDiagnostic completed successfully!")


if __name__ == '__main__':
    main()