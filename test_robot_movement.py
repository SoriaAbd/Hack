#!/usr/bin/env python3
"""
Quick test to verify robot movement is working correctly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from demo_simulation import DemoSimulation
import numpy as np
import time

def test_robot_movement():
    """Test that the robot can move and reach waypoints."""
    print("🤖 Testing Robot Movement")
    print("=" * 50)
    
    # Create demo simulation
    demo = DemoSimulation()
    
    # Run simulation for a few steps
    print(f"Initial robot position: {demo.robot_state.position}")
    print(f"Goal position: {demo.goal_position}")
    print(f"Initial distance to goal: {np.linalg.norm(demo.goal_position - demo.robot_state.position):.2f}m")
    
    if demo.planned_path:
        print(f"Path has {len(demo.planned_path.waypoints)} waypoints")
        if hasattr(demo, 'current_waypoint_index'):
            print(f"Starting at waypoint index: {demo.current_waypoint_index}")
    
    print("\nRunning simulation steps...")
    
    # Run for 300 steps (30 seconds)
    for step in range(300):
        demo.simulation_step()
        
        # Check every 20 steps
        if step % 20 == 0:
            pos = demo.robot_state.position
            goal_dist = np.linalg.norm(demo.goal_position - pos)
            vel = demo.robot_state.linear_velocity
            
            waypoint_info = ""
            if hasattr(demo, 'current_waypoint_index') and demo.planned_path:
                wp_idx = demo.current_waypoint_index
                total_wp = len(demo.planned_path.waypoints)
                waypoint_info = f" (WP {wp_idx+1}/{total_wp})"
            
            print(f"Step {step:3d}: Pos=({pos[0]:.2f}, {pos[1]:.2f}), "
                  f"Goal_dist={goal_dist:.2f}m, Vel={vel:.3f}m/s{waypoint_info}")
            
            # Check if robot is moving
            if step > 0 and vel < 0.01:
                print("⚠️  Robot appears to be stopped")
            
            # Check if goal reached
            if goal_dist < 0.5:
                print("🎯 Goal reached!")
                break
    
    # Final status
    final_pos = demo.robot_state.position
    final_goal_dist = np.linalg.norm(demo.goal_position - final_pos)
    
    print(f"\nFinal Results:")
    print(f"  Final position: ({final_pos[0]:.2f}, {final_pos[1]:.2f})")
    print(f"  Distance to goal: {final_goal_dist:.2f}m")
    print(f"  Goal position: ({demo.goal_position[0]:.2f}, {demo.goal_position[1]:.2f})")
    
    if final_goal_dist < 1.0:
        print("✅ SUCCESS: Robot moved significantly toward goal!")
    elif final_goal_dist < 15.0:
        print("⚠️  PARTIAL: Robot moved but didn't reach goal")
    else:
        print("❌ FAILED: Robot didn't move much")
    
    return final_goal_dist < 1.0

if __name__ == "__main__":
    success = test_robot_movement()
    sys.exit(0 if success else 1)