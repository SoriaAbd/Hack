#!/usr/bin/env python3
"""
ROS 2 node to run MuJoCo simulation.

This script creates and runs the MuJoCo simulation as a ROS 2 node,
handling the simulation loop and ROS 2 communication.
"""

import rclpy
from rclpy.executors import MultiThreadedExecutor
from adaptnav.simulation.mujoco_simulation import MuJoCoSimulation


def main(args=None):
    """Main function to run MuJoCo simulation node."""
    rclpy.init(args=args)
    
    # Create simulation node
    sim = MuJoCoSimulation(render_mode=None)  # Set to 'human' for visualization
    
    # Create executor
    executor = MultiThreadedExecutor()
    executor.add_node(sim)
    
    # Simulation parameters
    dt = 0.01  # 100 Hz simulation
    rate = sim.create_rate(100)
    
    sim.get_logger().info('Starting MuJoCo simulation loop...')
    
    try:
        while rclpy.ok():
            # Step simulation
            success = sim.step(dt)
            
            if not success:
                sim.get_logger().error('Simulation step failed, stopping...')
                break
            
            # Spin to process callbacks
            executor.spin_once(timeout_sec=0)
            
            # Sleep to maintain rate
            rate.sleep()
            
    except KeyboardInterrupt:
        sim.get_logger().info('Simulation interrupted by user')
    finally:
        sim.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
