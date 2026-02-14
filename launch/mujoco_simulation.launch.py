#!/usr/bin/env python3
"""
Launch file for MuJoCo warehouse simulation.

This launch file starts the MuJoCo simulation node with appropriate parameters.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for MuJoCo simulation."""
    
    # Declare launch arguments
    render_mode_arg = DeclareLaunchArgument(
        'render_mode',
        default_value='None',
        description='Rendering mode: None (headless), human (viewer), or rgb_array'
    )
    
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value='',
        description='Path to custom MJCF model file (empty for default warehouse model)'
    )
    
    # MuJoCo simulation node
    mujoco_sim_node = Node(
        package='adaptnav',
        executable='run_mujoco_sim.py',
        name='mujoco_simulation',
        output='screen',
        parameters=[{
            'render_mode': LaunchConfiguration('render_mode'),
            'model_path': LaunchConfiguration('model_path'),
        }]
    )
    
    return LaunchDescription([
        render_mode_arg,
        model_path_arg,
        mujoco_sim_node,
    ])
