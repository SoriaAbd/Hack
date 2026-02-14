"""
Launch file for AdaptNav navigation demonstration.

This launch file starts all required nodes for the complete navigation system:
- Simulation environment
- Sensor nodes
- Obstacle detector
- Global planner
- PPO agent
- Safety controller
- Navigation controller
- Visualization dashboard
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Generate launch description for AdaptNav navigation demo."""
    
    # Declare launch arguments
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value='default_config.yaml',
        description='Configuration file name'
    )
    
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation time'
    )
    
    # Get package share directory
    pkg_share = FindPackageShare('adaptnav')
    
    # Configuration file path
    config_file = PathJoinSubstitution([
        pkg_share,
        'config',
        LaunchConfiguration('config_file')
    ])
    
    # Nodes will be added as they are implemented
    # Example node structure:
    # simulation_node = Node(
    #     package='adaptnav',
    #     executable='simulation_node',
    #     name='simulation',
    #     parameters=[config_file, {'use_sim_time': LaunchConfiguration('use_sim_time')}],
    #     output='screen'
    # )
    
    return LaunchDescription([
        config_file_arg,
        use_sim_time_arg,
        # Nodes will be added here as they are implemented
    ])
