"""
Complete AdaptNav system launch file.

This launch file starts all required nodes for the complete navigation system:
- MuJoCo simulation environment
- Sensor simulation (LiDAR and depth camera)
- Obstacle detector with sensor fusion
- Global path planner (A*)
- PPO agent for local navigation
- Safety controller
- Navigation controller
- RViz2 visualization
- Web dashboard backend
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, ExecuteProcess
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.conditions import IfCondition


def generate_launch_description():
    """Generate launch description for complete AdaptNav system."""
    
    # Declare launch arguments
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value='default_config.yaml',
        description='Configuration file name'
    )
    
    warehouse_layout_arg = DeclareLaunchArgument(
        'warehouse_layout',
        default_value='medium_warehouse.yaml',
        description='Warehouse layout configuration file'
    )
    
    obstacle_scenario_arg = DeclareLaunchArgument(
        'obstacle_scenario',
        default_value='multi_obstacle.yaml',
        description='Obstacle scenario configuration file'
    )
    
    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz',
        default_value='true',
        description='Launch RViz2 visualization'
    )
    
    use_web_dashboard_arg = DeclareLaunchArgument(
        'use_web_dashboard',
        default_value='true',
        description='Launch web dashboard backend'
    )
    
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation time'
    )
    
    ppo_model_path_arg = DeclareLaunchArgument(
        'ppo_model_path',
        default_value='models/ppo_navigation_model.zip',
        description='Path to trained PPO model'
    )
    
    # Get package share directory
    pkg_share = FindPackageShare('adaptnav')
    
    # Configuration file paths
    config_file = PathJoinSubstitution([
        pkg_share,
        'config',
        LaunchConfiguration('config_file')
    ])
    
    warehouse_layout_file = PathJoinSubstitution([
        pkg_share,
        'config/warehouse_layouts',
        LaunchConfiguration('warehouse_layout')
    ])
    
    obstacle_scenario_file = PathJoinSubstitution([
        pkg_share,
        'config/obstacle_scenarios',
        LaunchConfiguration('obstacle_scenario')
    ])
    
    rviz_config_file = PathJoinSubstitution([
        pkg_share,
        'config',
        'adaptnav_rviz.rviz'
    ])
    
    # Include MuJoCo simulation launch file
    mujoco_simulation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                pkg_share,
                'launch',
                'mujoco_simulation.launch.py'
            ])
        ]),
        launch_arguments={
            'warehouse_layout': LaunchConfiguration('warehouse_layout'),
            'obstacle_scenario': LaunchConfiguration('obstacle_scenario'),
            'use_sim_time': LaunchConfiguration('use_sim_time')
        }.items()
    )
    
    # Obstacle detector node
    obstacle_detector_node = Node(
        package='adaptnav',
        executable='obstacle_detector',
        name='obstacle_detector',
        parameters=[
            config_file,
            {'use_sim_time': LaunchConfiguration('use_sim_time')}
        ],
        output='screen',
        respawn=True,
        respawn_delay=2.0
    )
    
    # Global path planner node
    global_planner_node = Node(
        package='adaptnav',
        executable='global_planner',
        name='global_planner',
        parameters=[
            config_file,
            {'use_sim_time': LaunchConfiguration('use_sim_time')}
        ],
        output='screen',
        respawn=True,
        respawn_delay=2.0
    )
    
    # PPO agent node
    ppo_agent_node = Node(
        package='adaptnav',
        executable='ppo_agent_node',
        name='ppo_agent',
        parameters=[
            config_file,
            {
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'model_path': LaunchConfiguration('ppo_model_path')
            }
        ],
        output='screen',
        respawn=True,
        respawn_delay=2.0
    )
    
    # Safety controller node
    safety_controller_node = Node(
        package='adaptnav',
        executable='safety_controller',
        name='safety_controller',
        parameters=[
            config_file,
            {'use_sim_time': LaunchConfiguration('use_sim_time')}
        ],
        output='screen',
        respawn=True,
        respawn_delay=1.0  # Quick restart for safety-critical component
    )
    
    # Navigation controller node
    navigation_controller_node = Node(
        package='adaptnav',
        executable='navigation_controller',
        name='navigation_controller',
        parameters=[
            config_file,
            {'use_sim_time': LaunchConfiguration('use_sim_time')}
        ],
        output='screen',
        respawn=True,
        respawn_delay=2.0
    )
    
    # RViz visualization node
    rviz_visualizer_node = Node(
        package='adaptnav',
        executable='rviz_visualizer',
        name='rviz_visualizer',
        parameters=[
            config_file,
            {'use_sim_time': LaunchConfiguration('use_sim_time')}
        ],
        output='screen',
        respawn=True,
        respawn_delay=2.0
    )
    
    # RViz2 GUI
    rviz2_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_file],
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_rviz'))
    )
    
    # Web dashboard backend node
    web_dashboard_node = Node(
        package='adaptnav',
        executable='web_dashboard_backend',
        name='web_dashboard_backend',
        parameters=[
            config_file,
            {'use_sim_time': LaunchConfiguration('use_sim_time')}
        ],
        output='screen',
        respawn=True,
        respawn_delay=2.0,
        condition=IfCondition(LaunchConfiguration('use_web_dashboard'))
    )
    
    # Static transform publishers for sensor frames
    base_to_laser_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_to_laser_tf',
        arguments=['0', '0', '0.1', '0', '0', '0', 'base_link', 'laser'],
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}]
    )
    
    base_to_camera_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_to_camera_tf',
        arguments=['0.2', '0', '0.15', '0', '0', '0', 'base_link', 'camera_link'],
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}]
    )
    
    # Map server (if using static map)
    map_server_node = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        parameters=[
            {'yaml_filename': warehouse_layout_file},
            {'use_sim_time': LaunchConfiguration('use_sim_time')}
        ],
        output='screen'
    )
    
    # Lifecycle manager for map server
    lifecycle_manager_node = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_localization',
        parameters=[
            {'autostart': True},
            {'node_names': ['map_server']},
            {'use_sim_time': LaunchConfiguration('use_sim_time')}
        ],
        output='screen'
    )
    
    return LaunchDescription([
        # Launch arguments
        config_file_arg,
        warehouse_layout_arg,
        obstacle_scenario_arg,
        use_rviz_arg,
        use_web_dashboard_arg,
        use_sim_time_arg,
        ppo_model_path_arg,
        
        # Simulation
        mujoco_simulation,
        
        # Static transforms
        base_to_laser_tf,
        base_to_camera_tf,
        
        # Map server
        map_server_node,
        lifecycle_manager_node,
        
        # Navigation stack nodes
        obstacle_detector_node,
        global_planner_node,
        ppo_agent_node,
        safety_controller_node,
        navigation_controller_node,
        
        # Visualization
        rviz_visualizer_node,
        rviz2_node,
        web_dashboard_node,
    ])