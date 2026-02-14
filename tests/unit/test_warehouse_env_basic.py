"""
Basic unit tests for WarehouseNavigationEnv that don't require ROS 2.

Tests the core functionality of the Gymnasium environment without
requiring the full simulation stack.
"""

import pytest
import numpy as np
import gymnasium as gym
from unittest.mock import Mock, patch

from adaptnav.rl.warehouse_env import WarehouseNavigationEnv


class TestWarehouseNavigationEnvBasic:
    """Basic test cases for WarehouseNavigationEnv."""
    
    def test_observation_space_definition(self):
        """Test observation space is correctly defined."""
        with patch('adaptnav.rl.warehouse_env.ROS_AVAILABLE', False):
            with pytest.raises(ValueError, match="ROS 2 not available"):
                env = WarehouseNavigationEnv()
    
    def test_action_space_definition(self):
        """Test action space is correctly defined."""
        # Test without creating the environment (just check the class definition)
        # We can test the space definitions by mocking the simulation
        with patch('adaptnav.rl.warehouse_env.ROS_AVAILABLE', True), \
             patch('adaptnav.rl.warehouse_env.MuJoCoSimulation') as mock_sim:
            
            mock_sim.return_value = Mock()
            env = WarehouseNavigationEnv()
            
            # Check observation space
            assert env.observation_space.shape == (372,)
            assert env.observation_space.dtype == np.float32
            
            # Check action space
            assert env.action_space.shape == (2,)
            assert env.action_space.dtype == np.float32
            assert np.array_equal(env.action_space.low, np.array([-1.0, -0.5]))
            assert np.array_equal(env.action_space.high, np.array([1.0, 0.5]))
    
    def test_reward_computation_logic(self):
        """Test reward computation without simulation."""
        with patch('adaptnav.rl.warehouse_env.ROS_AVAILABLE', True), \
             patch('adaptnav.rl.warehouse_env.MuJoCoSimulation') as mock_sim:
            
            mock_sim.return_value = Mock()
            env = WarehouseNavigationEnv(
                collision_penalty=10.0,
                goal_bonus=100.0,
                progress_weight=1.0,
                distance_weight=0.1,
                action_weight=0.01
            )
            
            # Set up test state
            env.goal_position = np.array([5.0, 0.0])
            env.initial_distance_to_goal = 5.0
            env.previous_distance_to_goal = 5.0
            
            # Mock robot state
            from adaptnav.core.robot_state import RobotState
            robot_state = Mock(spec=RobotState)
            robot_state.position = np.array([4.0, 0.0])  # Moved closer to goal
            
            # Test normal reward
            action = np.array([0.5, 0.1])
            reward = env._compute_reward(robot_state, action, collision=False)
            
            # Should have positive progress reward
            assert isinstance(reward, float)
            
            # Test collision penalty
            collision_reward = env._compute_reward(robot_state, action, collision=True)
            assert collision_reward < reward  # Should be lower due to collision penalty
            
            # Test goal reached bonus
            robot_state.position = np.array([5.0, 0.0])  # At goal
            goal_reward = env._compute_reward(robot_state, action, collision=False)
            assert goal_reward > reward  # Should be higher due to goal bonus
    
    def test_scenario_generation_logic(self):
        """Test random scenario generation logic."""
        with patch('adaptnav.rl.warehouse_env.ROS_AVAILABLE', True), \
             patch('adaptnav.rl.warehouse_env.MuJoCoSimulation') as mock_sim:
            
            mock_sim.return_value = Mock()
            env = WarehouseNavigationEnv()
            
            # Test scenario generation
            robot_pos, goal_pos, obstacles = env._generate_random_scenario()
            
            # Check return types
            assert isinstance(robot_pos, tuple)
            assert len(robot_pos) == 3  # x, y, theta
            assert isinstance(goal_pos, tuple)
            assert len(goal_pos) == 2  # x, y
            assert isinstance(obstacles, list)
            
            # Check positions are within bounds
            assert env.warehouse_bounds['x_min'] <= robot_pos[0] <= env.warehouse_bounds['x_max']
            assert env.warehouse_bounds['y_min'] <= robot_pos[1] <= env.warehouse_bounds['y_max']
            assert env.warehouse_bounds['x_min'] <= goal_pos[0] <= env.warehouse_bounds['x_max']
            assert env.warehouse_bounds['y_min'] <= goal_pos[1] <= env.warehouse_bounds['y_max']
            
            # Check distance constraint
            distance = np.sqrt((goal_pos[0] - robot_pos[0])**2 + (goal_pos[1] - robot_pos[1])**2)
            assert distance >= env.min_goal_distance
            assert distance <= env.max_goal_distance
    
    def test_obstacle_generation_logic(self):
        """Test obstacle generation logic."""
        with patch('adaptnav.rl.warehouse_env.ROS_AVAILABLE', True), \
             patch('adaptnav.rl.warehouse_env.MuJoCoSimulation') as mock_sim:
            
            mock_sim.return_value = Mock()
            env = WarehouseNavigationEnv()
            
            robot_pos = (0.0, 0.0, 0.0)
            goal_pos = (5.0, 0.0)
            
            obstacles = env._generate_random_obstacles(robot_pos, goal_pos)
            
            # Check obstacle structure
            for obstacle in obstacles:
                assert isinstance(obstacle, dict)
                assert 'type' in obstacle
                assert 'position' in obstacle
                assert 'velocity' in obstacle
                assert 'radius' in obstacle
                
                assert obstacle['type'] in ['worker', 'forklift']
                assert len(obstacle['position']) == 3  # x, y, z
                assert len(obstacle['velocity']) == 2  # vx, vy
                assert isinstance(obstacle['radius'], (int, float))
    
    def test_fallback_observation_creation(self):
        """Test fallback observation creation."""
        with patch('adaptnav.rl.warehouse_env.ROS_AVAILABLE', True), \
             patch('adaptnav.rl.warehouse_env.MuJoCoSimulation') as mock_sim:
            
            # Mock simulation
            mock_simulation = Mock()
            mock_sim.return_value = mock_simulation
            
            # Mock robot state
            from adaptnav.core.robot_state import RobotState
            mock_robot_state = Mock(spec=RobotState)
            mock_robot_state.position = np.array([0.0, 0.0])
            mock_robot_state.linear_velocity = 0.5
            mock_robot_state.angular_velocity = 0.1
            mock_simulation.get_ground_truth_robot_state.return_value = mock_robot_state
            
            # Mock obstacles
            from adaptnav.core.dynamic_obstacle import DynamicObstacle
            obstacle = Mock(spec=DynamicObstacle)
            obstacle.position = np.array([2.0, 0.0])
            mock_simulation.get_ground_truth_obstacles.return_value = [obstacle]
            
            env = WarehouseNavigationEnv()
            env.goal_position = np.array([5.0, 0.0])
            
            # Test fallback observation
            observation = env._create_fallback_observation()
            
            # Check observation shape and components
            assert observation.shape == (372,)
            assert observation.dtype == np.float32
            
            # Check components
            lidar_scan = observation[:360]
            goal_direction = observation[360:362]
            current_velocity = observation[362:364]
            obstacle_proximity = observation[364:372]
            
            assert lidar_scan.shape == (360,)
            assert goal_direction.shape == (2,)
            assert current_velocity.shape == (2,)
            assert obstacle_proximity.shape == (8,)
            
            # Check that values are reasonable
            assert np.all(lidar_scan >= 0.0)  # Normalized distances should be non-negative
            assert np.all(lidar_scan <= 1.0)  # Normalized distances should be <= 1
            
            # Check velocity values
            assert current_velocity[0] == 0.5  # Linear velocity
            assert current_velocity[1] == 0.1  # Angular velocity
    
    def test_action_clipping(self):
        """Test action clipping logic."""
        with patch('adaptnav.rl.warehouse_env.ROS_AVAILABLE', True), \
             patch('adaptnav.rl.warehouse_env.MuJoCoSimulation') as mock_sim:
            
            mock_sim.return_value = Mock()
            env = WarehouseNavigationEnv()
            
            # Test action clipping
            action = np.array([2.0, 1.0])  # Out of bounds
            clipped = np.clip(action, env.action_space.low, env.action_space.high)
            
            assert clipped[0] == 1.0  # Clipped to max linear velocity
            assert clipped[1] == 0.5  # Clipped to max angular velocity
            
            action = np.array([-2.0, -1.0])  # Out of bounds (negative)
            clipped = np.clip(action, env.action_space.low, env.action_space.high)
            
            assert clipped[0] == -1.0  # Clipped to min linear velocity
            assert clipped[1] == -0.5  # Clipped to min angular velocity
    
    def test_environment_parameters(self):
        """Test environment parameter initialization."""
        with patch('adaptnav.rl.warehouse_env.ROS_AVAILABLE', True), \
             patch('adaptnav.rl.warehouse_env.MuJoCoSimulation') as mock_sim:
            
            mock_sim.return_value = Mock()
            
            # Test default parameters
            env = WarehouseNavigationEnv()
            assert env.max_episode_steps == 1000
            assert env.goal_tolerance == 0.5
            assert env.collision_penalty == 10.0
            assert env.goal_bonus == 100.0
            
            # Test custom parameters
            env = WarehouseNavigationEnv(
                max_episode_steps=500,
                goal_tolerance=1.0,
                collision_penalty=20.0,
                goal_bonus=50.0
            )
            assert env.max_episode_steps == 500
            assert env.goal_tolerance == 1.0
            assert env.collision_penalty == 20.0
            assert env.goal_bonus == 50.0
    
    def test_gymnasium_interface_attributes(self):
        """Test that environment has required Gymnasium interface attributes."""
        with patch('adaptnav.rl.warehouse_env.ROS_AVAILABLE', True), \
             patch('adaptnav.rl.warehouse_env.MuJoCoSimulation') as mock_sim:
            
            mock_sim.return_value = Mock()
            env = WarehouseNavigationEnv()
            
            # Check required attributes
            assert hasattr(env, 'observation_space')
            assert hasattr(env, 'action_space')
            assert hasattr(env, 'reset')
            assert hasattr(env, 'step')
            assert hasattr(env, 'render')
            assert hasattr(env, 'close')
            assert hasattr(env, 'metadata')
            
            # Check space types
            assert isinstance(env.observation_space, gym.spaces.Box)
            assert isinstance(env.action_space, gym.spaces.Box)
    
    def test_reward_components(self):
        """Test individual reward components."""
        with patch('adaptnav.rl.warehouse_env.ROS_AVAILABLE', True), \
             patch('adaptnav.rl.warehouse_env.MuJoCoSimulation') as mock_sim:
            
            mock_sim.return_value = Mock()
            env = WarehouseNavigationEnv(
                collision_penalty=10.0,
                goal_bonus=100.0,
                progress_weight=1.0,
                distance_weight=0.1,
                action_weight=0.01
            )
            
            # Set up test scenario
            env.goal_position = np.array([10.0, 0.0])
            env.initial_distance_to_goal = 10.0
            env.previous_distance_to_goal = 10.0
            
            from adaptnav.core.robot_state import RobotState
            robot_state = Mock(spec=RobotState)
            
            # Test progress reward (moving closer)
            robot_state.position = np.array([9.0, 0.0])  # 1m closer
            action = np.array([0.0, 0.0])  # No action penalty
            reward = env._compute_reward(robot_state, action, collision=False)
            assert reward > 0  # Should be positive due to progress
            
            # Test distance penalty dominates when no progress
            robot_state.position = np.array([0.0, 0.0])  # Same distance
            env.previous_distance_to_goal = 10.0
            reward = env._compute_reward(robot_state, action, collision=False)
            assert reward < 0  # Should be negative due to distance penalty
            
            # Test action penalty
            large_action = np.array([1.0, 0.5])  # Large action
            small_action = np.array([0.1, 0.05])  # Small action
            
            reward_large = env._compute_reward(robot_state, large_action, collision=False)
            reward_small = env._compute_reward(robot_state, small_action, collision=False)
            assert reward_small > reward_large  # Smaller action should have higher reward
            
            # Test collision penalty
            reward_collision = env._compute_reward(robot_state, action, collision=True)
            reward_no_collision = env._compute_reward(robot_state, action, collision=False)
            assert reward_collision < reward_no_collision - 9.0  # Should be much lower
            
            # Test goal bonus
            robot_state.position = np.array([10.0, 0.0])  # At goal
            reward_at_goal = env._compute_reward(robot_state, action, collision=False)
            assert reward_at_goal > 90.0  # Should include large goal bonus