"""
Unit tests for WarehouseNavigationEnv.

Tests the Gymnasium environment implementation including:
- Environment initialization
- Observation and action space definitions
- Reset functionality with random scenario generation
- Step functionality with reward computation
- Termination conditions
"""

import pytest
import numpy as np
import gymnasium as gym
from unittest.mock import Mock, patch, MagicMock

from adaptnav.rl.warehouse_env import WarehouseNavigationEnv
from adaptnav.core.robot_state import RobotState
from adaptnav.core.dynamic_obstacle import DynamicObstacle


class TestWarehouseNavigationEnv:
    """Test cases for WarehouseNavigationEnv."""
    
    @patch('adaptnav.rl.warehouse_env.MuJoCoSimulation')
    def test_environment_initialization(self, mock_simulation_class):
        """Test environment initialization with correct spaces."""
        mock_simulation = Mock()
        mock_simulation_class.return_value = mock_simulation
        
        env = WarehouseNavigationEnv()
        
        # Check observation space
        assert env.observation_space.shape == (372,)
        assert env.observation_space.dtype == np.float32
        
        # Check action space
        assert env.action_space.shape == (2,)
        assert env.action_space.dtype == np.float32
        assert np.array_equal(env.action_space.low, np.array([-1.0, -0.5]))
        assert np.array_equal(env.action_space.high, np.array([1.0, 0.5]))
        
        # Check environment parameters
        assert env.max_episode_steps == 1000
        assert env.goal_tolerance == 0.5
        assert env.collision_penalty == 10.0
        assert env.goal_bonus == 100.0
    
    @patch('adaptnav.rl.warehouse_env.MuJoCoSimulation')
    def test_custom_parameters(self, mock_simulation_class):
        """Test environment initialization with custom parameters."""
        mock_simulation = Mock()
        mock_simulation_class.return_value = mock_simulation
        
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
    
    @patch('adaptnav.rl.warehouse_env.MuJoCoSimulation')
    def test_reset_functionality(self, mock_simulation_class):
        """Test environment reset with random scenario generation."""
        mock_simulation = Mock()
        mock_simulation_class.return_value = mock_simulation
        mock_simulation.reset.return_value = True
        
        # Mock robot state
        mock_robot_state = Mock(spec=RobotState)
        mock_robot_state.position = np.array([0.0, 0.0])
        mock_robot_state.linear_velocity = 0.0
        mock_robot_state.angular_velocity = 0.0
        mock_simulation.get_ground_truth_robot_state.return_value = mock_robot_state
        
        # Mock observation
        mock_simulation.get_observation.return_value = {
            'lidar_scan': Mock(),
            'odometry': Mock()
        }
        mock_simulation.get_ground_truth_obstacles.return_value = []
        
        env = WarehouseNavigationEnv()
        
        # Mock the fallback observation method
        with patch.object(env, '_create_fallback_observation') as mock_fallback:
            mock_fallback.return_value = np.zeros(372, dtype=np.float32)
            
            observation, info = env.reset(seed=42)
        
        # Check reset was called on simulation
        mock_simulation.reset.assert_called_once()
        
        # Check observation shape
        assert observation.shape == (372,)
        assert observation.dtype == np.float32
        
        # Check info dict
        assert 'robot_position' in info
        assert 'goal_position' in info
        assert 'initial_distance' in info
        assert 'obstacle_count' in info
        
        # Check episode state reset
        assert env.current_step == 0
        assert env.episode_reward == 0.0
        assert not env.collision_occurred
        assert not env.goal_reached
    
    @patch('adaptnav.rl.warehouse_env.MuJoCoSimulation')
    def test_step_functionality(self, mock_simulation_class):
        """Test environment step with action execution."""
        mock_simulation = Mock()
        mock_simulation_class.return_value = mock_simulation
        mock_simulation.reset.return_value = True
        mock_simulation.step.return_value = True
        mock_simulation.check_collision.return_value = False
        
        # Mock robot state
        mock_robot_state = Mock(spec=RobotState)
        mock_robot_state.position = np.array([1.0, 1.0])
        mock_robot_state.linear_velocity = 0.5
        mock_robot_state.angular_velocity = 0.1
        mock_simulation.get_ground_truth_robot_state.return_value = mock_robot_state
        
        # Mock observation
        mock_simulation.get_observation.return_value = {
            'lidar_scan': Mock(),
            'odometry': Mock()
        }
        mock_simulation.get_ground_truth_obstacles.return_value = []
        
        env = WarehouseNavigationEnv()
        
        # Mock the fallback observation method
        with patch.object(env, '_create_fallback_observation') as mock_fallback:
            mock_fallback.return_value = np.zeros(372, dtype=np.float32)
            
            # Reset environment
            env.reset(seed=42)
            
            # Take a step
            action = np.array([0.5, 0.2])
            observation, reward, terminated, truncated, info = env.step(action)
        
        # Check simulation step was called
        mock_simulation.step.assert_called_once_with(0.1)
        
        # Check observation shape
        assert observation.shape == (372,)
        assert observation.dtype == np.float32
        
        # Check reward is a float
        assert isinstance(reward, float)
        
        # Check termination flags
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        
        # Check info dict
        assert 'robot_position' in info
        assert 'goal_position' in info
        assert 'distance_to_goal' in info
        assert 'collision' in info
        assert 'goal_reached' in info
        assert 'episode_reward' in info
        assert 'step' in info
        
        # Check episode state updated
        assert env.current_step == 1
    
    @patch('adaptnav.rl.warehouse_env.MuJoCoSimulation')
    def test_collision_termination(self, mock_simulation_class):
        """Test episode termination on collision."""
        mock_simulation = Mock()
        mock_simulation_class.return_value = mock_simulation
        mock_simulation.reset.return_value = True
        mock_simulation.step.return_value = True
        mock_simulation.check_collision.return_value = True  # Collision!
        
        # Mock robot state
        mock_robot_state = Mock(spec=RobotState)
        mock_robot_state.position = np.array([1.0, 1.0])
        mock_robot_state.linear_velocity = 0.5
        mock_robot_state.angular_velocity = 0.1
        mock_simulation.get_ground_truth_robot_state.return_value = mock_robot_state
        
        # Mock observation
        mock_simulation.get_observation.return_value = {
            'lidar_scan': Mock(),
            'odometry': Mock()
        }
        mock_simulation.get_ground_truth_obstacles.return_value = []
        
        env = WarehouseNavigationEnv()
        
        # Mock the fallback observation method
        with patch.object(env, '_create_fallback_observation') as mock_fallback:
            mock_fallback.return_value = np.zeros(372, dtype=np.float32)
            
            # Reset environment
            env.reset(seed=42)
            
            # Take a step that causes collision
            action = np.array([0.5, 0.2])
            observation, reward, terminated, truncated, info = env.step(action)
        
        # Check episode terminated due to collision
        assert terminated
        assert not truncated
        assert info['collision']
        
        # Check collision penalty applied
        assert reward < 0  # Should be negative due to collision penalty
    
    @patch('adaptnav.rl.warehouse_env.MuJoCoSimulation')
    def test_goal_reached_termination(self, mock_simulation_class):
        """Test episode termination when goal is reached."""
        mock_simulation = Mock()
        mock_simulation_class.return_value = mock_simulation
        mock_simulation.reset.return_value = True
        mock_simulation.step.return_value = True
        mock_simulation.check_collision.return_value = False
        
        env = WarehouseNavigationEnv(goal_tolerance=1.0)
        
        # Mock robot state very close to goal
        mock_robot_state = Mock(spec=RobotState)
        mock_robot_state.position = np.array([0.0, 0.0])  # Will be close to goal after reset
        mock_robot_state.linear_velocity = 0.0
        mock_robot_state.angular_velocity = 0.0
        mock_simulation.get_ground_truth_robot_state.return_value = mock_robot_state
        
        # Mock observation
        mock_simulation.get_observation.return_value = {
            'lidar_scan': Mock(),
            'odometry': Mock()
        }
        mock_simulation.get_ground_truth_obstacles.return_value = []
        
        # Mock the fallback observation method
        with patch.object(env, '_create_fallback_observation') as mock_fallback:
            mock_fallback.return_value = np.zeros(372, dtype=np.float32)
            
            # Reset environment
            env.reset(seed=42)
            
            # Set goal very close to robot
            env.goal_position = np.array([0.1, 0.1])  # Within tolerance
            
            # Take a step
            action = np.array([0.0, 0.0])
            observation, reward, terminated, truncated, info = env.step(action)
        
        # Check episode terminated due to goal reached
        assert terminated
        assert not truncated
        assert info['goal_reached']
        
        # Check goal bonus applied
        assert reward > 50  # Should be positive due to goal bonus
    
    @patch('adaptnav.rl.warehouse_env.MuJoCoSimulation')
    def test_max_steps_truncation(self, mock_simulation_class):
        """Test episode truncation at max steps."""
        mock_simulation = Mock()
        mock_simulation_class.return_value = mock_simulation
        mock_simulation.reset.return_value = True
        mock_simulation.step.return_value = True
        mock_simulation.check_collision.return_value = False
        
        # Mock robot state
        mock_robot_state = Mock(spec=RobotState)
        mock_robot_state.position = np.array([10.0, 10.0])  # Far from goal
        mock_robot_state.linear_velocity = 0.0
        mock_robot_state.angular_velocity = 0.0
        mock_simulation.get_ground_truth_robot_state.return_value = mock_robot_state
        
        # Mock observation
        mock_simulation.get_observation.return_value = {
            'lidar_scan': Mock(),
            'odometry': Mock()
        }
        mock_simulation.get_ground_truth_obstacles.return_value = []
        
        env = WarehouseNavigationEnv(max_episode_steps=2)  # Very short episode
        
        # Mock the fallback observation method
        with patch.object(env, '_create_fallback_observation') as mock_fallback:
            mock_fallback.return_value = np.zeros(372, dtype=np.float32)
            
            # Reset environment
            env.reset(seed=42)
            
            # Take steps until truncation
            action = np.array([0.1, 0.0])
            
            # Step 1
            obs1, reward1, term1, trunc1, info1 = env.step(action)
            assert not term1 and not trunc1
            
            # Step 2 (should truncate)
            obs2, reward2, term2, trunc2, info2 = env.step(action)
            assert not term2 and trunc2
    
    def test_action_clipping(self):
        """Test that actions are clipped to valid range."""
        with patch('adaptnav.rl.warehouse_env.MuJoCoSimulation'):
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
    
    @patch('adaptnav.rl.warehouse_env.MuJoCoSimulation')
    def test_reward_computation(self, mock_simulation_class):
        """Test reward computation components."""
        mock_simulation = Mock()
        mock_simulation_class.return_value = mock_simulation
        
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
        
        # Test robot state
        robot_state = Mock(spec=RobotState)
        robot_state.position = np.array([4.0, 0.0])  # Moved closer to goal
        
        # Test reward computation
        action = np.array([0.5, 0.1])
        reward = env._compute_reward(robot_state, action, collision=False)
        
        # Should have positive progress reward and small penalties
        assert reward > 0  # Progress toward goal should dominate
        
        # Test collision penalty
        collision_reward = env._compute_reward(robot_state, action, collision=True)
        assert collision_reward < reward - 9.0  # Should be much lower due to collision penalty
        
        # Test goal reached bonus
        robot_state.position = np.array([5.0, 0.0])  # At goal
        goal_reward = env._compute_reward(robot_state, action, collision=False)
        assert goal_reward > 90.0  # Should include goal bonus
    
    @patch('adaptnav.rl.warehouse_env.MuJoCoSimulation')
    def test_scenario_generation(self, mock_simulation_class):
        """Test random scenario generation."""
        mock_simulation = Mock()
        mock_simulation_class.return_value = mock_simulation
        
        env = WarehouseNavigationEnv()
        
        # Generate multiple scenarios to test randomness
        scenarios = []
        for i in range(10):
            robot_pos, goal_pos, obstacles = env._generate_random_scenario()
            scenarios.append((robot_pos, goal_pos, obstacles))
        
        # Check that scenarios are different (at least some variation)
        robot_positions = [s[0] for s in scenarios]
        goal_positions = [s[1] for s in scenarios]
        
        # Should have some variation in positions
        robot_x_values = [pos[0] for pos in robot_positions]
        goal_x_values = [pos[0] for pos in goal_positions]
        
        assert len(set(robot_x_values)) > 1  # Should have different x positions
        assert len(set(goal_x_values)) > 1   # Should have different goal positions
        
        # Check distance constraints
        for robot_pos, goal_pos, _ in scenarios:
            distance = np.sqrt((goal_pos[0] - robot_pos[0])**2 + (goal_pos[1] - robot_pos[1])**2)
            assert distance >= env.min_goal_distance
            assert distance <= env.max_goal_distance
    
    @patch('adaptnav.rl.warehouse_env.MuJoCoSimulation')
    def test_fallback_observation(self, mock_simulation_class):
        """Test fallback observation creation."""
        mock_simulation = Mock()
        mock_simulation_class.return_value = mock_simulation
        
        # Mock robot state
        mock_robot_state = Mock(spec=RobotState)
        mock_robot_state.position = np.array([0.0, 0.0])
        mock_robot_state.linear_velocity = 0.5
        mock_robot_state.angular_velocity = 0.1
        mock_simulation.get_ground_truth_robot_state.return_value = mock_robot_state
        
        # Mock obstacles
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
        
        # Check that goal direction is normalized
        goal_dir_magnitude = np.linalg.norm(goal_direction)
        assert abs(goal_dir_magnitude - 1.0) < 1e-6  # Should be unit vector
        
        # Check velocity values
        assert current_velocity[0] == 0.5  # Linear velocity
        assert current_velocity[1] == 0.1  # Angular velocity


class TestWarehouseNavigationEnvIntegration:
    """Integration tests for WarehouseNavigationEnv."""
    
    @patch('adaptnav.rl.warehouse_env.MuJoCoSimulation')
    def test_gymnasium_interface_compliance(self, mock_simulation_class):
        """Test that environment complies with Gymnasium interface."""
        mock_simulation = Mock()
        mock_simulation_class.return_value = mock_simulation
        mock_simulation.reset.return_value = True
        mock_simulation.step.return_value = True
        mock_simulation.check_collision.return_value = False
        
        # Mock robot state
        mock_robot_state = Mock(spec=RobotState)
        mock_robot_state.position = np.array([0.0, 0.0])
        mock_robot_state.linear_velocity = 0.0
        mock_robot_state.angular_velocity = 0.0
        mock_simulation.get_ground_truth_robot_state.return_value = mock_robot_state
        
        # Mock observation
        mock_simulation.get_observation.return_value = {
            'lidar_scan': Mock(),
            'odometry': Mock()
        }
        mock_simulation.get_ground_truth_obstacles.return_value = []
        
        env = WarehouseNavigationEnv()
        
        # Mock the fallback observation method
        with patch.object(env, '_create_fallback_observation') as mock_fallback:
            mock_fallback.return_value = np.zeros(372, dtype=np.float32)
            
            # Test Gymnasium interface
            assert hasattr(env, 'observation_space')
            assert hasattr(env, 'action_space')
            assert hasattr(env, 'reset')
            assert hasattr(env, 'step')
            assert hasattr(env, 'render')
            assert hasattr(env, 'close')
            
            # Test reset returns correct format
            obs, info = env.reset()
            assert isinstance(obs, np.ndarray)
            assert isinstance(info, dict)
            
            # Test step returns correct format
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            assert isinstance(obs, np.ndarray)
            assert isinstance(reward, (int, float))
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            assert isinstance(info, dict)
    
    @patch('adaptnav.rl.warehouse_env.MuJoCoSimulation')
    def test_reproducible_episodes(self, mock_simulation_class):
        """Test that episodes are reproducible with same seed."""
        mock_simulation = Mock()
        mock_simulation_class.return_value = mock_simulation
        mock_simulation.reset.return_value = True
        
        # Mock robot state
        mock_robot_state = Mock(spec=RobotState)
        mock_robot_state.position = np.array([0.0, 0.0])
        mock_robot_state.linear_velocity = 0.0
        mock_robot_state.angular_velocity = 0.0
        mock_simulation.get_ground_truth_robot_state.return_value = mock_robot_state
        
        # Mock observation
        mock_simulation.get_observation.return_value = {
            'lidar_scan': Mock(),
            'odometry': Mock()
        }
        mock_simulation.get_ground_truth_obstacles.return_value = []
        
        env1 = WarehouseNavigationEnv()
        env2 = WarehouseNavigationEnv()
        
        # Mock the fallback observation method
        with patch.object(env1, '_create_fallback_observation') as mock_fallback1, \
             patch.object(env2, '_create_fallback_observation') as mock_fallback2:
            
            mock_fallback1.return_value = np.zeros(372, dtype=np.float32)
            mock_fallback2.return_value = np.zeros(372, dtype=np.float32)
            
            # Reset with same seed
            obs1, info1 = env1.reset(seed=42)
            obs2, info2 = env2.reset(seed=42)
            
            # Should have same goal positions (approximately)
            goal1 = np.array(info1['goal_position'])
            goal2 = np.array(info2['goal_position'])
            
            assert np.allclose(goal1, goal2, atol=1e-6)