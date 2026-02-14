"""
Unit tests for BaseSimulation class.

Tests the abstract base class interface and common functionality
for simulation backends.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys

# Check if ROS 2 is available
try:
    from geometry_msgs.msg import Twist, PoseStamped
    from custom_msgs.msg import ObstacleArray
    import rclpy
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    # Skip all tests if ROS is not available
    pytestmark = pytest.mark.skip(reason="ROS 2 not available")

if ROS_AVAILABLE:
    from adaptnav.simulation.base_simulation import BaseSimulation
    from adaptnav.core.robot_state import RobotState
    from adaptnav.core.dynamic_obstacle import DynamicObstacle


if not ROS_AVAILABLE:
    # Create dummy classes for when ROS is not available
    class ConcreteSimulation:
        pass
else:
    class ConcreteSimulation(BaseSimulation):
        """Concrete implementation of BaseSimulation for testing."""
        
        def __init__(self):
            super().__init__('test_simulation')
            self._robot_state = RobotState(
                position=np.array([0.0, 0.0]),
                orientation=0.0,
                linear_velocity=0.0,
                angular_velocity=0.0
            )
            self._obstacles = []
            self._collision = False
            self.set_initialized(True)
        
        def step(self, dt: float) -> bool:
            self.increment_step_count()
            return True
        
        def reset(self, robot_position=None, obstacle_configs=None) -> bool:
            self._step_count = 0
            if robot_position:
                self._robot_state.position = np.array([robot_position[0], robot_position[1]])
                self._robot_state.orientation = robot_position[2]
            if obstacle_configs:
                self._obstacles = [
                    DynamicObstacle(
                        id=i,
                        position=np.array(cfg['position']),
                        velocity=np.array(cfg.get('velocity', [0.0, 0.0])),
                        radius=cfg.get('radius', 0.5),
                        classification=cfg.get('type', 'unknown')
                    )
                    for i, cfg in enumerate(obstacle_configs)
                ]
            return True
        
        def get_observation(self):
            return {
                'lidar_scan': np.zeros(360),
                'odometry': self._robot_state,
                'timestamp': 0.0
            }
        
        def get_ground_truth_robot_state(self) -> RobotState:
            return self._robot_state
        
        def get_ground_truth_obstacles(self):
            return self._obstacles
        
        def check_collision(self) -> bool:
            return self._collision


@pytest.fixture
def rclpy_init():
    """Initialize rclpy for testing."""
    if not ROS_AVAILABLE:
        pytest.skip("ROS 2 not available")
    import rclpy
    if not rclpy.ok():
        rclpy.init()
    yield
    # Don't shutdown here as it may be used by other tests


@pytest.fixture
def simulation(rclpy_init):
    """Create a concrete simulation instance for testing."""
    if not ROS_AVAILABLE:
        pytest.skip("ROS 2 not available")
    sim = ConcreteSimulation()
    yield sim
    sim.destroy_node()


class TestBaseSimulationInitialization:
    """Test initialization of BaseSimulation."""
    
    def test_initialization_creates_publishers(self, simulation):
        """Test that initialization creates required ROS 2 publishers."""
        assert simulation._ground_truth_robot_pose_pub is not None
        assert simulation._ground_truth_obstacles_pub is not None
    
    def test_initialization_creates_subscriber(self, simulation):
        """Test that initialization creates velocity command subscriber."""
        assert simulation._cmd_vel_sub is not None
    
    def test_initial_state(self, simulation):
        """Test initial state of simulation."""
        assert simulation.is_initialized() is True
        assert simulation.get_step_count() == 0
        assert simulation.get_current_cmd_vel() is None


class TestBaseSimulationVelocityCommands:
    """Test velocity command handling."""
    
    def test_cmd_vel_callback_stores_command(self, simulation):
        """Test that velocity commands are stored."""
        cmd = Twist()
        cmd.linear.x = 0.5
        cmd.angular.z = 0.2
        
        simulation._cmd_vel_callback(cmd)
        
        stored_cmd = simulation.get_current_cmd_vel()
        assert stored_cmd is not None
        assert stored_cmd.linear.x == 0.5
        assert stored_cmd.angular.z == 0.2
    
    def test_cmd_vel_updates_on_new_command(self, simulation):
        """Test that new commands override old ones."""
        cmd1 = Twist()
        cmd1.linear.x = 0.5
        simulation._cmd_vel_callback(cmd1)
        
        cmd2 = Twist()
        cmd2.linear.x = 1.0
        simulation._cmd_vel_callback(cmd2)
        
        stored_cmd = simulation.get_current_cmd_vel()
        assert stored_cmd.linear.x == 1.0


class TestBaseSimulationLifecycle:
    """Test simulation lifecycle methods."""
    
    def test_step_increments_counter(self, simulation):
        """Test that step increments the step counter."""
        initial_count = simulation.get_step_count()
        simulation.step(0.1)
        assert simulation.get_step_count() == initial_count + 1
    
    def test_step_returns_success(self, simulation):
        """Test that step returns True on success."""
        result = simulation.step(0.1)
        assert result is True
    
    def test_reset_clears_step_count(self, simulation):
        """Test that reset clears the step counter."""
        simulation.step(0.1)
        simulation.step(0.1)
        assert simulation.get_step_count() > 0
        
        simulation.reset()
        assert simulation.get_step_count() == 0
    
    def test_reset_with_robot_position(self, simulation):
        """Test reset with custom robot position."""
        new_position = (5.0, 3.0, 1.57)
        simulation.reset(robot_position=new_position)
        
        robot_state = simulation.get_ground_truth_robot_state()
        assert np.allclose(robot_state.position, [5.0, 3.0])
        assert np.isclose(robot_state.orientation, 1.57)
    
    def test_reset_with_obstacle_configs(self, simulation):
        """Test reset with custom obstacle configurations."""
        obstacle_configs = [
            {'position': [1.0, 2.0], 'velocity': [0.5, 0.0], 'type': 'worker'},
            {'position': [3.0, 4.0], 'velocity': [0.0, 0.5], 'type': 'forklift'}
        ]
        simulation.reset(obstacle_configs=obstacle_configs)
        
        obstacles = simulation.get_ground_truth_obstacles()
        assert len(obstacles) == 2
        assert obstacles[0].classification == 'worker'
        assert obstacles[1].classification == 'forklift'


class TestBaseSimulationGroundTruth:
    """Test ground truth data access."""
    
    def test_get_ground_truth_robot_state(self, simulation):
        """Test getting ground truth robot state."""
        robot_state = simulation.get_ground_truth_robot_state()
        assert isinstance(robot_state, RobotState)
        assert robot_state.position is not None
        assert robot_state.orientation is not None
    
    def test_get_ground_truth_obstacles(self, simulation):
        """Test getting ground truth obstacles."""
        obstacle_configs = [
            {'position': [1.0, 2.0], 'velocity': [0.5, 0.0], 'type': 'worker'}
        ]
        simulation.reset(obstacle_configs=obstacle_configs)
        
        obstacles = simulation.get_ground_truth_obstacles()
        assert isinstance(obstacles, list)
        assert len(obstacles) == 1
        assert isinstance(obstacles[0], DynamicObstacle)
    
    def test_check_collision(self, simulation):
        """Test collision checking."""
        collision = simulation.check_collision()
        assert isinstance(collision, bool)


class TestBaseSimulationPublishing:
    """Test ROS 2 publishing functionality."""
    
    def test_publish_ground_truth_robot_pose(self, simulation):
        """Test publishing robot pose ground truth."""
        # Mock the publisher
        simulation._ground_truth_robot_pose_pub.publish = Mock()
        
        simulation.publish_ground_truth()
        
        # Verify publish was called
        assert simulation._ground_truth_robot_pose_pub.publish.called
        
        # Get the published message
        call_args = simulation._ground_truth_robot_pose_pub.publish.call_args
        msg = call_args[0][0]
        
        assert isinstance(msg, PoseStamped)
        assert msg.header.frame_id == 'map'
    
    def test_publish_ground_truth_obstacles(self, simulation):
        """Test publishing obstacle ground truth."""
        # Set up obstacles
        obstacle_configs = [
            {'position': [1.0, 2.0], 'velocity': [0.5, 0.0], 'type': 'worker'}
        ]
        simulation.reset(obstacle_configs=obstacle_configs)
        
        # Mock the publisher
        simulation._ground_truth_obstacles_pub.publish = Mock()
        
        simulation.publish_ground_truth()
        
        # Verify publish was called
        assert simulation._ground_truth_obstacles_pub.publish.called
        
        # Get the published message
        call_args = simulation._ground_truth_obstacles_pub.publish.call_args
        msg = call_args[0][0]
        
        assert isinstance(msg, ObstacleArray)
        assert msg.header.frame_id == 'map'
        assert len(msg.obstacles) == 1
        assert msg.obstacles[0].classification == 'worker'
        assert msg.obstacles[0].confidence == 1.0  # Ground truth has perfect confidence
    
    def test_publish_ground_truth_obstacle_covariance_is_zero(self, simulation):
        """Test that ground truth obstacles have zero covariance."""
        obstacle_configs = [
            {'position': [1.0, 2.0], 'velocity': [0.5, 0.0], 'type': 'worker'}
        ]
        simulation.reset(obstacle_configs=obstacle_configs)
        
        simulation._ground_truth_obstacles_pub.publish = Mock()
        simulation.publish_ground_truth()
        
        call_args = simulation._ground_truth_obstacles_pub.publish.call_args
        msg = call_args[0][0]
        
        # Ground truth should have zero covariance (perfect knowledge)
        assert all(cov == 0.0 for cov in msg.obstacles[0].covariance)


class TestBaseSimulationObservation:
    """Test observation retrieval."""
    
    def test_get_observation_returns_dict(self, simulation):
        """Test that get_observation returns a dictionary."""
        obs = simulation.get_observation()
        assert isinstance(obs, dict)
    
    def test_get_observation_contains_required_keys(self, simulation):
        """Test that observation contains required keys."""
        obs = simulation.get_observation()
        assert 'lidar_scan' in obs
        assert 'odometry' in obs
        assert 'timestamp' in obs


class TestBaseSimulationStateManagement:
    """Test state management methods."""
    
    def test_set_initialized(self, simulation):
        """Test setting initialization state."""
        simulation.set_initialized(False)
        assert simulation.is_initialized() is False
        
        simulation.set_initialized(True)
        assert simulation.is_initialized() is True
    
    def test_increment_step_count(self, simulation):
        """Test manual step count increment."""
        initial = simulation.get_step_count()
        simulation.increment_step_count()
        assert simulation.get_step_count() == initial + 1
