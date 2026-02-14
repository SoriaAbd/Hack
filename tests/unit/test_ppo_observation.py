"""
Unit tests for PPOObservation class.

Tests the observation structure used by the PPO agent, including:
- Observation construction and validation
- Vector flattening for neural network input
- ROS message conversion (when available)
- Edge cases and error handling
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock

from adaptnav.rl.ppo_observation import PPOObservation, ROS_AVAILABLE


class TestPPOObservation:
    """Test cases for PPOObservation class."""
    
    def test_observation_creation_valid(self):
        """Test creating a valid PPOObservation."""
        lidar_scan = np.ones(360, dtype=np.float32)
        goal_direction = np.array([1.0, 0.0], dtype=np.float32)
        current_velocity = np.array([0.5, 0.1], dtype=np.float32)
        obstacle_proximity = np.full(8, 5.0, dtype=np.float32)
        
        obs = PPOObservation(
            lidar_scan=lidar_scan,
            goal_direction=goal_direction,
            current_velocity=current_velocity,
            obstacle_proximity=obstacle_proximity
        )
        
        assert obs.lidar_scan.shape == (360,)
        assert obs.goal_direction.shape == (2,)
        assert obs.current_velocity.shape == (2,)
        assert obs.obstacle_proximity.shape == (8,)
        
        np.testing.assert_array_equal(obs.lidar_scan, lidar_scan)
        np.testing.assert_array_equal(obs.goal_direction, goal_direction)
        np.testing.assert_array_equal(obs.current_velocity, current_velocity)
        np.testing.assert_array_equal(obs.obstacle_proximity, obstacle_proximity)
    
    def test_observation_creation_invalid_shapes(self):
        """Test that invalid shapes raise ValueError."""
        # Invalid lidar_scan shape
        with pytest.raises(ValueError, match="lidar_scan must have shape \\(360,\\)"):
            PPOObservation(
                lidar_scan=np.ones(180),  # Wrong size
                goal_direction=np.array([1.0, 0.0]),
                current_velocity=np.array([0.5, 0.1]),
                obstacle_proximity=np.full(8, 5.0)
            )
        
        # Invalid goal_direction shape
        with pytest.raises(ValueError, match="goal_direction must have shape \\(2,\\)"):
            PPOObservation(
                lidar_scan=np.ones(360),
                goal_direction=np.array([1.0]),  # Wrong size
                current_velocity=np.array([0.5, 0.1]),
                obstacle_proximity=np.full(8, 5.0)
            )
        
        # Invalid current_velocity shape
        with pytest.raises(ValueError, match="current_velocity must have shape \\(2,\\)"):
            PPOObservation(
                lidar_scan=np.ones(360),
                goal_direction=np.array([1.0, 0.0]),
                current_velocity=np.array([0.5]),  # Wrong size
                obstacle_proximity=np.full(8, 5.0)
            )
        
        # Invalid obstacle_proximity shape
        with pytest.raises(ValueError, match="obstacle_proximity must have shape \\(8,\\)"):
            PPOObservation(
                lidar_scan=np.ones(360),
                goal_direction=np.array([1.0, 0.0]),
                current_velocity=np.array([0.5, 0.1]),
                obstacle_proximity=np.full(4, 5.0)  # Wrong size
            )
    
    def test_to_vector(self):
        """Test flattening observation to vector."""
        lidar_scan = np.arange(360, dtype=np.float32)
        goal_direction = np.array([1.0, -0.5], dtype=np.float32)
        current_velocity = np.array([0.8, 0.2], dtype=np.float32)
        obstacle_proximity = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float32)
        
        obs = PPOObservation(
            lidar_scan=lidar_scan,
            goal_direction=goal_direction,
            current_velocity=current_velocity,
            obstacle_proximity=obstacle_proximity
        )
        
        vector = obs.to_vector()
        
        # Check total dimension
        assert vector.shape == (372,), f"Expected shape (372,), got {vector.shape}"
        
        # Check that components are concatenated correctly
        np.testing.assert_array_equal(vector[:360], lidar_scan)
        np.testing.assert_array_equal(vector[360:362], goal_direction)
        np.testing.assert_array_equal(vector[362:364], current_velocity)
        np.testing.assert_array_equal(vector[364:372], obstacle_proximity)
    
    def test_create_empty(self):
        """Test creating empty observation with default values."""
        obs = PPOObservation.create_empty()
        
        assert obs.lidar_scan.shape == (360,)
        assert obs.goal_direction.shape == (2,)
        assert obs.current_velocity.shape == (2,)
        assert obs.obstacle_proximity.shape == (8,)
        
        # Check default values
        np.testing.assert_array_equal(obs.lidar_scan, np.ones(360, dtype=np.float32))
        np.testing.assert_array_equal(obs.goal_direction, np.array([1.0, 0.0], dtype=np.float32))
        np.testing.assert_array_equal(obs.current_velocity, np.array([0.0, 0.0], dtype=np.float32))
        np.testing.assert_array_equal(obs.obstacle_proximity, np.full(8, 10.0, dtype=np.float32))
    
    def test_equality(self):
        """Test equality comparison between observations."""
        obs1 = PPOObservation.create_empty()
        obs2 = PPOObservation.create_empty()
        
        # Should be equal
        assert obs1 == obs2
        
        # Modify one observation
        obs2.lidar_scan[0] = 0.5
        assert obs1 != obs2
        
        # Test with non-PPOObservation object
        assert obs1 != "not an observation"
        assert obs1 != 42
    
    def test_repr(self):
        """Test string representation."""
        obs = PPOObservation.create_empty()
        repr_str = repr(obs)
        
        assert "PPOObservation" in repr_str
        assert "lidar_scan" in repr_str
        assert "goal_direction" in repr_str
        assert "current_velocity" in repr_str
        assert "obstacle_proximity" in repr_str
    
    def test_process_lidar_scan(self):
        """Test LiDAR scan processing."""
        # Create mock LaserScan
        mock_scan = Mock()
        mock_scan.ranges = [1.0, 2.0, float('inf'), float('nan'), 5.0] * 72  # 360 readings
        mock_scan.range_min = 0.1
        mock_scan.range_max = 10.0
        
        processed = PPOObservation._process_lidar_scan(mock_scan)
        
        assert processed.shape == (360,)
        assert processed.dtype == np.float32
        assert np.all(processed >= 0.0)
        assert np.all(processed <= 1.0)
        assert np.all(np.isfinite(processed))
    
    def test_process_lidar_scan_interpolation(self):
        """Test LiDAR scan processing with interpolation."""
        # Create mock LaserScan with different number of readings
        mock_scan = Mock()
        mock_scan.ranges = [1.0, 2.0, 3.0, 4.0, 5.0] * 36  # 180 readings
        mock_scan.range_min = 0.1
        mock_scan.range_max = 10.0
        
        processed = PPOObservation._process_lidar_scan(mock_scan)
        
        # Should interpolate to exactly 360 readings
        assert processed.shape == (360,)
        assert processed.dtype == np.float32
    
    def test_compute_goal_direction(self):
        """Test goal direction computation."""
        # Create mock odometry
        mock_odom = Mock()
        mock_odom.pose.pose.position.x = 0.0
        mock_odom.pose.pose.position.y = 0.0
        mock_odom.pose.pose.orientation.x = 0.0
        mock_odom.pose.pose.orientation.y = 0.0
        mock_odom.pose.pose.orientation.z = 0.0
        mock_odom.pose.pose.orientation.w = 1.0  # No rotation
        
        # Create mock goal
        mock_goal = Mock()
        mock_goal.position.x = 1.0
        mock_goal.position.y = 0.0
        
        goal_dir = PPOObservation._compute_goal_direction(mock_odom, mock_goal)
        
        assert goal_dir.shape == (2,)
        assert goal_dir.dtype == np.float32
        # Should point forward (positive x direction)
        np.testing.assert_array_almost_equal(goal_dir, [1.0, 0.0], decimal=5)
    
    def test_compute_goal_direction_rotated_robot(self):
        """Test goal direction computation with rotated robot."""
        # Create mock odometry (robot rotated 90° counterclockwise)
        mock_odom = Mock()
        mock_odom.pose.pose.position.x = 0.0
        mock_odom.pose.pose.position.y = 0.0
        mock_odom.pose.pose.orientation.x = 0.0
        mock_odom.pose.pose.orientation.y = 0.0
        mock_odom.pose.pose.orientation.z = 0.7071068  # sin(π/4)
        mock_odom.pose.pose.orientation.w = 0.7071068  # cos(π/4)
        
        # Create mock goal (in world frame, goal is to the right of robot)
        mock_goal = Mock()
        mock_goal.position.x = 0.0
        mock_goal.position.y = 1.0
        
        goal_dir = PPOObservation._compute_goal_direction(mock_odom, mock_goal)
        
        assert goal_dir.shape == (2,)
        # In robot frame, goal should be forward
        np.testing.assert_array_almost_equal(goal_dir, [1.0, 0.0], decimal=4)
    
    def test_extract_velocity(self):
        """Test velocity extraction from odometry."""
        mock_odom = Mock()
        mock_odom.twist.twist.linear.x = 1.5
        mock_odom.twist.twist.angular.z = 0.3
        
        velocity = PPOObservation._extract_velocity(mock_odom)
        
        assert velocity.shape == (2,)
        assert velocity.dtype == np.float32
        np.testing.assert_array_almost_equal(velocity, [1.5, 0.3], decimal=5)
    
    def test_compute_obstacle_proximity_no_obstacles(self):
        """Test obstacle proximity computation with no obstacles."""
        mock_obstacles = Mock()
        mock_obstacles.obstacles = []
        
        mock_odom = Mock()
        mock_odom.pose.pose.position.x = 0.0
        mock_odom.pose.pose.position.y = 0.0
        mock_odom.pose.pose.orientation.x = 0.0
        mock_odom.pose.pose.orientation.y = 0.0
        mock_odom.pose.pose.orientation.z = 0.0
        mock_odom.pose.pose.orientation.w = 1.0
        
        proximity = PPOObservation._compute_obstacle_proximity(mock_obstacles, mock_odom)
        
        assert proximity.shape == (8,)
        assert proximity.dtype == np.float32
        # All sectors should have maximum distance
        np.testing.assert_array_equal(proximity, np.full(8, 10.0))
    
    def test_compute_obstacle_proximity_with_obstacles(self):
        """Test obstacle proximity computation with obstacles."""
        # Create mock obstacle in front of robot
        mock_obstacle = Mock()
        mock_obstacle.position.x = 2.0  # 2m in front
        mock_obstacle.position.y = 0.0
        
        mock_obstacles = Mock()
        mock_obstacles.obstacles = [mock_obstacle]
        
        mock_odom = Mock()
        mock_odom.pose.pose.position.x = 0.0
        mock_odom.pose.pose.position.y = 0.0
        mock_odom.pose.pose.orientation.x = 0.0
        mock_odom.pose.pose.orientation.y = 0.0
        mock_odom.pose.pose.orientation.z = 0.0
        mock_odom.pose.pose.orientation.w = 1.0
        
        proximity = PPOObservation._compute_obstacle_proximity(mock_obstacles, mock_odom)
        
        assert proximity.shape == (8,)
        # Forward sector (index 0) should have distance 2.0
        assert proximity[0] == pytest.approx(2.0, abs=1e-5)
        # Other sectors should have maximum distance
        for i in range(1, 8):
            assert proximity[i] == pytest.approx(10.0, abs=1e-5)


class TestPPOObservationROS:
    """Test cases for PPOObservation ROS integration."""
    
    @pytest.mark.skipif(not ROS_AVAILABLE, reason="ROS messages not available")
    def test_from_ros_messages_ros_available(self):
        """Test construction from ROS messages when ROS is available."""
        # This test will only run if ROS is actually available
        from sensor_msgs.msg import LaserScan
        from nav_msgs.msg import Odometry
        from geometry_msgs.msg import Pose, Point, Quaternion, Twist, TwistWithCovariance, PoseWithCovariance
        from custom_msgs.msg import ObstacleArray
        from std_msgs.msg import Header
        
        # Create real ROS messages
        scan = LaserScan()
        scan.ranges = [1.0] * 360
        scan.range_min = 0.1
        scan.range_max = 10.0
        
        odom = Odometry()
        odom.pose.pose.position = Point(x=0.0, y=0.0, z=0.0)
        odom.pose.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        odom.twist.twist.linear.x = 0.5
        odom.twist.twist.angular.z = 0.1
        
        goal = Pose()
        goal.position = Point(x=1.0, y=0.0, z=0.0)
        goal.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        
        obstacles = ObstacleArray()
        obstacles.obstacles = []
        
        # Should not raise an exception
        obs = PPOObservation.from_ros_messages(scan, odom, obstacles, goal)
        
        assert obs.lidar_scan.shape == (360,)
        assert obs.goal_direction.shape == (2,)
        assert obs.current_velocity.shape == (2,)
        assert obs.obstacle_proximity.shape == (8,)
    
    def test_from_ros_messages_ros_not_available(self):
        """Test that from_ros_messages raises error when ROS not available."""
        if ROS_AVAILABLE:
            pytest.skip("ROS is available, skipping this test")
        
        # Should raise ValueError when ROS not available
        with pytest.raises(ValueError, match="ROS messages not available"):
            PPOObservation.from_ros_messages(None, None, None, None)


class TestPPOObservationEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_goal_direction_at_goal(self):
        """Test goal direction when robot is at goal."""
        mock_odom = Mock()
        mock_odom.pose.pose.position.x = 1.0
        mock_odom.pose.pose.position.y = 1.0
        mock_odom.pose.pose.orientation.x = 0.0
        mock_odom.pose.pose.orientation.y = 0.0
        mock_odom.pose.pose.orientation.z = 0.0
        mock_odom.pose.pose.orientation.w = 1.0
        
        mock_goal = Mock()
        mock_goal.position.x = 1.0
        mock_goal.position.y = 1.0
        
        goal_dir = PPOObservation._compute_goal_direction(mock_odom, mock_goal)
        
        # Should be zero vector when at goal
        np.testing.assert_array_almost_equal(goal_dir, [0.0, 0.0], decimal=5)
    
    def test_lidar_scan_all_invalid(self):
        """Test LiDAR processing with all invalid readings."""
        mock_scan = Mock()
        mock_scan.ranges = [float('inf')] * 360
        mock_scan.range_min = 0.1
        mock_scan.range_max = 10.0
        
        processed = PPOObservation._process_lidar_scan(mock_scan)
        
        # All readings should be normalized to 1.0 (max range)
        np.testing.assert_array_equal(processed, np.ones(360, dtype=np.float32))
    
    def test_obstacle_proximity_sectors(self):
        """Test that obstacles are assigned to correct sectors."""
        # Create obstacles in different directions
        obstacles = []
        positions = [
            (1.0, 0.0),    # Forward (sector 0)
            (1.0, 1.0),    # Forward-right (sector 1)
            (0.0, 1.0),    # Right (sector 2)
            (-1.0, 1.0),   # Backward-right (sector 3)
            (-1.0, 0.0),   # Backward (sector 4)
            (-1.0, -1.0),  # Backward-left (sector 5)
            (0.0, -1.0),   # Left (sector 6)
            (1.0, -1.0),   # Forward-left (sector 7)
        ]
        
        for x, y in positions:
            mock_obstacle = Mock()
            mock_obstacle.position.x = x
            mock_obstacle.position.y = y
            obstacles.append(mock_obstacle)
        
        mock_obstacles = Mock()
        mock_obstacles.obstacles = obstacles
        
        mock_odom = Mock()
        mock_odom.pose.pose.position.x = 0.0
        mock_odom.pose.pose.position.y = 0.0
        mock_odom.pose.pose.orientation.x = 0.0
        mock_odom.pose.pose.orientation.y = 0.0
        mock_odom.pose.pose.orientation.z = 0.0
        mock_odom.pose.pose.orientation.w = 1.0
        
        proximity = PPOObservation._compute_obstacle_proximity(mock_obstacles, mock_odom)
        
        # Each sector should have an obstacle at approximately sqrt(2) distance
        # (except sectors 0, 2, 4, 6 which are at distance 1.0)
        expected_distances = [
            1.0,           # Forward
            np.sqrt(2),    # Forward-right
            1.0,           # Right
            np.sqrt(2),    # Backward-right
            1.0,           # Backward
            np.sqrt(2),    # Backward-left
            1.0,           # Left
            np.sqrt(2),    # Forward-left
        ]
        
        for i, expected in enumerate(expected_distances):
            assert proximity[i] == pytest.approx(expected, abs=1e-5)