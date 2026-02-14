"""
Unit tests for RobotState class.

Tests cover:
- Initialization and validation
- Distance to goal calculation
- ROS message conversion (if ROS available)
- Edge cases and error handling
"""

import pytest
import numpy as np
from adaptnav.core import RobotState


class TestRobotStateInitialization:
    """Test RobotState initialization and validation."""
    
    def test_basic_initialization(self):
        """Test basic initialization with required parameters."""
        position = np.array([1.0, 2.0])
        orientation = 0.5
        
        state = RobotState(position=position, orientation=orientation)
        
        assert np.allclose(state.position, position)
        assert state.orientation == 0.5
        assert state.linear_velocity == 0.0
        assert state.angular_velocity == 0.0
        assert state.timestamp is None
    
    def test_full_initialization(self):
        """Test initialization with all parameters."""
        position = np.array([3.0, 4.0])
        orientation = 1.57
        linear_velocity = 0.5
        angular_velocity = 0.2
        timestamp = 123.456
        
        state = RobotState(
            position=position,
            orientation=orientation,
            linear_velocity=linear_velocity,
            angular_velocity=angular_velocity,
            timestamp=timestamp
        )
        
        assert np.allclose(state.position, position)
        assert state.orientation == 1.57
        assert state.linear_velocity == 0.5
        assert state.angular_velocity == 0.2
        assert state.timestamp == 123.456
    
    def test_position_from_list(self):
        """Test that position can be initialized from a list."""
        state = RobotState(position=[5.0, 6.0], orientation=0.0)
        
        assert isinstance(state.position, np.ndarray)
        assert np.allclose(state.position, [5.0, 6.0])
    
    def test_invalid_position_dimension(self):
        """Test that invalid position dimensions raise ValueError."""
        with pytest.raises(ValueError, match="Position must be a 2D array"):
            RobotState(position=np.array([1.0]), orientation=0.0)
        
        with pytest.raises(ValueError, match="Position must be a 2D array"):
            RobotState(position=np.array([1.0, 2.0, 3.0]), orientation=0.0)
    
    def test_negative_velocities(self):
        """Test that negative velocities are allowed."""
        state = RobotState(
            position=[0.0, 0.0],
            orientation=0.0,
            linear_velocity=-0.5,
            angular_velocity=-0.3
        )
        
        assert state.linear_velocity == -0.5
        assert state.angular_velocity == -0.3


class TestDistanceToGoal:
    """Test distance_to_goal method."""
    
    def test_distance_to_same_position(self):
        """Test distance to goal at same position is zero."""
        state = RobotState(position=[1.0, 2.0], orientation=0.0)
        goal = np.array([1.0, 2.0])
        
        distance = state.distance_to_goal(goal)
        
        assert distance == 0.0
    
    def test_distance_horizontal(self):
        """Test distance calculation for horizontal movement."""
        state = RobotState(position=[0.0, 0.0], orientation=0.0)
        goal = np.array([3.0, 0.0])
        
        distance = state.distance_to_goal(goal)
        
        assert np.isclose(distance, 3.0)
    
    def test_distance_vertical(self):
        """Test distance calculation for vertical movement."""
        state = RobotState(position=[0.0, 0.0], orientation=0.0)
        goal = np.array([0.0, 4.0])
        
        distance = state.distance_to_goal(goal)
        
        assert np.isclose(distance, 4.0)
    
    def test_distance_diagonal(self):
        """Test distance calculation for diagonal movement."""
        state = RobotState(position=[0.0, 0.0], orientation=0.0)
        goal = np.array([3.0, 4.0])
        
        distance = state.distance_to_goal(goal)
        
        assert np.isclose(distance, 5.0)  # 3-4-5 triangle
    
    def test_distance_with_negative_coordinates(self):
        """Test distance calculation with negative coordinates."""
        state = RobotState(position=[-1.0, -1.0], orientation=0.0)
        goal = np.array([2.0, 3.0])
        
        distance = state.distance_to_goal(goal)
        
        expected = np.sqrt(3**2 + 4**2)
        assert np.isclose(distance, expected)
    
    def test_distance_goal_from_list(self):
        """Test that goal can be provided as a list."""
        state = RobotState(position=[0.0, 0.0], orientation=0.0)
        
        distance = state.distance_to_goal([3.0, 4.0])
        
        assert np.isclose(distance, 5.0)
    
    def test_invalid_goal_dimension(self):
        """Test that invalid goal dimensions raise ValueError."""
        state = RobotState(position=[0.0, 0.0], orientation=0.0)
        
        with pytest.raises(ValueError, match="Goal must be a 2D array"):
            state.distance_to_goal(np.array([1.0]))
        
        with pytest.raises(ValueError, match="Goal must be a 2D array"):
            state.distance_to_goal(np.array([1.0, 2.0, 3.0]))


class TestToPoseStamped:
    """Test to_pose_stamped method."""
    
    def test_to_pose_stamped_without_ros(self):
        """Test that to_pose_stamped raises ImportError when ROS not available."""
        state = RobotState(position=[1.0, 2.0], orientation=0.0)
        
        # Try to import ROS messages to check if available
        try:
            from geometry_msgs.msg import PoseStamped
            pytest.skip("ROS messages are available, skipping this test")
        except ImportError:
            # ROS not available, should raise ImportError
            with pytest.raises(ImportError, match="ROS messages not available"):
                state.to_pose_stamped()
    
    def test_to_pose_stamped_with_ros(self):
        """Test conversion to PoseStamped when ROS is available."""
        try:
            from geometry_msgs.msg import PoseStamped
        except ImportError:
            pytest.skip("ROS messages not available")
        
        state = RobotState(
            position=[1.5, 2.5],
            orientation=1.57,  # ~90 degrees
            timestamp=100.5
        )
        
        pose_stamped = state.to_pose_stamped(frame_id="map")
        
        # Check header
        assert pose_stamped.header.frame_id == "map"
        assert pose_stamped.header.stamp.sec == 100
        assert pose_stamped.header.stamp.nanosec == 500000000
        
        # Check position
        assert np.isclose(pose_stamped.pose.position.x, 1.5)
        assert np.isclose(pose_stamped.pose.position.y, 2.5)
        assert pose_stamped.pose.position.z == 0.0
        
        # Check orientation (quaternion)
        # For yaw = 1.57 rad, qz = sin(1.57/2) ≈ 0.7071, qw = cos(1.57/2) ≈ 0.7071
        assert np.isclose(pose_stamped.pose.orientation.x, 0.0)
        assert np.isclose(pose_stamped.pose.orientation.y, 0.0)
        assert np.isclose(pose_stamped.pose.orientation.z, np.sin(1.57 / 2), atol=1e-3)
        assert np.isclose(pose_stamped.pose.orientation.w, np.cos(1.57 / 2), atol=1e-3)
    
    def test_to_pose_stamped_zero_orientation(self):
        """Test conversion with zero orientation."""
        try:
            from geometry_msgs.msg import PoseStamped
        except ImportError:
            pytest.skip("ROS messages not available")
        
        state = RobotState(position=[0.0, 0.0], orientation=0.0)
        
        pose_stamped = state.to_pose_stamped()
        
        # For yaw = 0, qz = 0, qw = 1
        assert pose_stamped.pose.orientation.x == 0.0
        assert pose_stamped.pose.orientation.y == 0.0
        assert pose_stamped.pose.orientation.z == 0.0
        assert np.isclose(pose_stamped.pose.orientation.w, 1.0)
    
    def test_to_pose_stamped_custom_frame(self):
        """Test conversion with custom frame_id."""
        try:
            from geometry_msgs.msg import PoseStamped
        except ImportError:
            pytest.skip("ROS messages not available")
        
        state = RobotState(position=[0.0, 0.0], orientation=0.0)
        
        pose_stamped = state.to_pose_stamped(frame_id="odom")
        
        assert pose_stamped.header.frame_id == "odom"
    
    def test_to_pose_stamped_without_timestamp(self):
        """Test conversion when timestamp is None."""
        try:
            from geometry_msgs.msg import PoseStamped
        except ImportError:
            pytest.skip("ROS messages not available")
        
        state = RobotState(position=[0.0, 0.0], orientation=0.0, timestamp=None)
        
        pose_stamped = state.to_pose_stamped()
        
        # Timestamp should be zero when not provided
        assert pose_stamped.header.stamp.sec == 0
        assert pose_stamped.header.stamp.nanosec == 0


class TestRobotStateEquality:
    """Test equality and representation methods."""
    
    def test_equality_same_state(self):
        """Test that identical states are equal."""
        state1 = RobotState(
            position=[1.0, 2.0],
            orientation=0.5,
            linear_velocity=0.3,
            angular_velocity=0.1
        )
        state2 = RobotState(
            position=[1.0, 2.0],
            orientation=0.5,
            linear_velocity=0.3,
            angular_velocity=0.1
        )
        
        assert state1 == state2
    
    def test_equality_different_position(self):
        """Test that states with different positions are not equal."""
        state1 = RobotState(position=[1.0, 2.0], orientation=0.0)
        state2 = RobotState(position=[1.1, 2.0], orientation=0.0)
        
        assert state1 != state2
    
    def test_equality_different_orientation(self):
        """Test that states with different orientations are not equal."""
        state1 = RobotState(position=[1.0, 2.0], orientation=0.0)
        state2 = RobotState(position=[1.0, 2.0], orientation=0.1)
        
        assert state1 != state2
    
    def test_equality_with_tolerance(self):
        """Test that nearly equal states are considered equal."""
        state1 = RobotState(position=[1.0, 2.0], orientation=0.5)
        state2 = RobotState(position=[1.0 + 1e-10, 2.0], orientation=0.5)
        
        assert state1 == state2
    
    def test_equality_with_non_robot_state(self):
        """Test that comparison with non-RobotState returns False."""
        state = RobotState(position=[1.0, 2.0], orientation=0.0)
        
        assert state != "not a robot state"
        assert state != 42
        assert state != None
    
    def test_repr(self):
        """Test string representation."""
        state = RobotState(
            position=[1.5, 2.5],
            orientation=0.785,
            linear_velocity=0.5,
            angular_velocity=0.2,
            timestamp=100.0
        )
        
        repr_str = repr(state)
        
        assert "RobotState" in repr_str
        assert "1.5" in repr_str
        assert "2.5" in repr_str
        assert "0.785" in repr_str
        assert "0.5" in repr_str
        assert "0.2" in repr_str
        assert "100.0" in repr_str


class TestRobotStateEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_large_coordinates(self):
        """Test with large coordinate values."""
        state = RobotState(position=[1000.0, 2000.0], orientation=0.0)
        goal = np.array([1003.0, 2004.0])
        
        distance = state.distance_to_goal(goal)
        
        assert np.isclose(distance, 5.0)
    
    def test_orientation_wrapping(self):
        """Test with orientation values outside [0, 2π]."""
        # Orientation values outside [0, 2π] should still work
        state1 = RobotState(position=[0.0, 0.0], orientation=2 * np.pi)
        state2 = RobotState(position=[0.0, 0.0], orientation=-np.pi)
        
        # Should not raise errors
        assert state1.orientation == 2 * np.pi
        assert state2.orientation == -np.pi
    
    def test_zero_timestamp(self):
        """Test with zero timestamp."""
        state = RobotState(position=[0.0, 0.0], orientation=0.0, timestamp=0.0)
        
        assert state.timestamp == 0.0
    
    def test_very_small_distance(self):
        """Test distance calculation with very small differences."""
        state = RobotState(position=[1.0, 2.0], orientation=0.0)
        goal = np.array([1.0 + 1e-10, 2.0 + 1e-10])
        
        distance = state.distance_to_goal(goal)
        
        assert distance < 1e-9
