"""
Unit tests for Waypoint and Path classes.
"""

import pytest
import numpy as np
from adaptnav.core.path import Waypoint, Path


class TestWaypoint:
    """Test cases for the Waypoint class."""
    
    def test_waypoint_initialization(self):
        """Test basic waypoint creation."""
        wp = Waypoint(1.0, 2.0, 0.5)
        assert wp.x == 1.0
        assert wp.y == 2.0
        assert wp.theta == 0.5
    
    def test_waypoint_default_theta(self):
        """Test waypoint creation with default theta."""
        wp = Waypoint(3.0, 4.0)
        assert wp.x == 3.0
        assert wp.y == 4.0
        assert wp.theta == 0.0
    
    def test_waypoint_position(self):
        """Test position() method returns correct numpy array."""
        wp = Waypoint(5.0, 6.0, 1.0)
        pos = wp.position()
        assert isinstance(pos, np.ndarray)
        assert pos.shape == (2,)
        assert np.allclose(pos, [5.0, 6.0])
    
    def test_waypoint_distance_to(self):
        """Test distance_to() method."""
        wp = Waypoint(0.0, 0.0, 0.0)
        
        # Distance to origin should be 0
        assert np.isclose(wp.distance_to(np.array([0.0, 0.0])), 0.0)
        
        # Distance to (3, 4) should be 5 (Pythagorean triple)
        assert np.isclose(wp.distance_to(np.array([3.0, 4.0])), 5.0)
        
        # Distance to (1, 0) should be 1
        assert np.isclose(wp.distance_to(np.array([1.0, 0.0])), 1.0)
    
    def test_waypoint_equality(self):
        """Test waypoint equality comparison."""
        wp1 = Waypoint(1.0, 2.0, 0.5)
        wp2 = Waypoint(1.0, 2.0, 0.5)
        wp3 = Waypoint(1.0, 2.0, 0.6)
        wp4 = Waypoint(1.1, 2.0, 0.5)
        
        assert wp1 == wp2
        assert wp1 != wp3
        assert wp1 != wp4
        assert wp1 != "not a waypoint"
    
    def test_waypoint_repr(self):
        """Test waypoint string representation."""
        wp = Waypoint(1.5, 2.5, 0.785)
        repr_str = repr(wp)
        assert "Waypoint" in repr_str
        assert "1.50" in repr_str
        assert "2.50" in repr_str


class TestPath:
    """Test cases for the Path class."""
    
    def test_path_initialization(self):
        """Test basic path creation."""
        waypoints = [
            Waypoint(0.0, 0.0, 0.0),
            Waypoint(1.0, 0.0, 0.0),
            Waypoint(1.0, 1.0, 1.57)
        ]
        path = Path(waypoints)
        
        assert len(path.waypoints) == 3
        assert path.waypoints[0] == waypoints[0]
        assert path.waypoints[1] == waypoints[1]
        assert path.waypoints[2] == waypoints[2]
    
    def test_path_empty_waypoints_raises_error(self):
        """Test that creating a path with no waypoints raises an error."""
        with pytest.raises(ValueError, match="at least one waypoint"):
            Path([])
    
    def test_path_length_single_waypoint(self):
        """Test path length with single waypoint is zero."""
        path = Path([Waypoint(0.0, 0.0, 0.0)])
        assert np.isclose(path.total_length, 0.0)
    
    def test_path_length_straight_line(self):
        """Test path length calculation for straight line."""
        waypoints = [
            Waypoint(0.0, 0.0, 0.0),
            Waypoint(3.0, 0.0, 0.0),
            Waypoint(3.0, 4.0, 1.57)
        ]
        path = Path(waypoints)
        
        # Length should be 3 + 4 = 7
        assert np.isclose(path.total_length, 7.0)
    
    def test_path_length_complex(self):
        """Test path length calculation for complex path."""
        waypoints = [
            Waypoint(0.0, 0.0, 0.0),
            Waypoint(1.0, 0.0, 0.0),
            Waypoint(1.0, 1.0, 1.57),
            Waypoint(0.0, 1.0, 3.14)
        ]
        path = Path(waypoints)
        
        # Length should be 1 + 1 + 1 = 3
        assert np.isclose(path.total_length, 3.0)
    
    def test_get_closest_waypoint_at_waypoint(self):
        """Test finding closest waypoint when at a waypoint."""
        waypoints = [
            Waypoint(0.0, 0.0, 0.0),
            Waypoint(1.0, 0.0, 0.0),
            Waypoint(2.0, 0.0, 0.0)
        ]
        path = Path(waypoints)
        
        # Position at second waypoint
        idx, wp = path.get_closest_waypoint(np.array([1.0, 0.0]))
        assert idx == 1
        assert wp == waypoints[1]
    
    def test_get_closest_waypoint_between_waypoints(self):
        """Test finding closest waypoint when between waypoints."""
        waypoints = [
            Waypoint(0.0, 0.0, 0.0),
            Waypoint(2.0, 0.0, 0.0),
            Waypoint(4.0, 0.0, 0.0)
        ]
        path = Path(waypoints)
        
        # Position at 1.0, closer to first waypoint
        idx, wp = path.get_closest_waypoint(np.array([0.8, 0.0]))
        assert idx == 0
        
        # Position at 3.0, closer to third waypoint
        idx, wp = path.get_closest_waypoint(np.array([3.2, 0.0]))
        assert idx == 2
    
    def test_get_closest_waypoint_off_path(self):
        """Test finding closest waypoint when off the path."""
        waypoints = [
            Waypoint(0.0, 0.0, 0.0),
            Waypoint(1.0, 0.0, 0.0),
            Waypoint(2.0, 0.0, 0.0)
        ]
        path = Path(waypoints)
        
        # Position above the path, closest to middle waypoint
        idx, wp = path.get_closest_waypoint(np.array([1.0, 0.5]))
        assert idx == 1
        assert wp == waypoints[1]
    
    def test_get_lookahead_point_exact_distance(self):
        """Test lookahead point when waypoint is at exact distance."""
        waypoints = [
            Waypoint(0.0, 0.0, 0.0),
            Waypoint(1.0, 0.0, 0.0),
            Waypoint(2.0, 0.0, 0.0),
            Waypoint(3.0, 0.0, 0.0)
        ]
        path = Path(waypoints)
        
        # From origin, lookahead 2.0 should give third waypoint
        wp = path.get_lookahead_point(np.array([0.0, 0.0]), 2.0)
        assert wp == waypoints[2]
    
    def test_get_lookahead_point_between_waypoints(self):
        """Test lookahead point when distance falls between waypoints."""
        waypoints = [
            Waypoint(0.0, 0.0, 0.0),
            Waypoint(1.0, 0.0, 0.0),
            Waypoint(2.0, 0.0, 0.0),
            Waypoint(3.0, 0.0, 0.0)
        ]
        path = Path(waypoints)
        
        # From origin, lookahead 1.5 should give second waypoint (first >= 1.5)
        wp = path.get_lookahead_point(np.array([0.0, 0.0]), 1.5)
        assert wp == waypoints[2]
    
    def test_get_lookahead_point_exceeds_path(self):
        """Test lookahead point when distance exceeds path length."""
        waypoints = [
            Waypoint(0.0, 0.0, 0.0),
            Waypoint(1.0, 0.0, 0.0),
            Waypoint(2.0, 0.0, 0.0)
        ]
        path = Path(waypoints)
        
        # From origin, lookahead 10.0 should give last waypoint
        wp = path.get_lookahead_point(np.array([0.0, 0.0]), 10.0)
        assert wp == waypoints[-1]
    
    def test_get_lookahead_point_from_middle(self):
        """Test lookahead point when starting from middle of path."""
        waypoints = [
            Waypoint(0.0, 0.0, 0.0),
            Waypoint(1.0, 0.0, 0.0),
            Waypoint(2.0, 0.0, 0.0),
            Waypoint(3.0, 0.0, 0.0)
        ]
        path = Path(waypoints)
        
        # From position near second waypoint, lookahead 1.5
        wp = path.get_lookahead_point(np.array([1.1, 0.0]), 1.5)
        # Should find waypoint at distance >= 1.5 from [1.1, 0.0]
        # Third waypoint at [2.0, 0.0] is 0.9 away
        # Fourth waypoint at [3.0, 0.0] is 1.9 away (>= 1.5)
        assert wp == waypoints[3]
    
    def test_get_lookahead_point_small_distance(self):
        """Test lookahead point with very small distance."""
        waypoints = [
            Waypoint(0.0, 0.0, 0.0),
            Waypoint(1.0, 0.0, 0.0),
            Waypoint(2.0, 0.0, 0.0)
        ]
        path = Path(waypoints)
        
        # From origin, lookahead 0.1 should give first waypoint
        wp = path.get_lookahead_point(np.array([0.0, 0.0]), 0.1)
        assert wp == waypoints[1]
    
    def test_get_lookahead_point_invalid_distance(self):
        """Test that negative or zero lookahead distance raises error."""
        waypoints = [Waypoint(0.0, 0.0, 0.0), Waypoint(1.0, 0.0, 0.0)]
        path = Path(waypoints)
        
        with pytest.raises(ValueError, match="must be positive"):
            path.get_lookahead_point(np.array([0.0, 0.0]), 0.0)
        
        with pytest.raises(ValueError, match="must be positive"):
            path.get_lookahead_point(np.array([0.0, 0.0]), -1.0)
    
    def test_path_len(self):
        """Test __len__ method returns number of waypoints."""
        waypoints = [
            Waypoint(0.0, 0.0, 0.0),
            Waypoint(1.0, 0.0, 0.0),
            Waypoint(2.0, 0.0, 0.0)
        ]
        path = Path(waypoints)
        assert len(path) == 3
    
    def test_path_repr(self):
        """Test path string representation."""
        waypoints = [
            Waypoint(0.0, 0.0, 0.0),
            Waypoint(3.0, 4.0, 0.0)
        ]
        path = Path(waypoints)
        repr_str = repr(path)
        
        assert "Path" in repr_str
        assert "2" in repr_str  # 2 waypoints
        assert "5.00" in repr_str  # length is 5.0
    
    def test_path_with_timestamp(self):
        """Test path creation with timestamp."""
        waypoints = [Waypoint(0.0, 0.0, 0.0)]
        timestamp = 123.456
        path = Path(waypoints, timestamp=timestamp)
        
        assert path.timestamp == timestamp
    
    def test_path_without_timestamp(self):
        """Test path creation without timestamp."""
        waypoints = [Waypoint(0.0, 0.0, 0.0)]
        path = Path(waypoints)
        
        assert path.timestamp is None


class TestPathEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_get_closest_waypoint_empty_path_raises_error(self):
        """Test that get_closest_waypoint on empty path raises error."""
        # This should not be possible since Path.__init__ checks for empty,
        # but test the method's own check
        path = Path([Waypoint(0.0, 0.0, 0.0)])
        path.waypoints = []  # Force empty for testing
        
        with pytest.raises(ValueError, match="no waypoints"):
            path.get_closest_waypoint(np.array([0.0, 0.0]))
    
    def test_get_lookahead_point_empty_path_raises_error(self):
        """Test that get_lookahead_point on empty path raises error."""
        path = Path([Waypoint(0.0, 0.0, 0.0)])
        path.waypoints = []  # Force empty for testing
        
        with pytest.raises(ValueError, match="no waypoints"):
            path.get_lookahead_point(np.array([0.0, 0.0]), 1.0)
    
    def test_waypoint_with_negative_coordinates(self):
        """Test waypoint with negative coordinates."""
        wp = Waypoint(-5.0, -3.0, -1.57)
        assert wp.x == -5.0
        assert wp.y == -3.0
        assert wp.theta == -1.57
    
    def test_path_with_duplicate_waypoints(self):
        """Test path with duplicate waypoints."""
        waypoints = [
            Waypoint(0.0, 0.0, 0.0),
            Waypoint(0.0, 0.0, 0.0),
            Waypoint(1.0, 0.0, 0.0)
        ]
        path = Path(waypoints)
        
        # Length should be 0 + 1 = 1
        assert np.isclose(path.total_length, 1.0)
