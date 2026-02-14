"""
Unit tests for DynamicObstacle class.

Tests cover:
- Obstacle initialization
- Position prediction
- Distance calculations
- Input validation
- Edge cases
"""

import pytest
import numpy as np
from adaptnav.core import DynamicObstacle


class TestDynamicObstacleInitialization:
    """Test DynamicObstacle initialization and basic properties."""
    
    def test_basic_initialization(self):
        """Test basic obstacle initialization."""
        obstacle = DynamicObstacle(
            id=1,
            position=np.array([5.0, 3.0]),
            velocity=np.array([1.0, 0.5]),
            radius=0.5,
            classification="worker"
        )
        
        assert obstacle.id == 1
        assert np.allclose(obstacle.position, [5.0, 3.0])
        assert np.allclose(obstacle.velocity, [1.0, 0.5])
        assert obstacle.radius == 0.5
        assert obstacle.classification == "worker"
    
    def test_initialization_with_lists(self):
        """Test initialization with Python lists instead of numpy arrays."""
        obstacle = DynamicObstacle(
            id=2,
            position=[10.0, 20.0],
            velocity=[2.0, -1.0],
            radius=1.0,
            classification="forklift"
        )
        
        assert isinstance(obstacle.position, np.ndarray)
        assert isinstance(obstacle.velocity, np.ndarray)
        assert np.allclose(obstacle.position, [10.0, 20.0])
        assert np.allclose(obstacle.velocity, [2.0, -1.0])
    
    def test_default_classification(self):
        """Test that default classification is 'unknown'."""
        obstacle = DynamicObstacle(
            id=3,
            position=np.array([0.0, 0.0]),
            velocity=np.array([0.0, 0.0]),
            radius=0.5
        )
        
        assert obstacle.classification == "unknown"
    
    def test_zero_velocity(self):
        """Test obstacle with zero velocity (stationary)."""
        obstacle = DynamicObstacle(
            id=4,
            position=np.array([5.0, 5.0]),
            velocity=np.array([0.0, 0.0]),
            radius=0.5,
            classification="worker"
        )
        
        assert np.allclose(obstacle.velocity, [0.0, 0.0])
    
    def test_negative_velocity(self):
        """Test obstacle with negative velocity components."""
        obstacle = DynamicObstacle(
            id=5,
            position=np.array([5.0, 5.0]),
            velocity=np.array([-1.0, -2.0]),
            radius=0.5,
            classification="worker"
        )
        
        assert np.allclose(obstacle.velocity, [-1.0, -2.0])


class TestInputValidation:
    """Test input validation and error handling."""
    
    def test_invalid_position_dimension(self):
        """Test that 1D or 3D position raises ValueError."""
        with pytest.raises(ValueError, match="Position must be a 2D array"):
            DynamicObstacle(
                id=1,
                position=np.array([5.0]),  # 1D
                velocity=np.array([1.0, 0.5]),
                radius=0.5
            )
        
        with pytest.raises(ValueError, match="Position must be a 2D array"):
            DynamicObstacle(
                id=1,
                position=np.array([5.0, 3.0, 1.0]),  # 3D
                velocity=np.array([1.0, 0.5]),
                radius=0.5
            )
    
    def test_invalid_velocity_dimension(self):
        """Test that 1D or 3D velocity raises ValueError."""
        with pytest.raises(ValueError, match="Velocity must be a 2D array"):
            DynamicObstacle(
                id=1,
                position=np.array([5.0, 3.0]),
                velocity=np.array([1.0]),  # 1D
                radius=0.5
            )
        
        with pytest.raises(ValueError, match="Velocity must be a 2D array"):
            DynamicObstacle(
                id=1,
                position=np.array([5.0, 3.0]),
                velocity=np.array([1.0, 0.5, 0.2]),  # 3D
                radius=0.5
            )
    
    def test_negative_radius(self):
        """Test that negative radius raises ValueError."""
        with pytest.raises(ValueError, match="Radius must be positive"):
            DynamicObstacle(
                id=1,
                position=np.array([5.0, 3.0]),
                velocity=np.array([1.0, 0.5]),
                radius=-0.5
            )
    
    def test_zero_radius(self):
        """Test that zero radius raises ValueError."""
        with pytest.raises(ValueError, match="Radius must be positive"):
            DynamicObstacle(
                id=1,
                position=np.array([5.0, 3.0]),
                velocity=np.array([1.0, 0.5]),
                radius=0.0
            )


class TestPositionPrediction:
    """Test position prediction using constant velocity model."""
    
    def test_predict_position_zero_time(self):
        """Test prediction at t=0 returns current position."""
        obstacle = DynamicObstacle(
            id=1,
            position=np.array([5.0, 3.0]),
            velocity=np.array([1.0, 0.5]),
            radius=0.5
        )
        
        predicted = obstacle.predict_position(0.0)
        assert np.allclose(predicted, [5.0, 3.0])
    
    def test_predict_position_one_second(self):
        """Test prediction after 1 second."""
        obstacle = DynamicObstacle(
            id=1,
            position=np.array([5.0, 3.0]),
            velocity=np.array([1.0, 0.5]),
            radius=0.5
        )
        
        predicted = obstacle.predict_position(1.0)
        assert np.allclose(predicted, [6.0, 3.5])
    
    def test_predict_position_multiple_seconds(self):
        """Test prediction after multiple seconds."""
        obstacle = DynamicObstacle(
            id=1,
            position=np.array([0.0, 0.0]),
            velocity=np.array([2.0, -1.0]),
            radius=0.5
        )
        
        predicted = obstacle.predict_position(5.0)
        assert np.allclose(predicted, [10.0, -5.0])
    
    def test_predict_position_fractional_time(self):
        """Test prediction with fractional time."""
        obstacle = DynamicObstacle(
            id=1,
            position=np.array([1.0, 2.0]),
            velocity=np.array([3.0, 4.0]),
            radius=0.5
        )
        
        predicted = obstacle.predict_position(0.5)
        assert np.allclose(predicted, [2.5, 4.0])
    
    def test_predict_position_zero_velocity(self):
        """Test prediction with zero velocity (stationary obstacle)."""
        obstacle = DynamicObstacle(
            id=1,
            position=np.array([5.0, 5.0]),
            velocity=np.array([0.0, 0.0]),
            radius=0.5
        )
        
        predicted = obstacle.predict_position(10.0)
        assert np.allclose(predicted, [5.0, 5.0])
    
    def test_predict_position_negative_velocity(self):
        """Test prediction with negative velocity."""
        obstacle = DynamicObstacle(
            id=1,
            position=np.array([10.0, 10.0]),
            velocity=np.array([-2.0, -3.0]),
            radius=0.5
        )
        
        predicted = obstacle.predict_position(2.0)
        assert np.allclose(predicted, [6.0, 4.0])
    
    def test_predict_position_negative_time_raises_error(self):
        """Test that negative time raises ValueError."""
        obstacle = DynamicObstacle(
            id=1,
            position=np.array([5.0, 3.0]),
            velocity=np.array([1.0, 0.5]),
            radius=0.5
        )
        
        with pytest.raises(ValueError, match="Time interval dt must be non-negative"):
            obstacle.predict_position(-1.0)
    
    def test_predict_position_does_not_modify_state(self):
        """Test that prediction does not modify obstacle's internal state."""
        obstacle = DynamicObstacle(
            id=1,
            position=np.array([5.0, 3.0]),
            velocity=np.array([1.0, 0.5]),
            radius=0.5
        )
        
        original_position = obstacle.position.copy()
        obstacle.predict_position(5.0)
        
        # Original position should be unchanged
        assert np.allclose(obstacle.position, original_position)


class TestDistanceCalculation:
    """Test distance calculation to points."""
    
    def test_distance_to_same_point(self):
        """Test distance to obstacle's own position is zero."""
        obstacle = DynamicObstacle(
            id=1,
            position=np.array([5.0, 3.0]),
            velocity=np.array([1.0, 0.5]),
            radius=0.5
        )
        
        distance = obstacle.distance_to(np.array([5.0, 3.0]))
        assert abs(distance) < 1e-10
    
    def test_distance_to_point_on_x_axis(self):
        """Test distance to point along x-axis."""
        obstacle = DynamicObstacle(
            id=1,
            position=np.array([0.0, 0.0]),
            velocity=np.array([0.0, 0.0]),
            radius=0.5
        )
        
        distance = obstacle.distance_to(np.array([3.0, 0.0]))
        assert abs(distance - 3.0) < 1e-10
    
    def test_distance_to_point_on_y_axis(self):
        """Test distance to point along y-axis."""
        obstacle = DynamicObstacle(
            id=1,
            position=np.array([0.0, 0.0]),
            velocity=np.array([0.0, 0.0]),
            radius=0.5
        )
        
        distance = obstacle.distance_to(np.array([0.0, 4.0]))
        assert abs(distance - 4.0) < 1e-10
    
    def test_distance_to_diagonal_point(self):
        """Test distance to point on diagonal."""
        obstacle = DynamicObstacle(
            id=1,
            position=np.array([0.0, 0.0]),
            velocity=np.array([0.0, 0.0]),
            radius=0.5
        )
        
        distance = obstacle.distance_to(np.array([3.0, 4.0]))
        assert abs(distance - 5.0) < 1e-10  # 3-4-5 triangle
    
    def test_distance_to_negative_coordinates(self):
        """Test distance to point with negative coordinates."""
        obstacle = DynamicObstacle(
            id=1,
            position=np.array([5.0, 5.0]),
            velocity=np.array([0.0, 0.0]),
            radius=0.5
        )
        
        distance = obstacle.distance_to(np.array([2.0, 1.0]))
        expected = np.sqrt((5.0 - 2.0)**2 + (5.0 - 1.0)**2)
        assert abs(distance - expected) < 1e-10
    
    def test_distance_with_list_input(self):
        """Test distance calculation with list input."""
        obstacle = DynamicObstacle(
            id=1,
            position=np.array([0.0, 0.0]),
            velocity=np.array([0.0, 0.0]),
            radius=0.5
        )
        
        distance = obstacle.distance_to([3.0, 4.0])
        assert abs(distance - 5.0) < 1e-10
    
    def test_distance_to_invalid_point_dimension(self):
        """Test that invalid point dimension raises ValueError."""
        obstacle = DynamicObstacle(
            id=1,
            position=np.array([5.0, 3.0]),
            velocity=np.array([1.0, 0.5]),
            radius=0.5
        )
        
        with pytest.raises(ValueError, match="Point must be a 2D array"):
            obstacle.distance_to(np.array([5.0]))
        
        with pytest.raises(ValueError, match="Point must be a 2D array"):
            obstacle.distance_to(np.array([5.0, 3.0, 1.0]))


class TestObstacleEquality:
    """Test equality and hashing based on obstacle ID."""
    
    def test_equality_same_id(self):
        """Test that obstacles with same ID are equal."""
        obstacle1 = DynamicObstacle(
            id=1,
            position=np.array([5.0, 3.0]),
            velocity=np.array([1.0, 0.5]),
            radius=0.5
        )
        obstacle2 = DynamicObstacle(
            id=1,
            position=np.array([10.0, 20.0]),  # Different position
            velocity=np.array([2.0, 1.0]),    # Different velocity
            radius=1.0                         # Different radius
        )
        
        assert obstacle1 == obstacle2
    
    def test_inequality_different_id(self):
        """Test that obstacles with different IDs are not equal."""
        obstacle1 = DynamicObstacle(
            id=1,
            position=np.array([5.0, 3.0]),
            velocity=np.array([1.0, 0.5]),
            radius=0.5
        )
        obstacle2 = DynamicObstacle(
            id=2,
            position=np.array([5.0, 3.0]),  # Same position
            velocity=np.array([1.0, 0.5]),  # Same velocity
            radius=0.5                       # Same radius
        )
        
        assert obstacle1 != obstacle2
    
    def test_inequality_with_non_obstacle(self):
        """Test that obstacle is not equal to non-obstacle objects."""
        obstacle = DynamicObstacle(
            id=1,
            position=np.array([5.0, 3.0]),
            velocity=np.array([1.0, 0.5]),
            radius=0.5
        )
        
        assert obstacle != 1
        assert obstacle != "obstacle"
        assert obstacle != None
        assert obstacle != [5.0, 3.0]
    
    def test_hash_same_id(self):
        """Test that obstacles with same ID have same hash."""
        obstacle1 = DynamicObstacle(
            id=1,
            position=np.array([5.0, 3.0]),
            velocity=np.array([1.0, 0.5]),
            radius=0.5
        )
        obstacle2 = DynamicObstacle(
            id=1,
            position=np.array([10.0, 20.0]),
            velocity=np.array([2.0, 1.0]),
            radius=1.0
        )
        
        assert hash(obstacle1) == hash(obstacle2)
    
    def test_hash_different_id(self):
        """Test that obstacles with different IDs have different hashes."""
        obstacle1 = DynamicObstacle(
            id=1,
            position=np.array([5.0, 3.0]),
            velocity=np.array([1.0, 0.5]),
            radius=0.5
        )
        obstacle2 = DynamicObstacle(
            id=2,
            position=np.array([5.0, 3.0]),
            velocity=np.array([1.0, 0.5]),
            radius=0.5
        )
        
        # Different IDs should have different hashes (with high probability)
        assert hash(obstacle1) != hash(obstacle2)
    
    def test_use_in_set(self):
        """Test that obstacles can be used in sets (requires hashable)."""
        obstacle1 = DynamicObstacle(id=1, position=[0, 0], velocity=[0, 0], radius=0.5)
        obstacle2 = DynamicObstacle(id=2, position=[1, 1], velocity=[1, 1], radius=0.5)
        obstacle3 = DynamicObstacle(id=1, position=[2, 2], velocity=[2, 2], radius=0.5)
        
        obstacle_set = {obstacle1, obstacle2, obstacle3}
        
        # obstacle1 and obstacle3 have same ID, so set should have 2 elements
        assert len(obstacle_set) == 2
        assert obstacle1 in obstacle_set
        assert obstacle2 in obstacle_set


class TestObstacleRepresentation:
    """Test string representation of obstacles."""
    
    def test_repr(self):
        """Test __repr__ method."""
        obstacle = DynamicObstacle(
            id=42,
            position=np.array([5.0, 3.0]),
            velocity=np.array([1.0, 0.5]),
            radius=0.75,
            classification="worker"
        )
        
        repr_str = repr(obstacle)
        
        assert "DynamicObstacle" in repr_str
        assert "id=42" in repr_str
        assert "radius=0.75" in repr_str
        assert "classification='worker'" in repr_str


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_very_large_position(self):
        """Test obstacle with very large position values."""
        obstacle = DynamicObstacle(
            id=1,
            position=np.array([1000.0, 2000.0]),
            velocity=np.array([10.0, 20.0]),
            radius=0.5
        )
        
        predicted = obstacle.predict_position(10.0)
        assert np.allclose(predicted, [1100.0, 2200.0])
    
    def test_very_small_radius(self):
        """Test obstacle with very small radius."""
        obstacle = DynamicObstacle(
            id=1,
            position=np.array([5.0, 3.0]),
            velocity=np.array([1.0, 0.5]),
            radius=0.001
        )
        
        assert obstacle.radius == 0.001
    
    def test_very_large_radius(self):
        """Test obstacle with very large radius."""
        obstacle = DynamicObstacle(
            id=1,
            position=np.array([5.0, 3.0]),
            velocity=np.array([1.0, 0.5]),
            radius=100.0
        )
        
        assert obstacle.radius == 100.0
    
    def test_very_high_velocity(self):
        """Test obstacle with very high velocity."""
        obstacle = DynamicObstacle(
            id=1,
            position=np.array([0.0, 0.0]),
            velocity=np.array([100.0, 200.0]),
            radius=0.5
        )
        
        predicted = obstacle.predict_position(1.0)
        assert np.allclose(predicted, [100.0, 200.0])
    
    def test_float_precision(self):
        """Test that float precision is maintained."""
        obstacle = DynamicObstacle(
            id=1,
            position=np.array([1.23456789, 9.87654321]),
            velocity=np.array([0.11111111, 0.22222222]),
            radius=0.5
        )
        
        # Check that precision is maintained
        assert obstacle.position.dtype == np.float64
        assert obstacle.velocity.dtype == np.float64
    
    def test_classification_types(self):
        """Test all valid classification types."""
        classifications = ["worker", "forklift", "unknown"]
        
        for classification in classifications:
            obstacle = DynamicObstacle(
                id=1,
                position=np.array([0.0, 0.0]),
                velocity=np.array([0.0, 0.0]),
                radius=0.5,
                classification=classification
            )
            assert obstacle.classification == classification
