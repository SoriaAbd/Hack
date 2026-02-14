"""
Property-based tests for WarehouseMap class.

Feature: adaptnav-context-aware-warehouse-navigation
Property 7: Collision-Free Paths
Validates: Requirements 3.1
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, assume, settings
from adaptnav.core import WarehouseMap


# Custom strategies for generating test data
@st.composite
def warehouse_dimensions(draw):
    """Generate valid warehouse dimensions."""
    width = draw(st.floats(min_value=5.0, max_value=100.0))
    height = draw(st.floats(min_value=5.0, max_value=100.0))
    # Use finer resolution to avoid grid discretization issues
    resolution = draw(st.floats(min_value=0.05, max_value=0.2))
    return width, height, resolution


@st.composite
def position_in_warehouse(draw, width, height):
    """Generate a valid position within warehouse bounds."""
    x = draw(st.floats(min_value=0.0, max_value=width))
    y = draw(st.floats(min_value=0.0, max_value=height))
    return x, y


@st.composite
def obstacle_config(draw, width, height):
    """Generate a valid obstacle configuration."""
    # Obstacle center position
    x = draw(st.floats(min_value=0.0, max_value=width))
    y = draw(st.floats(min_value=0.0, max_value=height))
    
    # Obstacle dimensions (not too large)
    obstacle_width = draw(st.floats(min_value=0.5, max_value=min(10.0, width * 0.3)))
    obstacle_height = draw(st.floats(min_value=0.5, max_value=min(10.0, height * 0.3)))
    
    return x, y, obstacle_width, obstacle_height


@st.composite
def robot_radius(draw):
    """Generate a valid robot radius."""
    return draw(st.floats(min_value=0.1, max_value=1.0))


@pytest.mark.property
class TestWarehouseMapCollisionProperties:
    """Property-based tests for WarehouseMap collision checking."""
    
    @given(
        dims=warehouse_dimensions(),
        radius=robot_radius()
    )
    @settings(max_examples=100, deadline=60000)
    def test_empty_warehouse_is_collision_free(self, dims, radius):
        """
        Property: In an empty warehouse, all positions within bounds should be collision-free.
        
        **Validates: Requirements 3.1**
        """
        width, height, resolution = dims
        warehouse_map = WarehouseMap(width=width, height=height, resolution=resolution)
        
        # Generate a position well within bounds (accounting for robot radius)
        x = np.random.uniform(radius, width - radius)
        y = np.random.uniform(radius, height - radius)
        
        # Empty warehouse should be collision-free
        assert warehouse_map.is_collision_free(x, y, radius=radius), \
            f"Position ({x:.2f}, {y:.2f}) with radius {radius:.2f} should be collision-free in empty warehouse"
    
    @given(
        dims=warehouse_dimensions(),
        radius=robot_radius()
    )
    @settings(max_examples=100, deadline=60000)
    def test_position_inside_obstacle_collides(self, dims, radius):
        """
        Property: A position inside an obstacle should always be detected as a collision.
        
        **Validates: Requirements 3.1**
        """
        width, height, resolution = dims
        warehouse_map = WarehouseMap(width=width, height=height, resolution=resolution)
        
        # Place a large obstacle in the center (at least 3x3 grid cells to ensure coverage)
        obstacle_width = max(3.0 * resolution, min(5.0, width * 0.3))
        obstacle_height = max(3.0 * resolution, min(5.0, height * 0.3))
        obstacle_x = width / 2
        obstacle_y = height / 2
        
        warehouse_map.set_obstacle(
            x=obstacle_x,
            y=obstacle_y,
            width=obstacle_width,
            height=obstacle_height
        )
        
        # Test position at the center of the obstacle (should definitely be marked as occupied)
        assert not warehouse_map.is_collision_free(obstacle_x, obstacle_y, radius=radius), \
            f"Position at obstacle center ({obstacle_x:.2f}, {obstacle_y:.2f}) with radius {radius:.2f}m " \
            f"should collide (obstacle size: {obstacle_width:.2f}x{obstacle_height:.2f}m, resolution: {resolution:.2f}m)"
    
    @given(
        dims=warehouse_dimensions(),
        radius=robot_radius()
    )
    @settings(max_examples=100, deadline=60000)
    def test_out_of_bounds_positions_collide(self, dims, radius):
        """
        Property: Positions outside warehouse bounds should always be detected as collisions.
        
        **Validates: Requirements 3.1**
        """
        width, height, resolution = dims
        warehouse_map = WarehouseMap(width=width, height=height, resolution=resolution)
        
        # Test positions outside bounds
        out_of_bounds_positions = [
            (-1.0, height / 2),  # Left of bounds
            (width + 1.0, height / 2),  # Right of bounds
            (width / 2, -1.0),  # Below bounds
            (width / 2, height + 1.0),  # Above bounds
        ]
        
        for x, y in out_of_bounds_positions:
            assert not warehouse_map.is_collision_free(x, y, radius=radius), \
                f"Out-of-bounds position ({x:.2f}, {y:.2f}) should collide"
    
    @given(
        dims=warehouse_dimensions(),
        obs=st.data()
    )
    @settings(max_examples=100, deadline=60000)
    def test_collision_free_clearance_property(self, dims, obs):
        """
        Property: If a position is collision-free with radius R, then all positions
        within distance R from any obstacle should be detected as collisions.
        
        This tests the minimum clearance requirement (0.3m as per requirements).
        
        **Validates: Requirements 3.1**
        """
        width, height, resolution = dims
        warehouse_map = WarehouseMap(width=width, height=height, resolution=resolution)
        
        # Place a reasonably sized obstacle (at least 5x5 grid cells)
        obstacle_width = max(5.0 * resolution, 2.0)
        obstacle_height = max(5.0 * resolution, 2.0)
        obstacle_x = width / 2
        obstacle_y = height / 2
        
        warehouse_map.set_obstacle(
            x=obstacle_x,
            y=obstacle_y,
            width=obstacle_width,
            height=obstacle_height
        )
        
        # Required clearance (0.3m as per requirements)
        required_clearance = 0.3
        
        # Test a position very close to the obstacle edge
        # Position just outside the obstacle but within clearance distance
        # Add a small buffer to account for grid discretization
        test_x = obstacle_x + (obstacle_width / 2) + required_clearance * 0.3
        test_y = obstacle_y
        
        # Ensure test position is within bounds
        assume(0 <= test_x <= width and 0 <= test_y <= height)
        
        # With the required clearance radius, this should collide
        result = warehouse_map.is_collision_free(test_x, test_y, radius=required_clearance)
        
        # The position should collide because the robot's radius extends into the obstacle
        assert not result, \
            f"Position ({test_x:.2f}, {test_y:.2f}) with radius {required_clearance:.2f}m " \
            f"should collide with obstacle at ({obstacle_x:.2f}, {obstacle_y:.2f}) " \
            f"(obstacle edge at {obstacle_x + obstacle_width/2:.2f}, resolution: {resolution:.2f}m)"
    
    @given(
        dims=warehouse_dimensions(),
        radius=robot_radius()
    )
    @settings(max_examples=100, deadline=60000)
    def test_larger_radius_more_restrictive(self, dims, radius):
        """
        Property: If a position is collision-free with radius R1, and R2 > R1,
        then the position with radius R2 should be at least as restrictive
        (may collide even if R1 doesn't).
        
        **Validates: Requirements 3.1**
        """
        width, height, resolution = dims
        warehouse_map = WarehouseMap(width=width, height=height, resolution=resolution)
        
        # Place an obstacle
        obstacle_x = width / 2
        obstacle_y = height / 2
        obstacle_width = 2.0
        obstacle_height = 2.0
        
        warehouse_map.set_obstacle(
            x=obstacle_x,
            y=obstacle_y,
            width=obstacle_width,
            height=obstacle_height
        )
        
        # Test position near the obstacle
        test_x = obstacle_x + obstacle_width / 2 + radius + 0.5
        test_y = obstacle_y
        
        # Ensure test position is within bounds
        assume(0 <= test_x <= width and 0 <= test_y <= height)
        
        # Check with smaller radius
        smaller_radius = radius * 0.5
        result_small = warehouse_map.is_collision_free(test_x, test_y, radius=smaller_radius)
        
        # Check with larger radius
        larger_radius = radius * 1.5
        result_large = warehouse_map.is_collision_free(test_x, test_y, radius=larger_radius)
        
        # If smaller radius is collision-free, larger radius might not be
        # If smaller radius collides, larger radius must also collide
        if not result_small:
            assert not result_large, \
                f"If position collides with radius {smaller_radius:.2f}m, " \
                f"it must also collide with larger radius {larger_radius:.2f}m"
    
    @given(
        dims=warehouse_dimensions(),
        radius=robot_radius(),
        obs=st.data()
    )
    @settings(max_examples=100, deadline=60000)
    def test_collision_checking_is_deterministic(self, dims, radius, obs):
        """
        Property: Collision checking should be deterministic - calling is_collision_free
        multiple times with the same parameters should return the same result.
        
        **Validates: Requirements 3.1**
        """
        width, height, resolution = dims
        warehouse_map = WarehouseMap(width=width, height=height, resolution=resolution)
        
        # Randomly add some obstacles
        num_obstacles = obs.draw(st.integers(min_value=0, max_value=5))
        for _ in range(num_obstacles):
            obs_config = obs.draw(obstacle_config(width, height))
            warehouse_map.set_obstacle(*obs_config)
        
        # Generate a test position
        test_x = obs.draw(st.floats(min_value=0.0, max_value=width))
        test_y = obs.draw(st.floats(min_value=0.0, max_value=height))
        
        # Check collision multiple times
        results = [
            warehouse_map.is_collision_free(test_x, test_y, radius=radius)
            for _ in range(5)
        ]
        
        # All results should be identical
        assert all(r == results[0] for r in results), \
            f"Collision checking should be deterministic for position ({test_x:.2f}, {test_y:.2f})"
    
    @given(
        dims=warehouse_dimensions(),
        radius=robot_radius()
    )
    @settings(max_examples=100, deadline=60000)
    def test_symmetric_collision_checking(self, dims, radius):
        """
        Property: Collision checking should be symmetric - if position A is collision-free,
        then checking from nearby positions should show consistent results based on distance.
        
        **Validates: Requirements 3.1**
        """
        width, height, resolution = dims
        warehouse_map = WarehouseMap(width=width, height=height, resolution=resolution)
        
        # Place a centered obstacle (large enough to avoid boundary effects)
        obstacle_size = max(5.0 * resolution, 2.0)
        obstacle_x = width / 2
        obstacle_y = height / 2
        
        warehouse_map.set_obstacle(
            x=obstacle_x,
            y=obstacle_y,
            width=obstacle_size,
            height=obstacle_size
        )
        
        # Test positions at the same distance from obstacle center in different directions
        # Use a distance that's clearly outside the obstacle
        distance = obstacle_size / 2 + radius + 1.0
        
        positions = [
            (obstacle_x + distance, obstacle_y),  # Right
            (obstacle_x - distance, obstacle_y),  # Left
            (obstacle_x, obstacle_y + distance),  # Up
            (obstacle_x, obstacle_y - distance),  # Down
        ]
        
        # Filter positions within bounds (with margin for robot radius)
        valid_positions = [
            (x, y) for x, y in positions
            if radius <= x <= width - radius and radius <= y <= height - radius
        ]
        
        assume(len(valid_positions) >= 2)
        
        # Check collision for all valid positions
        results = [
            warehouse_map.is_collision_free(x, y, radius=radius)
            for x, y in valid_positions
        ]
        
        # Due to symmetry, all positions at the same distance should have the same result
        # (assuming square obstacle and positions along cardinal directions)
        assert all(r == results[0] for r in results), \
            f"Symmetric positions at distance {distance:.2f}m from obstacle should have consistent collision results. " \
            f"Results: {results}, Positions: {valid_positions}, Radius: {radius:.2f}m"
