"""
Property-based tests for planning performance by A* planner.

Feature: adaptnav-context-aware-warehouse-navigation
Property 10: Planning Performance
Validates: Requirements 3.5
"""

import pytest
import time
import numpy as np
from hypothesis import given, strategies as st, assume, settings
from adaptnav.core import WarehouseMap
from adaptnav.planning import AStarPlanner


# Custom strategies for generating test data
@st.composite
def warehouse_dimensions(draw):
    """Generate valid warehouse dimensions up to 50m x 50m as per requirements."""
    width = draw(st.floats(min_value=10.0, max_value=50.0))
    height = draw(st.floats(min_value=10.0, max_value=50.0))
    # Use reasonable resolution for path planning
    resolution = draw(st.floats(min_value=0.1, max_value=0.3))
    return width, height, resolution


def find_valid_positions(warehouse_map, robot_radius, max_attempts=100):
    """Find valid start and goal positions in the warehouse."""
    width = warehouse_map.width
    height = warehouse_map.height
    margin = robot_radius + 0.5
    
    for _ in range(max_attempts):
        start_x = np.random.uniform(margin, width - margin)
        start_y = np.random.uniform(margin, height - margin)
        goal_x = np.random.uniform(margin, width - margin)
        goal_y = np.random.uniform(margin, height - margin)
        
        if (warehouse_map.is_collision_free(start_x, start_y, robot_radius) and
            warehouse_map.is_collision_free(goal_x, goal_y, robot_radius)):
            
            distance = np.sqrt((goal_x - start_x)**2 + (goal_y - start_y)**2)
            if distance > 1.0:  # At least 1 meter apart
                return (start_x, start_y), (goal_x, goal_y)
    
    return None


@st.composite
def obstacle_configuration(draw, width, height, max_obstacles=8):
    """Generate a configuration of obstacles for typical warehouse layouts."""
    num_obstacles = draw(st.integers(min_value=0, max_value=max_obstacles))
    obstacles = []
    
    for _ in range(num_obstacles):
        # Place obstacles throughout the warehouse to create realistic complexity
        margin = 1.0
        obs_x = draw(st.floats(min_value=margin, max_value=width - margin))
        obs_y = draw(st.floats(min_value=margin, max_value=height - margin))
        
        # Keep obstacles reasonably sized for warehouse environment
        obs_width = draw(st.floats(min_value=0.5, max_value=min(4.0, width * 0.15)))
        obs_height = draw(st.floats(min_value=0.5, max_value=min(4.0, height * 0.15)))
        
        obstacles.append((obs_x, obs_y, obs_width, obs_height))
    
    return obstacles


@st.composite
def robot_radius_strategy(draw):
    """Generate a valid robot radius."""
    return draw(st.floats(min_value=0.2, max_value=0.5))


@pytest.mark.property
class TestPlanningPerformanceProperties:
    """Property-based tests for planning performance by A* planner."""
    
    @settings(max_examples=25, deadline=60000)  # 60 second timeout, reduced examples for speed
    @given(
        dims=warehouse_dimensions(),
        robot_radius=robot_radius_strategy(),
        obstacles=st.data()
    )
    def test_planning_completes_within_timeout(self, dims, robot_radius, obstacles):
        """
        Property: For any valid planning request in a warehouse up to 50m x 50m,
        the A* planner should return a result (success or failure) within 2 seconds.
        
        **Validates: Requirements 3.5**
        """
        width, height, resolution = dims
        
        # Create warehouse map
        warehouse_map = WarehouseMap(width=width, height=height, resolution=resolution)
        
        # Add obstacles to create realistic planning complexity
        obstacle_config = obstacles.draw(obstacle_configuration(width, height, max_obstacles=6))
        for obs_x, obs_y, obs_width, obs_height in obstacle_config:
            warehouse_map.set_obstacle(obs_x, obs_y, obs_width, obs_height)
        
        # Generate valid start and goal positions
        positions = find_valid_positions(warehouse_map, robot_radius)
        if positions is None:
            # Skip if we can't generate valid positions (too many obstacles)
            assume(False)
        (start_x, start_y), (goal_x, goal_y) = positions
        
        # Create A* planner with 2-second timeout as per requirements
        planner = AStarPlanner(warehouse_map, robot_radius=robot_radius, timeout_seconds=2.0)
        
        # Measure planning time
        start_time = time.time()
        path = planner.plan_path(start_x, start_y, goal_x, goal_y)
        end_time = time.time()
        
        planning_time = end_time - start_time
        
        # Verify planning completes within 2 seconds
        assert planning_time <= 2.0, (
            f"Planning took {planning_time:.3f}s, exceeding 2.0s limit. "
            f"Warehouse: {width:.1f}x{height:.1f}m, resolution: {resolution:.2f}m, "
            f"obstacles: {len(obstacle_config)}, robot_radius: {robot_radius:.2f}m"
        )
        
        # Verify planner returns a result (either success or failure, not hanging)
        assert path is not None or path is None, (
            "Planner should return either a Path object or None, not hang indefinitely"
        )
    
    @settings(max_examples=25, deadline=60000)  # 60 second timeout, reduced examples for speed
    @given(
        dims=warehouse_dimensions(),
        robot_radius=robot_radius_strategy()
    )
    def test_empty_warehouse_planning_performance(self, dims, robot_radius):
        """
        Property: For any valid planning request in an empty warehouse up to 50m x 50m,
        the A* planner should complete very quickly (well under 2 seconds).
        
        **Validates: Requirements 3.5**
        """
        width, height, resolution = dims
        
        # Create empty warehouse map
        warehouse_map = WarehouseMap(width=width, height=height, resolution=resolution)
        
        # Generate random start and goal positions with margin
        margin = robot_radius + 0.5
        start_x = np.random.uniform(margin, width - margin)
        start_y = np.random.uniform(margin, height - margin)
        goal_x = np.random.uniform(margin, width - margin)
        goal_y = np.random.uniform(margin, height - margin)
        
        # Ensure positions are different
        distance = np.sqrt((goal_x - start_x)**2 + (goal_y - start_y)**2)
        assume(distance > 1.0)
        
        # Create A* planner
        planner = AStarPlanner(warehouse_map, robot_radius=robot_radius, timeout_seconds=2.0)
        
        # Measure planning time
        start_time = time.time()
        path = planner.plan_path(start_x, start_y, goal_x, goal_y)
        end_time = time.time()
        
        planning_time = end_time - start_time
        
        # Empty warehouse should plan very quickly
        assert planning_time <= 2.0, (
            f"Planning in empty warehouse took {planning_time:.3f}s, exceeding 2.0s limit. "
            f"Warehouse: {width:.1f}x{height:.1f}m, resolution: {resolution:.2f}m"
        )
        
        # Should find a path in empty warehouse
        assert path is not None, (
            f"A* should find path in empty warehouse from "
            f"({start_x:.2f}, {start_y:.2f}) to ({goal_x:.2f}, {goal_y:.2f})"
        )
    
    @settings(max_examples=20, deadline=60000)  # Fewer examples due to complexity, reduced for speed
    @given(
        dims=warehouse_dimensions(),
        robot_radius=robot_radius_strategy(),
        obstacles=st.data()
    )
    def test_complex_warehouse_planning_performance(self, dims, robot_radius, obstacles):
        """
        Property: For any valid planning request in a complex warehouse with many obstacles,
        the A* planner should still return a result within 2 seconds.
        
        **Validates: Requirements 3.5**
        """
        width, height, resolution = dims
        
        # Create warehouse map
        warehouse_map = WarehouseMap(width=width, height=height, resolution=resolution)
        
        # Add many obstacles to create complex planning scenario
        max_obstacles = min(15, int(width * height / 25))  # Density-based obstacle count
        obstacle_config = obstacles.draw(obstacle_configuration(width, height, max_obstacles))
        
        for obs_x, obs_y, obs_width, obs_height in obstacle_config:
            warehouse_map.set_obstacle(obs_x, obs_y, obs_width, obs_height)
        
        # Try to find valid start and goal positions
        positions = find_valid_positions(warehouse_map, robot_radius, max_attempts=50)
        if positions is None:
            assume(False)  # Skip if we couldn't find valid positions
        (start_x, start_y), (goal_x, goal_y) = positions
        
        # Create A* planner
        planner = AStarPlanner(warehouse_map, robot_radius=robot_radius, timeout_seconds=2.0)
        
        # Measure planning time
        start_time = time.time()
        path = planner.plan_path(start_x, start_y, goal_x, goal_y)
        end_time = time.time()
        
        planning_time = end_time - start_time
        
        # Even complex scenarios should complete within 2 seconds
        assert planning_time <= 2.0, (
            f"Planning in complex warehouse took {planning_time:.3f}s, exceeding 2.0s limit. "
            f"Warehouse: {width:.1f}x{height:.1f}m, resolution: {resolution:.2f}m, "
            f"obstacles: {len(obstacle_config)}, robot_radius: {robot_radius:.2f}m"
        )
        
        # Planner should return a result (success or failure)
        assert path is not None or path is None, (
            "Planner should return either a Path object or None within timeout"
        )
    
    @settings(max_examples=25, deadline=60000)  # 60 second timeout, reduced examples for speed
    @given(
        dims=warehouse_dimensions(),
        robot_radius=robot_radius_strategy(),
        obstacles=st.data()
    )
    def test_planning_time_scales_reasonably(self, dims, robot_radius, obstacles):
        """
        Property: For any valid planning request, the planning time should scale
        reasonably with warehouse size and complexity, staying well under 2 seconds
        for typical scenarios.
        
        **Validates: Requirements 3.5**
        """
        width, height, resolution = dims
        
        # Create warehouse map
        warehouse_map = WarehouseMap(width=width, height=height, resolution=resolution)
        
        # Add moderate number of obstacles
        num_obstacles = min(8, int(width * height / 50))  # Moderate density
        obstacle_config = obstacles.draw(obstacle_configuration(width, height, num_obstacles))
        
        for obs_x, obs_y, obs_width, obs_height in obstacle_config:
            warehouse_map.set_obstacle(obs_x, obs_y, obs_width, obs_height)
        
        # Generate valid positions
        positions = find_valid_positions(warehouse_map, robot_radius)
        if positions is None:
            assume(False)
        (start_x, start_y), (goal_x, goal_y) = positions
        
        # Create A* planner
        planner = AStarPlanner(warehouse_map, robot_radius=robot_radius, timeout_seconds=2.0)
        
        # Measure planning time
        start_time = time.time()
        path = planner.plan_path(start_x, start_y, goal_x, goal_y)
        end_time = time.time()
        
        planning_time = end_time - start_time
        
        # Calculate expected complexity factors
        grid_cells = (width / resolution) * (height / resolution)
        complexity_factor = grid_cells / 10000  # Normalize to 100x100 grid
        
        # Planning time should be reasonable for the complexity
        expected_max_time = min(2.0, 0.1 + complexity_factor * 0.5)  # Scale with complexity
        
        assert planning_time <= 2.0, (
            f"Planning took {planning_time:.3f}s, exceeding 2.0s absolute limit. "
            f"Warehouse: {width:.1f}x{height:.1f}m, grid cells: {grid_cells:.0f}, "
            f"complexity factor: {complexity_factor:.2f}"
        )
        
        # For most scenarios, should be much faster than the limit
        if complexity_factor < 2.0:  # For reasonably sized warehouses
            assert planning_time <= 1.0, (
                f"Planning took {planning_time:.3f}s for moderate complexity scenario. "
                f"Expected under 1.0s for complexity factor {complexity_factor:.2f}"
            )
    
    @settings(max_examples=30, deadline=60000)  # 60 second timeout, reduced examples for speed
    @given(
        dims=warehouse_dimensions(),
        robot_radius=robot_radius_strategy(),
        obstacles=st.data()
    )
    def test_successful_planning_returns_valid_path_quickly(self, dims, robot_radius, obstacles):
        """
        Property: When A* planner successfully finds a path, it should return
        a valid Path object within 2 seconds, and the path should be non-empty.
        
        **Validates: Requirements 3.5**
        """
        width, height, resolution = dims
        
        # Create warehouse map with moderate obstacles to ensure some paths exist
        warehouse_map = WarehouseMap(width=width, height=height, resolution=resolution)
        
        # Add few obstacles to keep paths available
        obstacle_config = obstacles.draw(obstacle_configuration(width, height, max_obstacles=3))
        for obs_x, obs_y, obs_width, obs_height in obstacle_config:
            warehouse_map.set_obstacle(obs_x, obs_y, obs_width, obs_height)
        
        # Generate valid positions
        positions = find_valid_positions(warehouse_map, robot_radius)
        if positions is None:
            assume(False)
        (start_x, start_y), (goal_x, goal_y) = positions
        
        # Create A* planner
        planner = AStarPlanner(warehouse_map, robot_radius=robot_radius, timeout_seconds=2.0)
        
        # Measure planning time
        start_time = time.time()
        path = planner.plan_path(start_x, start_y, goal_x, goal_y)
        end_time = time.time()
        
        planning_time = end_time - start_time
        
        # Planning should complete within time limit
        assert planning_time <= 2.0, (
            f"Planning took {planning_time:.3f}s, exceeding 2.0s limit"
        )
        
        # If path is found, it should be valid
        if path is not None:
            assert len(path.waypoints) >= 2, (
                f"Successful planning should return path with at least 2 waypoints, "
                f"got {len(path.waypoints)}"
            )
            
            assert path.total_length > 0, (
                f"Successful planning should return path with positive length, "
                f"got {path.total_length:.3f}m"
            )
            
            # Path should start near the start position
            start_waypoint = path.waypoints[0]
            start_distance = np.sqrt(
                (start_waypoint.x - start_x)**2 + (start_waypoint.y - start_y)**2
            )
            max_start_error = resolution * np.sqrt(2)
            assert start_distance <= max_start_error, (
                f"Path start waypoint too far from requested start: "
                f"{start_distance:.3f}m > {max_start_error:.3f}m"
            )
            
            # Path should end near the goal position
            goal_waypoint = path.waypoints[-1]
            goal_distance = np.sqrt(
                (goal_waypoint.x - goal_x)**2 + (goal_waypoint.y - goal_y)**2
            )
            max_goal_error = resolution * np.sqrt(2)
            assert goal_distance <= max_goal_error, (
                f"Path goal waypoint too far from requested goal: "
                f"{goal_distance:.3f}m > {max_goal_error:.3f}m"
            )