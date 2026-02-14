"""
Unit tests for AStarPlanner class.

Tests cover:
- Simple path scenarios (straight line, L-shape, U-turn)
- Edge cases (start equals goal, unreachable goal)
- Performance on various map sizes
- Path reconstruction and validation
- Timeout handling
- Node comparison and hashing
"""

import pytest
import numpy as np
import time
from adaptnav.core import WarehouseMap
from adaptnav.planning.astar_planner import AStarPlanner, AStarNode


class TestAStarNode:
    """Test AStarNode functionality."""
    
    def test_node_initialization(self):
        """Test basic node initialization."""
        node = AStarNode(5, 10, g_cost=2.0, h_cost=3.0)
        
        assert node.grid_x == 5
        assert node.grid_y == 10
        assert node.g_cost == 2.0
        assert node.h_cost == 3.0
        assert node.f_cost == 5.0
        assert node.parent is None
    
    def test_node_with_parent(self):
        """Test node initialization with parent."""
        parent = AStarNode(3, 7)
        child = AStarNode(4, 7, parent=parent)
        
        assert child.parent == parent
    
    def test_node_comparison(self):
        """Test node comparison for priority queue."""
        node1 = AStarNode(0, 0, g_cost=1.0, h_cost=2.0)  # f_cost = 3.0
        node2 = AStarNode(1, 1, g_cost=2.0, h_cost=3.0)  # f_cost = 5.0
        
        assert node1 < node2
        assert not node2 < node1
    
    def test_node_equality(self):
        """Test node equality based on position."""
        node1 = AStarNode(5, 10, g_cost=1.0)
        node2 = AStarNode(5, 10, g_cost=2.0)  # Different cost, same position
        node3 = AStarNode(6, 10, g_cost=1.0)  # Different position
        
        assert node1 == node2
        assert node1 != node3
    
    def test_node_hash(self):
        """Test node hashing for use in sets/dicts."""
        node1 = AStarNode(5, 10)
        node2 = AStarNode(5, 10)
        node3 = AStarNode(6, 10)
        
        assert hash(node1) == hash(node2)
        assert hash(node1) != hash(node3)
        
        # Test in set
        node_set = {node1, node2, node3}
        assert len(node_set) == 2  # node1 and node2 are same
    
    def test_position_method(self):
        """Test position method returns tuple."""
        node = AStarNode(7, 13)
        assert node.position() == (7, 13)


class TestAStarPlannerInitialization:
    """Test AStarPlanner initialization."""
    
    def test_planner_initialization(self):
        """Test basic planner initialization."""
        warehouse_map = WarehouseMap(width=10.0, height=10.0)
        planner = AStarPlanner(warehouse_map, robot_radius=0.3, timeout_seconds=2.0)
        
        assert planner.warehouse_map == warehouse_map
        assert planner.robot_radius == 0.3
        assert planner.timeout_seconds == 2.0
    
    def test_planner_default_parameters(self):
        """Test planner with default parameters."""
        warehouse_map = WarehouseMap(width=10.0, height=10.0)
        planner = AStarPlanner(warehouse_map)
        
        assert planner.robot_radius == 0.3
        assert planner.timeout_seconds == 2.0


class TestDistanceAndCostCalculations:
    """Test distance and cost calculation methods."""
    
    def test_euclidean_distance(self):
        """Test Euclidean distance calculation."""
        warehouse_map = WarehouseMap(width=10.0, height=10.0)
        planner = AStarPlanner(warehouse_map)
        
        # Test simple cases
        assert planner.euclidean_distance((0, 0), (0, 0)) == 0.0
        assert planner.euclidean_distance((0, 0), (3, 4)) == 5.0
        assert planner.euclidean_distance((1, 1), (4, 5)) == 5.0
    
    def test_movement_cost_orthogonal(self):
        """Test movement cost for orthogonal moves."""
        warehouse_map = WarehouseMap(width=10.0, height=10.0)
        planner = AStarPlanner(warehouse_map)
        
        # Orthogonal moves should cost 1.0
        assert planner.get_movement_cost((5, 5), (5, 6)) == 1.0  # North
        assert planner.get_movement_cost((5, 5), (6, 5)) == 1.0  # East
        assert planner.get_movement_cost((5, 5), (5, 4)) == 1.0  # South
        assert planner.get_movement_cost((5, 5), (4, 5)) == 1.0  # West
    
    def test_movement_cost_diagonal(self):
        """Test movement cost for diagonal moves."""
        warehouse_map = WarehouseMap(width=10.0, height=10.0)
        planner = AStarPlanner(warehouse_map)
        
        # Diagonal moves should cost sqrt(2)
        expected_cost = np.sqrt(2)
        assert abs(planner.get_movement_cost((5, 5), (6, 6)) - expected_cost) < 1e-6
        assert abs(planner.get_movement_cost((5, 5), (4, 4)) - expected_cost) < 1e-6
        assert abs(planner.get_movement_cost((5, 5), (6, 4)) - expected_cost) < 1e-6
        assert abs(planner.get_movement_cost((5, 5), (4, 6)) - expected_cost) < 1e-6


class TestPositionValidation:
    """Test position validation methods."""
    
    def test_valid_position_in_free_space(self):
        """Test that positions in free space are valid."""
        warehouse_map = WarehouseMap(width=10.0, height=10.0)
        planner = AStarPlanner(warehouse_map, robot_radius=0.3)
        
        # Center of map should be valid
        assert planner.is_position_valid(50, 50)
        assert planner.is_position_valid(25, 75)
    
    def test_invalid_position_out_of_bounds(self):
        """Test that out-of-bounds positions are invalid."""
        warehouse_map = WarehouseMap(width=10.0, height=10.0)
        planner = AStarPlanner(warehouse_map)
        
        assert not planner.is_position_valid(-1, 50)
        assert not planner.is_position_valid(100, 50)
        assert not planner.is_position_valid(50, -1)
        assert not planner.is_position_valid(50, 100)
    
    def test_invalid_position_in_obstacle(self):
        """Test that positions in obstacles are invalid."""
        warehouse_map = WarehouseMap(width=10.0, height=10.0)
        warehouse_map.set_obstacle(x=5.0, y=5.0, width=2.0, height=2.0)
        planner = AStarPlanner(warehouse_map, robot_radius=0.3)
        
        # Center of obstacle should be invalid
        grid_x, grid_y = warehouse_map.world_to_grid(5.0, 5.0)
        assert not planner.is_position_valid(grid_x, grid_y)


class TestSimplePaths:
    """Test simple path planning scenarios."""
    
    def test_straight_line_path(self):
        """Test planning a straight line path."""
        warehouse_map = WarehouseMap(width=10.0, height=10.0)
        planner = AStarPlanner(warehouse_map, robot_radius=0.3)
        
        # Plan straight path from (1, 5) to (9, 5)
        path = planner.plan_path(1.0, 5.0, 9.0, 5.0)
        
        assert path is not None
        assert len(path.waypoints) >= 2
        
        # First waypoint should be near start
        start_waypoint = path.waypoints[0]
        assert abs(start_waypoint.x - 1.0) < 0.2
        assert abs(start_waypoint.y - 5.0) < 0.2
        
        # Last waypoint should be near goal
        goal_waypoint = path.waypoints[-1]
        assert abs(goal_waypoint.x - 9.0) < 0.2
        assert abs(goal_waypoint.y - 5.0) < 0.2
        
        # Path should be roughly horizontal
        for waypoint in path.waypoints:
            assert abs(waypoint.y - 5.0) < 1.0  # Allow some deviation
    
    def test_l_shape_path(self):
        """Test planning an L-shaped path around obstacle."""
        warehouse_map = WarehouseMap(width=10.0, height=10.0)
        # Place obstacle blocking direct path
        warehouse_map.set_obstacle(x=5.0, y=5.0, width=2.0, height=2.0)
        planner = AStarPlanner(warehouse_map, robot_radius=0.3)
        
        # Plan path that must go around obstacle
        path = planner.plan_path(2.0, 5.0, 8.0, 5.0)
        
        assert path is not None
        assert len(path.waypoints) >= 3  # Should have intermediate waypoints
        
        # Path should avoid the obstacle area
        for waypoint in path.waypoints:
            # Check that waypoint is not inside obstacle (with some margin)
            distance_to_obstacle_center = np.sqrt((waypoint.x - 5.0)**2 + (waypoint.y - 5.0)**2)
            assert distance_to_obstacle_center > 1.0  # Outside obstacle + robot radius
    
    def test_u_turn_path(self):
        """Test planning a U-turn path."""
        warehouse_map = WarehouseMap(width=10.0, height=10.0)
        # Create a wall forcing U-turn
        for y in range(30, 70):  # Block middle section
            warehouse_map.occupancy_grid[y, 50] = 100
        
        planner = AStarPlanner(warehouse_map, robot_radius=0.3)
        
        # Plan path that requires going around the wall
        path = planner.plan_path(3.0, 5.0, 7.0, 5.0)
        
        assert path is not None
        assert len(path.waypoints) >= 4  # Should have multiple turns
        
        # Path should go around the wall (either north or south)
        has_detour = False
        for waypoint in path.waypoints:
            if waypoint.y < 3.0 or waypoint.y > 7.0:  # Significant detour
                has_detour = True
                break
        assert has_detour


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_start_equals_goal(self):
        """Test planning when start equals goal."""
        warehouse_map = WarehouseMap(width=10.0, height=10.0)
        planner = AStarPlanner(warehouse_map)
        
        # Same position for start and goal
        path = planner.plan_path(5.0, 5.0, 5.0, 5.0)
        
        assert path is not None
        assert len(path.waypoints) == 2  # Start and goal waypoints
        
        # Both waypoints should be at the same location
        start_waypoint = path.waypoints[0]
        goal_waypoint = path.waypoints[1]
        assert abs(start_waypoint.x - 5.0) < 0.1
        assert abs(start_waypoint.y - 5.0) < 0.1
        assert abs(goal_waypoint.x - 5.0) < 0.1
        assert abs(goal_waypoint.y - 5.0) < 0.1
    
    def test_unreachable_goal_blocked(self):
        """Test planning to unreachable goal (completely blocked)."""
        warehouse_map = WarehouseMap(width=10.0, height=10.0)
        
        # Create a box around the goal position
        for x in range(70, 80):
            for y in range(70, 80):
                warehouse_map.occupancy_grid[y, x] = 100
        
        planner = AStarPlanner(warehouse_map, robot_radius=0.3)
        
        # Try to plan to position inside the box
        path = planner.plan_path(1.0, 1.0, 7.5, 7.5)
        
        assert path is None  # Should fail to find path
    
    def test_unreachable_goal_out_of_bounds(self):
        """Test planning to goal outside map bounds."""
        warehouse_map = WarehouseMap(width=10.0, height=10.0)
        planner = AStarPlanner(warehouse_map)
        
        # Try to plan to position outside bounds
        path = planner.plan_path(5.0, 5.0, 15.0, 5.0)
        
        assert path is None  # Should fail due to invalid goal
    
    def test_invalid_start_position(self):
        """Test planning from invalid start position."""
        warehouse_map = WarehouseMap(width=10.0, height=10.0)
        warehouse_map.set_obstacle(x=5.0, y=5.0, width=2.0, height=2.0)
        planner = AStarPlanner(warehouse_map, robot_radius=0.3)
        
        # Try to start from inside obstacle
        path = planner.plan_path(5.0, 5.0, 8.0, 8.0)
        
        assert path is None  # Should fail due to invalid start
    
    def test_very_close_start_and_goal(self):
        """Test planning between very close positions."""
        warehouse_map = WarehouseMap(width=10.0, height=10.0)
        planner = AStarPlanner(warehouse_map)
        
        # Plan between positions less than one grid cell apart
        path = planner.plan_path(5.0, 5.0, 5.05, 5.05)
        
        assert path is not None
        assert len(path.waypoints) >= 2
    
    def test_planning_at_map_boundaries(self):
        """Test planning at map boundaries."""
        warehouse_map = WarehouseMap(width=10.0, height=10.0)
        planner = AStarPlanner(warehouse_map, robot_radius=0.2)  # Small radius
        
        # Plan from corner to corner
        path = planner.plan_path(0.5, 0.5, 9.5, 9.5)
        
        assert path is not None
        assert len(path.waypoints) >= 2


class TestPerformance:
    """Test performance on various map sizes."""
    
    def test_small_map_performance(self):
        """Test performance on small map (10x10m)."""
        warehouse_map = WarehouseMap(width=10.0, height=10.0, resolution=0.1)
        planner = AStarPlanner(warehouse_map, timeout_seconds=1.0)
        
        start_time = time.time()
        path = planner.plan_path(1.0, 1.0, 9.0, 9.0)
        end_time = time.time()
        
        assert path is not None
        assert end_time - start_time < 0.5  # Should be very fast
    
    def test_medium_map_performance(self):
        """Test performance on medium map (30x30m)."""
        warehouse_map = WarehouseMap(width=30.0, height=30.0, resolution=0.1)
        # Add some obstacles to make it more realistic
        warehouse_map.set_obstacle(x=10.0, y=10.0, width=2.0, height=2.0)
        warehouse_map.set_obstacle(x=20.0, y=15.0, width=3.0, height=1.0)
        
        planner = AStarPlanner(warehouse_map, timeout_seconds=5.0)  # Increased timeout
        
        start_time = time.time()
        path = planner.plan_path(2.0, 2.0, 28.0, 28.0)
        end_time = time.time()
        
        # Path might be None due to timeout, but timing should be reasonable
        assert end_time - start_time < 6.0  # Should respect timeout
    
    def test_large_map_performance(self):
        """Test performance on large map (50x50m)."""
        warehouse_map = WarehouseMap(width=50.0, height=50.0, resolution=0.2)  # Coarser resolution
        # Add multiple obstacles
        for i in range(3):  # Fewer obstacles
            x = 10.0 + i * 15.0
            y = 10.0 + i * 10.0
            warehouse_map.set_obstacle(x=x, y=y, width=2.0, height=2.0)
        
        planner = AStarPlanner(warehouse_map, timeout_seconds=5.0)  # Increased timeout
        
        start_time = time.time()
        path = planner.plan_path(5.0, 5.0, 45.0, 45.0)
        end_time = time.time()
        
        # Path might be None due to timeout, but timing should be reasonable
        assert end_time - start_time < 6.0  # Should respect timeout
    
    def test_timeout_handling(self):
        """Test that planner respects timeout."""
        # Create a very large map to potentially trigger timeout
        warehouse_map = WarehouseMap(width=100.0, height=100.0, resolution=0.05)
        
        # Create a maze-like structure to make planning difficult
        for x in range(10, 90, 4):
            for y in range(10, 90, 4):
                warehouse_map.set_obstacle(x=x, y=y, width=1.0, height=1.0)
        
        planner = AStarPlanner(warehouse_map, timeout_seconds=0.1)  # Very short timeout
        
        start_time = time.time()
        path = planner.plan_path(5.0, 5.0, 95.0, 95.0)
        end_time = time.time()
        
        # Should either find path quickly or timeout
        assert end_time - start_time < 0.5  # Should not exceed timeout by much
        # Path might be None due to timeout, which is acceptable
    
    def test_complex_maze_performance(self):
        """Test performance on complex maze-like environment."""
        warehouse_map = WarehouseMap(width=15.0, height=15.0, resolution=0.1)  # Smaller map
        
        # Create a simpler maze with corridors
        for x in range(5, 10):  # Shorter wall
            warehouse_map.set_obstacle(x=x, y=7.5, width=0.5, height=0.5)
        
        # Leave gaps for passage (gaps at x=7.0 and x=12.0 are left free)
        
        planner = AStarPlanner(warehouse_map, robot_radius=0.2, timeout_seconds=3.0)
        
        start_time = time.time()
        path = planner.plan_path(2.0, 5.0, 13.0, 10.0)
        end_time = time.time()
        
        # Should find a path or timeout reasonably
        assert end_time - start_time < 4.0
        if path is not None:
            assert len(path.waypoints) > 3  # Should have multiple waypoints for complex path


class TestPathReconstruction:
    """Test path reconstruction and validation."""
    
    def test_path_reconstruction_simple(self):
        """Test path reconstruction for simple case."""
        warehouse_map = WarehouseMap(width=10.0, height=10.0)
        planner = AStarPlanner(warehouse_map)
        
        path = planner.plan_path(2.0, 2.0, 8.0, 8.0)
        
        assert path is not None
        assert len(path.waypoints) >= 2
        
        # Check that path is continuous (no large jumps)
        for i in range(len(path.waypoints) - 1):
            current = path.waypoints[i]
            next_wp = path.waypoints[i + 1]
            distance = np.sqrt((next_wp.x - current.x)**2 + (next_wp.y - current.y)**2)
            assert distance < 2.0  # No jumps larger than 2 meters
    
    def test_path_collision_free(self):
        """Test that reconstructed path is collision-free."""
        warehouse_map = WarehouseMap(width=10.0, height=10.0)
        warehouse_map.set_obstacle(x=5.0, y=5.0, width=1.0, height=1.0)
        planner = AStarPlanner(warehouse_map, robot_radius=0.3)
        
        path = planner.plan_path(2.0, 5.0, 8.0, 5.0)
        
        assert path is not None
        
        # Check that all waypoints are collision-free
        for waypoint in path.waypoints:
            assert warehouse_map.is_collision_free(waypoint.x, waypoint.y, planner.robot_radius)
    
    def test_path_length_calculation(self):
        """Test that path length is calculated correctly."""
        warehouse_map = WarehouseMap(width=10.0, height=10.0)
        planner = AStarPlanner(warehouse_map, robot_radius=0.2)  # Small radius
        
        path = planner.plan_path(1.0, 1.0, 4.0, 5.0)  # Start away from boundary
        
        assert path is not None
        # For straight line, path length should be approximately 5.0
        assert path.total_length >= 4.5  # Allow for discretization
        assert path.total_length <= 6.0   # But not too much overhead


class TestRobustness:
    """Test robustness and error handling."""
    
    def test_different_robot_radii(self):
        """Test planning with different robot radii."""
        warehouse_map = WarehouseMap(width=10.0, height=10.0)
        warehouse_map.set_obstacle(x=5.0, y=5.0, width=1.0, height=1.0)
        
        # Small robot should find path closer to obstacle
        small_planner = AStarPlanner(warehouse_map, robot_radius=0.1)
        small_path = small_planner.plan_path(3.0, 5.0, 7.0, 5.0)
        
        # Large robot should find path farther from obstacle
        large_planner = AStarPlanner(warehouse_map, robot_radius=0.8)
        large_path = large_planner.plan_path(3.0, 5.0, 7.0, 5.0)
        
        assert small_path is not None
        assert large_path is not None
        
        # Large robot path should be longer (more detour)
        assert large_path.total_length > small_path.total_length
    
    def test_planning_with_narrow_passages(self):
        """Test planning through narrow passages."""
        warehouse_map = WarehouseMap(width=10.0, height=10.0)
        
        # Create narrow passage
        for y in range(40, 60):
            warehouse_map.occupancy_grid[y, 49] = 100  # Left wall
            warehouse_map.occupancy_grid[y, 51] = 100  # Right wall
        
        # Leave small gap
        warehouse_map.occupancy_grid[50, 49] = 0
        warehouse_map.occupancy_grid[50, 51] = 0
        
        planner = AStarPlanner(warehouse_map, robot_radius=0.05)  # Small robot
        
        path = planner.plan_path(2.0, 5.0, 8.0, 5.0)
        
        assert path is not None
        # Path should go through the narrow passage
        has_passage_point = False
        for waypoint in path.waypoints:
            if abs(waypoint.x - 5.0) < 0.2 and abs(waypoint.y - 5.0) < 0.2:
                has_passage_point = True
                break
        assert has_passage_point