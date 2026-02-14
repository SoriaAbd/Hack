"""
Unit tests for WarehouseMap class.

Tests cover:
- Map initialization
- Coordinate conversions
- Collision checking
- Neighbor queries
- Obstacle placement
- YAML loading/saving
"""

import pytest
import numpy as np
import tempfile
import os
from adaptnav.core import WarehouseMap


class TestWarehouseMapInitialization:
    """Test WarehouseMap initialization and basic properties."""
    
    def test_basic_initialization(self):
        """Test basic map initialization with default parameters."""
        warehouse_map = WarehouseMap(width=10.0, height=10.0)
        
        assert warehouse_map.width == 10.0
        assert warehouse_map.height == 10.0
        assert warehouse_map.resolution == 0.1
        assert warehouse_map.origin == (0.0, 0.0)
        assert warehouse_map.grid_width == 100
        assert warehouse_map.grid_height == 100
        assert warehouse_map.occupancy_grid.shape == (100, 100)
    
    def test_custom_resolution(self):
        """Test map initialization with custom resolution."""
        warehouse_map = WarehouseMap(width=10.0, height=10.0, resolution=0.5)
        
        assert warehouse_map.resolution == 0.5
        assert warehouse_map.grid_width == 20
        assert warehouse_map.grid_height == 20
    
    def test_custom_origin(self):
        """Test map initialization with custom origin."""
        warehouse_map = WarehouseMap(width=10.0, height=10.0, origin=(-5.0, -5.0))
        
        assert warehouse_map.origin == (-5.0, -5.0)
    
    def test_initial_grid_is_free(self):
        """Test that initial occupancy grid is all free space."""
        warehouse_map = WarehouseMap(width=10.0, height=10.0)
        
        assert np.all(warehouse_map.occupancy_grid == 0)


class TestCoordinateConversions:
    """Test coordinate conversion methods."""
    
    def test_world_to_grid_at_origin(self):
        """Test world to grid conversion at origin."""
        warehouse_map = WarehouseMap(width=10.0, height=10.0, resolution=0.1)
        grid_x, grid_y = warehouse_map.world_to_grid(0.0, 0.0)
        
        assert grid_x == 0
        assert grid_y == 0
    
    def test_world_to_grid_positive(self):
        """Test world to grid conversion with positive coordinates."""
        warehouse_map = WarehouseMap(width=10.0, height=10.0, resolution=0.1)
        grid_x, grid_y = warehouse_map.world_to_grid(1.0, 2.0)
        
        assert grid_x == 10
        assert grid_y == 20
    
    def test_grid_to_world_at_origin(self):
        """Test grid to world conversion at origin (returns cell center)."""
        warehouse_map = WarehouseMap(width=10.0, height=10.0, resolution=0.1)
        x, y = warehouse_map.grid_to_world(0, 0)
        
        # Cell center is at 0.05, 0.05 for resolution 0.1
        assert abs(x - 0.05) < 1e-6
        assert abs(y - 0.05) < 1e-6
    
    def test_grid_to_world_positive(self):
        """Test grid to world conversion with positive coordinates."""
        warehouse_map = WarehouseMap(width=10.0, height=10.0, resolution=0.1)
        x, y = warehouse_map.grid_to_world(10, 20)
        
        assert abs(x - 1.05) < 1e-6
        assert abs(y - 2.05) < 1e-6
    
    def test_round_trip_conversion(self):
        """Test that world -> grid -> world conversion is consistent."""
        warehouse_map = WarehouseMap(width=10.0, height=10.0, resolution=0.1)
        
        # Start with grid coordinates
        original_grid_x, original_grid_y = 50, 75
        
        # Convert to world and back
        world_x, world_y = warehouse_map.grid_to_world(original_grid_x, original_grid_y)
        grid_x, grid_y = warehouse_map.world_to_grid(world_x, world_y)
        
        assert grid_x == original_grid_x
        assert grid_y == original_grid_y


class TestGridValidation:
    """Test grid cell validation."""
    
    def test_valid_cell_inside_bounds(self):
        """Test that cells inside bounds are valid."""
        warehouse_map = WarehouseMap(width=10.0, height=10.0, resolution=0.1)
        
        assert warehouse_map.is_valid_grid_cell(0, 0)
        assert warehouse_map.is_valid_grid_cell(50, 50)
        assert warehouse_map.is_valid_grid_cell(99, 99)
    
    def test_invalid_cell_outside_bounds(self):
        """Test that cells outside bounds are invalid."""
        warehouse_map = WarehouseMap(width=10.0, height=10.0, resolution=0.1)
        
        assert not warehouse_map.is_valid_grid_cell(-1, 0)
        assert not warehouse_map.is_valid_grid_cell(0, -1)
        assert not warehouse_map.is_valid_grid_cell(100, 0)
        assert not warehouse_map.is_valid_grid_cell(0, 100)
        assert not warehouse_map.is_valid_grid_cell(100, 100)


class TestCollisionChecking:
    """Test collision checking functionality."""
    
    def test_collision_free_in_empty_map(self):
        """Test that positions in empty map are collision-free."""
        warehouse_map = WarehouseMap(width=10.0, height=10.0)
        
        assert warehouse_map.is_collision_free(5.0, 5.0, radius=0.3)
        assert warehouse_map.is_collision_free(1.0, 1.0, radius=0.5)
    
    def test_collision_with_obstacle(self):
        """Test that positions inside obstacles are not collision-free."""
        warehouse_map = WarehouseMap(width=10.0, height=10.0)
        warehouse_map.set_obstacle(x=5.0, y=5.0, width=2.0, height=2.0)
        
        # Center of obstacle should collide
        assert not warehouse_map.is_collision_free(5.0, 5.0, radius=0.3)
    
    def test_collision_near_obstacle_edge(self):
        """Test collision checking near obstacle edges."""
        warehouse_map = WarehouseMap(width=10.0, height=10.0)
        warehouse_map.set_obstacle(x=5.0, y=5.0, width=2.0, height=2.0)
        
        # Just outside obstacle should be free (if robot radius is small)
        assert warehouse_map.is_collision_free(7.0, 5.0, radius=0.3)
        
        # Just outside but with large radius should collide
        assert not warehouse_map.is_collision_free(7.0, 5.0, radius=1.5)
    
    def test_collision_out_of_bounds(self):
        """Test that out-of-bounds positions are considered collisions."""
        warehouse_map = WarehouseMap(width=10.0, height=10.0)
        
        assert not warehouse_map.is_collision_free(-1.0, 5.0, radius=0.3)
        assert not warehouse_map.is_collision_free(11.0, 5.0, radius=0.3)
        assert not warehouse_map.is_collision_free(5.0, -1.0, radius=0.3)
        assert not warehouse_map.is_collision_free(5.0, 11.0, radius=0.3)
    
    def test_collision_with_different_radii(self):
        """Test collision checking with different robot radii."""
        warehouse_map = WarehouseMap(width=10.0, height=10.0)
        warehouse_map.set_obstacle(x=5.0, y=5.0, width=1.0, height=1.0)
        
        # Small radius might be free
        position_x, position_y = 6.0, 5.0
        
        # Test with increasing radii
        assert warehouse_map.is_collision_free(position_x, position_y, radius=0.1)
        # Larger radius should eventually collide
        assert not warehouse_map.is_collision_free(position_x, position_y, radius=1.0)


class TestNeighborQueries:
    """Test neighbor query functionality for path planning."""
    
    def test_neighbors_4_connected_center(self):
        """Test 4-connected neighbors in center of map."""
        warehouse_map = WarehouseMap(width=10.0, height=10.0)
        neighbors = warehouse_map.get_neighbors(50, 50, connectivity=4)
        
        assert len(neighbors) == 4
        assert (50, 51) in neighbors  # North
        assert (51, 50) in neighbors  # East
        assert (50, 49) in neighbors  # South
        assert (49, 50) in neighbors  # West
    
    def test_neighbors_8_connected_center(self):
        """Test 8-connected neighbors in center of map."""
        warehouse_map = WarehouseMap(width=10.0, height=10.0)
        neighbors = warehouse_map.get_neighbors(50, 50, connectivity=8)
        
        assert len(neighbors) == 8
        # Check all 8 directions
        expected = [
            (50, 51), (51, 51), (51, 50), (51, 49),
            (50, 49), (49, 49), (49, 50), (49, 51)
        ]
        for neighbor in expected:
            assert neighbor in neighbors
    
    def test_neighbors_at_corner(self):
        """Test neighbors at corner of map (fewer neighbors)."""
        warehouse_map = WarehouseMap(width=10.0, height=10.0)
        neighbors = warehouse_map.get_neighbors(0, 0, connectivity=8)
        
        # At corner, only 3 neighbors are valid
        assert len(neighbors) == 3
        assert (0, 1) in neighbors
        assert (1, 0) in neighbors
        assert (1, 1) in neighbors
    
    def test_neighbors_at_edge(self):
        """Test neighbors at edge of map."""
        warehouse_map = WarehouseMap(width=10.0, height=10.0)
        neighbors = warehouse_map.get_neighbors(0, 50, connectivity=8)
        
        # At edge, only 5 neighbors are valid
        assert len(neighbors) == 5
    
    def test_neighbors_exclude_obstacles(self):
        """Test that occupied neighbors are excluded."""
        warehouse_map = WarehouseMap(width=10.0, height=10.0)
        
        # Place obstacle to the north
        warehouse_map.occupancy_grid[51, 50] = 100
        
        neighbors = warehouse_map.get_neighbors(50, 50, connectivity=4)
        
        # Should have 3 neighbors (north is blocked)
        assert len(neighbors) == 3
        assert (50, 51) not in neighbors  # North is blocked
        assert (51, 50) in neighbors      # East is free
        assert (50, 49) in neighbors      # South is free
        assert (49, 50) in neighbors      # West is free


class TestObstaclePlacement:
    """Test obstacle placement functionality."""
    
    def test_set_single_obstacle(self):
        """Test placing a single rectangular obstacle."""
        warehouse_map = WarehouseMap(width=10.0, height=10.0, resolution=0.1)
        warehouse_map.set_obstacle(x=5.0, y=5.0, width=2.0, height=2.0)
        
        # Check that some cells are now occupied
        assert np.any(warehouse_map.occupancy_grid == 100)
        
        # Check that the center is occupied
        grid_x, grid_y = warehouse_map.world_to_grid(5.0, 5.0)
        assert warehouse_map.occupancy_grid[grid_y, grid_x] == 100
    
    def test_set_multiple_obstacles(self):
        """Test placing multiple obstacles."""
        warehouse_map = WarehouseMap(width=10.0, height=10.0)
        
        warehouse_map.set_obstacle(x=2.0, y=2.0, width=1.0, height=1.0)
        warehouse_map.set_obstacle(x=8.0, y=8.0, width=1.0, height=1.0)
        
        # Both obstacles should be present
        grid_x1, grid_y1 = warehouse_map.world_to_grid(2.0, 2.0)
        grid_x2, grid_y2 = warehouse_map.world_to_grid(8.0, 8.0)
        
        assert warehouse_map.occupancy_grid[grid_y1, grid_x1] == 100
        assert warehouse_map.occupancy_grid[grid_y2, grid_x2] == 100
    
    def test_obstacle_bounds_clamping(self):
        """Test that obstacles outside bounds are clamped."""
        warehouse_map = WarehouseMap(width=10.0, height=10.0)
        
        # Try to place obstacle partially outside bounds
        warehouse_map.set_obstacle(x=-1.0, y=5.0, width=2.0, height=2.0)
        
        # Should not raise error, obstacle should be clamped
        assert np.any(warehouse_map.occupancy_grid == 100)


class TestYAMLLoading:
    """Test YAML file loading and saving."""
    
    def test_load_from_yaml(self):
        """Test loading a map from YAML file."""
        # Create a temporary YAML file
        yaml_content = """
width: 20.0
height: 15.0
resolution: 0.2
origin: [1.0, 2.0]
obstacles:
  - type: rectangle
    x: 10.0
    y: 7.5
    width: 2.0
    height: 3.0
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name
        
        try:
            warehouse_map = WarehouseMap.from_yaml(temp_path)
            
            assert warehouse_map.width == 20.0
            assert warehouse_map.height == 15.0
            assert warehouse_map.resolution == 0.2
            assert warehouse_map.origin == (1.0, 2.0)
            
            # Check that obstacle was added
            assert np.any(warehouse_map.occupancy_grid == 100)
        finally:
            os.unlink(temp_path)
    
    def test_load_from_yaml_minimal(self):
        """Test loading a map with minimal YAML (no obstacles)."""
        yaml_content = """
width: 10.0
height: 10.0
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name
        
        try:
            warehouse_map = WarehouseMap.from_yaml(temp_path)
            
            assert warehouse_map.width == 10.0
            assert warehouse_map.height == 10.0
            assert warehouse_map.resolution == 0.1  # Default
            assert warehouse_map.origin == (0.0, 0.0)  # Default
            
            # No obstacles
            assert np.all(warehouse_map.occupancy_grid == 0)
        finally:
            os.unlink(temp_path)
    
    def test_save_to_yaml(self):
        """Test saving a map to YAML file."""
        warehouse_map = WarehouseMap(width=10.0, height=10.0, resolution=0.1, origin=(0.0, 0.0))
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name
        
        try:
            warehouse_map.to_yaml(temp_path)
            
            # Load it back and verify
            loaded_map = WarehouseMap.from_yaml(temp_path)
            
            assert loaded_map.width == warehouse_map.width
            assert loaded_map.height == warehouse_map.height
            assert loaded_map.resolution == warehouse_map.resolution
            assert loaded_map.origin == warehouse_map.origin
        finally:
            os.unlink(temp_path)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_zero_radius_collision_check(self):
        """Test collision checking with very small radius."""
        warehouse_map = WarehouseMap(width=10.0, height=10.0, resolution=0.1)
        warehouse_map.set_obstacle(x=5.0, y=5.0, width=1.0, height=1.0)
        
        # Position exactly at obstacle center should collide with small radius
        # Use a radius that's at least one grid cell
        assert not warehouse_map.is_collision_free(5.0, 5.0, radius=0.15)
    
    def test_very_small_map(self):
        """Test with a very small map."""
        warehouse_map = WarehouseMap(width=1.0, height=1.0, resolution=0.1)
        
        assert warehouse_map.grid_width == 10
        assert warehouse_map.grid_height == 10
        assert warehouse_map.is_collision_free(0.5, 0.5, radius=0.1)
    
    def test_large_resolution(self):
        """Test with large resolution (coarse grid)."""
        warehouse_map = WarehouseMap(width=10.0, height=10.0, resolution=1.0)
        
        assert warehouse_map.grid_width == 10
        assert warehouse_map.grid_height == 10
    
    def test_neighbors_with_all_blocked(self):
        """Test neighbors when all surrounding cells are blocked."""
        warehouse_map = WarehouseMap(width=10.0, height=10.0)
        
        # Block all neighbors
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                warehouse_map.occupancy_grid[50 + dy, 50 + dx] = 100
        
        neighbors = warehouse_map.get_neighbors(50, 50, connectivity=8)
        
        # Should have no neighbors
        assert len(neighbors) == 0
