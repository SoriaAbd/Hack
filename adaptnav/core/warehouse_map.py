"""
WarehouseMap class for static environment representation.

This module provides the WarehouseMap class which represents the static
warehouse environment using an occupancy grid. It supports collision checking,
neighbor queries for path planning, and loading maps from YAML files.
"""

import numpy as np
import yaml
from typing import List, Tuple, Optional


class WarehouseMap:
    """
    Represents a static warehouse environment using an occupancy grid.
    
    The occupancy grid uses the following convention:
    - 0: Free space
    - 100: Occupied space (obstacle)
    - -1: Unknown space
    
    Attributes:
        width: Width of the warehouse in meters
        height: Height of the warehouse in meters
        resolution: Size of each grid cell in meters (default: 0.1m)
        occupancy_grid: 2D numpy array representing the occupancy grid
        origin: (x, y) coordinates of the map origin in world frame
    """
    
    def __init__(
        self,
        width: float,
        height: float,
        resolution: float = 0.1,
        origin: Tuple[float, float] = (0.0, 0.0)
    ):
        """
        Initialize a WarehouseMap.
        
        Args:
            width: Width of the warehouse in meters
            height: Height of the warehouse in meters
            resolution: Grid cell size in meters (default: 0.1m)
            origin: (x, y) coordinates of the map origin in world frame
        """
        self.width = width
        self.height = height
        self.resolution = resolution
        self.origin = origin
        
        # Calculate grid dimensions
        self.grid_width = int(np.ceil(width / resolution))
        self.grid_height = int(np.ceil(height / resolution))
        
        # Initialize occupancy grid (0 = free, 100 = occupied, -1 = unknown)
        self.occupancy_grid = np.zeros((self.grid_height, self.grid_width), dtype=np.int8)
    
    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """
        Convert world coordinates to grid coordinates.
        
        Args:
            x: X coordinate in world frame (meters)
            y: Y coordinate in world frame (meters)
            
        Returns:
            Tuple of (grid_x, grid_y) coordinates
        """
        grid_x = int((x - self.origin[0]) / self.resolution)
        grid_y = int((y - self.origin[1]) / self.resolution)
        return grid_x, grid_y
    
    def grid_to_world(self, grid_x: int, grid_y: int) -> Tuple[float, float]:
        """
        Convert grid coordinates to world coordinates (cell center).
        
        Args:
            grid_x: X coordinate in grid frame
            grid_y: Y coordinate in grid frame
            
        Returns:
            Tuple of (x, y) coordinates in world frame (meters)
        """
        x = self.origin[0] + (grid_x + 0.5) * self.resolution
        y = self.origin[1] + (grid_y + 0.5) * self.resolution
        return x, y
    
    def is_valid_grid_cell(self, grid_x: int, grid_y: int) -> bool:
        """
        Check if grid coordinates are within bounds.
        
        Args:
            grid_x: X coordinate in grid frame
            grid_y: Y coordinate in grid frame
            
        Returns:
            True if coordinates are within grid bounds, False otherwise
        """
        return 0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height
    
    def is_collision_free(self, x: float, y: float, radius: float = 0.3) -> bool:
        """
        Check if a circular robot at position (x, y) is collision-free.
        
        This method checks all grid cells that could be occupied by a circular
        robot with the given radius centered at (x, y).
        
        Args:
            x: X coordinate in world frame (meters)
            y: Y coordinate in world frame (meters)
            radius: Robot radius in meters (default: 0.3m)
            
        Returns:
            True if the position is collision-free, False otherwise
        """
        # Convert center position to grid coordinates
        center_grid_x, center_grid_y = self.world_to_grid(x, y)
        
        # Calculate the number of grid cells to check based on radius
        grid_radius = int(np.ceil(radius / self.resolution))
        
        # Check all grid cells within the robot's radius
        for dy in range(-grid_radius, grid_radius + 1):
            for dx in range(-grid_radius, grid_radius + 1):
                grid_x = center_grid_x + dx
                grid_y = center_grid_y + dy
                
                # Skip if outside grid bounds
                if not self.is_valid_grid_cell(grid_x, grid_y):
                    return False  # Consider out-of-bounds as collision
                
                # Check if this grid cell is within the robot's circular footprint
                cell_world_x, cell_world_y = self.grid_to_world(grid_x, grid_y)
                distance = np.sqrt((cell_world_x - x)**2 + (cell_world_y - y)**2)
                
                if distance <= radius:
                    # Check if this cell is occupied
                    if self.occupancy_grid[grid_y, grid_x] != 0:
                        return False
        
        return True
    
    def get_neighbors(self, grid_x: int, grid_y: int, connectivity: int = 8) -> List[Tuple[int, int]]:
        """
        Get valid neighboring grid cells for path planning.
        
        Args:
            grid_x: X coordinate in grid frame
            grid_y: Y coordinate in grid frame
            connectivity: 4 for 4-connected, 8 for 8-connected (default: 8)
            
        Returns:
            List of (grid_x, grid_y) tuples for valid neighbors
        """
        neighbors = []
        
        # Define neighbor offsets based on connectivity
        if connectivity == 4:
            offsets = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        else:  # 8-connected
            offsets = [
                (0, 1), (1, 1), (1, 0), (1, -1),
                (0, -1), (-1, -1), (-1, 0), (-1, 1)
            ]
        
        for dx, dy in offsets:
            neighbor_x = grid_x + dx
            neighbor_y = grid_y + dy
            
            # Check if neighbor is within bounds
            if not self.is_valid_grid_cell(neighbor_x, neighbor_y):
                continue
            
            # Check if neighbor is free (not occupied)
            if self.occupancy_grid[neighbor_y, neighbor_x] == 0:
                neighbors.append((neighbor_x, neighbor_y))
        
        return neighbors
    
    def set_obstacle(self, x: float, y: float, width: float, height: float):
        """
        Add a rectangular obstacle to the map.
        
        Args:
            x: X coordinate of obstacle center in world frame (meters)
            y: Y coordinate of obstacle center in world frame (meters)
            width: Width of obstacle in meters
            height: Height of obstacle in meters
        """
        # Calculate grid bounds for the obstacle
        x_min = x - width / 2
        x_max = x + width / 2
        y_min = y - height / 2
        y_max = y + height / 2
        
        grid_x_min, grid_y_min = self.world_to_grid(x_min, y_min)
        grid_x_max, grid_y_max = self.world_to_grid(x_max, y_max)
        
        # Clamp to grid bounds
        grid_x_min = max(0, grid_x_min)
        grid_x_max = min(self.grid_width - 1, grid_x_max)
        grid_y_min = max(0, grid_y_min)
        grid_y_max = min(self.grid_height - 1, grid_y_max)
        
        # Mark cells as occupied
        self.occupancy_grid[grid_y_min:grid_y_max+1, grid_x_min:grid_x_max+1] = 100
    
    @classmethod
    def from_yaml(cls, filepath: str) -> 'WarehouseMap':
        """
        Load a warehouse map from a YAML file.
        
        Expected YAML format:
        ```yaml
        width: 50.0
        height: 50.0
        resolution: 0.1
        origin: [0.0, 0.0]
        obstacles:
          - type: rectangle
            x: 10.0
            y: 10.0
            width: 2.0
            height: 5.0
          - type: rectangle
            x: 20.0
            y: 15.0
            width: 3.0
            height: 1.0
        ```
        
        Args:
            filepath: Path to the YAML file
            
        Returns:
            WarehouseMap instance loaded from the file
        """
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        
        # Create map with basic parameters
        warehouse_map = cls(
            width=data['width'],
            height=data['height'],
            resolution=data.get('resolution', 0.1),
            origin=tuple(data.get('origin', [0.0, 0.0]))
        )
        
        # Add obstacles if present
        if 'obstacles' in data:
            for obstacle in data['obstacles']:
                if obstacle['type'] == 'rectangle':
                    warehouse_map.set_obstacle(
                        x=obstacle['x'],
                        y=obstacle['y'],
                        width=obstacle['width'],
                        height=obstacle['height']
                    )
        
        return warehouse_map
    
    def to_yaml(self, filepath: str):
        """
        Save the warehouse map to a YAML file.
        
        Note: This saves the basic map parameters but does not reconstruct
        individual obstacles from the occupancy grid.
        
        Args:
            filepath: Path to save the YAML file
        """
        data = {
            'width': float(self.width),
            'height': float(self.height),
            'resolution': float(self.resolution),
            'origin': [float(self.origin[0]), float(self.origin[1])]
        }
        
        with open(filepath, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
