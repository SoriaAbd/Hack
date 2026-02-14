"""
A* path planning algorithm implementation.

This module provides the AStarPlanner class which implements the A* search algorithm
for finding optimal paths through the warehouse environment. The planner operates
on occupancy grids and returns collision-free paths as sequences of waypoints.
"""

import heapq
import time
from typing import List, Tuple, Optional, Dict, Set
import numpy as np

from ..core.warehouse_map import WarehouseMap
from ..core.path import Path, Waypoint


class AStarNode:
    """
    Node class for A* search algorithm.
    
    Represents a single node in the search tree with position, costs, and parent
    information for path reconstruction.
    """
    
    def __init__(self, grid_x: int, grid_y: int, g_cost: float = 0.0, 
                 h_cost: float = 0.0, parent: Optional['AStarNode'] = None):
        """
        Initialize an A* node.
        
        Args:
            grid_x: X coordinate in grid frame
            grid_y: Y coordinate in grid frame
            g_cost: Cost from start to this node
            h_cost: Heuristic cost from this node to goal
            parent: Parent node for path reconstruction
        """
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.g_cost = g_cost
        self.h_cost = h_cost
        self.f_cost = g_cost + h_cost
        self.parent = parent
    
    def __lt__(self, other: 'AStarNode') -> bool:
        """Comparison for priority queue (lower f_cost has higher priority)."""
        return self.f_cost < other.f_cost
    
    def __eq__(self, other: 'AStarNode') -> bool:
        """Equality based on grid position."""
        return self.grid_x == other.grid_x and self.grid_y == other.grid_y
    
    def __hash__(self) -> int:
        """Hash based on grid position for use in sets/dicts."""
        return hash((self.grid_x, self.grid_y))
    
    def position(self) -> Tuple[int, int]:
        """Get position as tuple."""
        return (self.grid_x, self.grid_y)


class AStarPlanner:
    """
    A* path planning algorithm for warehouse navigation.
    
    This planner uses the A* search algorithm to find optimal paths through
    the warehouse environment. It operates on occupancy grids and returns
    collision-free paths as sequences of waypoints.
    
    Features:
    - Euclidean distance heuristic
    - 8-connected grid search
    - Configurable robot radius for collision checking
    - Timeout mechanism to prevent infinite search
    - Path reconstruction from search result
    """
    
    def __init__(self, warehouse_map: WarehouseMap, robot_radius: float = 0.3,
                 timeout_seconds: float = 2.0):
        """
        Initialize the A* planner.
        
        Args:
            warehouse_map: WarehouseMap containing the static environment
            robot_radius: Robot radius for collision checking (meters)
            timeout_seconds: Maximum planning time before timeout
        """
        self.warehouse_map = warehouse_map
        self.robot_radius = robot_radius
        self.timeout_seconds = timeout_seconds
    
    def euclidean_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """
        Compute Euclidean distance between two grid positions.
        
        Args:
            pos1: First position (grid_x, grid_y)
            pos2: Second position (grid_x, grid_y)
            
        Returns:
            Euclidean distance in grid units
        """
        dx = pos1[0] - pos2[0]
        dy = pos1[1] - pos2[1]
        return np.sqrt(dx * dx + dy * dy)
    
    def get_movement_cost(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> float:
        """
        Get the cost of moving from one grid cell to another.
        
        Diagonal moves cost sqrt(2), orthogonal moves cost 1.0.
        
        Args:
            from_pos: Starting position (grid_x, grid_y)
            to_pos: Target position (grid_x, grid_y)
            
        Returns:
            Movement cost
        """
        dx = abs(to_pos[0] - from_pos[0])
        dy = abs(to_pos[1] - from_pos[1])
        
        if dx == 1 and dy == 1:
            return np.sqrt(2)  # Diagonal move
        else:
            return 1.0  # Orthogonal move
    
    def is_position_valid(self, grid_x: int, grid_y: int) -> bool:
        """
        Check if a grid position is valid for the robot.
        
        This checks both grid bounds and collision-free status considering
        the robot's radius.
        
        Args:
            grid_x: X coordinate in grid frame
            grid_y: Y coordinate in grid frame
            
        Returns:
            True if position is valid and collision-free
        """
        # Check grid bounds
        if not self.warehouse_map.is_valid_grid_cell(grid_x, grid_y):
            return False
        
        # Convert to world coordinates and check collision
        world_x, world_y = self.warehouse_map.grid_to_world(grid_x, grid_y)
        return self.warehouse_map.is_collision_free(world_x, world_y, self.robot_radius)
    
    def reconstruct_path(self, goal_node: AStarNode) -> List[Waypoint]:
        """
        Reconstruct the path from start to goal using parent pointers.
        
        Args:
            goal_node: The goal node reached by A* search
            
        Returns:
            List of waypoints from start to goal
        """
        waypoints = []
        current = goal_node
        
        # Traverse back through parent pointers
        while current is not None:
            # Convert grid coordinates to world coordinates
            world_x, world_y = self.warehouse_map.grid_to_world(
                current.grid_x, current.grid_y
            )
            waypoints.append(Waypoint(world_x, world_y))
            current = current.parent
        
        # Reverse to get path from start to goal
        waypoints.reverse()
        return waypoints
    
    def plan_path(self, start_x: float, start_y: float, 
                  goal_x: float, goal_y: float) -> Optional[Path]:
        """
        Plan a path from start to goal using A* algorithm.
        
        Args:
            start_x: Start X coordinate in world frame (meters)
            start_y: Start Y coordinate in world frame (meters)
            goal_x: Goal X coordinate in world frame (meters)
            goal_y: Goal Y coordinate in world frame (meters)
            
        Returns:
            Path object if successful, None if no path found or timeout
        """
        start_time = time.time()
        
        # Convert world coordinates to grid coordinates
        start_grid_x, start_grid_y = self.warehouse_map.world_to_grid(start_x, start_y)
        goal_grid_x, goal_grid_y = self.warehouse_map.world_to_grid(goal_x, goal_y)
        
        # Check if start and goal positions are valid
        if not self.is_position_valid(start_grid_x, start_grid_y):
            print(f"Start position ({start_x:.2f}, {start_y:.2f}) is not valid")
            return None
        
        if not self.is_position_valid(goal_grid_x, goal_grid_y):
            print(f"Goal position ({goal_x:.2f}, {goal_y:.2f}) is not valid")
            return None
        
        # Check if start equals goal
        if start_grid_x == goal_grid_x and start_grid_y == goal_grid_y:
            waypoints = [Waypoint(start_x, start_y), Waypoint(goal_x, goal_y)]
            return Path(waypoints, time.time())
        
        # Initialize A* data structures
        open_set = []  # Priority queue of nodes to explore
        closed_set: Set[Tuple[int, int]] = set()  # Set of explored nodes
        g_costs: Dict[Tuple[int, int], float] = {}  # Best known g_cost for each position
        
        # Create start node
        start_node = AStarNode(
            start_grid_x, start_grid_y,
            g_cost=0.0,
            h_cost=self.euclidean_distance((start_grid_x, start_grid_y), 
                                         (goal_grid_x, goal_grid_y))
        )
        
        heapq.heappush(open_set, start_node)
        g_costs[(start_grid_x, start_grid_y)] = 0.0
        
        # A* main loop
        while open_set:
            # Check timeout
            if time.time() - start_time > self.timeout_seconds:
                print(f"A* planning timeout after {self.timeout_seconds} seconds")
                return None
            
            # Get node with lowest f_cost
            current = heapq.heappop(open_set)
            current_pos = current.position()
            
            # Skip if already processed (can happen due to duplicate entries)
            if current_pos in closed_set:
                continue
            
            # Add to closed set
            closed_set.add(current_pos)
            
            # Check if goal reached
            if current.grid_x == goal_grid_x and current.grid_y == goal_grid_y:
                waypoints = self.reconstruct_path(current)
                computation_time = time.time() - start_time
                print(f"A* found path with {len(waypoints)} waypoints in {computation_time:.3f}s")
                return Path(waypoints, time.time())
            
            # Explore neighbors
            neighbors = self.warehouse_map.get_neighbors(current.grid_x, current.grid_y)
            
            for neighbor_x, neighbor_y in neighbors:
                neighbor_pos = (neighbor_x, neighbor_y)
                
                # Skip if already processed
                if neighbor_pos in closed_set:
                    continue
                
                # Check if neighbor is valid (collision-free)
                if not self.is_position_valid(neighbor_x, neighbor_y):
                    continue
                
                # Calculate tentative g_cost
                movement_cost = self.get_movement_cost(current_pos, neighbor_pos)
                tentative_g_cost = current.g_cost + movement_cost
                
                # Skip if we've found a better path to this neighbor
                if (neighbor_pos in g_costs and 
                    tentative_g_cost >= g_costs[neighbor_pos]):
                    continue
                
                # This is the best path to neighbor so far
                g_costs[neighbor_pos] = tentative_g_cost
                
                # Calculate heuristic cost
                h_cost = self.euclidean_distance(neighbor_pos, (goal_grid_x, goal_grid_y))
                
                # Create neighbor node
                neighbor_node = AStarNode(
                    neighbor_x, neighbor_y,
                    g_cost=tentative_g_cost,
                    h_cost=h_cost,
                    parent=current
                )
                
                # Add to open set
                heapq.heappush(open_set, neighbor_node)
        
        # No path found
        print(f"A* could not find path from ({start_x:.2f}, {start_y:.2f}) to ({goal_x:.2f}, {goal_y:.2f})")
        return None