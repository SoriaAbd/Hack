"""
DynamicObstacle class for moving obstacle representation.

This module provides the DynamicObstacle class which represents dynamic
obstacles (workers, forklifts) in the warehouse environment. It supports
position prediction using a constant velocity model and distance calculations.
"""

import numpy as np
from typing import Literal


ObstacleClassification = Literal["worker", "forklift", "unknown"]


class DynamicObstacle:
    """
    Represents a dynamic obstacle in the warehouse environment.
    
    Dynamic obstacles are moving entities such as workers and forklifts that
    the robot must detect and avoid. This class tracks their position, velocity,
    and classification, and provides methods for prediction and distance calculation.
    
    Attributes:
        id: Unique tracking identifier for the obstacle
        position: [x, y] position in map frame (meters)
        velocity: [vx, vy] velocity in map frame (meters/second)
        radius: Bounding circle radius in meters
        classification: Type of obstacle ("worker", "forklift", "unknown")
    """
    
    def __init__(
        self,
        id: int,
        position: np.ndarray,
        velocity: np.ndarray,
        radius: float,
        classification: ObstacleClassification = "unknown"
    ):
        """
        Initialize a DynamicObstacle.
        
        Args:
            id: Unique tracking identifier
            position: [x, y] position in map frame (meters)
            velocity: [vx, vy] velocity in map frame (meters/second)
            radius: Bounding circle radius in meters
            classification: Type of obstacle ("worker", "forklift", "unknown")
        """
        self.id = id
        self.position = np.asarray(position, dtype=np.float64)
        self.velocity = np.asarray(velocity, dtype=np.float64)
        self.radius = float(radius)
        self.classification = classification
        
        # Validate inputs
        if self.position.shape != (2,):
            raise ValueError(f"Position must be a 2D array, got shape {self.position.shape}")
        if self.velocity.shape != (2,):
            raise ValueError(f"Velocity must be a 2D array, got shape {self.velocity.shape}")
        if self.radius <= 0:
            raise ValueError(f"Radius must be positive, got {self.radius}")
    
    def predict_position(self, dt: float) -> np.ndarray:
        """
        Predict position after dt seconds using constant velocity model.
        
        This method uses a simple constant velocity model to predict where
        the obstacle will be after a given time interval. This is useful for
        collision prediction and path planning.
        
        Args:
            dt: Time interval in seconds (must be non-negative)
            
        Returns:
            Predicted [x, y] position in map frame (meters)
            
        Raises:
            ValueError: If dt is negative
        """
        if dt < 0:
            raise ValueError(f"Time interval dt must be non-negative, got {dt}")
        
        return self.position + self.velocity * dt
    
    def distance_to(self, point: np.ndarray) -> float:
        """
        Compute distance from obstacle center to a point.
        
        This method calculates the Euclidean distance from the obstacle's
        center position to a given point in the map frame.
        
        Args:
            point: [x, y] coordinates in map frame (meters)
            
        Returns:
            Euclidean distance in meters
            
        Raises:
            ValueError: If point is not a 2D array
        """
        point = np.asarray(point, dtype=np.float64)
        if point.shape != (2,):
            raise ValueError(f"Point must be a 2D array, got shape {point.shape}")
        
        return np.linalg.norm(self.position - point)
    
    def __repr__(self) -> str:
        """Return string representation of the obstacle."""
        return (
            f"DynamicObstacle(id={self.id}, "
            f"position={self.position}, "
            f"velocity={self.velocity}, "
            f"radius={self.radius}, "
            f"classification='{self.classification}')"
        )
    
    def __eq__(self, other) -> bool:
        """Check equality based on obstacle ID."""
        if not isinstance(other, DynamicObstacle):
            return False
        return self.id == other.id
    
    def __hash__(self) -> int:
        """Hash based on obstacle ID."""
        return hash(self.id)
