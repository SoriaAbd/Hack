"""
Control module for AdaptNav.

This module provides safety controllers and velocity filtering for autonomous
warehouse navigation, ensuring safe operation around dynamic obstacles.
"""

from .safety_controller import SafetyController

__all__ = [
    'SafetyController'
]