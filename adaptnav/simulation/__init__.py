"""
Simulation environment for AdaptNav.

Provides physics-based warehouse simulation with support for:
- MuJoCo backend
- Isaac Sim backend (optional)
- Ground truth data access
- Collision detection
"""

from .base_simulation import BaseSimulation

try:
    from .mujoco_simulation import MuJoCoSimulation
    __all__ = ['BaseSimulation', 'MuJoCoSimulation']
except ImportError:
    # MuJoCo not installed
    __all__ = ['BaseSimulation']
