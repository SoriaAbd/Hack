"""
Property-based tests for ground truth availability.

Feature: adaptnav-context-aware-warehouse-navigation
Property 2: Ground Truth Availability
Validates: Requirements 1.4
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, assume, settings

try:
    import rclpy
    from adaptnav.simulation.mujoco_simulation import MuJoCoSimulation, MUJOCO_AVAILABLE
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    MUJOCO_AVAILABLE = False


# Warehouse bounds from warehouse.xml: 50m x 50m warehouse (-25 to 25 in x and y)
WAREHOUSE_MIN_X = -25.0
WAREHOUSE_MAX_X = 25.0
WAREHOUSE_MIN_Y = -25.0
WAREHOUSE_MAX_Y = 25.0


# Custom strategies for generating test data
@st.composite
def valid_robot_position(draw):
    """Generate a valid robot spawn position (x, y, theta)."""
    x = draw(st.floats(min_value=-20.0, max_value=20.0))
    y = draw(st.floats(min_value=-20.0, max_value=20.0))
    theta = draw(st.floats(min_value=-np.pi, max_value=np.pi))
    return (x, y, theta)


@st.composite
def valid_obstacle_config(draw):
    """Generate a valid obstacle configuration."""
    name = draw(st.sampled_from(['worker_1', 'worker_2', 'forklift_1']))
    x = draw(st.floats(min_value=-20.0, max_value=20.0))
    y = draw(st.floats(min_value=-20.0, max_value=20.0))
    vx = draw(st.floats(min_value=-2.0, max_value=2.0))
    vy = draw(st.floats(min_value=-2.0, max_value=2.0))
    
    return {
        'name': name,
        'position': [x, y],
        'velocity': [vx, vy]
    }


@st.composite
def simulation_time_steps(draw):
    """Generate a number of simulation time steps to advance."""
    # Test at different points in time: immediately after reset, and after some steps
    return draw(st.integers(min_value=0, max_value=50))


@pytest.mark.skipif(not ROS2_AVAILABLE or not MUJOCO_AVAILABLE,
                    reason="ROS 2 or MuJoCo not available")
@pytest.mark.property
class TestGroundTruthAvailabilityProperties:
    """Property-based tests for ground truth availability."""
    
    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup and teardown for each test."""
        if not rclpy.ok():
            rclpy.init()
        
        yield
    
    @given(
        robot_pos=valid_robot_position(),
        time_steps=simulation_time_steps()
    )
    @settings(max_examples=100, deadline=60000)
    def test_robot_ground_truth_always_available(self, robot_pos, time_steps):
        """
        Property 2: Ground Truth Availability - Robot
        
        For any entity (robot) in the simulation at any time, querying its
        ground truth position should return a valid position within the
        warehouse bounds.
        
        **Validates: Requirements 1.4**
        """
        sim = MuJoCoSimulation(render_mode=None)
        
        try:
            # Reset simulation
            success = sim.reset(robot_position=robot_pos)
            assert success, "Simulation reset should succeed"
            
            # Advance simulation by random number of steps
            for _ in range(time_steps):
                sim.step(dt=0.01)
            
            # Query ground truth robot state
            robot_state = sim.get_ground_truth_robot_state()
            
            # Verify position is not None/null
            assert robot_state is not None, \
                "Ground truth robot state should not be None"
            assert robot_state.position is not None, \
                "Robot position should not be None"
            
            # Verify position is a valid 2D array
            assert len(robot_state.position) == 2, \
                f"Robot position should be 2D, got {len(robot_state.position)}D"
            
            # Verify position values are finite (not NaN or Inf)
            assert np.isfinite(robot_state.position[0]), \
                f"Robot x position {robot_state.position[0]} is not finite"
            assert np.isfinite(robot_state.position[1]), \
                f"Robot y position {robot_state.position[1]} is not finite"
            
            # Verify position is within warehouse bounds
            x, y = robot_state.position
            assert WAREHOUSE_MIN_X <= x <= WAREHOUSE_MAX_X, \
                f"Robot x position {x:.2f} is outside warehouse bounds " \
                f"[{WAREHOUSE_MIN_X}, {WAREHOUSE_MAX_X}]"
            assert WAREHOUSE_MIN_Y <= y <= WAREHOUSE_MAX_Y, \
                f"Robot y position {y:.2f} is outside warehouse bounds " \
                f"[{WAREHOUSE_MIN_Y}, {WAREHOUSE_MAX_Y}]"
            
            # Verify orientation is finite
            assert np.isfinite(robot_state.orientation), \
                f"Robot orientation {robot_state.orientation} is not finite"
            
            # Verify timestamp is valid
            assert robot_state.timestamp >= 0, \
                f"Robot timestamp {robot_state.timestamp} should be non-negative"
        
        finally:
            sim.destroy_node()
    
    @given(
        robot_pos=valid_robot_position(),
        obstacle_configs=st.lists(
            valid_obstacle_config(),
            min_size=1,
            max_size=3,
            unique_by=lambda x: x['name']
        ),
        time_steps=simulation_time_steps()
    )
    @settings(max_examples=100, deadline=60000)
    def test_obstacles_ground_truth_always_available(self, robot_pos, obstacle_configs, time_steps):
        """
        Property 2: Ground Truth Availability - Obstacles
        
        For any entity (obstacles) in the simulation at any time, querying
        their ground truth positions should return valid positions within
        the warehouse bounds.
        
        **Validates: Requirements 1.4**
        """
        sim = MuJoCoSimulation(render_mode=None)
        
        try:
            # Reset simulation with obstacles
            success = sim.reset(robot_position=robot_pos, obstacle_configs=obstacle_configs)
            assert success, "Simulation reset should succeed"
            
            # Advance simulation by random number of steps
            for _ in range(time_steps):
                sim.step(dt=0.01)
            
            # Query ground truth obstacles
            obstacles = sim.get_ground_truth_obstacles()
            
            # Verify obstacles list is not None
            assert obstacles is not None, \
                "Ground truth obstacles should not be None"
            
            # Verify we have obstacles
            assert len(obstacles) > 0, \
                "Ground truth obstacles list should not be empty"
            
            # Check each obstacle
            for obs in obstacles:
                # Verify obstacle is not None
                assert obs is not None, \
                    f"Obstacle should not be None"
                
                # Verify position is not None
                assert obs.position is not None, \
                    f"Obstacle {obs.id} position should not be None"
                
                # Verify position is a valid 2D array
                assert len(obs.position) == 2, \
                    f"Obstacle {obs.id} position should be 2D, got {len(obs.position)}D"
                
                # Verify position values are finite (not NaN or Inf)
                assert np.isfinite(obs.position[0]), \
                    f"Obstacle {obs.id} x position {obs.position[0]} is not finite"
                assert np.isfinite(obs.position[1]), \
                    f"Obstacle {obs.id} y position {obs.position[1]} is not finite"
                
                # Verify position is within warehouse bounds
                x, y = obs.position
                assert WAREHOUSE_MIN_X <= x <= WAREHOUSE_MAX_X, \
                    f"Obstacle {obs.id} x position {x:.2f} is outside warehouse bounds " \
                    f"[{WAREHOUSE_MIN_X}, {WAREHOUSE_MAX_X}]"
                assert WAREHOUSE_MIN_Y <= y <= WAREHOUSE_MAX_Y, \
                    f"Obstacle {obs.id} y position {y:.2f} is outside warehouse bounds " \
                    f"[{WAREHOUSE_MIN_Y}, {WAREHOUSE_MAX_Y}]"
                
                # Verify velocity is not None and finite
                assert obs.velocity is not None, \
                    f"Obstacle {obs.id} velocity should not be None"
                assert len(obs.velocity) == 2, \
                    f"Obstacle {obs.id} velocity should be 2D, got {len(obs.velocity)}D"
                assert np.isfinite(obs.velocity[0]), \
                    f"Obstacle {obs.id} x velocity {obs.velocity[0]} is not finite"
                assert np.isfinite(obs.velocity[1]), \
                    f"Obstacle {obs.id} y velocity {obs.velocity[1]} is not finite"
                
                # Verify ID is valid
                assert obs.id > 0, \
                    f"Obstacle ID {obs.id} should be positive"
                
                # Verify classification is valid
                assert obs.classification in ['worker', 'forklift', 'unknown'], \
                    f"Obstacle {obs.id} has invalid classification: {obs.classification}"
        
        finally:
            sim.destroy_node()
    
    @given(
        robot_pos=valid_robot_position(),
        time_steps=simulation_time_steps()
    )
    @settings(max_examples=100, deadline=60000)
    def test_default_obstacles_ground_truth_always_available(self, robot_pos, time_steps):
        """
        Property 2: Ground Truth Availability - Default Obstacles
        
        For default obstacle configuration at any time, querying ground truth
        should return valid positions for all three obstacles.
        
        **Validates: Requirements 1.4**
        """
        sim = MuJoCoSimulation(render_mode=None)
        
        try:
            # Reset simulation with default obstacles
            success = sim.reset(robot_position=robot_pos)
            assert success, "Simulation reset should succeed"
            
            # Advance simulation by random number of steps
            for _ in range(time_steps):
                sim.step(dt=0.01)
            
            # Query ground truth obstacles
            obstacles = sim.get_ground_truth_obstacles()
            
            # Verify we have exactly 3 obstacles (default configuration)
            assert len(obstacles) == 3, \
                f"Default configuration should have 3 obstacles, got {len(obstacles)}"
            
            # Check each obstacle has valid ground truth
            for obs in obstacles:
                # Verify position is within bounds
                x, y = obs.position
                assert WAREHOUSE_MIN_X <= x <= WAREHOUSE_MAX_X, \
                    f"Obstacle {obs.id} x position {x:.2f} is outside warehouse bounds"
                assert WAREHOUSE_MIN_Y <= y <= WAREHOUSE_MAX_Y, \
                    f"Obstacle {obs.id} y position {y:.2f} is outside warehouse bounds"
                
                # Verify all values are finite
                assert np.all(np.isfinite(obs.position)), \
                    f"Obstacle {obs.id} position contains non-finite values"
                assert np.all(np.isfinite(obs.velocity)), \
                    f"Obstacle {obs.id} velocity contains non-finite values"
        
        finally:
            sim.destroy_node()
    
    @given(
        robot_pos=valid_robot_position(),
        time_steps=simulation_time_steps()
    )
    @settings(max_examples=100, deadline=60000)
    def test_ground_truth_available_throughout_simulation(self, robot_pos, time_steps):
        """
        Property 2: Ground Truth Availability - Continuous Availability
        
        Ground truth should be available at every point during simulation,
        not just at initialization.
        
        **Validates: Requirements 1.4**
        """
        sim = MuJoCoSimulation(render_mode=None)
        
        try:
            # Reset simulation
            success = sim.reset(robot_position=robot_pos)
            assert success, "Simulation reset should succeed"
            
            # Query ground truth at multiple points during simulation
            for step in range(time_steps):
                # Step simulation
                if step > 0:
                    sim.step(dt=0.01)
                
                # Query ground truth at this time step
                robot_state = sim.get_ground_truth_robot_state()
                obstacles = sim.get_ground_truth_obstacles()
                
                # Verify robot ground truth is available
                assert robot_state is not None, \
                    f"Robot ground truth not available at step {step}"
                assert np.all(np.isfinite(robot_state.position)), \
                    f"Robot position not finite at step {step}"
                
                # Verify robot is within bounds
                x, y = robot_state.position
                assert WAREHOUSE_MIN_X <= x <= WAREHOUSE_MAX_X, \
                    f"Robot x position {x:.2f} out of bounds at step {step}"
                assert WAREHOUSE_MIN_Y <= y <= WAREHOUSE_MAX_Y, \
                    f"Robot y position {y:.2f} out of bounds at step {step}"
                
                # Verify obstacles ground truth is available
                assert obstacles is not None, \
                    f"Obstacles ground truth not available at step {step}"
                assert len(obstacles) > 0, \
                    f"No obstacles at step {step}"
                
                for obs in obstacles:
                    assert np.all(np.isfinite(obs.position)), \
                        f"Obstacle {obs.id} position not finite at step {step}"
                    
                    # Verify obstacle is within bounds
                    ox, oy = obs.position
                    assert WAREHOUSE_MIN_X <= ox <= WAREHOUSE_MAX_X, \
                        f"Obstacle {obs.id} x position {ox:.2f} out of bounds at step {step}"
                    assert WAREHOUSE_MIN_Y <= oy <= WAREHOUSE_MAX_Y, \
                        f"Obstacle {obs.id} y position {oy:.2f} out of bounds at step {step}"
        
        finally:
            sim.destroy_node()
    
    @given(robot_pos=valid_robot_position())
    @settings(max_examples=100, deadline=60000)
    def test_ground_truth_ids_are_consistent(self, robot_pos):
        """
        Property 2: Ground Truth Availability - ID Consistency
        
        Obstacle IDs in ground truth should remain consistent throughout
        the simulation (same obstacle keeps same ID).
        
        **Validates: Requirements 1.4**
        """
        sim = MuJoCoSimulation(render_mode=None)
        
        try:
            # Reset simulation
            success = sim.reset(robot_position=robot_pos)
            assert success, "Simulation reset should succeed"
            
            # Get initial obstacle IDs
            initial_obstacles = sim.get_ground_truth_obstacles()
            initial_ids = set(obs.id for obs in initial_obstacles)
            
            # Verify IDs are unique
            assert len(initial_ids) == len(initial_obstacles), \
                "Obstacle IDs should be unique"
            
            # Step simulation and check IDs remain consistent
            for _ in range(10):
                sim.step(dt=0.01)
                
                current_obstacles = sim.get_ground_truth_obstacles()
                current_ids = set(obs.id for obs in current_obstacles)
                
                # IDs should remain the same
                assert current_ids == initial_ids, \
                    f"Obstacle IDs changed during simulation. " \
                    f"Initial: {initial_ids}, Current: {current_ids}"
                
                # Number of obstacles should remain the same
                assert len(current_obstacles) == len(initial_obstacles), \
                    f"Number of obstacles changed during simulation"
        
        finally:
            sim.destroy_node()
