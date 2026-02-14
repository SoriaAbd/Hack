"""
Property-based tests for simulation initialization.

Feature: adaptnav-context-aware-warehouse-navigation
Property 1: Initialization Consistency
Validates: Requirements 1.3
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


# Custom strategies for generating test data
@st.composite
def valid_robot_position(draw):
    """Generate a valid robot spawn position (x, y, theta)."""
    # Warehouse bounds are approximately -25 to 25 in x and y
    x = draw(st.floats(min_value=-20.0, max_value=20.0))
    y = draw(st.floats(min_value=-20.0, max_value=20.0))
    # Theta in radians
    theta = draw(st.floats(min_value=-np.pi, max_value=np.pi))
    return (x, y, theta)


@st.composite
def valid_obstacle_config(draw):
    """Generate a valid obstacle configuration."""
    # Obstacle name (one of the three obstacles in the warehouse model)
    name = draw(st.sampled_from(['worker_1', 'worker_2', 'forklift_1']))
    
    # Position within warehouse bounds
    x = draw(st.floats(min_value=-20.0, max_value=20.0))
    y = draw(st.floats(min_value=-20.0, max_value=20.0))
    
    # Velocity (reasonable range for warehouse obstacles)
    vx = draw(st.floats(min_value=-2.0, max_value=2.0))
    vy = draw(st.floats(min_value=-2.0, max_value=2.0))
    
    return {
        'name': name,
        'position': [x, y],
        'velocity': [vx, vy]
    }


@pytest.mark.skipif(not ROS2_AVAILABLE or not MUJOCO_AVAILABLE,
                    reason="ROS 2 or MuJoCo not available")
@pytest.mark.property
class TestSimulationInitializationProperties:
    """Property-based tests for simulation initialization consistency."""
    
    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup and teardown for each test."""
        # Initialize ROS 2
        if not rclpy.ok():
            rclpy.init()
        
        yield
        
        # Cleanup is handled by individual tests
    
    @given(robot_pos=valid_robot_position())
    @settings(max_examples=100, deadline=60000)
    def test_robot_initialization_consistency(self, robot_pos):
        """
        Property 1: Initialization Consistency - Robot Position
        
        For any valid warehouse configuration, when the simulation starts,
        the robot should be positioned at its specified initial position
        within a tolerance of 0.01m.
        
        **Validates: Requirements 1.3**
        """
        sim = MuJoCoSimulation(render_mode=None)
        
        try:
            # Reset simulation with specified robot position
            success = sim.reset(robot_position=robot_pos)
            assert success, "Simulation reset should succeed"
            
            # Get ground truth robot state
            robot_state = sim.get_ground_truth_robot_state()
            
            # Check position matches within tolerance
            expected_x, expected_y, expected_theta = robot_pos
            actual_x, actual_y = robot_state.position
            actual_theta = robot_state.orientation
            
            position_error = np.sqrt((actual_x - expected_x)**2 + (actual_y - expected_y)**2)
            
            assert position_error <= 0.01, \
                f"Robot position error {position_error:.4f}m exceeds tolerance 0.01m. " \
                f"Expected: ({expected_x:.3f}, {expected_y:.3f}), " \
                f"Actual: ({actual_x:.3f}, {actual_y:.3f})"
            
            # Check orientation matches within tolerance (0.01 radians ~ 0.57 degrees)
            # Normalize angle difference to [-pi, pi]
            angle_diff = (actual_theta - expected_theta + np.pi) % (2 * np.pi) - np.pi
            angle_error = abs(angle_diff)
            
            assert angle_error <= 0.01, \
                f"Robot orientation error {angle_error:.4f} rad exceeds tolerance 0.01 rad. " \
                f"Expected: {expected_theta:.3f}, Actual: {actual_theta:.3f}"
        
        finally:
            sim.destroy_node()
    
    @given(
        robot_pos=valid_robot_position(),
        obstacle_configs=st.lists(valid_obstacle_config(), min_size=1, max_size=3, unique_by=lambda x: x['name'])
    )
    @settings(max_examples=100, deadline=60000)
    def test_obstacle_initialization_consistency(self, robot_pos, obstacle_configs):
        """
        Property 1: Initialization Consistency - Obstacle Positions
        
        For any valid warehouse configuration, when the simulation starts,
        all obstacles should be positioned at their specified initial positions
        within a tolerance of 0.01m.
        
        **Validates: Requirements 1.3**
        """
        sim = MuJoCoSimulation(render_mode=None)
        
        try:
            # Reset simulation with specified robot and obstacle positions
            success = sim.reset(robot_position=robot_pos, obstacle_configs=obstacle_configs)
            assert success, "Simulation reset should succeed"
            
            # Get ground truth obstacles
            actual_obstacles = sim.get_ground_truth_obstacles()
            
            # Create a mapping from obstacle ID to expected configuration
            # The obstacle IDs are: worker_1=1, worker_2=2, forklift_1=3
            name_to_id = {'worker_1': 1, 'worker_2': 2, 'forklift_1': 3}
            expected_by_id = {name_to_id[cfg['name']]: cfg for cfg in obstacle_configs}
            
            # Check each obstacle
            for obs in actual_obstacles:
                if obs.id in expected_by_id:
                    expected = expected_by_id[obs.id]
                    expected_pos = np.array(expected['position'])
                    actual_pos = obs.position
                    
                    position_error = np.linalg.norm(actual_pos - expected_pos)
                    
                    assert position_error <= 0.01, \
                        f"Obstacle {obs.id} position error {position_error:.4f}m exceeds tolerance 0.01m. " \
                        f"Expected: ({expected_pos[0]:.3f}, {expected_pos[1]:.3f}), " \
                        f"Actual: ({actual_pos[0]:.3f}, {actual_pos[1]:.3f})"
        
        finally:
            sim.destroy_node()
    
    @given(robot_pos=valid_robot_position())
    @settings(max_examples=100, deadline=60000)
    def test_default_obstacle_initialization_consistency(self, robot_pos):
        """
        Property 1: Initialization Consistency - Default Obstacles
        
        When the simulation starts with default obstacle configuration,
        all three obstacles (2 workers + 1 forklift) should be present
        and positioned consistently.
        
        **Validates: Requirements 1.3**
        """
        sim = MuJoCoSimulation(render_mode=None)
        
        try:
            # Reset simulation with default obstacles
            success = sim.reset(robot_position=robot_pos)
            assert success, "Simulation reset should succeed"
            
            # Get ground truth obstacles
            obstacles = sim.get_ground_truth_obstacles()
            
            # Check we have exactly 3 obstacles
            assert len(obstacles) == 3, \
                f"Expected 3 obstacles in default configuration, got {len(obstacles)}"
            
            # Check obstacle IDs are unique
            obstacle_ids = [obs.id for obs in obstacles]
            assert len(set(obstacle_ids)) == 3, \
                f"Obstacle IDs should be unique, got {obstacle_ids}"
            
            # Check obstacle classifications
            classifications = [obs.classification for obs in obstacles]
            assert classifications.count('worker') == 2, \
                f"Expected 2 workers, got {classifications.count('worker')}"
            assert classifications.count('forklift') == 1, \
                f"Expected 1 forklift, got {classifications.count('forklift')}"
            
            # Check all obstacles have valid positions (within warehouse bounds)
            for obs in obstacles:
                assert len(obs.position) == 2, \
                    f"Obstacle {obs.id} position should be 2D, got {len(obs.position)}D"
                
                # Warehouse bounds are approximately -25 to 25
                assert -25.0 <= obs.position[0] <= 25.0, \
                    f"Obstacle {obs.id} x position {obs.position[0]:.2f} out of bounds"
                assert -25.0 <= obs.position[1] <= 25.0, \
                    f"Obstacle {obs.id} y position {obs.position[1]:.2f} out of bounds"
        
        finally:
            sim.destroy_node()
    
    @given(robot_pos=valid_robot_position())
    @settings(max_examples=100, deadline=60000)
    def test_initialization_is_deterministic(self, robot_pos):
        """
        Property 1: Initialization Consistency - Determinism
        
        Resetting the simulation multiple times with the same configuration
        should produce identical initial states.
        
        **Validates: Requirements 1.3**
        """
        sim = MuJoCoSimulation(render_mode=None)
        
        try:
            # Reset simulation multiple times with same configuration
            positions = []
            orientations = []
            
            for _ in range(3):
                success = sim.reset(robot_position=robot_pos)
                assert success, "Simulation reset should succeed"
                
                robot_state = sim.get_ground_truth_robot_state()
                positions.append(robot_state.position.copy())
                orientations.append(robot_state.orientation)
            
            # Check all positions are identical
            for i in range(1, len(positions)):
                position_diff = np.linalg.norm(positions[i] - positions[0])
                assert position_diff < 1e-6, \
                    f"Reset {i} produced different position: " \
                    f"diff={position_diff:.9f}m from first reset"
            
            # Check all orientations are identical
            for i in range(1, len(orientations)):
                angle_diff = abs(orientations[i] - orientations[0])
                assert angle_diff < 1e-6, \
                    f"Reset {i} produced different orientation: " \
                    f"diff={angle_diff:.9f} rad from first reset"
        
        finally:
            sim.destroy_node()
    
    @given(robot_pos=valid_robot_position())
    @settings(max_examples=100, deadline=60000)
    def test_robot_initial_velocity_is_zero(self, robot_pos):
        """
        Property 1: Initialization Consistency - Initial Velocity
        
        When the simulation starts, the robot should have zero velocity
        (both linear and angular).
        
        **Validates: Requirements 1.3**
        """
        sim = MuJoCoSimulation(render_mode=None)
        
        try:
            # Reset simulation
            success = sim.reset(robot_position=robot_pos)
            assert success, "Simulation reset should succeed"
            
            # Get ground truth robot state
            robot_state = sim.get_ground_truth_robot_state()
            
            # Check velocities are zero (within small tolerance for numerical precision)
            assert abs(robot_state.linear_velocity) < 0.01, \
                f"Robot initial linear velocity {robot_state.linear_velocity:.4f} m/s should be ~0"
            assert abs(robot_state.angular_velocity) < 0.01, \
                f"Robot initial angular velocity {robot_state.angular_velocity:.4f} rad/s should be ~0"
        
        finally:
            sim.destroy_node()
    
    @given(
        robot_pos=valid_robot_position(),
        obstacle_configs=st.lists(valid_obstacle_config(), min_size=1, max_size=3, unique_by=lambda x: x['name'])
    )
    @settings(max_examples=100, deadline=60000)
    def test_simulation_is_initialized_after_reset(self, robot_pos, obstacle_configs):
        """
        Property 1: Initialization Consistency - Initialization Flag
        
        After a successful reset, the simulation should report that it is initialized.
        
        **Validates: Requirements 1.3**
        """
        sim = MuJoCoSimulation(render_mode=None)
        
        try:
            # Reset simulation
            success = sim.reset(robot_position=robot_pos, obstacle_configs=obstacle_configs)
            assert success, "Simulation reset should succeed"
            
            # Check initialization flag
            assert sim.is_initialized(), \
                "Simulation should report as initialized after successful reset"
            
            # Check step count is reset
            assert sim.get_step_count() == 0, \
                f"Step count should be 0 after reset, got {sim.get_step_count()}"
        
        finally:
            sim.destroy_node()
