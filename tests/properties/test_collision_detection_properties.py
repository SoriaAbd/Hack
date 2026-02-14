"""
Property-based tests for collision detection.

Feature: adaptnav-context-aware-warehouse-navigation
Property 3: Collision Detection
Validates: Requirements 1.5
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, assume, settings

try:
    import rclpy
    from adaptnav.simulation.mujoco_simulation import MuJoCoSimulation, MUJOCO_AVAILABLE
    from adaptnav.core.dynamic_obstacle import DynamicObstacle
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    MUJOCO_AVAILABLE = False


# Robot and obstacle dimensions (approximate bounding radii)
ROBOT_RADIUS = 0.3  # Robot is approximately 0.6m diameter
WORKER_RADIUS = 0.25  # Workers are approximately 0.5m diameter
FORKLIFT_RADIUS = 0.5  # Forklifts are approximately 1.0m diameter


# Custom strategies for generating test data
@st.composite
def valid_robot_position(draw):
    """Generate a valid robot spawn position (x, y, theta)."""
    x = draw(st.floats(min_value=-20.0, max_value=20.0))
    y = draw(st.floats(min_value=-20.0, max_value=20.0))
    theta = draw(st.floats(min_value=-np.pi, max_value=np.pi))
    return (x, y, theta)


@st.composite
def overlapping_obstacle_config(draw, robot_pos):
    """
    Generate an obstacle configuration that overlaps with the robot.
    
    Args:
        robot_pos: Robot position (x, y, theta)
    
    Returns:
        Obstacle configuration dict
    """
    robot_x, robot_y, _ = robot_pos
    
    # Choose obstacle type
    obstacle_name = draw(st.sampled_from(['worker_1', 'worker_2', 'forklift_1']))
    
    # Determine obstacle radius
    if 'worker' in obstacle_name:
        obstacle_radius = WORKER_RADIUS
    else:
        obstacle_radius = FORKLIFT_RADIUS
    
    # Generate position that overlaps with robot
    # Overlap means distance < robot_radius + obstacle_radius
    overlap_threshold = ROBOT_RADIUS + obstacle_radius
    
    # Generate a distance that ensures overlap (0 to 90% of overlap threshold)
    distance = draw(st.floats(min_value=0.0, max_value=overlap_threshold * 0.9))
    
    # Generate random angle
    angle = draw(st.floats(min_value=0, max_value=2 * np.pi))
    
    # Calculate obstacle position
    obs_x = robot_x + distance * np.cos(angle)
    obs_y = robot_y + distance * np.sin(angle)
    
    # Ensure obstacle is within warehouse bounds
    obs_x = np.clip(obs_x, -20.0, 20.0)
    obs_y = np.clip(obs_y, -20.0, 20.0)
    
    # Generate velocity (not critical for collision detection)
    vx = draw(st.floats(min_value=-2.0, max_value=2.0))
    vy = draw(st.floats(min_value=-2.0, max_value=2.0))
    
    return {
        'name': obstacle_name,
        'position': [obs_x, obs_y],
        'velocity': [vx, vy]
    }


@st.composite
def non_overlapping_obstacle_config(draw, robot_pos):
    """
    Generate an obstacle configuration that does NOT overlap with the robot.
    
    Args:
        robot_pos: Robot position (x, y, theta)
    
    Returns:
        Obstacle configuration dict
    """
    robot_x, robot_y, _ = robot_pos
    
    # Choose obstacle type
    obstacle_name = draw(st.sampled_from(['worker_1', 'worker_2', 'forklift_1']))
    
    # Determine obstacle radius
    if 'worker' in obstacle_name:
        obstacle_radius = WORKER_RADIUS
    else:
        obstacle_radius = FORKLIFT_RADIUS
    
    # Generate position that does NOT overlap with robot
    # No overlap means distance >= robot_radius + obstacle_radius + safety margin
    overlap_threshold = ROBOT_RADIUS + obstacle_radius
    safety_margin = 0.2  # 20cm safety margin to ensure clear separation
    
    # Generate a distance that ensures no overlap (at least threshold + margin)
    distance = draw(st.floats(min_value=overlap_threshold + safety_margin, max_value=15.0))
    
    # Generate random angle
    angle = draw(st.floats(min_value=0, max_value=2 * np.pi))
    
    # Calculate obstacle position
    obs_x = robot_x + distance * np.cos(angle)
    obs_y = robot_y + distance * np.sin(angle)
    
    # Ensure obstacle is within warehouse bounds
    assume(-20.0 <= obs_x <= 20.0)
    assume(-20.0 <= obs_y <= 20.0)
    
    # Generate velocity
    vx = draw(st.floats(min_value=-2.0, max_value=2.0))
    vy = draw(st.floats(min_value=-2.0, max_value=2.0))
    
    return {
        'name': obstacle_name,
        'position': [obs_x, obs_y],
        'velocity': [vx, vy]
    }


@pytest.mark.skipif(not ROS2_AVAILABLE or not MUJOCO_AVAILABLE,
                    reason="ROS 2 or MuJoCo not available")
@pytest.mark.property
class TestCollisionDetectionProperties:
    """Property-based tests for collision detection."""
    
    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup and teardown for each test."""
        if not rclpy.ok():
            rclpy.init()
        
        yield
    
    @given(robot_pos=valid_robot_position())
    @settings(max_examples=100, deadline=60000)
    def test_overlapping_objects_detected_as_collision(self, robot_pos):
        """
        Property 3: Collision Detection - Overlapping Objects
        
        For any two objects with overlapping bounding volumes, the physics
        engine should detect and report a collision.
        
        **Validates: Requirements 1.5**
        """
        sim = MuJoCoSimulation(render_mode=None)
        
        try:
            # Generate an overlapping obstacle configuration
            obstacle_config = overlapping_obstacle_config(robot_pos).example()
            
            # Reset simulation with robot and overlapping obstacle
            success = sim.reset(robot_position=robot_pos, obstacle_configs=[obstacle_config])
            assert success, "Simulation reset should succeed"
            
            # Step simulation to allow physics to settle and detect collision
            # Multiple steps to ensure physics engine processes contacts
            for _ in range(5):
                sim.step(dt=0.01)
            
            # Check if collision is detected
            collision_detected = sim.check_collision()
            
            # Get actual positions for debugging
            robot_state = sim.get_ground_truth_robot_state()
            obstacles = sim.get_ground_truth_obstacles()
            
            # Find the obstacle we placed
            obstacle = None
            name_to_id = {'worker_1': 1, 'worker_2': 2, 'forklift_1': 3}
            expected_id = name_to_id[obstacle_config['name']]
            for obs in obstacles:
                if obs.id == expected_id:
                    obstacle = obs
                    break
            
            assert obstacle is not None, f"Could not find obstacle {obstacle_config['name']}"
            
            # Calculate actual distance between robot and obstacle
            distance = np.linalg.norm(robot_state.position - obstacle.position)
            
            # Determine expected overlap
            if 'worker' in obstacle_config['name']:
                obstacle_radius = WORKER_RADIUS
            else:
                obstacle_radius = FORKLIFT_RADIUS
            
            overlap_threshold = ROBOT_RADIUS + obstacle_radius
            is_overlapping = distance < overlap_threshold
            
            # Assert collision is detected when objects overlap
            assert collision_detected or not is_overlapping, \
                f"Collision not detected for overlapping objects! " \
                f"Robot at ({robot_state.position[0]:.3f}, {robot_state.position[1]:.3f}), " \
                f"Obstacle {obstacle.id} at ({obstacle.position[0]:.3f}, {obstacle.position[1]:.3f}), " \
                f"Distance: {distance:.3f}m, Overlap threshold: {overlap_threshold:.3f}m, " \
                f"Is overlapping: {is_overlapping}"
        
        finally:
            sim.destroy_node()
    
    @given(robot_pos=valid_robot_position())
    @settings(max_examples=100, deadline=60000)
    def test_non_overlapping_objects_not_detected_as_collision(self, robot_pos):
        """
        Property 3: Collision Detection - Non-Overlapping Objects
        
        For any two objects with non-overlapping bounding volumes (with
        sufficient separation), the physics engine should NOT report a collision.
        
        **Validates: Requirements 1.5**
        """
        sim = MuJoCoSimulation(render_mode=None)
        
        try:
            # Generate a non-overlapping obstacle configuration
            obstacle_config = non_overlapping_obstacle_config(robot_pos).example()
            
            # Reset simulation with robot and non-overlapping obstacle
            success = sim.reset(robot_position=robot_pos, obstacle_configs=[obstacle_config])
            assert success, "Simulation reset should succeed"
            
            # Step simulation to allow physics to settle
            for _ in range(5):
                sim.step(dt=0.01)
            
            # Check if collision is detected
            collision_detected = sim.check_collision()
            
            # Get actual positions for debugging
            robot_state = sim.get_ground_truth_robot_state()
            obstacles = sim.get_ground_truth_obstacles()
            
            # Find the obstacle we placed
            obstacle = None
            name_to_id = {'worker_1': 1, 'worker_2': 2, 'forklift_1': 3}
            expected_id = name_to_id[obstacle_config['name']]
            for obs in obstacles:
                if obs.id == expected_id:
                    obstacle = obs
                    break
            
            assert obstacle is not None, f"Could not find obstacle {obstacle_config['name']}"
            
            # Calculate actual distance between robot and obstacle
            distance = np.linalg.norm(robot_state.position - obstacle.position)
            
            # Determine expected overlap
            if 'worker' in obstacle_config['name']:
                obstacle_radius = WORKER_RADIUS
            else:
                obstacle_radius = FORKLIFT_RADIUS
            
            overlap_threshold = ROBOT_RADIUS + obstacle_radius
            is_overlapping = distance < overlap_threshold
            
            # Assert no collision is detected when objects don't overlap
            assert not collision_detected or is_overlapping, \
                f"False collision detected for non-overlapping objects! " \
                f"Robot at ({robot_state.position[0]:.3f}, {robot_state.position[1]:.3f}), " \
                f"Obstacle {obstacle.id} at ({obstacle.position[0]:.3f}, {obstacle.position[1]:.3f}), " \
                f"Distance: {distance:.3f}m, Overlap threshold: {overlap_threshold:.3f}m, " \
                f"Is overlapping: {is_overlapping}"
        
        finally:
            sim.destroy_node()
    
    @given(
        robot_pos=valid_robot_position(),
        time_steps=st.integers(min_value=1, max_value=20)
    )
    @settings(max_examples=100, deadline=60000)
    def test_collision_detection_persists_during_overlap(self, robot_pos, time_steps):
        """
        Property 3: Collision Detection - Persistence
        
        When two objects are overlapping, the collision should be detected
        consistently across multiple simulation steps (not just once).
        
        **Validates: Requirements 1.5**
        """
        sim = MuJoCoSimulation(render_mode=None)
        
        try:
            # Generate an overlapping obstacle configuration
            obstacle_config = overlapping_obstacle_config(robot_pos).example()
            
            # Reset simulation with robot and overlapping obstacle
            success = sim.reset(robot_position=robot_pos, obstacle_configs=[obstacle_config])
            assert success, "Simulation reset should succeed"
            
            # Track collision detection across multiple steps
            collision_detections = []
            
            for step in range(time_steps):
                sim.step(dt=0.01)
                collision_detected = sim.check_collision()
                collision_detections.append(collision_detected)
            
            # Get final positions
            robot_state = sim.get_ground_truth_robot_state()
            obstacles = sim.get_ground_truth_obstacles()
            
            # Find the obstacle we placed
            obstacle = None
            name_to_id = {'worker_1': 1, 'worker_2': 2, 'forklift_1': 3}
            expected_id = name_to_id[obstacle_config['name']]
            for obs in obstacles:
                if obs.id == expected_id:
                    obstacle = obs
                    break
            
            # Calculate final distance
            distance = np.linalg.norm(robot_state.position - obstacle.position)
            
            # Determine if objects are still overlapping
            if 'worker' in obstacle_config['name']:
                obstacle_radius = WORKER_RADIUS
            else:
                obstacle_radius = FORKLIFT_RADIUS
            
            overlap_threshold = ROBOT_RADIUS + obstacle_radius
            is_overlapping = distance < overlap_threshold
            
            # If objects are overlapping, collision should be detected in most steps
            # (allowing for some physics settling time)
            if is_overlapping:
                collision_rate = sum(collision_detections) / len(collision_detections)
                assert collision_rate >= 0.5, \
                    f"Collision detection inconsistent for overlapping objects. " \
                    f"Detected in {sum(collision_detections)}/{len(collision_detections)} steps. " \
                    f"Final distance: {distance:.3f}m, Threshold: {overlap_threshold:.3f}m"
        
        finally:
            sim.destroy_node()
    
    @given(robot_pos=valid_robot_position())
    @settings(max_examples=100, deadline=60000)
    def test_collision_detection_with_multiple_obstacles(self, robot_pos):
        """
        Property 3: Collision Detection - Multiple Obstacles
        
        When the robot overlaps with any obstacle in a multi-obstacle
        environment, the collision should be detected.
        
        **Validates: Requirements 1.5**
        """
        sim = MuJoCoSimulation(render_mode=None)
        
        try:
            # Generate one overlapping obstacle and one non-overlapping obstacle
            overlapping_config = overlapping_obstacle_config(robot_pos).example()
            
            # Ensure we use different obstacle names
            available_names = ['worker_1', 'worker_2', 'forklift_1']
            available_names.remove(overlapping_config['name'])
            
            # Create a non-overlapping obstacle with a different name
            non_overlapping_config = {
                'name': available_names[0],
                'position': [robot_pos[0] + 10.0, robot_pos[1] + 10.0],
                'velocity': [0.0, 0.0]
            }
            
            obstacle_configs = [overlapping_config, non_overlapping_config]
            
            # Reset simulation
            success = sim.reset(robot_position=robot_pos, obstacle_configs=obstacle_configs)
            assert success, "Simulation reset should succeed"
            
            # Step simulation
            for _ in range(5):
                sim.step(dt=0.01)
            
            # Check if collision is detected
            collision_detected = sim.check_collision()
            
            # Get positions
            robot_state = sim.get_ground_truth_robot_state()
            obstacles = sim.get_ground_truth_obstacles()
            
            # Find the overlapping obstacle
            name_to_id = {'worker_1': 1, 'worker_2': 2, 'forklift_1': 3}
            overlapping_id = name_to_id[overlapping_config['name']]
            
            overlapping_obstacle = None
            for obs in obstacles:
                if obs.id == overlapping_id:
                    overlapping_obstacle = obs
                    break
            
            assert overlapping_obstacle is not None, \
                f"Could not find overlapping obstacle {overlapping_config['name']}"
            
            # Calculate distance to overlapping obstacle
            distance = np.linalg.norm(robot_state.position - overlapping_obstacle.position)
            
            # Determine expected overlap
            if 'worker' in overlapping_config['name']:
                obstacle_radius = WORKER_RADIUS
            else:
                obstacle_radius = FORKLIFT_RADIUS
            
            overlap_threshold = ROBOT_RADIUS + obstacle_radius
            is_overlapping = distance < overlap_threshold
            
            # Assert collision is detected
            assert collision_detected or not is_overlapping, \
                f"Collision not detected in multi-obstacle environment! " \
                f"Robot at ({robot_state.position[0]:.3f}, {robot_state.position[1]:.3f}), " \
                f"Overlapping obstacle at ({overlapping_obstacle.position[0]:.3f}, " \
                f"{overlapping_obstacle.position[1]:.3f}), " \
                f"Distance: {distance:.3f}m, Threshold: {overlap_threshold:.3f}m"
        
        finally:
            sim.destroy_node()
    
    @given(robot_pos=valid_robot_position())
    @settings(max_examples=100, deadline=60000)
    def test_no_collision_with_default_separated_obstacles(self, robot_pos):
        """
        Property 3: Collision Detection - Default Configuration
        
        When using default obstacle positions (which are separated from
        typical robot spawn positions), no collision should be detected
        initially.
        
        **Validates: Requirements 1.5**
        """
        sim = MuJoCoSimulation(render_mode=None)
        
        try:
            # Reset with default obstacles (they should be separated)
            success = sim.reset(robot_position=robot_pos)
            assert success, "Simulation reset should succeed"
            
            # Step simulation briefly
            for _ in range(3):
                sim.step(dt=0.01)
            
            # Get positions
            robot_state = sim.get_ground_truth_robot_state()
            obstacles = sim.get_ground_truth_obstacles()
            
            # Calculate minimum distance to any obstacle
            min_distance = float('inf')
            for obs in obstacles:
                distance = np.linalg.norm(robot_state.position - obs.position)
                min_distance = min(min_distance, distance)
            
            # Check collision detection
            collision_detected = sim.check_collision()
            
            # If minimum distance is large enough, no collision should be detected
            # Use the largest possible overlap threshold (robot + forklift)
            max_overlap_threshold = ROBOT_RADIUS + FORKLIFT_RADIUS
            
            if min_distance >= max_overlap_threshold + 0.2:  # 20cm safety margin
                assert not collision_detected, \
                    f"False collision detected with well-separated obstacles! " \
                    f"Minimum distance: {min_distance:.3f}m, " \
                    f"Max overlap threshold: {max_overlap_threshold:.3f}m"
        
        finally:
            sim.destroy_node()
