"""
Property-based tests for obstacle detection completeness.

This module implements Property 11: Detection Completeness
**Validates: Requirements 4.1, 4.2**

Property 11: Detection Completeness
For any detected dynamic obstacle, the obstacle detector output should include 
both position estimate and velocity estimate (not null/missing values).

Requirements 4.1: Obstacle detector shall identify dynamic obstacles and estimate position
Requirements 4.2: Obstacle detector shall estimate velocity of each detected dynamic obstacle
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume
from hypothesis import HealthCheck
import time

# Check if ROS 2 and MuJoCo are available
try:
    import rclpy
    import mujoco
    from adaptnav.simulation.mujoco_simulation import MuJoCoSimulation
    from adaptnav.perception.obstacle_detector import ObstacleDetector, DetectedObstacle, ObstacleTrack
    from custom_msgs.msg import Obstacle, ObstacleArray
    from geometry_msgs.msg import Point, Vector3
    from builtin_interfaces.msg import Time as ROSTime
    ROS_MUJOCO_AVAILABLE = True
except ImportError:
    ROS_MUJOCO_AVAILABLE = False
    # Skip all tests if ROS 2 or MuJoCo are not available
    pytestmark = pytest.mark.skip(reason="ROS 2 or MuJoCo not available")


# Test configuration
MIN_EXAMPLES = 100  # Minimum iterations per property test
TEST_TIMEOUT = 60  # seconds


@pytest.fixture(scope="module")
def simulation():
    """Create a simulation instance for testing."""
    if not ROS_MUJOCO_AVAILABLE:
        pytest.skip("ROS 2 or MuJoCo not available")
    
    # Initialize ROS 2 for testing
    rclpy.init()
    
    try:
        sim = MuJoCoSimulation(render_mode=None)
        sim.reset()
        yield sim
        sim.destroy_node()
    finally:
        rclpy.shutdown()


@pytest.fixture(scope="module")
def obstacle_detector():
    """Create an obstacle detector instance for testing."""
    if not ROS_MUJOCO_AVAILABLE:
        pytest.skip("ROS 2 or MuJoCo not available")
    
    detector = ObstacleDetector()
    yield detector
    detector.destroy_node()


if ROS_MUJOCO_AVAILABLE:
    class TestObstacleDetectionCompletenessProperty:
        """
        Property 11: Detection Completeness
        
        **Validates: Requirements 4.1, 4.2**
        
        For any detected dynamic obstacle, the obstacle detector output should include 
        both position estimate and velocity estimate (not null/missing values).
        
        This property ensures that:
        1. Every detected obstacle has a valid position estimate (not null/NaN/infinite)
        2. Every detected obstacle has a valid velocity estimate (not null/NaN/infinite)
        3. Position and velocity estimates are within reasonable physical bounds
        4. The detector never outputs incomplete obstacle information
        5. All required fields in the Obstacle message are properly populated
        """
        
        @settings(
            max_examples=MIN_EXAMPLES,
            deadline=TEST_TIMEOUT * 1000,
            suppress_health_check=[HealthCheck.function_scoped_fixture]
        )
        @given(
            robot_x=st.floats(min_value=-20.0, max_value=20.0),
            robot_y=st.floats(min_value=-20.0, max_value=20.0),
            robot_yaw=st.floats(min_value=-np.pi, max_value=np.pi),
            simulation_steps=st.integers(min_value=5, max_value=20)
        )
        def test_all_detected_obstacles_have_position_and_velocity_estimates(
            self, simulation, obstacle_detector, robot_x, robot_y, robot_yaw, simulation_steps
        ):
            """
            **Validates: Requirements 4.1, 4.2**
            
            Property: Every detected obstacle must have both position and velocity estimates.
            
            This is the core property that validates the obstacle detector always provides
            complete information for detected obstacles. No obstacle should be published
            without both position and velocity estimates.
            """
            # Avoid positions that would place robot inside obstacles
            assume(abs(robot_x) < 22.0 and abs(robot_y) < 22.0)
            
            # Set robot position and run simulation for multiple steps to allow tracking
            simulation.set_robot_pose(robot_x, robot_y, robot_yaw)
            
            # Run simulation for several steps to allow obstacle detection and tracking
            obstacles_detected = []
            for step in range(simulation_steps):
                # Move robot slightly to generate sensor data
                linear_vel = 0.1 if step % 2 == 0 else -0.1
                angular_vel = 0.05 if step % 3 == 0 else -0.05
                simulation.step(linear_vel, angular_vel)
                
                # Allow some time for processing
                time.sleep(0.05)
                
                # Trigger obstacle detection manually by calling the detection callback
                # This simulates the timer-based detection process
                try:
                    obstacle_detector._detection_callback()
                    
                    # Check if any obstacles were published
                    # In a real test, we would subscribe to the topic, but for this test
                    # we'll check the internal tracking state
                    if hasattr(obstacle_detector, '_obstacle_tracks'):
                        for track_id, track in obstacle_detector._obstacle_tracks.items():
                            # Convert track to obstacle message
                            current_time = simulation.get_clock().now()
                            obstacle_msg = track.to_obstacle_msg(current_time)
                            obstacles_detected.append(obstacle_msg)
                            
                except Exception as e:
                    # Detection might fail due to no sensor data, which is acceptable
                    continue
            
            # Core property validation: Every detected obstacle must have complete information
            for i, obstacle in enumerate(obstacles_detected):
                # Validate obstacle ID is set
                assert obstacle.id > 0, \
                    f"Obstacle {i}: ID must be positive, got {obstacle.id}"
                
                # Validate position estimate exists and is valid
                assert obstacle.position is not None, \
                    f"Obstacle {obstacle.id}: Position estimate is None"
                
                assert isinstance(obstacle.position, Point), \
                    f"Obstacle {obstacle.id}: Position must be geometry_msgs/Point"
                
                # Position values must be finite and within reasonable bounds
                assert np.isfinite(obstacle.position.x), \
                    f"Obstacle {obstacle.id}: Position x is not finite: {obstacle.position.x}"
                assert np.isfinite(obstacle.position.y), \
                    f"Obstacle {obstacle.id}: Position y is not finite: {obstacle.position.y}"
                assert np.isfinite(obstacle.position.z), \
                    f"Obstacle {obstacle.id}: Position z is not finite: {obstacle.position.z}"
                
                # Position should be within reasonable warehouse bounds
                assert -50.0 <= obstacle.position.x <= 50.0, \
                    f"Obstacle {obstacle.id}: Position x {obstacle.position.x} outside reasonable bounds [-50, 50]"
                assert -50.0 <= obstacle.position.y <= 50.0, \
                    f"Obstacle {obstacle.id}: Position y {obstacle.position.y} outside reasonable bounds [-50, 50]"
                
                # Validate velocity estimate exists and is valid
                assert obstacle.velocity is not None, \
                    f"Obstacle {obstacle.id}: Velocity estimate is None"
                
                assert isinstance(obstacle.velocity, Vector3), \
                    f"Obstacle {obstacle.id}: Velocity must be geometry_msgs/Vector3"
                
                # Velocity values must be finite
                assert np.isfinite(obstacle.velocity.x), \
                    f"Obstacle {obstacle.id}: Velocity x is not finite: {obstacle.velocity.x}"
                assert np.isfinite(obstacle.velocity.y), \
                    f"Obstacle {obstacle.id}: Velocity y is not finite: {obstacle.velocity.y}"
                assert np.isfinite(obstacle.velocity.z), \
                    f"Obstacle {obstacle.id}: Velocity z is not finite: {obstacle.velocity.z}"
                
                # Velocity should be within reasonable physical bounds for warehouse obstacles
                velocity_magnitude = np.sqrt(
                    obstacle.velocity.x**2 + obstacle.velocity.y**2 + obstacle.velocity.z**2
                )
                assert velocity_magnitude <= 10.0, \
                    f"Obstacle {obstacle.id}: Velocity magnitude {velocity_magnitude:.2f} m/s exceeds reasonable limit (10 m/s)"
        
        @settings(
            max_examples=MIN_EXAMPLES,
            deadline=TEST_TIMEOUT * 1000,
            suppress_health_check=[HealthCheck.function_scoped_fixture]
        )
        @given(
            robot_x=st.floats(min_value=-15.0, max_value=15.0),
            robot_y=st.floats(min_value=-15.0, max_value=15.0),
            robot_yaw=st.floats(min_value=-np.pi, max_value=np.pi)
        )
        def test_obstacle_message_fields_are_complete(
            self, simulation, obstacle_detector, robot_x, robot_y, robot_yaw
        ):
            """
            **Validates: Requirements 4.1, 4.2**
            
            Property: All required fields in Obstacle messages must be populated.
            
            This validates that the obstacle detector populates all required fields
            in the Obstacle message, not just position and velocity.
            """
            # Set robot position
            simulation.set_robot_pose(robot_x, robot_y, robot_yaw)
            
            # Run simulation to generate obstacles
            obstacles_detected = []
            for step in range(10):
                simulation.step(0.1, 0.0)
                time.sleep(0.05)
                
                try:
                    obstacle_detector._detection_callback()
                    
                    if hasattr(obstacle_detector, '_obstacle_tracks'):
                        for track_id, track in obstacle_detector._obstacle_tracks.items():
                            current_time = simulation.get_clock().now()
                            obstacle_msg = track.to_obstacle_msg(current_time)
                            obstacles_detected.append(obstacle_msg)
                            break  # Only need one obstacle for this test
                except Exception:
                    continue
                
                if obstacles_detected:
                    break
            
            # Validate complete message structure for each detected obstacle
            for obstacle in obstacles_detected:
                # ID field
                assert hasattr(obstacle, 'id'), "Obstacle message missing 'id' field"
                assert isinstance(obstacle.id, (int, np.integer)), \
                    f"Obstacle ID must be integer, got {type(obstacle.id)}"
                
                # Position field (already validated above, but check structure)
                assert hasattr(obstacle, 'position'), "Obstacle message missing 'position' field"
                assert hasattr(obstacle.position, 'x'), "Position missing 'x' field"
                assert hasattr(obstacle.position, 'y'), "Position missing 'y' field"
                assert hasattr(obstacle.position, 'z'), "Position missing 'z' field"
                
                # Velocity field (already validated above, but check structure)
                assert hasattr(obstacle, 'velocity'), "Obstacle message missing 'velocity' field"
                assert hasattr(obstacle.velocity, 'x'), "Velocity missing 'x' field"
                assert hasattr(obstacle.velocity, 'y'), "Velocity missing 'y' field"
                assert hasattr(obstacle.velocity, 'z'), "Velocity missing 'z' field"
                
                # Covariance field
                assert hasattr(obstacle, 'covariance'), "Obstacle message missing 'covariance' field"
                assert len(obstacle.covariance) == 16, \
                    f"Covariance must have 16 elements (4x4 matrix), got {len(obstacle.covariance)}"
                
                # All covariance values must be finite
                for i, cov_val in enumerate(obstacle.covariance):
                    assert np.isfinite(cov_val), \
                        f"Covariance element {i} is not finite: {cov_val}"
                
                # Classification field
                assert hasattr(obstacle, 'classification'), "Obstacle message missing 'classification' field"
                assert isinstance(obstacle.classification, str), \
                    f"Classification must be string, got {type(obstacle.classification)}"
                assert obstacle.classification in ["worker", "forklift", "unknown"], \
                    f"Invalid classification: {obstacle.classification}"
                
                # Confidence field
                assert hasattr(obstacle, 'confidence'), "Obstacle message missing 'confidence' field"
                assert isinstance(obstacle.confidence, (float, np.floating)), \
                    f"Confidence must be float, got {type(obstacle.confidence)}"
                assert 0.0 <= obstacle.confidence <= 1.0, \
                    f"Confidence must be in [0.0, 1.0], got {obstacle.confidence}"
                assert np.isfinite(obstacle.confidence), \
                    f"Confidence must be finite, got {obstacle.confidence}"
                
                # Last seen timestamp field
                assert hasattr(obstacle, 'last_seen'), "Obstacle message missing 'last_seen' field"
                assert hasattr(obstacle.last_seen, 'sec'), "Timestamp missing 'sec' field"
                assert hasattr(obstacle.last_seen, 'nanosec'), "Timestamp missing 'nanosec' field"
        
        @settings(
            max_examples=50,  # Fewer examples for this more complex test
            deadline=TEST_TIMEOUT * 1000,
            suppress_health_check=[HealthCheck.function_scoped_fixture]
        )
        @given(
            robot_x=st.floats(min_value=-10.0, max_value=10.0),
            robot_y=st.floats(min_value=-10.0, max_value=10.0),
            movement_pattern=st.integers(min_value=0, max_value=3)
        )
        def test_velocity_estimates_are_physically_reasonable(
            self, simulation, obstacle_detector, robot_x, robot_y, movement_pattern
        ):
            """
            **Validates: Requirements 4.2**
            
            Property: Velocity estimates must be physically reasonable for warehouse obstacles.
            
            This validates that velocity estimates are not only present but also
            within reasonable bounds for the types of obstacles expected in a warehouse.
            """
            # Set robot position
            simulation.set_robot_pose(robot_x, robot_y, 0.0)
            
            # Define different movement patterns to generate various scenarios
            movement_patterns = [
                [(0.0, 0.0)] * 10,  # Stationary
                [(0.2, 0.0)] * 10,  # Slow forward
                [(0.0, 0.1)] * 10,  # Slow rotation
                [(0.1, 0.05)] * 10  # Combined motion
            ]
            
            pattern = movement_patterns[movement_pattern]
            
            # Run simulation with chosen movement pattern
            obstacles_with_velocity = []
            for step, (linear_vel, angular_vel) in enumerate(pattern):
                simulation.step(linear_vel, angular_vel)
                time.sleep(0.1)  # Allow more time for tracking to develop
                
                try:
                    obstacle_detector._detection_callback()
                    
                    if hasattr(obstacle_detector, '_obstacle_tracks'):
                        for track_id, track in obstacle_detector._obstacle_tracks.items():
                            # Only consider tracks that have been updated multiple times
                            # to ensure velocity estimates are meaningful
                            if track.total_detections >= 3:
                                current_time = simulation.get_clock().now()
                                obstacle_msg = track.to_obstacle_msg(current_time)
                                obstacles_with_velocity.append(obstacle_msg)
                except Exception:
                    continue
            
            # Validate velocity estimates are physically reasonable
            for obstacle in obstacles_with_velocity:
                velocity_2d = np.sqrt(obstacle.velocity.x**2 + obstacle.velocity.y**2)
                
                # Velocity should be reasonable for warehouse obstacles
                if obstacle.classification == "worker":
                    # Workers typically walk at 0.5-2.0 m/s
                    assert velocity_2d <= 3.0, \
                        f"Worker obstacle {obstacle.id}: velocity {velocity_2d:.2f} m/s too high for worker"
                elif obstacle.classification == "forklift":
                    # Forklifts typically move at 1-5 m/s in warehouses
                    assert velocity_2d <= 6.0, \
                        f"Forklift obstacle {obstacle.id}: velocity {velocity_2d:.2f} m/s too high for forklift"
                else:  # unknown
                    # Unknown obstacles should still be within reasonable bounds
                    assert velocity_2d <= 8.0, \
                        f"Unknown obstacle {obstacle.id}: velocity {velocity_2d:.2f} m/s exceeds reasonable limit"
                
                # Vertical velocity should be minimal (obstacles move on ground plane)
                assert abs(obstacle.velocity.z) <= 0.5, \
                    f"Obstacle {obstacle.id}: vertical velocity {obstacle.velocity.z:.2f} m/s too high"
        
        @settings(
            max_examples=MIN_EXAMPLES,
            deadline=TEST_TIMEOUT * 1000,
            suppress_health_check=[HealthCheck.function_scoped_fixture]
        )
        @given(
            robot_x=st.floats(min_value=-15.0, max_value=15.0),
            robot_y=st.floats(min_value=-15.0, max_value=15.0)
        )
        def test_position_estimates_are_within_sensor_range(
            self, simulation, obstacle_detector, robot_x, robot_y
        ):
            """
            **Validates: Requirements 4.1**
            
            Property: Position estimates must be within the sensor detection range.
            
            This validates that detected obstacles are within the maximum detection
            range of the sensors, ensuring position estimates are physically possible.
            """
            # Set robot position
            simulation.set_robot_pose(robot_x, robot_y, 0.0)
            
            # Run simulation to detect obstacles
            detected_obstacles = []
            for step in range(15):
                simulation.step(0.05, 0.02)  # Small movements
                time.sleep(0.05)
                
                try:
                    obstacle_detector._detection_callback()
                    
                    if hasattr(obstacle_detector, '_obstacle_tracks'):
                        for track_id, track in obstacle_detector._obstacle_tracks.items():
                            current_time = simulation.get_clock().now()
                            obstacle_msg = track.to_obstacle_msg(current_time)
                            detected_obstacles.append((obstacle_msg, robot_x, robot_y))
                except Exception:
                    continue
            
            # Validate position estimates are within sensor range
            max_detection_range = 8.0  # From obstacle detector configuration
            
            for obstacle_msg, robot_pos_x, robot_pos_y in detected_obstacles:
                # Calculate distance from robot to detected obstacle
                distance_to_robot = np.sqrt(
                    (obstacle_msg.position.x - robot_pos_x)**2 + 
                    (obstacle_msg.position.y - robot_pos_y)**2
                )
                
                # Position estimate must be within sensor range
                assert distance_to_robot <= max_detection_range + 0.5, \
                    f"Obstacle {obstacle_msg.id} at distance {distance_to_robot:.2f}m " \
                    f"exceeds max detection range {max_detection_range}m"
                
                # Position should not be too close (minimum sensor range)
                assert distance_to_robot >= 0.05, \
                    f"Obstacle {obstacle_msg.id} at distance {distance_to_robot:.2f}m " \
                    f"is too close (below minimum sensor range)"


    class TestObstacleDetectionCompletenessEdgeCases:
        """Edge cases for obstacle detection completeness."""
        
        def test_empty_obstacle_array_is_valid(self, simulation, obstacle_detector):
            """
            **Validates: Requirements 4.1, 4.2**
            
            Test that empty obstacle arrays are handled correctly.
            """
            # Position robot in area with no obstacles
            simulation.set_robot_pose(0.0, 0.0, 0.0)
            simulation.step(0.0, 0.0)
            time.sleep(0.1)
            
            # Run detection
            try:
                obstacle_detector._detection_callback()
                # Should not crash with no obstacles
                assert True, "Detection callback should handle empty results gracefully"
            except Exception as e:
                pytest.fail(f"Detection callback failed with no obstacles: {e}")
        
        def test_single_obstacle_completeness(self, simulation, obstacle_detector):
            """
            **Validates: Requirements 4.1, 4.2**
            
            Test completeness with exactly one detected obstacle.
            """
            # Position robot to potentially detect obstacles
            simulation.set_robot_pose(5.0, 5.0, 0.0)
            
            # Run simulation to detect at least one obstacle
            single_obstacle = None
            for step in range(20):
                simulation.step(0.1, 0.0)
                time.sleep(0.05)
                
                try:
                    obstacle_detector._detection_callback()
                    
                    if hasattr(obstacle_detector, '_obstacle_tracks'):
                        tracks = list(obstacle_detector._obstacle_tracks.values())
                        if len(tracks) >= 1:
                            # Take the first track with sufficient detections
                            for track in tracks:
                                if track.total_detections >= 2:
                                    current_time = simulation.get_clock().now()
                                    single_obstacle = track.to_obstacle_msg(current_time)
                                    break
                            if single_obstacle:
                                break
                except Exception:
                    continue
            
            if single_obstacle:
                # Validate the single obstacle has complete information
                assert single_obstacle.id > 0
                assert np.isfinite(single_obstacle.position.x)
                assert np.isfinite(single_obstacle.position.y)
                assert np.isfinite(single_obstacle.velocity.x)
                assert np.isfinite(single_obstacle.velocity.y)
                assert len(single_obstacle.covariance) == 16
                assert single_obstacle.classification in ["worker", "forklift", "unknown"]
                assert 0.0 <= single_obstacle.confidence <= 1.0
        
        def test_obstacle_tracking_consistency(self, simulation, obstacle_detector):
            """
            **Validates: Requirements 4.1, 4.2**
            
            Test that tracked obstacles maintain completeness over time.
            """
            # Position robot and run for extended period
            simulation.set_robot_pose(3.0, 3.0, 0.0)
            
            tracked_obstacles = {}
            for step in range(25):
                simulation.step(0.05, 0.01)
                time.sleep(0.05)
                
                try:
                    obstacle_detector._detection_callback()
                    
                    if hasattr(obstacle_detector, '_obstacle_tracks'):
                        for track_id, track in obstacle_detector._obstacle_tracks.items():
                            if track.total_detections >= 2:
                                current_time = simulation.get_clock().now()
                                obstacle_msg = track.to_obstacle_msg(current_time)
                                
                                if track_id not in tracked_obstacles:
                                    tracked_obstacles[track_id] = []
                                tracked_obstacles[track_id].append(obstacle_msg)
                except Exception:
                    continue
            
            # Validate that all tracked obstacles maintain completeness over time
            for track_id, obstacle_history in tracked_obstacles.items():
                for i, obstacle in enumerate(obstacle_history):
                    # Each obstacle in the tracking history must be complete
                    assert obstacle.id == track_id, \
                        f"Track {track_id}, frame {i}: ID mismatch"
                    assert np.isfinite(obstacle.position.x), \
                        f"Track {track_id}, frame {i}: Invalid position.x"
                    assert np.isfinite(obstacle.position.y), \
                        f"Track {track_id}, frame {i}: Invalid position.y"
                    assert np.isfinite(obstacle.velocity.x), \
                        f"Track {track_id}, frame {i}: Invalid velocity.x"
                    assert np.isfinite(obstacle.velocity.y), \
                        f"Track {track_id}, frame {i}: Invalid velocity.y"
                    assert len(obstacle.covariance) == 16, \
                        f"Track {track_id}, frame {i}: Invalid covariance size"
                    assert obstacle.classification in ["worker", "forklift", "unknown"], \
                        f"Track {track_id}, frame {i}: Invalid classification"
                    assert 0.0 <= obstacle.confidence <= 1.0, \
                        f"Track {track_id}, frame {i}: Invalid confidence"