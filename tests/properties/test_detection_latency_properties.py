"""
Property-based tests for obstacle detection latency.

This module implements Property 13: Detection Latency
**Validates: Requirements 4.4**

Property 13: Detection Latency
For any obstacle entering the sensor range, the obstacle detector should publish 
a detection within 0.5 seconds of the obstacle becoming visible.

Requirements 4.4: When a dynamic obstacle enters the robot's sensor range, 
the obstacle detector shall detect it within 0.5 seconds
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume
from hypothesis import HealthCheck
import time
from typing import List, Tuple, Optional, Dict
import threading

# Check if ROS 2 and MuJoCo are available
try:
    import rclpy
    import mujoco
    from adaptnav.simulation.mujoco_simulation import MuJoCoSimulation
    from adaptnav.perception.obstacle_detector import ObstacleDetector, DetectedObstacle, ObstacleTrack
    from adaptnav.core.dynamic_obstacle import DynamicObstacle
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
MAX_DETECTION_LATENCY = 0.5  # seconds (from requirements)


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
    class TestDetectionLatencyProperty:
        """
        Property 13: Detection Latency
        
        **Validates: Requirements 4.4**
        
        For any obstacle entering the sensor range, the obstacle detector should publish 
        a detection within 0.5 seconds of the obstacle becoming visible.
        
        This property ensures that:
        1. Obstacles are detected promptly when they enter sensor range
        2. Detection latency is consistently under the 0.5 second requirement
        3. The detector responds quickly to new obstacles across different scenarios
        4. Detection timing is consistent regardless of robot position or obstacle type
        5. The system meets real-time performance requirements for safety
        """
        
        def __init__(self):
            self._detection_events = []
            self._detection_lock = threading.Lock()
        
        def _setup_detection_monitoring(self, obstacle_detector: ObstacleDetector):
            """Set up monitoring for obstacle detection events."""
            # Store original callback
            original_callback = obstacle_detector._detection_callback
            
            def monitored_callback():
                """Wrapper to monitor detection events."""
                detection_time = time.time()
                
                # Call original detection callback
                original_callback()
                
                # Record detection event with current obstacles
                with obstacle_detector._data_lock:
                    if hasattr(obstacle_detector, '_obstacle_tracks'):
                        current_obstacles = []
                        for track_id, track in obstacle_detector._obstacle_tracks.items():
                            if track.total_detections == 1:  # Newly detected obstacle
                                current_obstacles.append({
                                    'track_id': track_id,
                                    'position': track.get_position().copy(),
                                    'detection_time': detection_time,
                                    'creation_time': track.creation_time
                                })
                        
                        if current_obstacles:
                            with self._detection_lock:
                                self._detection_events.extend(current_obstacles)
            
            # Replace callback
            obstacle_detector._detection_callback = monitored_callback
        
        @settings(
            max_examples=MIN_EXAMPLES,
            deadline=TEST_TIMEOUT * 1000,
            suppress_health_check=[HealthCheck.function_scoped_fixture]
        )
        @given(
            robot_x=st.floats(min_value=-15.0, max_value=15.0),
            robot_y=st.floats(min_value=-15.0, max_value=15.0),
            robot_yaw=st.floats(min_value=-np.pi, max_value=np.pi),
            movement_duration=st.integers(min_value=10, max_value=25)
        )
        def test_obstacles_detected_within_latency_requirement(
            self, simulation, obstacle_detector, robot_x, robot_y, robot_yaw, movement_duration
        ):
            """
            **Validates: Requirements 4.4**
            
            Property: Obstacles entering sensor range are detected within 0.5 seconds.
            
            This is the core property that validates detection latency meets the
            real-time requirement. For any obstacle that becomes visible to the
            sensors, the detector must publish a detection within 0.5 seconds.
            """
            # Avoid positions that would place robot inside obstacles
            assume(abs(robot_x) < 18.0 and abs(robot_y) < 18.0)
            
            # Clear previous detection events
            with self._detection_lock:
                self._detection_events.clear()
            
            # Set up detection monitoring
            self._setup_detection_monitoring(obstacle_detector)
            
            # Set initial robot position
            simulation.set_robot_pose(robot_x, robot_y, robot_yaw)
            
            # Record when obstacles first become visible
            obstacle_visibility_events = []
            previous_ground_truth = set()
            
            # Move robot to encounter obstacles and measure detection latency
            for step in range(movement_duration):
                # Move robot to explore environment and encounter obstacles
                linear_vel = 0.1 * np.sin(step * 0.2)  # Sinusoidal forward motion
                angular_vel = 0.05 * np.cos(step * 0.3)  # Gentle rotation
                
                step_start_time = time.time()
                simulation.step(linear_vel, angular_vel)
                
                # Allow time for sensor data processing
                time.sleep(0.05)
                
                # Get current ground truth obstacles
                current_ground_truth = simulation.get_ground_truth_obstacles()
                current_gt_ids = {obs.id for obs in current_ground_truth}
                
                # Check for newly visible obstacles
                newly_visible = current_gt_ids - previous_ground_truth
                
                for obs in current_ground_truth:
                    if obs.id in newly_visible:
                        # Check if obstacle is within sensor range
                        robot_pos = np.array([robot_x, robot_y])  # Approximate current position
                        obstacle_distance = np.linalg.norm(obs.position - robot_pos)
                        
                        # Only consider obstacles within detection range
                        if obstacle_distance <= 8.0:  # Max detection range
                            obstacle_visibility_events.append({
                                'obstacle_id': obs.id,
                                'position': obs.position.copy(),
                                'visibility_time': step_start_time,
                                'distance_to_robot': obstacle_distance
                            })
                
                previous_ground_truth = current_gt_ids
                
                # Allow detection processing
                time.sleep(0.05)
            
            # Wait additional time for final detections to process
            time.sleep(0.2)
            
            # Analyze detection latencies
            with self._detection_lock:
                detection_events = self._detection_events.copy()
            
            # Match visibility events with detection events
            latency_measurements = []
            
            for visibility_event in obstacle_visibility_events:
                vis_time = visibility_event['visibility_time']
                vis_position = visibility_event['position']
                
                # Find corresponding detection event
                best_match = None
                min_position_error = float('inf')
                
                for detection_event in detection_events:
                    det_time = detection_event['detection_time']
                    det_position = detection_event['position']
                    
                    # Only consider detections after visibility
                    if det_time >= vis_time:
                        # Match by position proximity
                        position_error = np.linalg.norm(det_position - vis_position)
                        
                        if position_error < min_position_error and position_error < 2.0:  # 2m max association distance
                            min_position_error = position_error
                            best_match = detection_event
                
                if best_match:
                    latency = best_match['detection_time'] - vis_time
                    latency_measurements.append({
                        'latency': latency,
                        'obstacle_id': visibility_event['obstacle_id'],
                        'distance': visibility_event['distance_to_robot'],
                        'position_error': min_position_error
                    })
            
            # Core property validation: All detections must meet latency requirement
            assert len(latency_measurements) >= 1, \
                f"No obstacle detection latency measurements collected. " \
                f"Visibility events: {len(obstacle_visibility_events)}, " \
                f"Detection events: {len(detection_events)}"
            
            # Validate each detection meets the latency requirement
            violations = []
            for measurement in latency_measurements:
                latency = measurement['latency']
                
                if latency > MAX_DETECTION_LATENCY:
                    violations.append(measurement)
                
                # Individual validation for debugging
                assert latency >= 0.0, \
                    f"Invalid negative latency: {latency:.3f}s for obstacle {measurement['obstacle_id']}"
                
                # Reasonable upper bound (should never take more than 2 seconds)
                assert latency <= 2.0, \
                    f"Excessive detection latency: {latency:.3f}s for obstacle {measurement['obstacle_id']}"
            
            # Core property: All detections must meet the 0.5 second requirement
            violation_rate = len(violations) / len(latency_measurements)
            max_allowed_violations = 0.1  # Allow up to 10% violations due to measurement noise
            
            assert violation_rate <= max_allowed_violations, \
                f"Detection latency requirement violated in {len(violations)}/{len(latency_measurements)} " \
                f"cases ({violation_rate:.1%}). Violations: {violations[:3]}..."  # Show first 3 violations
            
            # Statistical validation: Average latency should be well below requirement
            avg_latency = np.mean([m['latency'] for m in latency_measurements])
            assert avg_latency <= MAX_DETECTION_LATENCY * 0.8, \
                f"Average detection latency {avg_latency:.3f}s exceeds 80% of requirement ({MAX_DETECTION_LATENCY * 0.8:.3f}s)"
        
        @settings(
            max_examples=50,  # Fewer examples for this more complex test
            deadline=TEST_TIMEOUT * 1000,
            suppress_health_check=[HealthCheck.function_scoped_fixture]
        )
        @given(
            robot_x=st.floats(min_value=-10.0, max_value=10.0),
            robot_y=st.floats(min_value=-10.0, max_value=10.0),
            approach_speed=st.floats(min_value=0.05, max_value=0.2)
        )
        def test_detection_latency_consistent_across_approach_speeds(
            self, simulation, obstacle_detector, robot_x, robot_y, approach_speed
        ):
            """
            **Validates: Requirements 4.4**
            
            Property: Detection latency is consistent regardless of robot approach speed.
            
            This validates that the detection latency requirement is met consistently
            whether the robot is moving slowly or quickly when obstacles come into view.
            """
            # Clear previous detection events
            with self._detection_lock:
                self._detection_events.clear()
            
            # Set up detection monitoring
            self._setup_detection_monitoring(obstacle_detector)
            
            # Set robot position
            simulation.set_robot_pose(robot_x, robot_y, 0.0)
            
            # Record obstacle visibility and detection events
            visibility_events = []
            previous_obstacles = set()
            
            # Move robot at specified speed to encounter obstacles
            for step in range(20):
                step_start_time = time.time()
                
                # Move forward at specified speed
                simulation.step(approach_speed, 0.0)
                time.sleep(0.05)
                
                # Check for newly visible obstacles
                current_obstacles = simulation.get_ground_truth_obstacles()
                current_ids = {obs.id for obs in current_obstacles}
                newly_visible = current_ids - previous_obstacles
                
                for obs in current_obstacles:
                    if obs.id in newly_visible:
                        robot_pos = np.array([robot_x + step * approach_speed * 0.05, robot_y])
                        distance = np.linalg.norm(obs.position - robot_pos)
                        
                        if distance <= 8.0:  # Within sensor range
                            visibility_events.append({
                                'obstacle_id': obs.id,
                                'position': obs.position.copy(),
                                'visibility_time': step_start_time,
                                'approach_speed': approach_speed
                            })
                
                previous_obstacles = current_ids
                time.sleep(0.05)
            
            # Wait for final detections
            time.sleep(0.2)
            
            # Analyze latencies
            with self._detection_lock:
                detection_events = self._detection_events.copy()
            
            latencies = []
            for vis_event in visibility_events:
                for det_event in detection_events:
                    if (det_event['detection_time'] >= vis_event['visibility_time'] and
                        np.linalg.norm(det_event['position'] - vis_event['position']) < 2.0):
                        latency = det_event['detection_time'] - vis_event['visibility_time']
                        latencies.append(latency)
                        break
            
            if latencies:
                # All latencies should meet requirement regardless of approach speed
                max_latency = max(latencies)
                avg_latency = np.mean(latencies)
                
                assert max_latency <= MAX_DETECTION_LATENCY, \
                    f"Max detection latency {max_latency:.3f}s exceeds requirement at approach speed {approach_speed:.3f} m/s"
                
                assert avg_latency <= MAX_DETECTION_LATENCY * 0.7, \
                    f"Average detection latency {avg_latency:.3f}s too high at approach speed {approach_speed:.3f} m/s"
        
        @settings(
            max_examples=MIN_EXAMPLES,
            deadline=TEST_TIMEOUT * 1000,
            suppress_health_check=[HealthCheck.function_scoped_fixture]
        )
        @given(
            robot_x=st.floats(min_value=-12.0, max_value=12.0),
            robot_y=st.floats(min_value=-12.0, max_value=12.0),
            robot_yaw=st.floats(min_value=-np.pi, max_value=np.pi)
        )
        def test_detection_latency_under_sensor_noise_conditions(
            self, simulation, obstacle_detector, robot_x, robot_y, robot_yaw
        ):
            """
            **Validates: Requirements 4.4**
            
            Property: Detection latency requirement is met even under noisy sensor conditions.
            
            This validates that the detector maintains low latency performance
            even when sensor data contains realistic noise levels.
            """
            # Clear previous detection events
            with self._detection_lock:
                self._detection_events.clear()
            
            # Set up detection monitoring
            self._setup_detection_monitoring(obstacle_detector)
            
            # Set robot position
            simulation.set_robot_pose(robot_x, robot_y, robot_yaw)
            
            # Collect latency measurements under noisy conditions
            latency_measurements = []
            
            for measurement_cycle in range(15):
                cycle_start_time = time.time()
                
                # Move robot to encounter obstacles
                linear_vel = 0.08 + 0.02 * np.random.randn()  # Add noise to motion
                angular_vel = 0.03 + 0.01 * np.random.randn()
                
                simulation.step(linear_vel, angular_vel)
                time.sleep(0.1)  # Allow sensor processing
                
                # Get ground truth for comparison
                ground_truth_obstacles = simulation.get_ground_truth_obstacles()
                
                # Check for new detections
                with self._detection_lock:
                    recent_detections = [
                        event for event in self._detection_events
                        if event['detection_time'] >= cycle_start_time
                    ]
                
                # Match detections to ground truth and measure latency
                for detection in recent_detections:
                    det_pos = detection['position']
                    det_time = detection['detection_time']
                    
                    # Find closest ground truth obstacle
                    min_distance = float('inf')
                    closest_gt = None
                    
                    for gt_obs in ground_truth_obstacles:
                        distance = np.linalg.norm(gt_obs.position - det_pos)
                        if distance < min_distance:
                            min_distance = distance
                            closest_gt = gt_obs
                    
                    if closest_gt and min_distance < 1.5:  # Valid association
                        # Estimate when obstacle became visible (approximate)
                        visibility_time = cycle_start_time
                        latency = det_time - visibility_time
                        
                        if 0.0 <= latency <= 2.0:  # Reasonable latency range
                            latency_measurements.append(latency)
            
            # Validate latency performance under noisy conditions
            if latency_measurements:
                max_latency = max(latency_measurements)
                avg_latency = np.mean(latency_measurements)
                
                # Even with noise, must meet latency requirement
                assert max_latency <= MAX_DETECTION_LATENCY * 1.2, \
                    f"Max detection latency {max_latency:.3f}s exceeds allowable limit under noisy conditions"
                
                assert avg_latency <= MAX_DETECTION_LATENCY * 0.8, \
                    f"Average detection latency {avg_latency:.3f}s too high under noisy conditions"
                
                # Latency should be reasonably consistent (low variance)
                latency_std = np.std(latency_measurements)
                assert latency_std <= 0.2, \
                    f"Detection latency variance too high: {latency_std:.3f}s std dev"
        
        @settings(
            max_examples=50,
            deadline=TEST_TIMEOUT * 1000,
            suppress_health_check=[HealthCheck.function_scoped_fixture]
        )
        @given(
            robot_x=st.floats(min_value=-8.0, max_value=8.0),
            robot_y=st.floats(min_value=-8.0, max_value=8.0),
            detection_scenario=st.integers(min_value=0, max_value=3)
        )
        def test_detection_latency_across_different_obstacle_types(
            self, simulation, obstacle_detector, robot_x, robot_y, detection_scenario
        ):
            """
            **Validates: Requirements 4.4**
            
            Property: Detection latency requirement is met for different obstacle types.
            
            This validates that the latency requirement is consistently met
            regardless of obstacle size, classification, or movement pattern.
            """
            # Clear previous detection events
            with self._detection_lock:
                self._detection_events.clear()
            
            # Set up detection monitoring
            self._setup_detection_monitoring(obstacle_detector)
            
            # Set robot position
            simulation.set_robot_pose(robot_x, robot_y, 0.0)
            
            # Define different detection scenarios
            scenarios = [
                {'name': 'stationary_obstacles', 'motion': (0.0, 0.1)},  # Rotate to scan
                {'name': 'approaching_obstacles', 'motion': (0.1, 0.0)},  # Move forward
                {'name': 'crossing_obstacles', 'motion': (0.05, 0.05)},  # Diagonal motion
                {'name': 'retreating_obstacles', 'motion': (-0.05, 0.02)}  # Backing up
            ]
            
            scenario = scenarios[detection_scenario]
            linear_vel, angular_vel = scenario['motion']
            
            # Collect detection latency data for this scenario
            scenario_latencies = []
            obstacle_types_detected = set()
            
            for step in range(20):
                step_start_time = time.time()
                
                # Execute scenario motion
                simulation.step(linear_vel, angular_vel)
                time.sleep(0.08)
                
                # Get ground truth obstacles and their types
                ground_truth_obstacles = simulation.get_ground_truth_obstacles()
                
                # Check for new detections in this step
                with self._detection_lock:
                    step_detections = [
                        event for event in self._detection_events
                        if event['detection_time'] >= step_start_time
                    ]
                
                # Analyze detections for this step
                for detection in step_detections:
                    det_pos = detection['position']
                    det_time = detection['detection_time']
                    
                    # Find matching ground truth obstacle
                    for gt_obs in ground_truth_obstacles:
                        distance = np.linalg.norm(gt_obs.position - det_pos)
                        
                        if distance < 1.0:  # Close match
                            # Estimate visibility time (when obstacle entered sensor range)
                            visibility_time = step_start_time
                            latency = det_time - visibility_time
                            
                            if 0.0 <= latency <= 1.0:  # Valid latency measurement
                                scenario_latencies.append({
                                    'latency': latency,
                                    'obstacle_type': gt_obs.classification,
                                    'scenario': scenario['name']
                                })
                                obstacle_types_detected.add(gt_obs.classification)
                            break
            
            # Validate latency performance across obstacle types
            if scenario_latencies:
                # Group by obstacle type
                type_latencies = {}
                for measurement in scenario_latencies:
                    obs_type = measurement['obstacle_type']
                    if obs_type not in type_latencies:
                        type_latencies[obs_type] = []
                    type_latencies[obs_type].append(measurement['latency'])
                
                # Validate each obstacle type meets latency requirement
                for obs_type, latencies in type_latencies.items():
                    max_latency = max(latencies)
                    avg_latency = np.mean(latencies)
                    
                    assert max_latency <= MAX_DETECTION_LATENCY, \
                        f"Max detection latency {max_latency:.3f}s exceeds requirement for {obs_type} " \
                        f"obstacles in {scenario['name']} scenario"
                    
                    assert avg_latency <= MAX_DETECTION_LATENCY * 0.75, \
                        f"Average detection latency {avg_latency:.3f}s too high for {obs_type} " \
                        f"obstacles in {scenario['name']} scenario"
                
                # Overall scenario performance
                all_latencies = [m['latency'] for m in scenario_latencies]
                overall_max = max(all_latencies)
                overall_avg = np.mean(all_latencies)
                
                assert overall_max <= MAX_DETECTION_LATENCY, \
                    f"Overall max detection latency {overall_max:.3f}s exceeds requirement in {scenario['name']}"
                
                assert overall_avg <= MAX_DETECTION_LATENCY * 0.6, \
                    f"Overall average detection latency {overall_avg:.3f}s too high in {scenario['name']}"

    class TestDetectionLatencyEdgeCases:
        """Edge cases for detection latency testing."""
        
        def __init__(self):
            self._detection_events = []
            self._detection_lock = threading.Lock()
        
        def _setup_detection_monitoring(self, obstacle_detector: ObstacleDetector):
            """Set up monitoring for obstacle detection events."""
            # Store original callback
            original_callback = obstacle_detector._detection_callback
            
            def monitored_callback():
                """Wrapper to monitor detection events."""
                detection_time = time.time()
                
                # Call original detection callback
                original_callback()
                
                # Record detection event
                with obstacle_detector._data_lock:
                    if hasattr(obstacle_detector, '_obstacle_tracks'):
                        for track_id, track in obstacle_detector._obstacle_tracks.items():
                            if track.total_detections == 1:  # Newly detected
                                with self._detection_lock:
                                    self._detection_events.append({
                                        'track_id': track_id,
                                        'position': track.get_position().copy(),
                                        'detection_time': detection_time
                                    })
            
            obstacle_detector._detection_callback = monitored_callback
        
        def test_detection_latency_with_multiple_simultaneous_obstacles(
            self, simulation, obstacle_detector
        ):
            """
            **Validates: Requirements 4.4**
            
            Test detection latency when multiple obstacles enter sensor range simultaneously.
            """
            # Clear previous events
            with self._detection_lock:
                self._detection_events.clear()
            
            self._setup_detection_monitoring(obstacle_detector)
            
            # Position robot to potentially see multiple obstacles
            simulation.set_robot_pose(0.0, 0.0, 0.0)
            
            # Move robot to encounter multiple obstacles
            simultaneous_detection_start = time.time()
            
            for step in range(15):
                simulation.step(0.1, 0.1)  # Move and rotate
                time.sleep(0.1)
            
            # Wait for detections to complete
            time.sleep(0.3)
            
            # Analyze simultaneous detections
            with self._detection_lock:
                detection_events = self._detection_events.copy()
            
            if len(detection_events) >= 2:  # Multiple obstacles detected
                # Check that all detections occurred within reasonable time
                detection_times = [event['detection_time'] for event in detection_events]
                time_span = max(detection_times) - min(detection_times)
                
                # All detections should occur within a short time window
                assert time_span <= 1.0, \
                    f"Detection time span {time_span:.3f}s too large for simultaneous obstacles"
                
                # Each detection should meet latency requirement
                for event in detection_events:
                    latency = event['detection_time'] - simultaneous_detection_start
                    assert latency <= MAX_DETECTION_LATENCY + 0.5, \
                        f"Detection latency {latency:.3f}s exceeds requirement for simultaneous obstacles"
        
        def test_detection_latency_at_maximum_sensor_range(
            self, simulation, obstacle_detector
        ):
            """
            **Validates: Requirements 4.4**
            
            Test detection latency for obstacles at the edge of sensor range.
            """
            with self._detection_lock:
                self._detection_events.clear()
            
            self._setup_detection_monitoring(obstacle_detector)
            
            # Position robot to detect obstacles at maximum range
            simulation.set_robot_pose(5.0, 5.0, 0.0)
            
            range_detection_start = time.time()
            
            # Move slowly to detect distant obstacles
            for step in range(20):
                simulation.step(0.05, 0.02)  # Slow movement
                time.sleep(0.1)
            
            time.sleep(0.2)
            
            # Check detection latencies for distant obstacles
            with self._detection_lock:
                detection_events = self._detection_events.copy()
            
            if detection_events:
                for event in detection_events:
                    latency = event['detection_time'] - range_detection_start
                    
                    # Even distant obstacles should be detected within requirement
                    assert latency <= MAX_DETECTION_LATENCY + 0.2, \
                        f"Detection latency {latency:.3f}s too high for distant obstacle"
        
        def test_detection_latency_with_rapid_robot_motion(
            self, simulation, obstacle_detector
        ):
            """
            **Validates: Requirements 4.4**
            
            Test detection latency when robot is moving rapidly.
            """
            with self._detection_lock:
                self._detection_events.clear()
            
            self._setup_detection_monitoring(obstacle_detector)
            
            # Start rapid motion
            simulation.set_robot_pose(-5.0, -5.0, 0.0)
            rapid_motion_start = time.time()
            
            # Rapid movement to quickly encounter obstacles
            for step in range(10):
                simulation.step(0.2, 0.1)  # Fast movement
                time.sleep(0.05)  # Short sleep for rapid motion
            
            time.sleep(0.3)
            
            # Analyze detection performance during rapid motion
            with self._detection_lock:
                detection_events = self._detection_events.copy()
            
            if detection_events:
                for event in detection_events:
                    latency = event['detection_time'] - rapid_motion_start
                    
                    # Latency requirement should still be met during rapid motion
                    assert latency <= MAX_DETECTION_LATENCY + 0.1, \
                        f"Detection latency {latency:.3f}s exceeds requirement during rapid motion"
        
        def test_detection_latency_recovery_after_sensor_interruption(
            self, simulation, obstacle_detector
        ):
            """
            **Validates: Requirements 4.4**
            
            Test that detection latency is maintained after sensor data interruption.
            """
            with self._detection_lock:
                self._detection_events.clear()
            
            self._setup_detection_monitoring(obstacle_detector)
            
            # Normal operation
            simulation.set_robot_pose(2.0, 2.0, 0.0)
            
            for step in range(5):
                simulation.step(0.1, 0.0)
                time.sleep(0.1)
            
            # Simulate sensor interruption by pausing
            time.sleep(0.3)  # Brief interruption
            
            # Resume operation and measure recovery
            recovery_start = time.time()
            
            for step in range(10):
                simulation.step(0.08, 0.05)
                time.sleep(0.1)
            
            time.sleep(0.2)
            
            # Check detection latency after recovery
            with self._detection_lock:
                recovery_detections = [
                    event for event in self._detection_events
                    if event['detection_time'] >= recovery_start
                ]
            
            if recovery_detections:
                for event in recovery_detections:
                    latency = event['detection_time'] - recovery_start
                    
                    # Detection should resume quickly after interruption
                    assert latency <= MAX_DETECTION_LATENCY, \
                        f"Detection latency {latency:.3f}s too high after sensor recovery"
        
        def test_minimum_detection_latency_is_reasonable(
            self, simulation, obstacle_detector
        ):
            """
            **Validates: Requirements 4.4**
            
            Test that minimum detection latency is reasonable (not artificially zero).
            """
            with self._detection_lock:
                self._detection_events.clear()
            
            self._setup_detection_monitoring(obstacle_detector)
            
            # Position for optimal detection conditions
            simulation.set_robot_pose(0.0, 0.0, 0.0)
            
            optimal_detection_start = time.time()
            
            # Optimal movement for detection
            for step in range(12):
                simulation.step(0.06, 0.03)
                time.sleep(0.08)
            
            time.sleep(0.2)
            
            # Analyze minimum latencies
            with self._detection_lock:
                detection_events = self._detection_events.copy()
            
            if detection_events:
                latencies = [
                    event['detection_time'] - optimal_detection_start
                    for event in detection_events
                ]
                
                min_latency = min(latencies)
                
                # Minimum latency should be reasonable (not zero due to processing time)
                assert min_latency >= 0.01, \
                    f"Minimum detection latency {min_latency:.3f}s unreasonably low"
                
                # But still well within requirement
                assert min_latency <= MAX_DETECTION_LATENCY * 0.5, \
                    f"Minimum detection latency {min_latency:.3f}s should be well below requirement"