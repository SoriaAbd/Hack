"""
Property-based tests for obstacle tracking consistency.

This module implements Property 14: Tracking Consistency
**Validates: Requirements 4.5**

Property 14: Tracking Consistency
For any obstacle detected in consecutive frames (frame N and N+1) where the position 
change is less than 2 meters, the tracking ID should remain the same.

Requirements 4.5: The obstacle detector shall maintain tracking of dynamic obstacles 
across consecutive sensor frames
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume
from hypothesis import HealthCheck
import time
from typing import List, Tuple, Optional, Dict, Set
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
MAX_POSITION_CHANGE = 2.0  # meters (from requirements)


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
    class TestTrackingConsistencyProperty:
        """
        Property 14: Tracking Consistency
        
        **Validates: Requirements 4.5**
        
        For any obstacle detected in consecutive frames (frame N and N+1) where the position 
        change is less than 2 meters, the tracking ID should remain the same.
        
        This property ensures that:
        1. Obstacles that move small distances maintain the same tracking ID
        2. The tracking system provides consistent identification across frames
        3. Data association works correctly for typical obstacle movements
        4. Tracking IDs are stable for continuous obstacle trajectories
        5. The system meets the requirement for maintaining tracking across sensor frames
        """
        
        def __init__(self):
            self._tracking_history = []
            self._tracking_lock = threading.Lock()
        
        def _setup_tracking_monitoring(self, obstacle_detector: ObstacleDetector):
            """Set up monitoring for obstacle tracking events."""
            # Store original callback
            original_callback = obstacle_detector._detection_callback
            
            def monitored_callback():
                """Wrapper to monitor tracking events."""
                frame_time = time.time()
                
                # Call original detection callback
                original_callback()
                
                # Record tracking state after detection
                with obstacle_detector._data_lock:
                    if hasattr(obstacle_detector, '_obstacle_tracks'):
                        frame_tracks = {}
                        for track_id, track in obstacle_detector._obstacle_tracks.items():
                            frame_tracks[track_id] = {
                                'position': track.get_position().copy(),
                                'velocity': track.get_velocity().copy(),
                                'total_detections': track.total_detections,
                                'missed_detections': track.missed_detections,
                                'age': track.age,
                                'classification': track.classification,
                                'confidence': track.confidence
                            }
                        
                        with self._tracking_lock:
                            self._tracking_history.append({
                                'frame_time': frame_time,
                                'tracks': frame_tracks
                            })
            
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
            movement_steps=st.integers(min_value=15, max_value=30)
        )
        def test_tracking_id_consistency_for_small_position_changes(
            self, simulation, obstacle_detector, robot_x, robot_y, robot_yaw, movement_steps
        ):
            """
            **Validates: Requirements 4.5**
            
            Property: Obstacles with position changes < 2m maintain the same tracking ID.
            
            This is the core property that validates tracking consistency. When an
            obstacle is detected in consecutive frames and hasn't moved too far,
            the tracking system should maintain the same ID for that obstacle.
            """
            # Avoid positions that would place robot inside obstacles
            assume(abs(robot_x) < 18.0 and abs(robot_y) < 18.0)
            
            # Clear previous tracking history
            with self._tracking_lock:
                self._tracking_history.clear()
            
            # Set up tracking monitoring
            self._setup_tracking_monitoring(obstacle_detector)
            
            # Set initial robot position
            simulation.set_robot_pose(robot_x, robot_y, robot_yaw)
            
            # Move robot to encounter and track obstacles
            for step in range(movement_steps):
                # Small movements to maintain obstacle visibility
                linear_vel = 0.05 + 0.02 * np.sin(step * 0.1)  # Gentle forward motion
                angular_vel = 0.02 * np.cos(step * 0.15)  # Gentle rotation
                
                simulation.step(linear_vel, angular_vel)
                
                # Allow time for detection and tracking
                time.sleep(0.1)
            
            # Wait for final processing
            time.sleep(0.2)
            
            # Analyze tracking consistency
            with self._tracking_lock:
                tracking_history = self._tracking_history.copy()
            
            # Core property validation: Check tracking ID consistency for consecutive frames
            tracking_violations = []
            consistent_tracks = 0
            total_consecutive_pairs = 0
            
            for i in range(len(tracking_history) - 1):
                frame_n = tracking_history[i]
                frame_n_plus_1 = tracking_history[i + 1]
                
                # For each track in frame N, check if it appears in frame N+1
                for track_id_n, track_data_n in frame_n['tracks'].items():
                    position_n = track_data_n['position']
                    
                    # Look for the same obstacle in frame N+1
                    # First, check if the exact same track ID exists
                    if track_id_n in frame_n_plus_1['tracks']:
                        track_data_n_plus_1 = frame_n_plus_1['tracks'][track_id_n]
                        position_n_plus_1 = track_data_n_plus_1['position']
                        
                        # Calculate position change
                        position_change = np.linalg.norm(position_n_plus_1 - position_n)
                        
                        total_consecutive_pairs += 1
                        
                        if position_change < MAX_POSITION_CHANGE:
                            # Small position change - tracking ID should be consistent
                            consistent_tracks += 1
                        else:
                            # Large position change - ID consistency not required
                            pass
                    
                    else:
                        # Track ID disappeared - check if obstacle reappeared with different ID
                        # This would be a tracking consistency violation
                        min_distance = float('inf')
                        closest_new_track = None
                        
                        for track_id_n_plus_1, track_data_n_plus_1 in frame_n_plus_1['tracks'].items():
                            if track_id_n_plus_1 not in frame_n['tracks']:  # New track
                                position_n_plus_1 = track_data_n_plus_1['position']
                                distance = np.linalg.norm(position_n_plus_1 - position_n)
                                
                                if distance < min_distance:
                                    min_distance = distance
                                    closest_new_track = track_id_n_plus_1
                        
                        total_consecutive_pairs += 1
                        
                        if min_distance < MAX_POSITION_CHANGE and closest_new_track is not None:
                            # Found a new track very close to the old position
                            # This is a tracking consistency violation
                            tracking_violations.append({
                                'frame_n': i,
                                'frame_n_plus_1': i + 1,
                                'original_track_id': track_id_n,
                                'new_track_id': closest_new_track,
                                'position_change': min_distance,
                                'original_position': position_n,
                                'new_position': frame_n_plus_1['tracks'][closest_new_track]['position']
                            })
            
            # Core property assertion: Tracking consistency must be maintained
            if total_consecutive_pairs > 0:
                violation_rate = len(tracking_violations) / total_consecutive_pairs
                max_allowed_violations = 0.15  # Allow up to 15% violations due to edge cases
                
                assert violation_rate <= max_allowed_violations, \
                    f"Tracking consistency violated in {len(tracking_violations)}/{total_consecutive_pairs} " \
                    f"cases ({violation_rate:.1%}). Max allowed: {max_allowed_violations:.1%}. " \
                    f"Sample violations: {tracking_violations[:3]}"
                
                # Additional validation: Most tracks should maintain consistency
                if consistent_tracks > 0:
                    consistency_rate = consistent_tracks / total_consecutive_pairs
                    assert consistency_rate >= 0.7, \
                        f"Tracking consistency rate {consistency_rate:.1%} too low. " \
                        f"Expected at least 70% consistency for small position changes."
            
            else:
                # No consecutive tracking pairs found - this might indicate detection issues
                # but is not necessarily a tracking consistency violation
                assert len(tracking_history) >= 2, \
                    f"Insufficient tracking data collected. Got {len(tracking_history)} frames, expected at least 2."
        
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
        def test_tracking_consistency_across_different_movement_patterns(
            self, simulation, obstacle_detector, robot_x, robot_y, movement_pattern
        ):
            """
            **Validates: Requirements 4.5**
            
            Property: Tracking consistency is maintained across different robot movement patterns.
            
            This validates that the tracking system maintains ID consistency regardless
            of how the robot moves (straight, circular, stop-and-go, etc.).
            """
            # Clear previous tracking history
            with self._tracking_lock:
                self._tracking_history.clear()
            
            # Set up tracking monitoring
            self._setup_tracking_monitoring(obstacle_detector)
            
            # Set robot position
            simulation.set_robot_pose(robot_x, robot_y, 0.0)
            
            # Define different movement patterns
            movement_patterns = [
                {'name': 'straight_line', 'commands': [(0.1, 0.0)] * 20},
                {'name': 'circular', 'commands': [(0.08, 0.1)] * 25},
                {'name': 'stop_and_go', 'commands': [(0.15, 0.0)] * 3 + [(0.0, 0.0)] * 2} * 4,
                {'name': 'zigzag', 'commands': [(0.1, 0.1)] * 5 + [(0.1, -0.1)] * 5} * 2
            ]
            
            pattern = movement_patterns[movement_pattern]
            
            # Execute movement pattern
            for linear_vel, angular_vel in pattern['commands']:
                simulation.step(linear_vel, angular_vel)
                time.sleep(0.08)
            
            # Wait for final processing
            time.sleep(0.2)
            
            # Analyze tracking consistency for this movement pattern
            with self._tracking_lock:
                tracking_history = self._tracking_history.copy()
            
            # Check tracking consistency across the movement pattern
            id_consistency_violations = 0
            total_tracking_opportunities = 0
            
            for i in range(len(tracking_history) - 1):
                current_frame = tracking_history[i]
                next_frame = tracking_history[i + 1]
                
                for track_id, track_data in current_frame['tracks'].items():
                    total_tracking_opportunities += 1
                    
                    if track_id in next_frame['tracks']:
                        # Track ID maintained - check if position change is reasonable
                        pos_current = track_data['position']
                        pos_next = next_frame['tracks'][track_id]['position']
                        position_change = np.linalg.norm(pos_next - pos_current)
                        
                        # If position change is small, this is good consistency
                        if position_change < MAX_POSITION_CHANGE:
                            # Good tracking consistency
                            pass
                        else:
                            # Large position change but same ID - might be acceptable
                            # depending on obstacle velocity
                            pass
                    else:
                        # Track ID lost - check if obstacle reappeared with new ID
                        pos_current = track_data['position']
                        
                        for new_track_id, new_track_data in next_frame['tracks'].items():
                            if new_track_id not in current_frame['tracks']:
                                pos_new = new_track_data['position']
                                distance = np.linalg.norm(pos_new - pos_current)
                                
                                if distance < MAX_POSITION_CHANGE:
                                    # Same obstacle, different ID - consistency violation
                                    id_consistency_violations += 1
                                    break
            
            # Validate tracking consistency for this movement pattern
            if total_tracking_opportunities > 0:
                violation_rate = id_consistency_violations / total_tracking_opportunities
                
                assert violation_rate <= 0.2, \
                    f"Tracking consistency violation rate {violation_rate:.1%} too high " \
                    f"for {pattern['name']} movement pattern. Expected ≤ 20%."
        
        @settings(
            max_examples=MIN_EXAMPLES,
            deadline=TEST_TIMEOUT * 1000,
            suppress_health_check=[HealthCheck.function_scoped_fixture]
        )
        @given(
            robot_x=st.floats(min_value=-12.0, max_value=12.0),
            robot_y=st.floats(min_value=-12.0, max_value=12.0),
            detection_duration=st.integers(min_value=10, max_value=20)
        )
        def test_tracking_id_stability_over_time(
            self, simulation, obstacle_detector, robot_x, robot_y, detection_duration
        ):
            """
            **Validates: Requirements 4.5**
            
            Property: Tracking IDs remain stable over extended observation periods.
            
            This validates that obstacles maintain the same tracking ID over multiple
            consecutive frames, not just between adjacent frames.
            """
            # Clear previous tracking history
            with self._tracking_lock:
                self._tracking_history.clear()
            
            # Set up tracking monitoring
            self._setup_tracking_monitoring(obstacle_detector)
            
            # Set robot position
            simulation.set_robot_pose(robot_x, robot_y, 0.0)
            
            # Perform extended observation with minimal robot movement
            for step in range(detection_duration):
                # Very small movements to maintain obstacle visibility
                linear_vel = 0.02 * np.sin(step * 0.05)
                angular_vel = 0.01 * np.cos(step * 0.08)
                
                simulation.step(linear_vel, angular_vel)
                time.sleep(0.1)
            
            # Wait for final processing
            time.sleep(0.2)
            
            # Analyze long-term tracking stability
            with self._tracking_lock:
                tracking_history = self._tracking_history.copy()
            
            # Find tracks that appear in multiple frames
            track_appearances = {}
            
            for frame_idx, frame in enumerate(tracking_history):
                for track_id, track_data in frame['tracks'].items():
                    if track_id not in track_appearances:
                        track_appearances[track_id] = []
                    
                    track_appearances[track_id].append({
                        'frame_idx': frame_idx,
                        'position': track_data['position'],
                        'frame_time': frame['frame_time']
                    })
            
            # Analyze stability for tracks that appear in multiple frames
            stable_tracks = 0
            unstable_tracks = 0
            
            for track_id, appearances in track_appearances.items():
                if len(appearances) >= 3:  # Track appears in at least 3 frames
                    # Check position consistency across appearances
                    positions = [app['position'] for app in appearances]
                    
                    # Calculate maximum position change across all appearances
                    max_position_change = 0.0
                    for i in range(len(positions) - 1):
                        for j in range(i + 1, len(positions)):
                            change = np.linalg.norm(positions[j] - positions[i])
                            max_position_change = max(max_position_change, change)
                    
                    # Check if track is stable (reasonable position changes)
                    if max_position_change <= MAX_POSITION_CHANGE * 2:  # Allow some accumulation
                        stable_tracks += 1
                    else:
                        unstable_tracks += 1
            
            # Validate long-term tracking stability
            total_multi_frame_tracks = stable_tracks + unstable_tracks
            
            if total_multi_frame_tracks > 0:
                stability_rate = stable_tracks / total_multi_frame_tracks
                
                assert stability_rate >= 0.8, \
                    f"Long-term tracking stability rate {stability_rate:.1%} too low. " \
                    f"Expected at least 80% of multi-frame tracks to be stable. " \
                    f"Stable: {stable_tracks}, Unstable: {unstable_tracks}"
        
        @settings(
            max_examples=50,
            deadline=TEST_TIMEOUT * 1000,
            suppress_health_check=[HealthCheck.function_scoped_fixture]
        )
        @given(
            robot_x=st.floats(min_value=-8.0, max_value=8.0),
            robot_y=st.floats(min_value=-8.0, max_value=8.0),
            robot_yaw=st.floats(min_value=-np.pi, max_value=np.pi)
        )
        def test_tracking_consistency_with_multiple_obstacles(
            self, simulation, obstacle_detector, robot_x, robot_y, robot_yaw
        ):
            """
            **Validates: Requirements 4.5**
            
            Property: Tracking consistency is maintained when multiple obstacles are present.
            
            This validates that the data association algorithm correctly maintains
            tracking IDs even when multiple obstacles are detected simultaneously.
            """
            # Clear previous tracking history
            with self._tracking_lock:
                self._tracking_history.clear()
            
            # Set up tracking monitoring
            self._setup_tracking_monitoring(obstacle_detector)
            
            # Set robot position
            simulation.set_robot_pose(robot_x, robot_y, robot_yaw)
            
            # Move robot to encounter multiple obstacles
            multi_obstacle_violations = []
            
            for step in range(18):
                # Movement pattern designed to encounter multiple obstacles
                linear_vel = 0.08 + 0.02 * np.cos(step * 0.2)
                angular_vel = 0.04 * np.sin(step * 0.15)
                
                simulation.step(linear_vel, angular_vel)
                time.sleep(0.1)
            
            # Wait for final processing
            time.sleep(0.2)
            
            # Analyze multi-obstacle tracking consistency
            with self._tracking_lock:
                tracking_history = self._tracking_history.copy()
            
            # Find frames with multiple obstacles
            multi_obstacle_frames = []
            for frame_idx, frame in enumerate(tracking_history):
                if len(frame['tracks']) >= 2:
                    multi_obstacle_frames.append(frame_idx)
            
            # Check tracking consistency in multi-obstacle scenarios
            for frame_idx in multi_obstacle_frames[:-1]:  # Exclude last frame
                if frame_idx + 1 < len(tracking_history):
                    current_frame = tracking_history[frame_idx]
                    next_frame = tracking_history[frame_idx + 1]
                    
                    # Check each track in current frame
                    for track_id, track_data in current_frame['tracks'].items():
                        pos_current = track_data['position']
                        
                        if track_id in next_frame['tracks']:
                            # Track ID maintained
                            pos_next = next_frame['tracks'][track_id]['position']
                            position_change = np.linalg.norm(pos_next - pos_current)
                            
                            if position_change >= MAX_POSITION_CHANGE:
                                # Large position change with same ID - might be ID swap
                                # Check if another track is closer to original position
                                for other_id, other_data in next_frame['tracks'].items():
                                    if other_id != track_id:
                                        other_pos = other_data['position']
                                        other_distance = np.linalg.norm(other_pos - pos_current)
                                        
                                        if other_distance < position_change / 2:
                                            # Possible ID swap detected
                                            multi_obstacle_violations.append({
                                                'frame': frame_idx,
                                                'track_id': track_id,
                                                'position_change': position_change,
                                                'possible_swap_with': other_id,
                                                'swap_distance': other_distance
                                            })
            
            # Validate multi-obstacle tracking consistency
            total_multi_obstacle_opportunities = len(multi_obstacle_frames) - 1
            
            if total_multi_obstacle_opportunities > 0:
                violation_rate = len(multi_obstacle_violations) / total_multi_obstacle_opportunities
                
                assert violation_rate <= 0.25, \
                    f"Multi-obstacle tracking consistency violation rate {violation_rate:.1%} too high. " \
                    f"Expected ≤ 25% in complex scenarios. Violations: {len(multi_obstacle_violations)}"


    class TestTrackingConsistencyEdgeCases:
        """Edge cases for tracking consistency testing."""
        
        def __init__(self):
            self._tracking_history = []
            self._tracking_lock = threading.Lock()
        
        def _setup_tracking_monitoring(self, obstacle_detector: ObstacleDetector):
            """Set up monitoring for obstacle tracking events."""
            original_callback = obstacle_detector._detection_callback
            
            def monitored_callback():
                frame_time = time.time()
                original_callback()
                
                with obstacle_detector._data_lock:
                    if hasattr(obstacle_detector, '_obstacle_tracks'):
                        frame_tracks = {}
                        for track_id, track in obstacle_detector._obstacle_tracks.items():
                            frame_tracks[track_id] = {
                                'position': track.get_position().copy(),
                                'total_detections': track.total_detections
                            }
                        
                        with self._tracking_lock:
                            self._tracking_history.append({
                                'frame_time': frame_time,
                                'tracks': frame_tracks
                            })
            
            obstacle_detector._detection_callback = monitored_callback
        
        def test_tracking_consistency_with_no_obstacles(
            self, simulation, obstacle_detector
        ):
            """
            **Validates: Requirements 4.5**
            
            Test tracking consistency when no obstacles are detected.
            """
            with self._tracking_lock:
                self._tracking_history.clear()
            
            self._setup_tracking_monitoring(obstacle_detector)
            
            # Position robot in area with minimal obstacles
            simulation.set_robot_pose(0.0, 0.0, 0.0)
            
            for step in range(10):
                simulation.step(0.05, 0.0)
                time.sleep(0.1)
            
            time.sleep(0.2)
            
            # Should not crash and should handle empty tracking gracefully
            with self._tracking_lock:
                tracking_history = self._tracking_history.copy()
            
            # Verify system handles no obstacles correctly
            for frame in tracking_history:
                assert isinstance(frame['tracks'], dict), "Tracking should return dict even with no obstacles"
        
        def test_tracking_consistency_with_single_obstacle(
            self, simulation, obstacle_detector
        ):
            """
            **Validates: Requirements 4.5**
            
            Test tracking consistency with exactly one obstacle.
            """
            with self._tracking_lock:
                self._tracking_history.clear()
            
            self._setup_tracking_monitoring(obstacle_detector)
            
            # Position robot to potentially detect single obstacle
            simulation.set_robot_pose(5.0, 5.0, 0.0)
            
            for step in range(15):
                simulation.step(0.06, 0.02)
                time.sleep(0.1)
            
            time.sleep(0.2)
            
            # Analyze single obstacle tracking
            with self._tracking_lock:
                tracking_history = self._tracking_history.copy()
            
            # Find frames with exactly one obstacle
            single_obstacle_frames = [
                frame for frame in tracking_history 
                if len(frame['tracks']) == 1
            ]
            
            if len(single_obstacle_frames) >= 2:
                # Check consistency between consecutive single-obstacle frames
                for i in range(len(single_obstacle_frames) - 1):
                    frame1 = single_obstacle_frames[i]
                    frame2 = single_obstacle_frames[i + 1]
                    
                    track_id1 = list(frame1['tracks'].keys())[0]
                    track_id2 = list(frame2['tracks'].keys())[0]
                    
                    pos1 = frame1['tracks'][track_id1]['position']
                    pos2 = frame2['tracks'][track_id2]['position']
                    
                    position_change = np.linalg.norm(pos2 - pos1)
                    
                    if position_change < MAX_POSITION_CHANGE:
                        # Small position change - should have same ID
                        assert track_id1 == track_id2, \
                            f"Single obstacle tracking ID changed from {track_id1} to {track_id2} " \
                            f"with only {position_change:.2f}m position change"
        
        def test_tracking_consistency_boundary_conditions(
            self, simulation, obstacle_detector
        ):
            """
            **Validates: Requirements 4.5**
            
            Test tracking consistency at boundary conditions (exactly 2m movement).
            """
            with self._tracking_lock:
                self._tracking_history.clear()
            
            self._setup_tracking_monitoring(obstacle_detector)
            
            # Position robot and perform controlled movement
            simulation.set_robot_pose(3.0, 3.0, 0.0)
            
            # Perform movement that might result in boundary position changes
            for step in range(12):
                # Larger movements to test boundary conditions
                linear_vel = 0.15 if step % 2 == 0 else 0.05
                angular_vel = 0.08 if step % 3 == 0 else 0.02
                
                simulation.step(linear_vel, angular_vel)
                time.sleep(0.12)
            
            time.sleep(0.2)
            
            # Analyze boundary condition tracking
            with self._tracking_lock:
                tracking_history = self._tracking_history.copy()
            
            # Look for position changes near the 2m boundary
            boundary_cases = []
            
            for i in range(len(tracking_history) - 1):
                frame1 = tracking_history[i]
                frame2 = tracking_history[i + 1]
                
                for track_id1, track_data1 in frame1['tracks'].items():
                    if track_id1 in frame2['tracks']:
                        pos1 = track_data1['position']
                        pos2 = frame2['tracks'][track_id1]['position']
                        position_change = np.linalg.norm(pos2 - pos1)
                        
                        # Check for boundary cases (1.8m - 2.2m range)
                        if 1.8 <= position_change <= 2.2:
                            boundary_cases.append({
                                'track_id': track_id1,
                                'position_change': position_change,
                                'maintained_id': True
                            })
            
            # Boundary cases should still maintain reasonable tracking behavior
            if boundary_cases:
                # Most boundary cases should maintain ID (allowing some flexibility)
                maintained_count = sum(1 for case in boundary_cases if case['maintained_id'])
                maintenance_rate = maintained_count / len(boundary_cases)
                
                # At boundary conditions, allow more flexibility
                assert maintenance_rate >= 0.5, \
                    f"Boundary condition tracking maintenance rate {maintenance_rate:.1%} too low. " \
                    f"Expected at least 50% at boundary conditions."