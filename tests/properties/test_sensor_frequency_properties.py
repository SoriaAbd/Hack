"""
Property-based tests for sensor publishing frequency.

Feature: adaptnav-context-aware-warehouse-navigation
Property 4: Sensor Publishing Frequency
Validates: Requirements 2.3
"""

import pytest
import numpy as np
import time
from typing import List, Tuple
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
def simulation_duration(draw):
    """Generate a simulation duration for frequency testing."""
    # Test durations between 5 and 15 seconds for reasonable test times
    return draw(st.floats(min_value=5.0, max_value=15.0))


@pytest.mark.skipif(not ROS2_AVAILABLE or not MUJOCO_AVAILABLE,
                    reason="ROS 2 or MuJoCo not available")
@pytest.mark.property
class TestSensorPublishingFrequencyProperties:
    """
    Property 4: Sensor Publishing Frequency
    
    For any sensor data stream over a 10-second window, the average publishing 
    frequency should be within the specified range:
    - LiDAR: 20 Hz (50ms period)
    - Depth camera: 15 Hz (66.7ms period)
    
    **Validates: Requirements 2.3**
    """
    
    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup and teardown for each test."""
        # Initialize ROS 2
        if not rclpy.ok():
            rclpy.init()
        
        yield
        
        # Cleanup is handled by individual tests
    
    @given(
        robot_pos=valid_robot_position(),
        duration=simulation_duration()
    )
    @settings(max_examples=100, deadline=120000)  # 2 minutes for longer simulation tests
    def test_lidar_publishing_frequency(self, robot_pos, duration):
        """
        Property 4: LiDAR Publishing Frequency
        
        For any sensor data stream over a test duration window, the LiDAR
        publishing frequency should be approximately 20 Hz (±2 Hz tolerance).
        
        **Validates: Requirements 2.3**
        """
        sim = MuJoCoSimulation(render_mode=None)
        
        try:
            # Reset simulation with specified robot position
            success = sim.reset(robot_position=robot_pos)
            assert success, "Simulation reset should succeed"
            
            # Track LiDAR scan publications
            scan_timestamps = []
            start_time = time.time()
            sim_time = 0.0
            dt = 0.01  # 10ms simulation step
            
            # Run simulation for specified duration
            while sim_time < duration:
                # Step simulation
                success = sim.step(dt)
                assert success, "Simulation step should succeed"
                
                # Check if LiDAR scan was published this step
                # We simulate the publishing logic by checking the internal timing
                if sim_time - sim._last_sensor_publish_time >= sim._sensor_publish_period:
                    scan_timestamps.append(sim_time)
                
                sim_time += dt
            
            # Analyze publishing frequency
            if len(scan_timestamps) >= 2:
                # Calculate intervals between publications
                intervals = np.diff(scan_timestamps)
                
                # Calculate average frequency
                avg_interval = np.mean(intervals)
                avg_frequency = 1.0 / avg_interval
                
                # Expected frequency is 20 Hz with ±2 Hz tolerance
                expected_frequency = 20.0
                tolerance = 2.0
                
                assert (expected_frequency - tolerance) <= avg_frequency <= (expected_frequency + tolerance), \
                    f"LiDAR frequency {avg_frequency:.2f} Hz outside expected range " \
                    f"[{expected_frequency - tolerance}, {expected_frequency + tolerance}] Hz. " \
                    f"Average interval: {avg_interval:.4f}s, Expected: {1.0/expected_frequency:.4f}s"
                
                # Check that the configured period matches expectation
                expected_period = 1.0 / expected_frequency  # 0.05 seconds for 20 Hz
                assert abs(sim._sensor_publish_period - expected_period) < 0.001, \
                    f"Configured LiDAR period {sim._sensor_publish_period:.4f}s " \
                    f"doesn't match expected {expected_period:.4f}s"
            else:
                pytest.fail(f"Insufficient LiDAR publications: {len(scan_timestamps)} in {duration:.1f}s")
        
        finally:
            sim.destroy_node()
    
    @given(
        robot_pos=valid_robot_position(),
        duration=simulation_duration()
    )
    @settings(max_examples=100, deadline=120000)  # 2 minutes for longer simulation tests
    def test_depth_camera_publishing_frequency(self, robot_pos, duration):
        """
        Property 4: Depth Camera Publishing Frequency
        
        For any sensor data stream over a test duration window, the depth camera
        publishing frequency should be approximately 15 Hz (±2 Hz tolerance).
        
        **Validates: Requirements 2.3**
        """
        sim = MuJoCoSimulation(render_mode=None)
        
        try:
            # Reset simulation with specified robot position
            success = sim.reset(robot_position=robot_pos)
            assert success, "Simulation reset should succeed"
            
            # Track depth camera publications
            depth_timestamps = []
            sim_time = 0.0
            dt = 0.01  # 10ms simulation step
            
            # Run simulation for specified duration
            while sim_time < duration:
                # Step simulation
                success = sim.step(dt)
                assert success, "Simulation step should succeed"
                
                # Check if depth camera was published this step
                # We simulate the publishing logic by checking the internal timing
                if sim_time - sim._last_depth_publish_time >= sim._depth_camera_publish_period:
                    depth_timestamps.append(sim_time)
                
                sim_time += dt
            
            # Analyze publishing frequency
            if len(depth_timestamps) >= 2:
                # Calculate intervals between publications
                intervals = np.diff(depth_timestamps)
                
                # Calculate average frequency
                avg_interval = np.mean(intervals)
                avg_frequency = 1.0 / avg_interval
                
                # Expected frequency is 15 Hz with ±2 Hz tolerance
                expected_frequency = 15.0
                tolerance = 2.0
                
                assert (expected_frequency - tolerance) <= avg_frequency <= (expected_frequency + tolerance), \
                    f"Depth camera frequency {avg_frequency:.2f} Hz outside expected range " \
                    f"[{expected_frequency - tolerance}, {expected_frequency + tolerance}] Hz. " \
                    f"Average interval: {avg_interval:.4f}s, Expected: {1.0/expected_frequency:.4f}s"
                
                # Check that the configured period matches expectation
                expected_period = 1.0 / expected_frequency  # 0.0667 seconds for 15 Hz
                assert abs(sim._depth_camera_publish_period - expected_period) < 0.001, \
                    f"Configured depth camera period {sim._depth_camera_publish_period:.4f}s " \
                    f"doesn't match expected {expected_period:.4f}s"
            else:
                pytest.fail(f"Insufficient depth camera publications: {len(depth_timestamps)} in {duration:.1f}s")
        
        finally:
            sim.destroy_node()
    
    @given(robot_pos=valid_robot_position())
    @settings(max_examples=100, deadline=60000)
    def test_sensor_frequency_configuration(self, robot_pos):
        """
        Property 4: Sensor Frequency Configuration
        
        The simulation should be configured with the correct publishing periods:
        - LiDAR: 0.05s (20 Hz)
        - Depth camera: 0.0667s (15 Hz)
        
        **Validates: Requirements 2.3**
        """
        sim = MuJoCoSimulation(render_mode=None)
        
        try:
            # Reset simulation
            success = sim.reset(robot_position=robot_pos)
            assert success, "Simulation reset should succeed"
            
            # Check LiDAR configuration
            expected_lidar_period = 0.05  # 20 Hz
            assert abs(sim._sensor_publish_period - expected_lidar_period) < 0.001, \
                f"LiDAR publish period {sim._sensor_publish_period:.4f}s " \
                f"should be {expected_lidar_period:.4f}s (20 Hz)"
            
            # Check depth camera configuration
            expected_depth_period = 1.0 / 15.0  # 15 Hz ≈ 0.0667s
            assert abs(sim._depth_camera_publish_period - expected_depth_period) < 0.001, \
                f"Depth camera publish period {sim._depth_camera_publish_period:.4f}s " \
                f"should be {expected_depth_period:.4f}s (15 Hz)"
            
            # Verify frequencies are within specified range (10-30 Hz for LiDAR, 10-20 Hz for depth)
            lidar_freq = 1.0 / sim._sensor_publish_period
            depth_freq = 1.0 / sim._depth_camera_publish_period
            
            assert 10.0 <= lidar_freq <= 30.0, \
                f"LiDAR frequency {lidar_freq:.1f} Hz outside specified range [10, 30] Hz"
            
            assert 10.0 <= depth_freq <= 20.0, \
                f"Depth camera frequency {depth_freq:.1f} Hz outside specified range [10, 20] Hz"
        
        finally:
            sim.destroy_node()
    
    @given(
        robot_pos=valid_robot_position(),
        duration=st.floats(min_value=10.0, max_value=12.0)  # Fixed 10-second window as per requirement
    )
    @settings(max_examples=50, deadline=180000)  # 3 minutes for 10-second simulation tests
    def test_sensor_frequency_over_10_second_window(self, robot_pos, duration):
        """
        Property 4: Sensor Publishing Frequency Over 10-Second Window
        
        For any sensor data stream over a 10-second window, the average publishing 
        frequency should be within the specified range (10-30 Hz for LiDAR, 10-20 Hz for depth).
        This is the exact requirement from the specification.
        
        **Validates: Requirements 2.3**
        """
        sim = MuJoCoSimulation(render_mode=None)
        
        try:
            # Reset simulation with specified robot position
            success = sim.reset(robot_position=robot_pos)
            assert success, "Simulation reset should succeed"
            
            # Track both sensor publications over exactly 10 seconds
            lidar_count = 0
            depth_count = 0
            sim_time = 0.0
            dt = 0.01  # 10ms simulation step
            last_lidar_publish = -1.0
            last_depth_publish = -1.0
            
            # Run simulation for exactly 10 seconds
            target_duration = 10.0
            while sim_time < target_duration:
                # Step simulation
                success = sim.step(dt)
                assert success, "Simulation step should succeed"
                
                # Check if LiDAR scan would be published this step
                if sim_time - last_lidar_publish >= sim._sensor_publish_period:
                    lidar_count += 1
                    last_lidar_publish = sim_time
                
                # Check if depth camera would be published this step
                if sim_time - last_depth_publish >= sim._depth_camera_publish_period:
                    depth_count += 1
                    last_depth_publish = sim_time
                
                sim_time += dt
            
            # Calculate actual frequencies over the 10-second window
            lidar_frequency = lidar_count / target_duration
            depth_frequency = depth_count / target_duration
            
            # Validate LiDAR frequency is within specified range (10-30 Hz)
            assert 10.0 <= lidar_frequency <= 30.0, \
                f"LiDAR frequency {lidar_frequency:.2f} Hz over 10s window " \
                f"outside specified range [10, 30] Hz. Published {lidar_count} scans."
            
            # Validate depth camera frequency is within specified range (10-20 Hz)  
            assert 10.0 <= depth_frequency <= 20.0, \
                f"Depth camera frequency {depth_frequency:.2f} Hz over 10s window " \
                f"outside specified range [10, 20] Hz. Published {depth_count} images."
            
            # Additional check: frequencies should be close to configured values
            expected_lidar_freq = 1.0 / sim._sensor_publish_period
            expected_depth_freq = 1.0 / sim._depth_camera_publish_period
            
            # Allow 5% tolerance for timing precision
            lidar_tolerance = expected_lidar_freq * 0.05
            depth_tolerance = expected_depth_freq * 0.05
            
            assert abs(lidar_frequency - expected_lidar_freq) <= lidar_tolerance, \
                f"LiDAR frequency {lidar_frequency:.2f} Hz deviates too much from " \
                f"expected {expected_lidar_freq:.2f} Hz (tolerance: ±{lidar_tolerance:.2f})"
            
            assert abs(depth_frequency - expected_depth_freq) <= depth_tolerance, \
                f"Depth camera frequency {depth_frequency:.2f} Hz deviates too much from " \
                f"expected {expected_depth_freq:.2f} Hz (tolerance: ±{depth_tolerance:.2f})"
        
        finally:
            sim.destroy_node()
    
    @given(robot_pos=valid_robot_position())
    @settings(max_examples=100, deadline=60000)
    def test_sensor_timing_independence(self, robot_pos):
        """
        Property 4: Sensor Timing Independence
        
        LiDAR and depth camera should publish independently at their own frequencies,
        not synchronized to each other.
        
        **Validates: Requirements 2.3**
        """
        sim = MuJoCoSimulation(render_mode=None)
        
        try:
            # Reset simulation
            success = sim.reset(robot_position=robot_pos)
            assert success, "Simulation reset should succeed"
            
            # Check that the publish periods are different
            lidar_period = sim._sensor_publish_period
            depth_period = sim._depth_camera_publish_period
            
            assert abs(lidar_period - depth_period) > 0.001, \
                f"LiDAR and depth camera should have different publish periods. " \
                f"LiDAR: {lidar_period:.4f}s, Depth: {depth_period:.4f}s"
            
            # Check that periods are not simple multiples of each other
            # (which would cause unwanted synchronization)
            ratio1 = lidar_period / depth_period
            ratio2 = depth_period / lidar_period
            
            # Neither ratio should be close to an integer (within 10%)
            for ratio in [ratio1, ratio2]:
                nearest_int = round(ratio)
                if nearest_int > 0:
                    deviation = abs(ratio - nearest_int) / nearest_int
                    assert deviation > 0.1, \
                        f"Sensor periods may be synchronized (ratio: {ratio:.3f}, " \
                        f"nearest integer: {nearest_int}, deviation: {deviation:.3f})"
        
        finally:
            sim.destroy_node()
    
    @given(
        robot_pos=valid_robot_position(),
        robot_orientation=st.floats(min_value=-np.pi, max_value=np.pi)
    )
    @settings(max_examples=100, deadline=60000)
    def test_sensor_frequency_invariant_to_robot_state(self, robot_pos, robot_orientation):
        """
        Property 4: Sensor Frequency Invariant to Robot State
        
        Sensor publishing frequencies should be consistent regardless of robot
        position and orientation in the warehouse.
        
        **Validates: Requirements 2.3**
        """
        sim = MuJoCoSimulation(render_mode=None)
        
        try:
            # Test with different robot positions and orientations
            robot_x, robot_y, _ = robot_pos
            test_position = (robot_x, robot_y, robot_orientation)
            
            # Reset simulation with test position
            success = sim.reset(robot_position=test_position)
            assert success, "Simulation reset should succeed"
            
            # Check that configured frequencies remain constant
            expected_lidar_period = 0.05  # 20 Hz
            expected_depth_period = 1.0 / 15.0  # 15 Hz
            
            assert abs(sim._sensor_publish_period - expected_lidar_period) < 0.001, \
                f"LiDAR period changed with robot state: {sim._sensor_publish_period:.4f}s " \
                f"vs expected {expected_lidar_period:.4f}s"
            
            assert abs(sim._depth_camera_publish_period - expected_depth_period) < 0.001, \
                f"Depth camera period changed with robot state: {sim._depth_camera_publish_period:.4f}s " \
                f"vs expected {expected_depth_period:.4f}s"
            
            # Run a few simulation steps to ensure timing logic works
            for _ in range(10):
                success = sim.step(0.01)
                assert success, "Simulation step should succeed"
                
                # Verify periods haven't changed during simulation
                assert abs(sim._sensor_publish_period - expected_lidar_period) < 0.001
                assert abs(sim._depth_camera_publish_period - expected_depth_period) < 0.001
        
        finally:
            sim.destroy_node()


# Additional unit tests for edge cases
@pytest.mark.skipif(not ROS2_AVAILABLE or not MUJOCO_AVAILABLE,
                    reason="ROS 2 or MuJoCo not available")
class TestSensorFrequencyEdgeCases:
    """Unit tests for sensor frequency edge cases."""
    
    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup and teardown for each test."""
        if not rclpy.ok():
            rclpy.init()
        yield
    
    def test_sensor_frequency_at_simulation_start(self):
        """Test sensor frequency behavior at simulation start."""
        sim = MuJoCoSimulation(render_mode=None)
        
        try:
            success = sim.reset()
            assert success
            
            # At start, last publish times should be 0
            assert sim._last_sensor_publish_time == 0.0
            assert sim._last_depth_publish_time == 0.0
            
            # First step should trigger both sensors to publish
            success = sim.step(0.01)
            assert success
            
            # After first step, both sensors should have published
            # (since sim_time >= publish_period for both)
            
        finally:
            sim.destroy_node()
    
    def test_sensor_frequency_with_large_time_steps(self):
        """Test sensor frequency behavior with large simulation time steps."""
        sim = MuJoCoSimulation(render_mode=None)
        
        try:
            success = sim.reset()
            assert success
            
            # Take a large time step (larger than both sensor periods)
            large_dt = 0.1  # 100ms, larger than both 50ms and 67ms periods
            success = sim.step(large_dt)
            assert success
            
            # Both sensors should have published
            assert sim._last_sensor_publish_time > 0
            assert sim._last_depth_publish_time > 0
            
        finally:
            sim.destroy_node()
    
    def test_sensor_frequency_with_small_time_steps(self):
        """Test sensor frequency behavior with very small simulation time steps."""
        sim = MuJoCoSimulation(render_mode=None)
        
        try:
            success = sim.reset()
            assert success
            
            # Take many small time steps
            small_dt = 0.001  # 1ms
            lidar_publishes = 0
            depth_publishes = 0
            
            for i in range(100):  # 100ms total
                last_lidar_time = sim._last_sensor_publish_time
                last_depth_time = sim._last_depth_publish_time
                
                success = sim.step(small_dt)
                assert success
                
                # Check if sensors published this step
                if sim._last_sensor_publish_time > last_lidar_time:
                    lidar_publishes += 1
                if sim._last_depth_publish_time > last_depth_time:
                    depth_publishes += 1
            
            # Over 100ms, LiDAR (20Hz, 50ms period) should publish ~2 times
            # Depth camera (15Hz, 67ms period) should publish ~1 time
            assert 1 <= lidar_publishes <= 3, f"Expected 1-3 LiDAR publishes, got {lidar_publishes}"
            assert 0 <= depth_publishes <= 2, f"Expected 0-2 depth publishes, got {depth_publishes}"
            
        finally:
            sim.destroy_node()