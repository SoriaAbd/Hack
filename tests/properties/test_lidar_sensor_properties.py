"""
Property-based tests for LiDAR sensor simulation.

Tests validate correctness properties for the LiDAR sensor implementation
including range limitations, noise characteristics, and occlusion effects.
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume
from hypothesis import HealthCheck

from adaptnav.simulation.mujoco_simulation import MuJoCoSimulation


# Test configuration
MIN_EXAMPLES = 100  # Minimum iterations per property test
TEST_TIMEOUT = 60  # seconds


@pytest.fixture(scope="module")
def simulation():
    """Create a simulation instance for testing."""
    sim = MuJoCoSimulation(render_mode=None)
    sim.reset()
    yield sim
    sim.destroy_node()


class TestLiDARRangeLimitations:
    """
    Property 5: LiDAR Range Limitations
    
    For any LiDAR scan, all range readings should be either within the valid 
    range [0.1m, 10m] or marked as invalid/infinite.
    
    **Validates: Requirements 2.4**
    """
    
    @settings(
        max_examples=MIN_EXAMPLES,
        deadline=TEST_TIMEOUT * 1000,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    @given(
        robot_x=st.floats(min_value=-20.0, max_value=20.0),
        robot_y=st.floats(min_value=-20.0, max_value=20.0),
        robot_yaw=st.floats(min_value=-np.pi, max_value=np.pi)
    )
    def test_all_ranges_within_limits(self, simulation, robot_x, robot_y, robot_yaw):
        """
        Property: All LiDAR range readings must be within [0.1m, 10m] or inf.
        
        This property ensures that the LiDAR sensor respects its physical
        range limitations and properly handles out-of-range measurements.
        """
        # Set robot position
        simulation.set_robot_pose(robot_x, robot_y, robot_yaw)
        simulation.step(0.0, 0.0)  # Update simulation
        
        # Get LiDAR scan
        obs = simulation.get_observation()
        lidar_scan = obs['lidar_scan']
        
        # Check all ranges
        for i, range_val in enumerate(lidar_scan.ranges):
            # Range must be within valid limits or infinite
            assert (0.1 <= range_val <= 10.0) or np.isinf(range_val), \
                f"Range at angle {i}° is {range_val}, outside valid range [0.1, 10.0]"
    
    @settings(
        max_examples=MIN_EXAMPLES,
        deadline=TEST_TIMEOUT * 1000,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    @given(
        robot_x=st.floats(min_value=-20.0, max_value=20.0),
        robot_y=st.floats(min_value=-20.0, max_value=20.0),
        robot_yaw=st.floats(min_value=-np.pi, max_value=np.pi)
    )
    def test_scan_has_360_rays(self, simulation, robot_x, robot_y, robot_yaw):
        """
        Property: LiDAR scan must have exactly 360 rays (1° resolution).
        
        This ensures the sensor provides the specified angular resolution.
        """
        # Set robot position
        simulation.set_robot_pose(robot_x, robot_y, robot_yaw)
        simulation.step(0.0, 0.0)
        
        # Get LiDAR scan
        obs = simulation.get_observation()
        lidar_scan = obs['lidar_scan']
        
        # Must have exactly 360 rays
        assert len(lidar_scan.ranges) == 360, \
            f"Expected 360 rays, got {len(lidar_scan.ranges)}"


class TestLiDARScanMetadata:
    """Test LiDAR scan metadata properties."""
    
    @settings(
        max_examples=MIN_EXAMPLES,
        deadline=TEST_TIMEOUT * 1000,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    @given(
        robot_x=st.floats(min_value=-20.0, max_value=20.0),
        robot_y=st.floats(min_value=-20.0, max_value=20.0)
    )
    def test_scan_metadata_consistency(self, simulation, robot_x, robot_y):
        """
        Property: LiDAR scan metadata must be consistent across all scans.
        
        This ensures the sensor configuration remains stable.
        """
        # Set robot position
        simulation.set_robot_pose(robot_x, robot_y, 0.0)
        simulation.step(0.0, 0.0)
        
        # Get LiDAR scan
        obs = simulation.get_observation()
        lidar_scan = obs['lidar_scan']
        
        # Check metadata
        assert lidar_scan.angle_min == 0.0
        assert abs(lidar_scan.angle_max - 2 * np.pi) < 1e-3
        assert lidar_scan.range_min == 0.1
        assert lidar_scan.range_max == 10.0
        assert lidar_scan.header.frame_id == 'laser'
        
        # Check angle increment
        expected_increment = 2 * np.pi / 360
        assert abs(lidar_scan.angle_increment - expected_increment) < 1e-6


class TestLiDARNoiseModel:
    """Test LiDAR noise characteristics."""
    
    @settings(
        max_examples=50,  # Fewer examples for statistical tests
        deadline=TEST_TIMEOUT * 1000,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    @given(
        robot_x=st.floats(min_value=-20.0, max_value=20.0),
        robot_y=st.floats(min_value=-20.0, max_value=20.0)
    )
    def test_noise_is_bounded(self, simulation, robot_x, robot_y):
        """
        Property: Noise should not cause readings to violate range limits.
        
        Even with Gaussian noise, readings must be clamped to valid range.
        """
        # Set robot position
        simulation.set_robot_pose(robot_x, robot_y, 0.0)
        
        # Take multiple scans to observe noise
        scans = []
        for _ in range(10):
            simulation.step(0.0, 0.0)
            obs = simulation.get_observation()
            scans.append(obs['lidar_scan'].ranges)
        
        # Check all readings are within bounds
        for scan in scans:
            for range_val in scan:
                assert 0.1 <= range_val <= 10.0 or np.isinf(range_val), \
                    f"Noisy range {range_val} outside valid bounds"


class TestLiDAROcclusion:
    """Test LiDAR occlusion effects."""
    
    @settings(
        max_examples=MIN_EXAMPLES,
        deadline=TEST_TIMEOUT * 1000,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    @given(
        robot_x=st.floats(min_value=-20.0, max_value=20.0),
        robot_y=st.floats(min_value=-20.0, max_value=20.0),
        robot_yaw=st.floats(min_value=-np.pi, max_value=np.pi)
    )
    def test_detects_nearby_obstacles(self, simulation, robot_x, robot_y, robot_yaw):
        """
        Property: LiDAR should detect obstacles within range.
        
        When the robot is in the warehouse, at least some rays should
        detect obstacles (walls, shelves, or dynamic obstacles).
        """
        # Avoid positions too close to walls (within 0.5m)
        assume(abs(robot_x) < 24.0 and abs(robot_y) < 24.0)
        
        # Set robot position
        simulation.set_robot_pose(robot_x, robot_y, robot_yaw)
        simulation.step(0.0, 0.0)
        
        # Get LiDAR scan
        obs = simulation.get_observation()
        lidar_scan = obs['lidar_scan']
        
        # Count rays that detect obstacles
        detected_obstacles = sum(1 for r in lidar_scan.ranges if r < 10.0)
        
        # In a warehouse environment, we should detect something
        # (walls, shelves, or dynamic obstacles)
        assert detected_obstacles > 0, \
            f"No obstacles detected at position ({robot_x:.2f}, {robot_y:.2f})"
    
    @settings(
        max_examples=MIN_EXAMPLES,
        deadline=TEST_TIMEOUT * 1000,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    @given(
        robot_yaw=st.floats(min_value=-np.pi, max_value=np.pi)
    )
    def test_detects_walls_near_boundary(self, simulation, robot_yaw):
        """
        Property: LiDAR should detect walls when robot is near warehouse boundary.
        
        When positioned near a wall, the LiDAR should detect it at close range.
        """
        # Position robot near north wall (but not too close to avoid collision)
        robot_x = 0.0
        robot_y = 20.0  # 5m from north wall at y=25
        
        simulation.set_robot_pose(robot_x, robot_y, robot_yaw)
        simulation.step(0.0, 0.0)
        
        # Get LiDAR scan
        obs = simulation.get_observation()
        lidar_scan = obs['lidar_scan']
        
        # Find minimum range (should detect nearby wall)
        min_range = min(lidar_scan.ranges)
        
        # Should detect wall within reasonable distance
        assert min_range < 8.0, \
            f"Expected to detect nearby wall, but minimum range is {min_range:.2f}m"


class TestLiDARPublishingFrequency:
    """
    Property 4: Sensor Publishing Frequency
    
    For any sensor data stream over a 10-second window, the average publishing 
    frequency should be within the specified range (20 Hz for LiDAR).
    
    **Validates: Requirements 2.3**
    
    Note: This is a simplified test that checks the configured frequency.
    A full integration test would measure actual publishing rates.
    """
    
    def test_configured_frequency(self, simulation):
        """
        Property: LiDAR should be configured to publish at 20 Hz.
        
        This checks the sensor publish period configuration.
        """
        # Check the configured publish period
        expected_period = 0.05  # 20 Hz = 1/20 = 0.05 seconds
        assert simulation._sensor_publish_period == expected_period, \
            f"Expected publish period {expected_period}s, got {simulation._sensor_publish_period}s"
        
        # Verify this corresponds to 20 Hz
        frequency = 1.0 / simulation._sensor_publish_period
        assert abs(frequency - 20.0) < 0.1, \
            f"Expected 20 Hz, got {frequency:.1f} Hz"


class TestLiDARRayCasting:
    """Test ray casting implementation details."""
    
    @settings(
        max_examples=MIN_EXAMPLES,
        deadline=TEST_TIMEOUT * 1000,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    @given(
        robot_x=st.floats(min_value=-20.0, max_value=20.0),
        robot_y=st.floats(min_value=-20.0, max_value=20.0),
        robot_yaw=st.floats(min_value=-np.pi, max_value=np.pi)
    )
    def test_scan_is_deterministic_without_noise(self, simulation, robot_x, robot_y, robot_yaw):
        """
        Property: Without noise, consecutive scans from same position should be similar.
        
        This tests that the ray casting is deterministic (noise is the only
        source of variation).
        """
        # Set robot position
        simulation.set_robot_pose(robot_x, robot_y, robot_yaw)
        
        # Take two scans
        simulation.step(0.0, 0.0)
        obs1 = simulation.get_observation()
        scan1 = obs1['lidar_scan'].ranges
        
        simulation.step(0.0, 0.0)
        obs2 = simulation.get_observation()
        scan2 = obs2['lidar_scan'].ranges
        
        # Scans should be similar (within noise tolerance)
        # Noise std is 0.01m, so 3-sigma is 0.03m
        differences = [abs(r1 - r2) for r1, r2 in zip(scan1, scan2)]
        max_diff = max(differences)
        
        # Most differences should be small (within 3-sigma of noise)
        assert max_diff < 0.1, \
            f"Consecutive scans differ by {max_diff:.3f}m, expected < 0.1m"


# Additional unit tests for edge cases
class TestLiDAREdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_scan_at_warehouse_center(self, simulation):
        """Test LiDAR scan from center of warehouse."""
        simulation.set_robot_pose(0.0, 0.0, 0.0)
        simulation.step(0.0, 0.0)
        
        obs = simulation.get_observation()
        lidar_scan = obs['lidar_scan']
        
        # Should detect obstacles in all directions
        assert len(lidar_scan.ranges) == 360
        assert all(0.1 <= r <= 10.0 for r in lidar_scan.ranges)
    
    def test_scan_near_corner(self, simulation):
        """Test LiDAR scan from near a corner."""
        # Position near southwest corner
        simulation.set_robot_pose(-20.0, -20.0, 0.0)
        simulation.step(0.0, 0.0)
        
        obs = simulation.get_observation()
        lidar_scan = obs['lidar_scan']
        
        # Should detect nearby walls
        min_range = min(lidar_scan.ranges)
        assert min_range < 6.0, "Should detect nearby walls"
    
    def test_scan_between_shelves(self, simulation):
        """Test LiDAR scan from between shelf rows."""
        # Position between shelf rows
        simulation.set_robot_pose(-10.0, 0.0, 0.0)
        simulation.step(0.0, 0.0)
        
        obs = simulation.get_observation()
        lidar_scan = obs['lidar_scan']
        
        # Should detect shelves on both sides
        assert len(lidar_scan.ranges) == 360
        # At least some rays should detect nearby shelves
        close_detections = sum(1 for r in lidar_scan.ranges if r < 5.0)
        assert close_detections > 0, "Should detect nearby shelves"
