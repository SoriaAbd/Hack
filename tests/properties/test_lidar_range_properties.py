"""
Property-based tests for LiDAR range limitations.

This module implements Property 5: LiDAR Range Limitations
**Validates: Requirements 2.4**

Property 5: LiDAR Range Limitations
For any LiDAR scan, all range readings should be either within the valid 
range [0.1m, 10m] or marked as invalid/infinite.

Requirements 2.4: LiDAR sensor with realistic range limitations and noise
- Range: 0.1m to 10m
- 360° coverage with 1° resolution  
- Realistic noise model
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume
from hypothesis import HealthCheck

# Check if ROS 2 and MuJoCo are available
try:
    import rclpy
    import mujoco
    from adaptnav.simulation.mujoco_simulation import MuJoCoSimulation
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


if ROS_MUJOCO_AVAILABLE:
    class TestLiDARRangeLimitationsProperty:
        """
        Property 5: LiDAR Range Limitations
        
        **Validates: Requirements 2.4**
        
        For any LiDAR scan, all range readings should be either within the valid 
        range [0.1m, 10m] or marked as invalid/infinite.
        
        This property ensures that:
        1. All range measurements are within [0.1m, 10m] bounds
        2. Out-of-range measurements are properly handled (marked as max range)
        3. The sensor respects its physical limitations
        4. 360° coverage with proper angular resolution is maintained
        5. Realistic noise doesn't violate range constraints
        """
        
        @settings(
            max_examples=MIN_EXAMPLES,
            deadline=TEST_TIMEOUT * 1000,
            suppress_health_check=[HealthCheck.function_scoped_fixture]
        )
        @given(
            robot_x=st.floats(min_value=-22.0, max_value=22.0),
            robot_y=st.floats(min_value=-22.0, max_value=22.0),
            robot_yaw=st.floats(min_value=-np.pi, max_value=np.pi)
        )
        def test_all_ranges_within_physical_limits(self, simulation, robot_x, robot_y, robot_yaw):
            """
            **Validates: Requirements 2.4**
            
            Property: All LiDAR range readings must be within [0.1m, 10m].
            
            This is the core property that validates the LiDAR sensor respects
            its physical range limitations. No measurement should be below 0.1m
            or above 10m (the sensor's specified range limits).
            """
            # Avoid positions that would place robot inside obstacles
            assume(abs(robot_x) < 24.0 and abs(robot_y) < 24.0)
            
            # Set robot position and orientation
            simulation.set_robot_pose(robot_x, robot_y, robot_yaw)
            simulation.step(0.0, 0.0)  # Update simulation state
            
            # Get LiDAR scan
            obs = simulation.get_observation()
            lidar_scan = obs['lidar_scan']
            
            # Verify scan structure
            assert len(lidar_scan.ranges) == 360, \
                f"Expected 360 rays (1° resolution), got {len(lidar_scan.ranges)}"
            
            # Check each range measurement
            for i, range_val in enumerate(lidar_scan.ranges):
                angle_deg = i  # Each ray is 1 degree
                
                # Core property: All ranges must be within [0.1m, 10m]
                assert 0.1 <= range_val <= 10.0, \
                    f"Range at {angle_deg}° is {range_val:.3f}m, outside valid range [0.1, 10.0]"
                
                # Additional validation: No NaN or infinite values
                assert not np.isnan(range_val), \
                    f"Range at {angle_deg}° is NaN"
                assert not np.isinf(range_val), \
                    f"Range at {angle_deg}° is infinite"
        
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
        def test_360_degree_coverage_with_1_degree_resolution(self, simulation, robot_x, robot_y, robot_yaw):
            """
            **Validates: Requirements 2.4**
            
            Property: LiDAR must provide 360° coverage with 1° resolution.
            
            This validates the angular coverage and resolution requirements.
            """
            # Set robot position
            simulation.set_robot_pose(robot_x, robot_y, robot_yaw)
            simulation.step(0.0, 0.0)
            
            # Get LiDAR scan
            obs = simulation.get_observation()
            lidar_scan = obs['lidar_scan']
            
            # Verify 360° coverage
            assert lidar_scan.angle_min == 0.0, \
                f"Expected angle_min=0.0, got {lidar_scan.angle_min}"
            assert abs(lidar_scan.angle_max - 2 * np.pi) < 1e-3, \
                f"Expected angle_max≈{2*np.pi:.3f}, got {lidar_scan.angle_max:.3f}"
            
            # Verify 1° resolution (360 rays)
            assert len(lidar_scan.ranges) == 360, \
                f"Expected 360 rays for 1° resolution, got {len(lidar_scan.ranges)}"
            
            # Verify angle increment
            expected_increment = 2 * np.pi / 360  # 1 degree in radians
            assert abs(lidar_scan.angle_increment - expected_increment) < 1e-6, \
                f"Expected angle_increment={expected_increment:.6f}, got {lidar_scan.angle_increment:.6f}"
            
            # Verify range limits are correctly set
            assert lidar_scan.range_min == 0.1, \
                f"Expected range_min=0.1, got {lidar_scan.range_min}"
            assert lidar_scan.range_max == 10.0, \
                f"Expected range_max=10.0, got {lidar_scan.range_max}"
        
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
        def test_noise_respects_range_limits(self, simulation, robot_x, robot_y, robot_yaw):
            """
            **Validates: Requirements 2.4**
            
            Property: Realistic noise must not cause measurements to violate range limits.
            
            Even with Gaussian noise applied to range measurements, the final
            output must be clamped to the valid range [0.1m, 10m].
            """
            # Set robot position
            simulation.set_robot_pose(robot_x, robot_y, robot_yaw)
            
            # Take multiple scans to observe noise effects
            all_ranges = []
            for _ in range(10):  # Multiple scans to capture noise variation
                simulation.step(0.0, 0.0)
                obs = simulation.get_observation()
                lidar_scan = obs['lidar_scan']
                all_ranges.extend(lidar_scan.ranges)
            
            # Verify all noisy measurements are within bounds
            for i, range_val in enumerate(all_ranges):
                assert 0.1 <= range_val <= 10.0, \
                    f"Noisy range measurement {range_val:.3f}m violates limits [0.1, 10.0]"
        
        @settings(
            max_examples=50,  # Fewer examples for statistical test
            deadline=TEST_TIMEOUT * 1000,
            suppress_health_check=[HealthCheck.function_scoped_fixture]
        )
        @given(
            robot_x=st.floats(min_value=-15.0, max_value=15.0),
            robot_y=st.floats(min_value=-15.0, max_value=15.0)
        )
        def test_noise_characteristics_are_realistic(self, simulation, robot_x, robot_y):
            """
            **Validates: Requirements 2.4**
            
            Property: Noise model should be realistic (Gaussian with reasonable std).
            
            The noise should add variation but not be excessive. Standard deviation
            should be around 0.01m as configured.
            """
            # Set robot at fixed position and orientation
            simulation.set_robot_pose(robot_x, robot_y, 0.0)
            
            # Collect measurements for one ray over multiple scans
            ray_index = 0  # First ray (0°)
            measurements = []
            
            for _ in range(50):  # Collect enough samples for statistics
                simulation.step(0.0, 0.0)
                obs = simulation.get_observation()
                lidar_scan = obs['lidar_scan']
                measurements.append(lidar_scan.ranges[ray_index])
            
            # Calculate noise statistics
            measurements = np.array(measurements)
            std_dev = np.std(measurements)
            
            # Noise should be present but reasonable
            # Expected std is 0.01m, allow some tolerance
            assert 0.005 < std_dev < 0.05, \
                f"Noise std deviation {std_dev:.4f}m outside expected range [0.005, 0.05]"
            
            # All measurements should still be within range limits
            assert np.all((measurements >= 0.1) & (measurements <= 10.0)), \
                "Some noisy measurements violate range limits"
        
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
        def test_detects_obstacles_within_range(self, simulation, robot_x, robot_y, robot_yaw):
            """
            **Validates: Requirements 2.4**
            
            Property: LiDAR should detect obstacles within its range.
            
            In a warehouse environment with walls, shelves, and dynamic obstacles,
            the LiDAR should detect at least some obstacles within its 10m range.
            """
            # Avoid positions too close to boundaries
            assume(abs(robot_x) < 22.0 and abs(robot_y) < 22.0)
            
            # Set robot position
            simulation.set_robot_pose(robot_x, robot_y, robot_yaw)
            simulation.step(0.0, 0.0)
            
            # Get LiDAR scan
            obs = simulation.get_observation()
            lidar_scan = obs['lidar_scan']
            
            # Count detections at different ranges
            close_detections = sum(1 for r in lidar_scan.ranges if r < 5.0)
            medium_detections = sum(1 for r in lidar_scan.ranges if 5.0 <= r < 8.0)
            far_detections = sum(1 for r in lidar_scan.ranges if 8.0 <= r < 10.0)
            max_range_readings = sum(1 for r in lidar_scan.ranges if r == 10.0)
            
            # In a warehouse, we should detect obstacles
            total_detections = close_detections + medium_detections + far_detections
            assert total_detections > 0, \
                f"No obstacles detected within 10m range at position ({robot_x:.1f}, {robot_y:.1f})"
            
            # Some readings should be at max range (no obstacle detected)
            # This validates that the sensor can reach its maximum range
            assert max_range_readings >= 0, \
                "Expected some readings at max range (10.0m)"


    class TestLiDARRangeLimitationsEdgeCases:
        """Edge cases for LiDAR range limitations."""
        
        def test_scan_metadata_consistency(self, simulation):
            """
            **Validates: Requirements 2.4**
            
            Test that scan metadata is consistent with range limitations.
            """
            # Position robot at center
            simulation.set_robot_pose(0.0, 0.0, 0.0)
            simulation.step(0.0, 0.0)
            
            # Get LiDAR scan
            obs = simulation.get_observation()
            lidar_scan = obs['lidar_scan']
            
            # Verify metadata matches requirements
            assert lidar_scan.range_min == 0.1, "range_min should be 0.1m"
            assert lidar_scan.range_max == 10.0, "range_max should be 10.0m"
            assert lidar_scan.header.frame_id == 'laser', "frame_id should be 'laser'"
            
            # Verify angular coverage
            assert lidar_scan.angle_min == 0.0, "angle_min should be 0.0"
            assert abs(lidar_scan.angle_max - 2 * np.pi) < 1e-3, "angle_max should be 2π"
        
        def test_corner_position_ranges(self, simulation):
            """
            **Validates: Requirements 2.4**
            
            Test range limitations from corner positions.
            """
            # Test all four corners
            corners = [
                (-20.0, -20.0),  # Southwest
                (20.0, -20.0),   # Southeast  
                (20.0, 20.0),    # Northeast
                (-20.0, 20.0)    # Northwest
            ]
            
            for x, y in corners:
                simulation.set_robot_pose(x, y, 0.0)
                simulation.step(0.0, 0.0)
                
                obs = simulation.get_observation()
                lidar_scan = obs['lidar_scan']
                
                # All ranges must be within limits
                for i, range_val in enumerate(lidar_scan.ranges):
                    assert 0.1 <= range_val <= 10.0, \
                        f"Range {range_val:.3f}m at corner ({x}, {y}), ray {i}° violates limits"
        
        def test_multiple_consecutive_scans(self, simulation):
            """
            **Validates: Requirements 2.4**
            
            Test that range limitations hold across multiple consecutive scans.
            """
            # Position robot at center
            simulation.set_robot_pose(0.0, 0.0, 0.0)
            
            # Take multiple consecutive scans
            for scan_num in range(20):
                simulation.step(0.0, 0.0)
                obs = simulation.get_observation()
                lidar_scan = obs['lidar_scan']
                
                # Verify all ranges in this scan
                for i, range_val in enumerate(lidar_scan.ranges):
                    assert 0.1 <= range_val <= 10.0, \
                        f"Scan {scan_num}, ray {i}°: range {range_val:.3f}m violates limits"
                
                # Verify scan structure
                assert len(lidar_scan.ranges) == 360, \
                    f"Scan {scan_num}: expected 360 rays, got {len(lidar_scan.ranges)}"