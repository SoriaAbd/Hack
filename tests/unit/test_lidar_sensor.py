"""
Unit tests for LiDAR sensor simulation.

Tests specific scenarios and edge cases for the LiDAR sensor implementation.
"""

import pytest
import numpy as np

from adaptnav.simulation.mujoco_simulation import MuJoCoSimulation


@pytest.fixture
def simulation():
    """Create a simulation instance for testing."""
    sim = MuJoCoSimulation(render_mode=None)
    sim.reset()
    yield sim
    sim.destroy_node()


class TestLiDARBasicFunctionality:
    """Test basic LiDAR functionality."""
    
    def test_lidar_scan_structure(self, simulation):
        """Test that LiDAR scan has correct structure."""
        obs = simulation.get_observation()
        lidar_scan = obs['lidar_scan']
        
        # Check message type and fields
        assert hasattr(lidar_scan, 'header')
        assert hasattr(lidar_scan, 'ranges')
        assert hasattr(lidar_scan, 'angle_min')
        assert hasattr(lidar_scan, 'angle_max')
        assert hasattr(lidar_scan, 'angle_increment')
        assert hasattr(lidar_scan, 'range_min')
        assert hasattr(lidar_scan, 'range_max')
    
    def test_lidar_360_degree_coverage(self, simulation):
        """Test that LiDAR provides 360° coverage."""
        obs = simulation.get_observation()
        lidar_scan = obs['lidar_scan']
        
        assert lidar_scan.angle_min == 0.0
        assert abs(lidar_scan.angle_max - 2 * np.pi) < 1e-3
        assert len(lidar_scan.ranges) == 360
    
    def test_lidar_one_degree_resolution(self, simulation):
        """Test that LiDAR has 1° angular resolution."""
        obs = simulation.get_observation()
        lidar_scan = obs['lidar_scan']
        
        expected_increment = 2 * np.pi / 360  # 1 degree in radians
        assert abs(lidar_scan.angle_increment - expected_increment) < 1e-6
    
    def test_lidar_range_limits(self, simulation):
        """Test that LiDAR has correct range limits."""
        obs = simulation.get_observation()
        lidar_scan = obs['lidar_scan']
        
        assert lidar_scan.range_min == 0.1  # 0.1m minimum
        assert lidar_scan.range_max == 10.0  # 10m maximum


class TestLiDARWallDetection:
    """Test LiDAR detection of walls."""
    
    def test_detects_north_wall(self, simulation):
        """Test detection of north wall."""
        # Position robot facing north wall
        simulation.set_robot_pose(0.0, 20.0, np.pi / 2)
        simulation.step(0.0, 0.0)
        
        obs = simulation.get_observation()
        lidar_scan = obs['lidar_scan']
        
        # Ray at 90° (north) should detect wall at ~5m
        # Index 90 corresponds to 90° (north direction relative to robot)
        north_ray_idx = 90
        north_range = lidar_scan.ranges[north_ray_idx]
        
        # Should detect wall within reasonable distance
        assert 4.0 < north_range < 6.0, \
            f"Expected wall at ~5m, got {north_range:.2f}m"
    
    def test_detects_south_wall(self, simulation):
        """Test detection of south wall."""
        # Position robot facing south wall
        simulation.set_robot_pose(0.0, -20.0, -np.pi / 2)
        simulation.step(0.0, 0.0)
        
        obs = simulation.get_observation()
        lidar_scan = obs['lidar_scan']
        
        # Should detect wall in forward direction
        min_range = min(lidar_scan.ranges)
        assert min_range < 6.0, "Should detect nearby south wall"
    
    def test_detects_east_wall(self, simulation):
        """Test detection of east wall."""
        # Position robot facing east wall
        simulation.set_robot_pose(20.0, 0.0, 0.0)
        simulation.step(0.0, 0.0)
        
        obs = simulation.get_observation()
        lidar_scan = obs['lidar_scan']
        
        # Should detect wall in forward direction
        min_range = min(lidar_scan.ranges)
        assert min_range < 6.0, "Should detect nearby east wall"
    
    def test_detects_west_wall(self, simulation):
        """Test detection of west wall."""
        # Position robot facing west wall
        simulation.set_robot_pose(-20.0, 0.0, np.pi)
        simulation.step(0.0, 0.0)
        
        obs = simulation.get_observation()
        lidar_scan = obs['lidar_scan']
        
        # Should detect wall in forward direction
        min_range = min(lidar_scan.ranges)
        assert min_range < 6.0, "Should detect nearby west wall"


class TestLiDARShelfDetection:
    """Test LiDAR detection of shelves."""
    
    def test_detects_shelves_from_center(self, simulation):
        """Test detection of shelves from warehouse center."""
        # Position at center
        simulation.set_robot_pose(0.0, 0.0, 0.0)
        simulation.step(0.0, 0.0)
        
        obs = simulation.get_observation()
        lidar_scan = obs['lidar_scan']
        
        # Should detect shelves in multiple directions
        close_detections = sum(1 for r in lidar_scan.ranges if r < 8.0)
        assert close_detections > 50, \
            f"Expected to detect shelves, got {close_detections} close detections"
    
    def test_detects_shelf_directly_ahead(self, simulation):
        """Test detection of shelf directly ahead."""
        # Position facing shelf at x=-15
        simulation.set_robot_pose(-10.0, 0.0, np.pi)
        simulation.step(0.0, 0.0)
        
        obs = simulation.get_observation()
        lidar_scan = obs['lidar_scan']
        
        # Forward rays should detect shelf at ~5m
        # Check rays in forward direction (around 180°)
        forward_rays = lidar_scan.ranges[170:190]
        min_forward = min(forward_rays)
        
        assert min_forward < 7.0, \
            f"Expected to detect shelf ahead, minimum range is {min_forward:.2f}m"


class TestLiDARDynamicObstacles:
    """Test LiDAR detection of dynamic obstacles."""
    
    def test_detects_workers(self, simulation):
        """Test detection of worker obstacles."""
        # Workers are at (-8, -5) and (8, 5)
        # Position robot near worker 1
        simulation.set_robot_pose(-10.0, -5.0, 0.0)
        simulation.step(0.0, 0.0)
        
        obs = simulation.get_observation()
        lidar_scan = obs['lidar_scan']
        
        # Should detect worker within 3m
        min_range = min(lidar_scan.ranges)
        assert min_range < 3.0, \
            f"Expected to detect nearby worker, minimum range is {min_range:.2f}m"
    
    def test_detects_forklift(self, simulation):
        """Test detection of forklift obstacle."""
        # Forklift starts at (0, -15)
        # Position robot near forklift
        simulation.set_robot_pose(0.0, -12.0, -np.pi / 2)
        simulation.step(0.0, 0.0)
        
        obs = simulation.get_observation()
        lidar_scan = obs['lidar_scan']
        
        # Should detect forklift within 5m
        min_range = min(lidar_scan.ranges)
        assert min_range < 5.0, \
            f"Expected to detect nearby forklift, minimum range is {min_range:.2f}m"


class TestLiDARNoiseCharacteristics:
    """Test LiDAR noise model."""
    
    def test_noise_adds_variation(self, simulation):
        """Test that noise adds variation to measurements."""
        # Position robot at fixed location
        simulation.set_robot_pose(0.0, 0.0, 0.0)
        
        # Take multiple scans
        scans = []
        for _ in range(20):
            simulation.step(0.0, 0.0)
            obs = simulation.get_observation()
            scans.append(obs['lidar_scan'].ranges)
        
        # Calculate variance for each ray
        variances = []
        for ray_idx in range(360):
            ray_values = [scan[ray_idx] for scan in scans]
            variance = np.var(ray_values)
            variances.append(variance)
        
        # Average variance should be non-zero (noise is present)
        avg_variance = np.mean(variances)
        assert avg_variance > 0, "Expected noise to add variation"
        
        # Variance should be reasonable (not too large)
        # Noise std is 0.01m, so variance should be ~0.0001
        assert avg_variance < 0.01, \
            f"Variance {avg_variance:.6f} is too large"
    
    def test_noise_is_gaussian_like(self, simulation):
        """Test that noise has Gaussian-like properties."""
        # Position robot at fixed location
        simulation.set_robot_pose(0.0, 0.0, 0.0)
        
        # Take many scans and collect measurements for one ray
        ray_idx = 0
        measurements = []
        for _ in range(100):
            simulation.step(0.0, 0.0)
            obs = simulation.get_observation()
            measurements.append(obs['lidar_scan'].ranges[ray_idx])
        
        # Calculate statistics
        mean_val = np.mean(measurements)
        std_val = np.std(measurements)
        
        # Standard deviation should be close to configured noise (0.01m)
        # Allow some tolerance due to finite sampling
        assert 0.005 < std_val < 0.02, \
            f"Expected std ~0.01m, got {std_val:.4f}m"


class TestLiDAROcclusion:
    """Test LiDAR occlusion effects."""
    
    def test_closer_obstacle_occludes_farther(self, simulation):
        """Test that closer obstacles occlude farther ones."""
        # Position robot so a shelf is between it and a wall
        # Robot at (-12, 0), shelf at (-15, 0), wall at (-25, 0)
        simulation.set_robot_pose(-12.0, 0.0, np.pi)
        simulation.step(0.0, 0.0)
        
        obs = simulation.get_observation()
        lidar_scan = obs['lidar_scan']
        
        # Forward rays should detect shelf (~3m), not wall (~13m)
        forward_rays = lidar_scan.ranges[170:190]
        min_forward = min(forward_rays)
        
        assert min_forward < 5.0, \
            f"Expected to detect shelf at ~3m, got {min_forward:.2f}m"
        assert min_forward > 1.0, \
            f"Detection too close: {min_forward:.2f}m"


class TestLiDARFrameId:
    """Test LiDAR frame ID."""
    
    def test_frame_id_is_laser(self, simulation):
        """Test that LiDAR scan has correct frame_id."""
        obs = simulation.get_observation()
        lidar_scan = obs['lidar_scan']
        
        assert lidar_scan.header.frame_id == 'laser', \
            f"Expected frame_id 'laser', got '{lidar_scan.header.frame_id}'"


class TestLiDARTimestamp:
    """Test LiDAR timestamp."""
    
    def test_timestamp_increases(self, simulation):
        """Test that LiDAR timestamps increase over time."""
        # Get first scan
        obs1 = simulation.get_observation()
        scan1 = obs1['lidar_scan']
        time1 = scan1.header.stamp.sec + scan1.header.stamp.nanosec * 1e-9
        
        # Step simulation
        simulation.step(0.0, 0.0)
        
        # Get second scan
        obs2 = simulation.get_observation()
        scan2 = obs2['lidar_scan']
        time2 = scan2.header.stamp.sec + scan2.header.stamp.nanosec * 1e-9
        
        # Time should increase
        assert time2 >= time1, \
            f"Timestamp should increase: {time1} -> {time2}"


class TestLiDARRangeValidation:
    """Test range validation and clamping."""
    
    def test_no_negative_ranges(self, simulation):
        """Test that no ranges are negative."""
        # Test from multiple positions
        positions = [
            (0.0, 0.0, 0.0),
            (-10.0, -10.0, 0.0),
            (10.0, 10.0, 0.0),
            (-20.0, 20.0, np.pi / 4)
        ]
        
        for x, y, yaw in positions:
            simulation.set_robot_pose(x, y, yaw)
            simulation.step(0.0, 0.0)
            
            obs = simulation.get_observation()
            lidar_scan = obs['lidar_scan']
            
            # No negative ranges
            for i, r in enumerate(lidar_scan.ranges):
                assert r >= 0.0, \
                    f"Negative range {r} at position ({x}, {y}), ray {i}"
    
    def test_no_ranges_below_minimum(self, simulation):
        """Test that no ranges are below minimum (0.1m)."""
        # Test from multiple positions
        positions = [
            (0.0, 0.0, 0.0),
            (-10.0, -10.0, 0.0),
            (10.0, 10.0, 0.0)
        ]
        
        for x, y, yaw in positions:
            simulation.set_robot_pose(x, y, yaw)
            simulation.step(0.0, 0.0)
            
            obs = simulation.get_observation()
            lidar_scan = obs['lidar_scan']
            
            # No ranges below minimum (unless infinite)
            for i, r in enumerate(lidar_scan.ranges):
                if not np.isinf(r):
                    assert r >= 0.1, \
                        f"Range {r} below minimum at position ({x}, {y}), ray {i}"
    
    def test_no_ranges_above_maximum(self, simulation):
        """Test that no finite ranges are above maximum (10m)."""
        # Test from multiple positions
        positions = [
            (0.0, 0.0, 0.0),
            (-10.0, -10.0, 0.0),
            (10.0, 10.0, 0.0)
        ]
        
        for x, y, yaw in positions:
            simulation.set_robot_pose(x, y, yaw)
            simulation.step(0.0, 0.0)
            
            obs = simulation.get_observation()
            lidar_scan = obs['lidar_scan']
            
            # No finite ranges above maximum
            for i, r in enumerate(lidar_scan.ranges):
                if not np.isinf(r):
                    assert r <= 10.0, \
                        f"Range {r} above maximum at position ({x}, {y}), ray {i}"
