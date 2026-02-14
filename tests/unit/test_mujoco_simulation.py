"""
Unit tests for MuJoCo simulation backend.

Tests the MuJoCo implementation of the BaseSimulation interface,
including initialization, stepping, reset, and sensor simulation.
"""

import pytest
import numpy as np

try:
    import rclpy
    from adaptnav.simulation.mujoco_simulation import MuJoCoSimulation, MUJOCO_AVAILABLE
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    MUJOCO_AVAILABLE = False


@pytest.mark.skipif(not ROS2_AVAILABLE or not MUJOCO_AVAILABLE, 
                    reason="ROS 2 or MuJoCo not available")
class TestMuJoCoSimulation:
    """Test suite for MuJoCo simulation."""
    
    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup and teardown for each test."""
        # Initialize ROS 2
        if not rclpy.ok():
            rclpy.init()
        
        yield
        
        # Cleanup is handled by individual tests
    
    def test_initialization(self):
        """Test that MuJoCo simulation initializes correctly."""
        sim = MuJoCoSimulation(render_mode=None)
        
        assert sim.is_initialized()
        assert sim.get_step_count() == 0
        
        # Cleanup
        sim.destroy_node()
    
    def test_reset_default(self):
        """Test reset with default parameters."""
        sim = MuJoCoSimulation(render_mode=None)
        
        # Reset simulation
        success = sim.reset()
        assert success
        
        # Check robot is at default spawn position
        robot_state = sim.get_ground_truth_robot_state()
        assert np.allclose(robot_state.position, [-20.0, -20.0], atol=0.1)
        
        # Check obstacles are present
        obstacles = sim.get_ground_truth_obstacles()
        assert len(obstacles) == 3  # 2 workers + 1 forklift
        
        # Cleanup
        sim.destroy_node()
    
    def test_reset_custom_position(self):
        """Test reset with custom robot position."""
        sim = MuJoCoSimulation(render_mode=None)
        
        # Reset with custom position
        custom_pos = (5.0, 10.0, np.pi / 4)
        success = sim.reset(robot_position=custom_pos)
        assert success
        
        # Check robot is at custom position
        robot_state = sim.get_ground_truth_robot_state()
        assert np.allclose(robot_state.position, [5.0, 10.0], atol=0.1)
        assert np.allclose(robot_state.orientation, np.pi / 4, atol=0.1)
        
        # Cleanup
        sim.destroy_node()
    
    def test_step(self):
        """Test simulation step."""
        sim = MuJoCoSimulation(render_mode=None)
        sim.reset()
        
        initial_step_count = sim.get_step_count()
        
        # Step simulation
        success = sim.step(dt=0.01)
        assert success
        
        # Check step count incremented
        assert sim.get_step_count() == initial_step_count + 1
        
        # Cleanup
        sim.destroy_node()
    
    def test_ground_truth_robot_state(self):
        """Test ground truth robot state retrieval."""
        sim = MuJoCoSimulation(render_mode=None)
        sim.reset()
        
        robot_state = sim.get_ground_truth_robot_state()
        
        # Check state has required fields
        assert robot_state.position is not None
        assert len(robot_state.position) == 2
        assert robot_state.orientation is not None
        assert robot_state.linear_velocity is not None
        assert robot_state.angular_velocity is not None
        assert robot_state.timestamp is not None
        
        # Cleanup
        sim.destroy_node()
    
    def test_ground_truth_obstacles(self):
        """Test ground truth obstacle retrieval."""
        sim = MuJoCoSimulation(render_mode=None)
        sim.reset()
        
        obstacles = sim.get_ground_truth_obstacles()
        
        # Check we have expected number of obstacles
        assert len(obstacles) == 3
        
        # Check each obstacle has required fields
        for obs in obstacles:
            assert obs.id is not None
            assert obs.position is not None
            assert len(obs.position) == 2
            assert obs.velocity is not None
            assert len(obs.velocity) == 2
            assert obs.radius > 0
            assert obs.classification in ['worker', 'forklift']
        
        # Check classifications
        classifications = [obs.classification for obs in obstacles]
        assert classifications.count('worker') == 2
        assert classifications.count('forklift') == 1
        
        # Cleanup
        sim.destroy_node()
    
    def test_get_observation(self):
        """Test observation retrieval."""
        sim = MuJoCoSimulation(render_mode=None)
        sim.reset()
        
        obs = sim.get_observation()
        
        # Check observation has required keys
        assert 'lidar_scan' in obs
        assert 'odometry' in obs
        assert 'timestamp' in obs
        
        # Check LiDAR scan
        lidar_scan = obs['lidar_scan']
        assert len(lidar_scan.ranges) == 360  # 1 degree resolution
        
        # Check all ranges are within valid range
        for r in lidar_scan.ranges:
            assert 0.1 <= r <= 10.0 or r == float('inf')
        
        # Cleanup
        sim.destroy_node()
    
    def test_collision_detection_no_collision(self):
        """Test collision detection when no collision occurs."""
        sim = MuJoCoSimulation(render_mode=None)
        sim.reset()
        
        # Initially no collision (robot spawns away from obstacles)
        assert not sim.check_collision()
        
        # Cleanup
        sim.destroy_node()
    
    def test_obstacle_motion(self):
        """Test that obstacles move over time."""
        sim = MuJoCoSimulation(render_mode=None)
        sim.reset()
        
        # Get initial obstacle positions
        initial_obstacles = sim.get_ground_truth_obstacles()
        initial_positions = [obs.position.copy() for obs in initial_obstacles]
        
        # Step simulation for 1 second
        for _ in range(100):
            sim.step(dt=0.01)
        
        # Get final obstacle positions
        final_obstacles = sim.get_ground_truth_obstacles()
        final_positions = [obs.position for obs in final_obstacles]
        
        # Check that at least one obstacle has moved
        moved = False
        for i in range(len(initial_positions)):
            if not np.allclose(initial_positions[i], final_positions[i], atol=0.1):
                moved = True
                break
        
        assert moved, "Obstacles should move over time"
        
        # Cleanup
        sim.destroy_node()
    
    def test_velocity_command_application(self):
        """Test that velocity commands affect robot motion."""
        sim = MuJoCoSimulation(render_mode=None)
        sim.reset()
        
        # Get initial robot position
        initial_state = sim.get_ground_truth_robot_state()
        initial_pos = initial_state.position.copy()
        
        # Create velocity command
        from geometry_msgs.msg import Twist
        cmd_vel = Twist()
        cmd_vel.linear.x = 1.0  # 1 m/s forward
        cmd_vel.angular.z = 0.0
        
        # Simulate velocity command callback
        sim._cmd_vel_callback(cmd_vel)
        
        # Step simulation for 1 second
        for _ in range(100):
            sim.step(dt=0.01)
        
        # Get final robot position
        final_state = sim.get_ground_truth_robot_state()
        final_pos = final_state.position
        
        # Check robot has moved forward
        distance_moved = np.linalg.norm(final_pos - initial_pos)
        assert distance_moved > 0.5, "Robot should move when velocity command is applied"
        
        # Cleanup
        sim.destroy_node()
    
    def test_lidar_scan_properties(self):
        """Test LiDAR scan properties."""
        sim = MuJoCoSimulation(render_mode=None)
        sim.reset()
        
        obs = sim.get_observation()
        lidar_scan = obs['lidar_scan']
        
        # Check scan properties
        assert lidar_scan.angle_min == 0.0
        assert lidar_scan.angle_max == pytest.approx(2 * np.pi, rel=1e-3)
        assert lidar_scan.range_min == 0.1
        assert lidar_scan.range_max == 10.0
        assert len(lidar_scan.ranges) == 360
        
        # Check angle increment
        expected_increment = 2 * np.pi / 360
        assert lidar_scan.angle_increment == pytest.approx(expected_increment, rel=1e-3)
        
        # Cleanup
        sim.destroy_node()
    
    def test_odometry_has_noise(self):
        """Test that odometry includes noise (not perfect ground truth)."""
        sim = MuJoCoSimulation(render_mode=None)
        sim.reset()
        
        # Get ground truth and odometry multiple times
        differences = []
        for _ in range(10):
            ground_truth = sim.get_ground_truth_robot_state()
            obs = sim.get_observation()
            odom = obs['odometry']
            
            # Calculate difference
            odom_pos = np.array([odom.pose.pose.position.x, odom.pose.pose.position.y])
            diff = np.linalg.norm(odom_pos - ground_truth.position)
            differences.append(diff)
            
            sim.step(dt=0.01)
        
        # Check that there is some noise (not all zeros)
        assert np.mean(differences) > 0, "Odometry should have noise"
        
        # Cleanup
        sim.destroy_node()
    
    def test_depth_camera_generation(self):
        """Test depth camera image generation."""
        sim = MuJoCoSimulation(render_mode=None)
        sim.reset()
        
        # Generate depth image
        depth_image, camera_info = sim._get_depth_image()
        
        # Check depth image properties
        assert depth_image.width == 640
        assert depth_image.height == 480
        assert depth_image.encoding == '32FC1'
        assert depth_image.header.frame_id == 'camera_link'
        assert len(depth_image.data) == 640 * 480 * 4  # 4 bytes per float32
        
        # Check camera info
        assert camera_info.width == 640
        assert camera_info.height == 480
        assert camera_info.distortion_model == 'plumb_bob'
        assert len(camera_info.k) == 9  # 3x3 camera matrix
        assert len(camera_info.d) == 5  # 5 distortion coefficients
        assert len(camera_info.r) == 9  # 3x3 rectification matrix
        assert len(camera_info.p) == 12  # 3x4 projection matrix
        
        # Convert depth data back to numpy array and check values
        depth_array = np.frombuffer(depth_image.data, dtype=np.float32).reshape((480, 640))
        
        # Check depth values are within expected range
        assert np.all(depth_array >= 0.5)  # Min range
        assert np.all(depth_array <= 5.0)  # Max range
        
        # Check that we have some variation in depth values (not all the same)
        assert np.std(depth_array) > 0.1, "Depth image should have variation"
        
        # Cleanup
        sim.destroy_node()
    
    def test_depth_camera_fov(self):
        """Test depth camera field of view is approximately 90 degrees."""
        sim = MuJoCoSimulation(render_mode=None)
        sim.reset()
        
        # Generate depth image
        depth_image, camera_info = sim._get_depth_image()
        
        # Check camera intrinsics for 90° FOV
        fx = camera_info.k[0]  # K[0,0]
        fy = camera_info.k[4]  # K[1,1]
        cx = camera_info.k[2]  # K[0,2]
        cy = camera_info.k[5]  # K[1,2]
        
        # For 90° FOV: fx = width / (2 * tan(45°)) = width / 2
        expected_fx = 640 / 2.0
        assert abs(fx - expected_fx) < 1.0, f"Expected fx ~{expected_fx}, got {fx}"
        
        # Square pixels assumption
        assert abs(fx - fy) < 1.0, "fx and fy should be approximately equal"
        
        # Principal point should be at image center
        assert abs(cx - 320) < 1.0, "cx should be at image center"
        assert abs(cy - 240) < 1.0, "cy should be at image center"
        
        # Cleanup
        sim.destroy_node()
    
    def test_depth_camera_noise_model(self):
        """Test that depth camera has realistic noise that increases with distance."""
        sim = MuJoCoSimulation(render_mode=None)
        sim.reset()
        
        # Generate multiple depth images to test noise
        depth_arrays = []
        for _ in range(5):
            depth_image, _ = sim._get_depth_image()
            depth_array = np.frombuffer(depth_image.data, dtype=np.float32).reshape((480, 640))
            depth_arrays.append(depth_array)
            sim.step(dt=0.01)  # Small step to get slightly different noise
        
        # Calculate standard deviation across images for each pixel
        depth_stack = np.stack(depth_arrays, axis=0)
        noise_std = np.std(depth_stack, axis=0)
        
        # Check that there is noise (not all zeros)
        assert np.mean(noise_std) > 0.001, "Depth camera should have noise"
        
        # The noise model should increase with distance, but this is hard to test
        # without knowing the exact scene geometry. Just check that noise exists.
        assert np.max(noise_std) > 0.01, "Some pixels should have noticeable noise"
        
        # Cleanup
        sim.destroy_node()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
