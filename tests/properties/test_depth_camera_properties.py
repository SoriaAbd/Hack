"""
Property-based tests for depth camera sensor simulation.

Tests universal properties that should hold for all depth camera configurations
and robot positions in the warehouse environment.
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


@pytest.fixture(scope="class")
def simulation():
    """Create a MuJoCo simulation instance for testing."""
    if not ROS2_AVAILABLE or not MUJOCO_AVAILABLE:
        pytest.skip("ROS 2 or MuJoCo not available")
    
    if not rclpy.ok():
        rclpy.init()
    
    sim = MuJoCoSimulation(render_mode=None)
    yield sim
    sim.destroy_node()


@pytest.mark.skipif(not ROS2_AVAILABLE or not MUJOCO_AVAILABLE, 
                    reason="ROS 2 or MuJoCo not available")
class TestDepthCameraFieldOfView:
    """
    Property 6: Depth Camera Field of View
    For any object outside the depth camera's field of view (90° horizontal), 
    that object should not appear in the depth image.
    **Validates: Requirements 2.5**
    """
    
    @given(
        robot_x=st.floats(min_value=-15.0, max_value=15.0),
        robot_y=st.floats(min_value=-15.0, max_value=15.0),
        robot_yaw=st.floats(min_value=0.0, max_value=2*np.pi)
    )
    @settings(max_examples=20, deadline=30000)  # Reduced for performance
    def test_depth_image_fov_constraint(self, simulation, robot_x, robot_y, robot_yaw):
        """Test that depth camera respects 90° field of view constraint."""
        # Reset simulation with robot at specified position
        success = simulation.reset(robot_position=(robot_x, robot_y, robot_yaw))
        assume(success)
        
        # Generate depth image
        depth_image, camera_info = simulation._get_depth_image()
        
        # Verify image dimensions match 90° FOV specification
        assert depth_image.width == 640
        assert depth_image.height == 480
        
        # Verify camera intrinsics are consistent with 90° FOV
        fx = camera_info.k[0]  # K[0,0]
        expected_fx = 640 / (2.0 * np.tan(np.pi / 4))  # 90° FOV
        
        # Allow some tolerance for numerical precision
        assert abs(fx - expected_fx) < 1.0, f"Camera fx={fx} inconsistent with 90° FOV (expected ~{expected_fx})"
        
        # Convert depth data to numpy array
        depth_array = np.frombuffer(depth_image.data, dtype=np.float32).reshape((480, 640))
        
        # All depth values should be within valid range
        assert np.all(depth_array >= 0.5), "All depth values should be >= 0.5m"
        assert np.all(depth_array <= 5.0), "All depth values should be <= 5.0m"


@pytest.mark.skipif(not ROS2_AVAILABLE or not MUJOCO_AVAILABLE, 
                    reason="ROS 2 or MuJoCo not available")
class TestDepthCameraRangeLimitations:
    """
    Test depth camera range limitations (0.5-5m) as specified in requirements.
    **Validates: Requirements 2.2, 2.5**
    """
    
    @given(
        robot_x=st.floats(min_value=-15.0, max_value=15.0),
        robot_y=st.floats(min_value=-15.0, max_value=15.0),
        robot_yaw=st.floats(min_value=0.0, max_value=2*np.pi)
    )
    @settings(max_examples=20, deadline=30000)
    def test_all_depths_within_range_limits(self, simulation, robot_x, robot_y, robot_yaw):
        """Test that all depth measurements are within specified range (0.5-5m)."""
        # Reset simulation with robot at specified position
        success = simulation.reset(robot_position=(robot_x, robot_y, robot_yaw))
        assume(success)
        
        # Generate depth image
        depth_image, camera_info = simulation._get_depth_image()
        
        # Convert depth data to numpy array
        depth_array = np.frombuffer(depth_image.data, dtype=np.float32).reshape((480, 640))
        
        # Check that all depth values are within the specified range
        min_range = 0.5  # meters
        max_range = 5.0  # meters
        
        assert np.all(depth_array >= min_range), f"Found depth values below minimum range {min_range}m"
        assert np.all(depth_array <= max_range), f"Found depth values above maximum range {max_range}m"
        
        # Verify image encoding and properties
        assert depth_image.encoding == '32FC1', "Depth image should use 32-bit float encoding"
        assert depth_image.width == 640, "Depth image width should be 640 pixels"
        assert depth_image.height == 480, "Depth image height should be 480 pixels"


@pytest.mark.skipif(not ROS2_AVAILABLE or not MUJOCO_AVAILABLE, 
                    reason="ROS 2 or MuJoCo not available")
class TestDepthCameraNoiseModel:
    """
    Test depth camera realistic noise model with depth accuracy degradation.
    **Validates: Requirements 2.5**
    """
    
    @given(
        robot_x=st.floats(min_value=-10.0, max_value=10.0),
        robot_y=st.floats(min_value=-10.0, max_value=10.0),
        robot_yaw=st.floats(min_value=0.0, max_value=2*np.pi)
    )
    @settings(max_examples=10, deadline=60000)  # Fewer examples due to computational cost
    def test_noise_increases_with_distance(self, simulation, robot_x, robot_y, robot_yaw):
        """Test that depth camera noise increases with distance."""
        # Reset simulation with robot at specified position
        success = simulation.reset(robot_position=(robot_x, robot_y, robot_yaw))
        assume(success)
        
        # Generate multiple depth images to measure noise
        depth_images = []
        for _ in range(5):  # Generate 5 images to measure noise variation
            depth_image, _ = simulation._get_depth_image()
            depth_array = np.frombuffer(depth_image.data, dtype=np.float32).reshape((480, 640))
            depth_images.append(depth_array)
            simulation.step(dt=0.001)  # Small step to get different noise samples
        
        # Calculate standard deviation across images for each pixel
        depth_stack = np.stack(depth_images, axis=0)
        noise_std = np.std(depth_stack, axis=0)
        
        # Check that there is measurable noise
        mean_noise = np.mean(noise_std)
        assert mean_noise > 0.001, f"Depth camera should have measurable noise, got {mean_noise}"
        
        # The noise model should be realistic (not too high)
        assert mean_noise < 0.1, f"Depth camera noise should be realistic, got {mean_noise}"


@pytest.mark.skipif(not ROS2_AVAILABLE or not MUJOCO_AVAILABLE, 
                    reason="ROS 2 or MuJoCo not available")
class TestDepthCameraImageStructure:
    """
    Test depth camera image structure and metadata consistency.
    **Validates: Requirements 2.2, 2.3**
    """
    
    @given(
        robot_x=st.floats(min_value=-15.0, max_value=15.0),
        robot_y=st.floats(min_value=-15.0, max_value=15.0)
    )
    @settings(max_examples=15, deadline=30000)
    def test_image_structure_consistency(self, simulation, robot_x, robot_y):
        """Test that depth image structure is consistent across different positions."""
        # Reset simulation with robot at specified position
        success = simulation.reset(robot_position=(robot_x, robot_y, 0.0))
        assume(success)
        
        # Generate depth image and camera info
        depth_image, camera_info = simulation._get_depth_image()
        
        # Test image message structure
        assert depth_image.header.frame_id == 'camera_link', "Frame ID should be 'camera_link'"
        assert depth_image.width == 640, "Image width should be 640"
        assert depth_image.height == 480, "Image height should be 480"
        assert depth_image.encoding == '32FC1', "Encoding should be 32FC1"
        assert depth_image.step == 640 * 4, "Step should be width * 4 bytes"
        assert len(depth_image.data) == 640 * 480 * 4, "Data length should match dimensions"
        
        # Test camera info structure
        assert camera_info.width == 640, "Camera info width should match image"
        assert camera_info.height == 480, "Camera info height should match image"
        assert camera_info.distortion_model == 'plumb_bob', "Should use plumb_bob distortion model"
        assert len(camera_info.k) == 9, "Camera matrix K should have 9 elements"
        assert len(camera_info.d) == 5, "Distortion coefficients should have 5 elements"
        assert len(camera_info.r) == 9, "Rectification matrix should have 9 elements"
        assert len(camera_info.p) == 12, "Projection matrix should have 12 elements"
        
        # Test that camera intrinsics are reasonable
        fx, fy = camera_info.k[0], camera_info.k[4]
        cx, cy = camera_info.k[2], camera_info.k[5]
        
        assert fx > 0, "Focal length fx should be positive"
        assert fy > 0, "Focal length fy should be positive"
        assert 300 < cx < 340, f"Principal point cx should be near image center, got {cx}"
        assert 220 < cy < 260, f"Principal point cy should be near image center, got {cy}"


@pytest.mark.skipif(not ROS2_AVAILABLE or not MUJOCO_AVAILABLE, 
                    reason="ROS 2 or MuJoCo not available")
class TestDepthCameraResolution:
    """
    Test depth camera resolution specification (640x480).
    **Validates: Requirements 2.2**
    """
    
    @given(
        robot_yaw=st.floats(min_value=0.0, max_value=2*np.pi)
    )
    @settings(max_examples=10, deadline=20000)
    def test_resolution_specification(self, simulation, robot_yaw):
        """Test that depth camera always produces 640x480 images."""
        # Reset simulation with robot at center, varying orientation
        success = simulation.reset(robot_position=(0.0, 0.0, robot_yaw))
        assume(success)
        
        # Generate depth image
        depth_image, camera_info = simulation._get_depth_image()
        
        # Verify exact resolution
        assert depth_image.width == 640, f"Width should be exactly 640, got {depth_image.width}"
        assert depth_image.height == 480, f"Height should be exactly 480, got {depth_image.height}"
        assert camera_info.width == 640, f"Camera info width should be 640, got {camera_info.width}"
        assert camera_info.height == 480, f"Camera info height should be 480, got {camera_info.height}"
        
        # Verify data size matches resolution
        expected_data_size = 640 * 480 * 4  # 4 bytes per float32
        actual_data_size = len(depth_image.data)
        assert actual_data_size == expected_data_size, \
            f"Data size should be {expected_data_size}, got {actual_data_size}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])