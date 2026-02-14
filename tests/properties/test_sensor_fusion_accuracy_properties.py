"""
Property-based tests for sensor fusion accuracy.

This module implements Property 12: Sensor Fusion Accuracy
**Validates: Requirements 4.3**

Property 12: Sensor Fusion Accuracy
For any obstacle visible to both LiDAR and depth camera, the fused position estimate 
should have lower error than either single-sensor estimate alone (when compared to ground truth).

Requirements 4.3: Obstacle detector shall fuse data from LiDAR and depth camera to improve detection accuracy
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume
from hypothesis import HealthCheck
import time
from typing import List, Tuple, Optional

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
    class TestSensorFusionAccuracyProperty:
        """
        Property 12: Sensor Fusion Accuracy
        
        **Validates: Requirements 4.3**
        
        For any obstacle visible to both LiDAR and depth camera, the fused position estimate 
        should have lower error than either single-sensor estimate alone (when compared to ground truth).
        
        This property ensures that:
        1. Sensor fusion improves position accuracy compared to individual sensors
        2. The fused estimate has lower error than LiDAR-only estimate
        3. The fused estimate has lower error than depth camera-only estimate
        4. The improvement is consistent across different robot positions and obstacle configurations
        5. The fusion algorithm provides meaningful accuracy gains
        """
        
        @settings(
            max_examples=MIN_EXAMPLES,
            deadline=TEST_TIMEOUT * 1000,
            suppress_health_check=[HealthCheck.function_scoped_fixture]
        )
        @given(
            robot_x=st.floats(min_value=-15.0, max_value=15.0),
            robot_y=st.floats(min_value=-15.0, max_value=15.0),
            robot_yaw=st.floats(min_value=-np.pi, max_value=np.pi),
            simulation_steps=st.integers(min_value=10, max_value=25)
        )
        def test_fused_position_estimate_has_lower_error_than_individual_sensors(
            self, simulation, obstacle_detector, robot_x, robot_y, robot_yaw, simulation_steps
        ):
            """
            **Validates: Requirements 4.3**
            
            Property: Fused sensor estimates have lower position error than individual sensor estimates.
            
            This is the core property that validates sensor fusion improves accuracy.
            For obstacles visible to both sensors, the fused position should be more
            accurate than using either LiDAR or depth camera alone.
            """
            # Avoid positions that would place robot inside obstacles or too close to walls
            assume(abs(robot_x) < 18.0 and abs(robot_y) < 18.0)
            
            # Set robot position
            simulation.set_robot_pose(robot_x, robot_y, robot_yaw)
            
            # Collect sensor fusion accuracy data over multiple steps
            fusion_accuracy_data = []
            
            for step in range(simulation_steps):
                # Move robot slightly to generate varied sensor perspectives
                linear_vel = 0.05 * np.sin(step * 0.3)  # Gentle sinusoidal motion
                angular_vel = 0.02 * np.cos(step * 0.2)  # Gentle rotation
                simulation.step(linear_vel, angular_vel)
                
                # Allow time for sensor data processing
                time.sleep(0.1)
                
                # Get ground truth obstacle positions
                ground_truth_obstacles = simulation.get_ground_truth_obstacles()
                if not ground_truth_obstacles:
                    continue
                
                # Get sensor data for individual sensor processing
                with obstacle_detector._data_lock:
                    scan = obstacle_detector._latest_scan
                    depth_image = obstacle_detector._latest_depth_image
                    camera_info = obstacle_detector._latest_camera_info
                
                if scan is None or depth_image is None or camera_info is None:
                    continue
                
                # Extract point clouds from individual sensors
                lidar_points = obstacle_detector._extract_lidar_points(scan)
                depth_points = obstacle_detector._extract_depth_points(depth_image, camera_info)
                
                # Skip if either sensor has insufficient data
                if len(lidar_points) < 5 or len(depth_points) < 5:
                    continue
                
                # Process with individual sensors
                lidar_only_detections = self._process_single_sensor_detections(
                    obstacle_detector, lidar_points, simulation.get_clock().now()
                )
                depth_only_detections = self._process_single_sensor_detections(
                    obstacle_detector, depth_points, simulation.get_clock().now()
                )
                
                # Process with fused sensors
                fused_points = obstacle_detector._fuse_point_clouds(lidar_points, depth_points)
                fused_detections = self._process_single_sensor_detections(
                    obstacle_detector, fused_points, simulation.get_clock().now()
                )
                
                # Compare accuracy for obstacles visible to both sensors
                accuracy_comparison = self._compare_detection_accuracy(
                    ground_truth_obstacles,
                    lidar_only_detections,
                    depth_only_detections,
                    fused_detections
                )
                
                if accuracy_comparison:
                    fusion_accuracy_data.extend(accuracy_comparison)
            
            # Validate that we have sufficient data for meaningful comparison
            assert len(fusion_accuracy_data) >= 3, \
                f"Insufficient sensor fusion data collected: {len(fusion_accuracy_data)} comparisons. " \
                f"Need at least 3 for meaningful analysis."
            
            # Core property validation: Fused estimates should be more accurate
            fusion_improvements = 0
            total_comparisons = len(fusion_accuracy_data)
            
            for comparison in fusion_accuracy_data:
                gt_pos = comparison['ground_truth_position']
                lidar_error = comparison['lidar_error']
                depth_error = comparison['depth_error']
                fused_error = comparison['fused_error']
                
                # Fused estimate should be better than both individual sensors
                if fused_error < lidar_error and fused_error < depth_error:
                    fusion_improvements += 1
                
                # Individual validation for debugging
                assert np.isfinite(lidar_error), \
                    f"LiDAR error is not finite: {lidar_error}"
                assert np.isfinite(depth_error), \
                    f"Depth camera error is not finite: {depth_error}"
                assert np.isfinite(fused_error), \
                    f"Fused error is not finite: {fused_error}"
                
                # Errors should be reasonable (not extremely large)
                max_reasonable_error = 5.0  # 5 meters maximum error
                assert lidar_error <= max_reasonable_error, \
                    f"LiDAR error {lidar_error:.2f}m exceeds reasonable limit"
                assert depth_error <= max_reasonable_error, \
                    f"Depth camera error {depth_error:.2f}m exceeds reasonable limit"
                assert fused_error <= max_reasonable_error, \
                    f"Fused error {fused_error:.2f}m exceeds reasonable limit"
            
            # Core property: Fusion should improve accuracy in majority of cases
            improvement_rate = fusion_improvements / total_comparisons
            min_improvement_rate = 0.6  # Expect fusion to be better in at least 60% of cases
            
            assert improvement_rate >= min_improvement_rate, \
                f"Sensor fusion improvement rate {improvement_rate:.2%} is below minimum " \
                f"{min_improvement_rate:.2%}. Fusion improved accuracy in {fusion_improvements}/{total_comparisons} cases."
        
        def _process_single_sensor_detections(
            self, detector: ObstacleDetector, points: np.ndarray, current_time
        ) -> List[DetectedObstacle]:
            """
            Process point cloud with obstacle detection pipeline to get detections.
            
            Args:
                detector: Obstacle detector instance
                points: Point cloud data (N, 2)
                current_time: Current ROS time
                
            Returns:
                List of detected obstacles
            """
            if len(points) == 0:
                return []
            
            # Cluster points to detect obstacles
            clusters = detector._cluster_points(points)
            
            # Convert clusters to obstacle detections
            detections = detector._clusters_to_obstacles(clusters, current_time)
            
            return detections
        
        def _compare_detection_accuracy(
            self,
            ground_truth_obstacles: List[DynamicObstacle],
            lidar_detections: List[DetectedObstacle],
            depth_detections: List[DetectedObstacle],
            fused_detections: List[DetectedObstacle]
        ) -> List[dict]:
            """
            Compare detection accuracy between individual sensors and fusion.
            
            Args:
                ground_truth_obstacles: True obstacle positions
                lidar_detections: Detections from LiDAR only
                depth_detections: Detections from depth camera only
                fused_detections: Detections from sensor fusion
                
            Returns:
                List of accuracy comparison dictionaries
            """
            comparisons = []
            
            # For each ground truth obstacle, find the closest detection from each method
            for gt_obstacle in ground_truth_obstacles:
                gt_pos = gt_obstacle.position
                
                # Find closest detection from each sensor method
                lidar_match = self._find_closest_detection(gt_pos, lidar_detections)
                depth_match = self._find_closest_detection(gt_pos, depth_detections)
                fused_match = self._find_closest_detection(gt_pos, fused_detections)
                
                # Only compare if obstacle is detected by all methods
                # (indicating it's visible to both sensors)
                if lidar_match and depth_match and fused_match:
                    lidar_error = np.linalg.norm(lidar_match.position - gt_pos)
                    depth_error = np.linalg.norm(depth_match.position - gt_pos)
                    fused_error = np.linalg.norm(fused_match.position - gt_pos)
                    
                    # Only include if all detections are within reasonable range
                    # (to avoid comparing completely wrong detections)
                    max_association_distance = 2.0  # meters
                    if (lidar_error <= max_association_distance and 
                        depth_error <= max_association_distance and 
                        fused_error <= max_association_distance):
                        
                        comparisons.append({
                            'ground_truth_position': gt_pos.copy(),
                            'lidar_error': lidar_error,
                            'depth_error': depth_error,
                            'fused_error': fused_error,
                            'obstacle_id': gt_obstacle.id,
                            'obstacle_classification': gt_obstacle.classification
                        })
            
            return comparisons
        
        def _find_closest_detection(
            self, gt_position: np.ndarray, detections: List[DetectedObstacle]
        ) -> Optional[DetectedObstacle]:
            """
            Find the detection closest to a ground truth position.
            
            Args:
                gt_position: Ground truth position [x, y]
                detections: List of detections to search
                
            Returns:
                Closest detection or None if no detections
            """
            if not detections:
                return None
            
            min_distance = float('inf')
            closest_detection = None
            
            for detection in detections:
                distance = np.linalg.norm(detection.position - gt_position)
                if distance < min_distance:
                    min_distance = distance
                    closest_detection = detection
            
            return closest_detection
        
        @settings(
            max_examples=50,  # Fewer examples for this more complex test
            deadline=TEST_TIMEOUT * 1000,
            suppress_health_check=[HealthCheck.function_scoped_fixture]
        )
        @given(
            robot_x=st.floats(min_value=-10.0, max_value=10.0),
            robot_y=st.floats(min_value=-10.0, max_value=10.0),
            robot_yaw=st.floats(min_value=-np.pi, max_value=np.pi)
        )
        def test_sensor_fusion_provides_consistent_accuracy_improvement(
            self, simulation, obstacle_detector, robot_x, robot_y, robot_yaw
        ):
            """
            **Validates: Requirements 4.3**
            
            Property: Sensor fusion provides consistent accuracy improvement across scenarios.
            
            This validates that the accuracy improvement from sensor fusion is not
            just a statistical fluke but a consistent benefit across different
            robot positions and obstacle configurations.
            """
            # Set robot position
            simulation.set_robot_pose(robot_x, robot_y, robot_yaw)
            
            # Collect accuracy data across multiple scenarios
            scenario_results = []
            
            for scenario in range(15):  # Test multiple scenarios
                # Vary robot motion to create different sensor perspectives
                if scenario < 5:
                    # Forward motion scenarios
                    linear_vel = 0.1
                    angular_vel = 0.0
                elif scenario < 10:
                    # Rotation scenarios
                    linear_vel = 0.0
                    angular_vel = 0.1
                else:
                    # Combined motion scenarios
                    linear_vel = 0.05
                    angular_vel = 0.05
                
                simulation.step(linear_vel, angular_vel)
                time.sleep(0.1)
                
                # Get ground truth and sensor data
                ground_truth_obstacles = simulation.get_ground_truth_obstacles()
                if not ground_truth_obstacles:
                    continue
                
                with obstacle_detector._data_lock:
                    scan = obstacle_detector._latest_scan
                    depth_image = obstacle_detector._latest_depth_image
                    camera_info = obstacle_detector._latest_camera_info
                
                if scan is None or depth_image is None or camera_info is None:
                    continue
                
                # Process with different sensor configurations
                lidar_points = obstacle_detector._extract_lidar_points(scan)
                depth_points = obstacle_detector._extract_depth_points(depth_image, camera_info)
                
                if len(lidar_points) < 3 or len(depth_points) < 3:
                    continue
                
                # Get detections from each method
                current_time = simulation.get_clock().now()
                lidar_detections = self._process_single_sensor_detections(
                    obstacle_detector, lidar_points, current_time
                )
                depth_detections = self._process_single_sensor_detections(
                    obstacle_detector, depth_points, current_time
                )
                fused_points = obstacle_detector._fuse_point_clouds(lidar_points, depth_points)
                fused_detections = self._process_single_sensor_detections(
                    obstacle_detector, fused_points, current_time
                )
                
                # Analyze accuracy for this scenario
                accuracy_data = self._compare_detection_accuracy(
                    ground_truth_obstacles, lidar_detections, depth_detections, fused_detections
                )
                
                if accuracy_data:
                    # Calculate average improvement for this scenario
                    improvements = []
                    for comparison in accuracy_data:
                        lidar_error = comparison['lidar_error']
                        depth_error = comparison['depth_error']
                        fused_error = comparison['fused_error']
                        
                        # Calculate relative improvement
                        best_individual = min(lidar_error, depth_error)
                        if best_individual > 0:
                            improvement = (best_individual - fused_error) / best_individual
                            improvements.append(improvement)
                    
                    if improvements:
                        avg_improvement = np.mean(improvements)
                        scenario_results.append(avg_improvement)
            
            # Validate consistency across scenarios
            assert len(scenario_results) >= 5, \
                f"Insufficient scenarios tested: {len(scenario_results)}. Need at least 5."
            
            # Most scenarios should show improvement
            positive_improvements = sum(1 for imp in scenario_results if imp > 0)
            improvement_consistency = positive_improvements / len(scenario_results)
            
            assert improvement_consistency >= 0.6, \
                f"Sensor fusion improvement consistency {improvement_consistency:.2%} is too low. " \
                f"Only {positive_improvements}/{len(scenario_results)} scenarios showed improvement."
            
            # Average improvement should be meaningful
            avg_improvement = np.mean(scenario_results)
            min_avg_improvement = 0.05  # At least 5% average improvement
            
            assert avg_improvement >= min_avg_improvement, \
                f"Average sensor fusion improvement {avg_improvement:.2%} is below minimum " \
                f"{min_avg_improvement:.2%}."
        
        @settings(
            max_examples=MIN_EXAMPLES,
            deadline=TEST_TIMEOUT * 1000,
            suppress_health_check=[HealthCheck.function_scoped_fixture]
        )
        @given(
            robot_x=st.floats(min_value=-12.0, max_value=12.0),
            robot_y=st.floats(min_value=-12.0, max_value=12.0)
        )
        def test_fusion_accuracy_improvement_is_statistically_significant(
            self, simulation, obstacle_detector, robot_x, robot_y
        ):
            """
            **Validates: Requirements 4.3**
            
            Property: Sensor fusion accuracy improvement is statistically significant.
            
            This validates that the accuracy improvement from sensor fusion is
            not due to random chance but represents a genuine improvement in
            detection accuracy.
            """
            # Set robot position
            simulation.set_robot_pose(robot_x, robot_y, 0.0)
            
            # Collect error measurements for statistical analysis
            lidar_errors = []
            depth_errors = []
            fused_errors = []
            
            for measurement in range(20):  # Collect multiple measurements
                # Small random movements to vary sensor perspective
                linear_vel = np.random.uniform(-0.05, 0.05)
                angular_vel = np.random.uniform(-0.03, 0.03)
                simulation.step(linear_vel, angular_vel)
                time.sleep(0.05)
                
                # Get sensor data and ground truth
                ground_truth_obstacles = simulation.get_ground_truth_obstacles()
                if not ground_truth_obstacles:
                    continue
                
                with obstacle_detector._data_lock:
                    scan = obstacle_detector._latest_scan
                    depth_image = obstacle_detector._latest_depth_image
                    camera_info = obstacle_detector._latest_camera_info
                
                if scan is None or depth_image is None or camera_info is None:
                    continue
                
                # Process sensors
                lidar_points = obstacle_detector._extract_lidar_points(scan)
                depth_points = obstacle_detector._extract_depth_points(depth_image, camera_info)
                
                if len(lidar_points) < 3 or len(depth_points) < 3:
                    continue
                
                current_time = simulation.get_clock().now()
                lidar_detections = self._process_single_sensor_detections(
                    obstacle_detector, lidar_points, current_time
                )
                depth_detections = self._process_single_sensor_detections(
                    obstacle_detector, depth_points, current_time
                )
                fused_points = obstacle_detector._fuse_point_clouds(lidar_points, depth_points)
                fused_detections = self._process_single_sensor_detections(
                    obstacle_detector, fused_points, current_time
                )
                
                # Collect error measurements
                accuracy_data = self._compare_detection_accuracy(
                    ground_truth_obstacles, lidar_detections, depth_detections, fused_detections
                )
                
                for comparison in accuracy_data:
                    lidar_errors.append(comparison['lidar_error'])
                    depth_errors.append(comparison['depth_error'])
                    fused_errors.append(comparison['fused_error'])
            
            # Validate sufficient data for statistical analysis
            assert len(fused_errors) >= 10, \
                f"Insufficient error measurements: {len(fused_errors)}. Need at least 10."
            
            # Convert to numpy arrays for analysis
            lidar_errors = np.array(lidar_errors)
            depth_errors = np.array(depth_errors)
            fused_errors = np.array(fused_errors)
            
            # Statistical validation: fused errors should be significantly lower
            mean_lidar_error = np.mean(lidar_errors)
            mean_depth_error = np.mean(depth_errors)
            mean_fused_error = np.mean(fused_errors)
            
            # Fused error should be lower than both individual sensors on average
            assert mean_fused_error < mean_lidar_error, \
                f"Fused error {mean_fused_error:.3f} not lower than LiDAR error {mean_lidar_error:.3f}"
            assert mean_fused_error < mean_depth_error, \
                f"Fused error {mean_fused_error:.3f} not lower than depth error {mean_depth_error:.3f}"
            
            # Calculate improvement percentages
            lidar_improvement = (mean_lidar_error - mean_fused_error) / mean_lidar_error
            depth_improvement = (mean_depth_error - mean_fused_error) / mean_depth_error
            
            # Improvements should be meaningful (at least 10%)
            min_improvement = 0.10
            assert lidar_improvement >= min_improvement, \
                f"LiDAR improvement {lidar_improvement:.2%} below minimum {min_improvement:.2%}"
            assert depth_improvement >= min_improvement, \
                f"Depth improvement {depth_improvement:.2%} below minimum {min_improvement:.2%}"
            
            # Variance should be reasonable (fusion shouldn't increase uncertainty dramatically)
            std_lidar = np.std(lidar_errors)
            std_depth = np.std(depth_errors)
            std_fused = np.std(fused_errors)
            
            # Fused standard deviation should not be much higher than individual sensors
            max_std_increase = 1.5  # Allow up to 50% increase in std dev
            assert std_fused <= max_std_increase * min(std_lidar, std_depth), \
                f"Fused error std dev {std_fused:.3f} too high compared to individual sensors " \
                f"(LiDAR: {std_lidar:.3f}, Depth: {std_depth:.3f})"


    class TestSensorFusionAccuracyEdgeCases:
        """Edge cases for sensor fusion accuracy testing."""
        
        def test_fusion_with_single_sensor_data(self, simulation, obstacle_detector):
            """
            **Validates: Requirements 4.3**
            
            Test that fusion gracefully handles cases where only one sensor has data.
            """
            simulation.set_robot_pose(0.0, 0.0, 0.0)
            simulation.step(0.0, 0.0)
            time.sleep(0.1)
            
            # Test with only LiDAR data
            with obstacle_detector._data_lock:
                scan = obstacle_detector._latest_scan
            
            if scan is not None:
                lidar_points = obstacle_detector._extract_lidar_points(scan)
                empty_depth_points = np.empty((0, 2))
                
                # Fusion should work with only LiDAR data
                fused_points = obstacle_detector._fuse_point_clouds(lidar_points, empty_depth_points)
                assert len(fused_points) == len(lidar_points), \
                    "Fusion with only LiDAR data should return LiDAR points"
                np.testing.assert_array_equal(fused_points, lidar_points)
        
        def test_fusion_with_no_sensor_data(self, simulation, obstacle_detector):
            """
            **Validates: Requirements 4.3**
            
            Test that fusion handles the case where no sensor data is available.
            """
            empty_lidar = np.empty((0, 2))
            empty_depth = np.empty((0, 2))
            
            # Fusion should return empty array when no data available
            fused_points = obstacle_detector._fuse_point_clouds(empty_lidar, empty_depth)
            assert len(fused_points) == 0, \
                "Fusion with no sensor data should return empty array"
        
        def test_fusion_accuracy_with_noisy_sensors(self, simulation, obstacle_detector):
            """
            **Validates: Requirements 4.3**
            
            Test that fusion provides accuracy benefits even with noisy sensor data.
            """
            simulation.set_robot_pose(5.0, 5.0, 0.0)
            
            # Run simulation to get sensor data
            for step in range(10):
                simulation.step(0.02, 0.01)
                time.sleep(0.05)
            
            # Get ground truth and sensor data
            ground_truth_obstacles = simulation.get_ground_truth_obstacles()
            if not ground_truth_obstacles:
                pytest.skip("No ground truth obstacles available")
            
            with obstacle_detector._data_lock:
                scan = obstacle_detector._latest_scan
                depth_image = obstacle_detector._latest_depth_image
                camera_info = obstacle_detector._latest_camera_info
            
            if scan is None or depth_image is None or camera_info is None:
                pytest.skip("Sensor data not available")
            
            # Extract clean sensor data
            lidar_points = obstacle_detector._extract_lidar_points(scan)
            depth_points = obstacle_detector._extract_depth_points(depth_image, camera_info)
            
            if len(lidar_points) < 5 or len(depth_points) < 5:
                pytest.skip("Insufficient sensor data")
            
            # Add artificial noise to simulate degraded sensor conditions
            noise_level = 0.1  # 10cm noise
            noisy_lidar = lidar_points + np.random.normal(0, noise_level, lidar_points.shape)
            noisy_depth = depth_points + np.random.normal(0, noise_level, depth_points.shape)
            
            # Process with noisy individual sensors and fusion
            current_time = simulation.get_clock().now()
            
            # Create a helper instance for processing
            helper = TestSensorFusionAccuracyProperty()
            
            noisy_lidar_detections = helper._process_single_sensor_detections(
                obstacle_detector, noisy_lidar, current_time
            )
            noisy_depth_detections = helper._process_single_sensor_detections(
                obstacle_detector, noisy_depth, current_time
            )
            
            # Fuse noisy sensor data
            fused_noisy_points = obstacle_detector._fuse_point_clouds(noisy_lidar, noisy_depth)
            fused_noisy_detections = helper._process_single_sensor_detections(
                obstacle_detector, fused_noisy_points, current_time
            )
            
            # Even with noisy data, fusion should provide some benefit
            if (noisy_lidar_detections and noisy_depth_detections and 
                fused_noisy_detections and ground_truth_obstacles):
                
                accuracy_data = helper._compare_detection_accuracy(
                    ground_truth_obstacles, 
                    noisy_lidar_detections, 
                    noisy_depth_detections, 
                    fused_noisy_detections
                )
                
                if accuracy_data:
                    # At least some cases should show improvement even with noise
                    improvements = 0
                    for comparison in accuracy_data:
                        lidar_error = comparison['lidar_error']
                        depth_error = comparison['depth_error']
                        fused_error = comparison['fused_error']
                        
                        if fused_error < min(lidar_error, depth_error):
                            improvements += 1
                    
                    # Should have at least some improvement even with noisy data
                    improvement_rate = improvements / len(accuracy_data)
                    assert improvement_rate > 0, \
                        "Sensor fusion should provide some accuracy benefit even with noisy data"