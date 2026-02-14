#!/usr/bin/env python3
"""
Unit tests for obstacle detector algorithms.

Tests the core algorithms without ROS dependencies:
- Point cloud extraction from LiDAR and depth camera
- DBSCAN clustering for obstacle detection
- Obstacle tracking and velocity estimation with Kalman filter
"""

import pytest
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.optimize import linear_sum_assignment
import time


class KalmanFilter:
    """
    2D Kalman filter for tracking obstacle position and velocity.
    
    State vector: [x, y, vx, vy]
    - x, y: position in meters
    - vx, vy: velocity in m/s
    """
    
    def __init__(self, initial_position: np.ndarray, dt: float = 0.1):
        """
        Initialize Kalman filter.
        
        Args:
            initial_position: Initial [x, y] position
            dt: Time step for prediction (seconds)
        """
        self.dt = dt
        
        # State vector [x, y, vx, vy]
        self.state = np.array([
            initial_position[0],  # x
            initial_position[1],  # y
            0.0,                  # vx (initial velocity unknown)
            0.0                   # vy (initial velocity unknown)
        ])
        
        # State covariance matrix (4x4)
        self.P = np.eye(4)
        self.P[0, 0] = 0.1  # x position uncertainty
        self.P[1, 1] = 0.1  # y position uncertainty
        self.P[2, 2] = 1.0  # vx velocity uncertainty (high initially)
        self.P[3, 3] = 1.0  # vy velocity uncertainty (high initially)
        
        # State transition matrix (constant velocity model)
        self.F = np.array([
            [1, 0, dt, 0],   # x = x + vx*dt
            [0, 1, 0, dt],   # y = y + vy*dt
            [0, 0, 1, 0],    # vx = vx
            [0, 0, 0, 1]     # vy = vy
        ])
        
        # Process noise covariance (4x4)
        # Models uncertainty in constant velocity assumption
        q = 0.5  # Process noise standard deviation
        self.Q = np.array([
            [dt**4/4, 0, dt**3/2, 0],
            [0, dt**4/4, 0, dt**3/2],
            [dt**3/2, 0, dt**2, 0],
            [0, dt**3/2, 0, dt**2]
        ]) * q**2
        
        # Measurement matrix (observe position only)
        self.H = np.array([
            [1, 0, 0, 0],  # Measure x
            [0, 1, 0, 0]   # Measure y
        ])
        
        # Measurement noise covariance (2x2)
        r = 0.2  # Measurement noise standard deviation (20cm)
        self.R = np.eye(2) * r**2
        
        self.last_update_time = time.time()
    
    def predict(self, dt: float = None) -> None:
        """
        Predict next state using motion model.
        
        Args:
            dt: Time step (uses default if None)
        """
        if dt is not None:
            # Update state transition matrix with new dt
            self.F[0, 2] = dt
            self.F[1, 3] = dt
            
            # Update process noise with new dt
            q = 0.5
            self.Q = np.array([
                [dt**4/4, 0, dt**3/2, 0],
                [0, dt**4/4, 0, dt**3/2],
                [dt**3/2, 0, dt**2, 0],
                [0, dt**3/2, 0, dt**2]
            ]) * q**2
        
        # Predict state: x_k = F * x_{k-1}
        self.state = self.F @ self.state
        
        # Predict covariance: P_k = F * P_{k-1} * F^T + Q
        self.P = self.F @ self.P @ self.F.T + self.Q
    
    def update(self, measurement: np.ndarray) -> None:
        """
        Update filter with new measurement.
        
        Args:
            measurement: Observed [x, y] position
        """
        # Innovation (measurement residual)
        z = measurement.reshape(-1, 1)
        y = z - (self.H @ self.state.reshape(-1, 1))
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state: x_k = x_k + K * y
        self.state = (self.state.reshape(-1, 1) + K @ y).flatten()
        
        # Update covariance: P_k = (I - K * H) * P_k
        I = np.eye(4)
        self.P = (I - K @ self.H) @ self.P
        
        self.last_update_time = time.time()
    
    def get_position(self) -> np.ndarray:
        """Get current position estimate."""
        return self.state[:2]
    
    def get_velocity(self) -> np.ndarray:
        """Get current velocity estimate."""
        return self.state[2:]
    
    def get_position_covariance(self) -> np.ndarray:
        """Get position covariance (2x2)."""
        return self.P[:2, :2]
    
    def get_velocity_covariance(self) -> np.ndarray:
        """Get velocity covariance (2x2)."""
        return self.P[2:, 2:]
    
    def get_full_covariance(self) -> np.ndarray:
        """Get full state covariance (4x4)."""
        return self.P.copy()


class MockLaserScan:
    """Mock LiDAR scan message."""
    def __init__(self):
        self.angle_min = -np.pi
        self.angle_max = np.pi
        self.angle_increment = np.pi / 180
        self.range_min = 0.1
        self.range_max = 10.0
        self.ranges = []


class MockCameraInfo:
    """Mock camera info message."""
    def __init__(self):
        self.width = 640
        self.height = 480
        self.k = [320.0, 0.0, 320.0, 0.0, 320.0, 240.0, 0.0, 0.0, 1.0]


class DetectedObstacle:
    """Represents a detected obstacle before tracking."""
    
    def __init__(self, position: np.ndarray, radius: float, classification: str, 
                 confidence: float, timestamp):
        self.position = position
        self.radius = radius
        self.classification = classification
        self.confidence = confidence
        self.timestamp = timestamp


class ObstacleTrack:
    """Tracks an obstacle over time using Kalman filter for position and velocity estimation."""
    
    def __init__(self, track_id: int, initial_detection: DetectedObstacle, current_time: float):
        self.track_id = track_id
        self.classification = initial_detection.classification
        self.confidence = initial_detection.confidence
        self.last_update_time = current_time
        self.creation_time = current_time
        
        # Initialize Kalman filter with initial position
        self.kalman_filter = KalmanFilter(initial_detection.position)
        
        # Track age and missed detections
        self.age = 0  # Number of frames since creation
        self.missed_detections = 0  # Consecutive frames without detection
        self.total_detections = 1  # Total number of detections associated with this track
    
    def predict(self, current_time: float) -> None:
        """
        Predict obstacle state at current time.
        
        Args:
            current_time: Current timestamp
        """
        dt = current_time - self.last_update_time
        if dt > 0:
            self.kalman_filter.predict(dt)
        self.age += 1
    
    def update(self, detection: DetectedObstacle, current_time: float) -> None:
        """
        Update track with new detection using Kalman filter.
        
        Args:
            detection: New obstacle detection
            current_time: Current timestamp
        """
        # Update Kalman filter with new position measurement
        self.kalman_filter.update(detection.position)
        
        # Update track properties
        self.classification = detection.classification
        self.confidence = detection.confidence
        self.last_update_time = current_time
        self.missed_detections = 0
        self.total_detections += 1
    
    def mark_missed(self) -> None:
        """Mark that this track was not associated with any detection this frame."""
        self.missed_detections += 1
        self.age += 1
    
    def get_position(self) -> np.ndarray:
        """Get current position estimate."""
        return self.kalman_filter.get_position()
    
    def get_velocity(self) -> np.ndarray:
        """Get current velocity estimate."""
        return self.kalman_filter.get_velocity()
    
    def get_position_uncertainty(self) -> float:
        """Get position uncertainty (trace of position covariance)."""
        pos_cov = self.kalman_filter.get_position_covariance()
        return np.trace(pos_cov)
    
    def get_velocity_magnitude(self) -> float:
        """Get velocity magnitude."""
        velocity = self.get_velocity()
        return np.linalg.norm(velocity)
    
    def is_stationary(self, threshold: float = 0.1) -> bool:
        """Check if obstacle is stationary."""
        return self.get_velocity_magnitude() < threshold
    
    def should_delete(self, max_missed: int = 5, max_age_without_detection: int = 10) -> bool:
        """
        Determine if track should be deleted.
        
        Args:
            max_missed: Maximum consecutive missed detections
            max_age_without_detection: Maximum age if no recent detections
            
        Returns:
            True if track should be deleted
        """
        # Delete if too many consecutive missed detections
        if self.missed_detections >= max_missed:
            return True
        
        # Delete if track is old and hasn't been detected recently
        if self.age > max_age_without_detection and self.missed_detections > 2:
            return True
        
        return False


class ObstacleDetectorAlgorithms:
    """Core obstacle detection algorithms without ROS dependencies."""
    
    def __init__(self):
        # Detection parameters
        self._dbscan_eps = 0.5  # Maximum distance between points in a cluster (meters)
        self._dbscan_min_samples = 3  # Minimum points to form a cluster
        self._max_detection_range = 8.0  # Maximum range for obstacle detection (meters)
        self._min_obstacle_size = 0.2  # Minimum obstacle radius (meters)
        self._max_obstacle_size = 2.0  # Maximum obstacle radius (meters)
    
    def extract_lidar_points(self, scan: MockLaserScan) -> np.ndarray:
        """Extract 2D points from LiDAR scan."""
        ranges = np.array(scan.ranges)
        angles = np.linspace(scan.angle_min, scan.angle_max, len(ranges))
        
        # Filter out invalid ranges
        valid_mask = (ranges >= scan.range_min) & (ranges <= scan.range_max) & np.isfinite(ranges)
        valid_ranges = ranges[valid_mask]
        valid_angles = angles[valid_mask]
        
        # Filter by maximum detection range
        range_mask = valid_ranges <= self._max_detection_range
        valid_ranges = valid_ranges[range_mask]
        valid_angles = valid_angles[range_mask]
        
        # Convert to Cartesian coordinates (robot frame)
        x = valid_ranges * np.cos(valid_angles)
        y = valid_ranges * np.sin(valid_angles)
        
        return np.column_stack([x, y])
    
    def extract_depth_points(self, depth_image: np.ndarray, camera_info: MockCameraInfo) -> np.ndarray:
        """Extract 3D points from depth camera and project to 2D."""
        height, width = depth_image.shape
        
        # Camera intrinsic parameters
        fx = camera_info.k[0]  # Focal length x
        fy = camera_info.k[4]  # Focal length y
        cx = camera_info.k[2]  # Principal point x
        cy = camera_info.k[5]  # Principal point y
        
        # Create coordinate grids
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        
        # Get valid depth values
        valid_mask = (depth_image > 0.1) & (depth_image < self._max_detection_range) & np.isfinite(depth_image)
        
        if not np.any(valid_mask):
            return np.empty((0, 2))
        
        # Extract valid coordinates and depths
        u_valid = u[valid_mask]
        v_valid = v[valid_mask]
        z_valid = depth_image[valid_mask]
        
        # Convert to 3D points in camera frame
        x_cam = (u_valid - cx) * z_valid / fx
        y_cam = (v_valid - cy) * z_valid / fy
        z_cam = z_valid
        
        # Transform from camera frame to robot frame
        # Assuming camera is mounted facing forward at robot center
        # Camera frame: x=right, y=down, z=forward
        # Robot frame: x=forward, y=left, z=up
        x_robot = z_cam  # Camera z becomes robot x (forward)
        y_robot = -x_cam  # Camera -x becomes robot y (left)
        
        # Filter by range
        range_mask = np.sqrt(x_robot**2 + y_robot**2) <= self._max_detection_range
        
        return np.column_stack([x_robot[range_mask], y_robot[range_mask]])
    
    def fuse_point_clouds(self, lidar_points: np.ndarray, depth_points: np.ndarray) -> np.ndarray:
        """Fuse LiDAR and depth camera point clouds."""
        if len(lidar_points) == 0 and len(depth_points) == 0:
            return np.empty((0, 2))
        elif len(lidar_points) == 0:
            return depth_points
        elif len(depth_points) == 0:
            return lidar_points
        else:
            # Simple concatenation - could be improved with more sophisticated fusion
            return np.vstack([lidar_points, depth_points])
    
    def cluster_points(self, points: np.ndarray) -> list:
        """Cluster points using DBSCAN to identify obstacles."""
        if len(points) < self._dbscan_min_samples:
            return []
        
        # Apply DBSCAN clustering
        clustering = DBSCAN(eps=self._dbscan_eps, min_samples=self._dbscan_min_samples)
        labels = clustering.fit_predict(points)
        
        # Extract clusters (ignore noise points with label -1)
        clusters = []
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            if label == -1:  # Skip noise points
                continue
            
            cluster_points = points[labels == label]
            
            # Filter clusters by size
            cluster_center = np.mean(cluster_points, axis=0)
            distances = np.linalg.norm(cluster_points - cluster_center, axis=1)
            cluster_radius = np.max(distances)
            
            if self._min_obstacle_size <= cluster_radius <= self._max_obstacle_size:
                clusters.append(cluster_points)
        
        return clusters
    
    def clusters_to_obstacles(self, clusters: list, current_time) -> list:
        """Convert point clusters to obstacle detections."""
        detections = []
        
        for i, cluster in enumerate(clusters):
            # Compute obstacle properties
            center = np.mean(cluster, axis=0)
            distances = np.linalg.norm(cluster - center, axis=1)
            radius = np.max(distances)
            
            # Estimate confidence based on number of points and compactness
            num_points = len(cluster)
            compactness = np.std(distances) / radius if radius > 0 else 0
            confidence = min(1.0, (num_points / 10.0) * (1.0 - compactness))
            
            # Simple classification based on size
            if radius < 0.7:
                classification = "worker"
            elif radius > 1.2:
                classification = "forklift"
            else:
                classification = "unknown"
            
            detection = DetectedObstacle(
                position=center,
                radius=radius,
                classification=classification,
                confidence=confidence,
                timestamp=current_time
            )
            
            detections.append(detection)
        
        return detections


@pytest.fixture
def detector():
    """Create an obstacle detector algorithms instance."""
    return ObstacleDetectorAlgorithms()


@pytest.fixture
def sample_laser_scan():
    """Create a sample LiDAR scan message."""
    scan = MockLaserScan()
    
    # Create scan with obstacles at specific positions
    num_rays = 360
    ranges = np.full(num_rays, 10.0)  # Default max range
    
    # Add obstacle at 2m distance, 0 degrees (front)
    for i in range(175, 185):  # 10 degree spread
        ranges[i] = 2.0
    
    # Add obstacle at 3m distance, 90 degrees (left)
    for i in range(85, 95):  # 10 degree spread
        ranges[i] = 3.0
    
    scan.ranges = ranges.tolist()
    return scan


@pytest.fixture
def sample_depth_image():
    """Create a sample depth image."""
    # Create 640x480 depth image
    depth_image = np.full((480, 640), 5.0, dtype=np.float32)  # Default 5m depth
    
    # Add obstacle in center of image at 2m depth
    depth_image[200:280, 280:360] = 2.0
    
    return depth_image


@pytest.fixture
def sample_camera_info():
    """Create sample camera calibration info."""
    return MockCameraInfo()


class TestObstacleDetectorAlgorithms:
    """Test cases for the core obstacle detection algorithms."""
    
    def test_extract_lidar_points(self, detector, sample_laser_scan):
        """Test LiDAR point extraction."""
        points = detector.extract_lidar_points(sample_laser_scan)
        
        # Should extract points from the scan
        assert len(points) > 0
        assert points.shape[1] == 2  # 2D points [x, y]
        
        # Check that points are within expected range
        distances = np.linalg.norm(points, axis=1)
        assert np.all(distances >= sample_laser_scan.range_min)
        assert np.all(distances <= detector._max_detection_range)
        
        # Should have points near the obstacle positions we created
        # Obstacle at 2m front should be around (2, 0)
        front_points = points[np.abs(points[:, 1]) < 0.5]  # Points near y=0
        front_distances = np.linalg.norm(front_points, axis=1)
        assert np.any(np.abs(front_distances - 2.0) < 0.1)
    
    def test_extract_depth_points(self, detector, sample_depth_image, sample_camera_info):
        """Test depth camera point extraction."""
        points = detector.extract_depth_points(sample_depth_image, sample_camera_info)
        
        # Should extract points from the depth image
        assert len(points) > 0
        assert points.shape[1] == 2  # 2D points [x, y]
        
        # Check that points are within expected range
        distances = np.linalg.norm(points, axis=1)
        assert np.all(distances <= detector._max_detection_range)
        
        # Should have points near the obstacle we created in center at 2m depth
        center_points = points[np.abs(points[:, 1]) < 0.5]  # Points near y=0 (center)
        center_distances = np.linalg.norm(center_points, axis=1)
        assert np.any(np.abs(center_distances - 2.0) < 0.5)
    
    def test_fuse_point_clouds(self, detector):
        """Test point cloud fusion."""
        lidar_points = np.array([[1.0, 0.0], [2.0, 0.0]])
        depth_points = np.array([[0.0, 1.0], [0.0, 2.0]])
        
        fused = detector.fuse_point_clouds(lidar_points, depth_points)
        
        # Should combine both point clouds
        assert len(fused) == 4
        assert np.array_equal(fused[:2], lidar_points)
        assert np.array_equal(fused[2:], depth_points)
        
        # Test with empty arrays
        empty = np.empty((0, 2))
        fused_empty = detector.fuse_point_clouds(empty, depth_points)
        assert np.array_equal(fused_empty, depth_points)
        
        fused_empty2 = detector.fuse_point_clouds(lidar_points, empty)
        assert np.array_equal(fused_empty2, lidar_points)
    
    def test_cluster_points(self, detector):
        """Test DBSCAN clustering."""
        # Create points forming two distinct clusters with sufficient size
        # Make clusters larger to pass the size filter (min_obstacle_size = 0.2)
        cluster1 = np.array([
            [1.0, 1.0], [1.3, 1.0], [1.0, 1.3], [1.3, 1.3], 
            [1.6, 1.0], [1.0, 1.6], [1.6, 1.6]
        ])
        cluster2 = np.array([
            [5.0, 5.0], [5.3, 5.0], [5.0, 5.3], [5.3, 5.3], 
            [5.6, 5.0], [5.0, 5.6], [5.6, 5.6]
        ])
        noise = np.array([[10.0, 10.0]])  # Isolated point (noise)
        
        points = np.vstack([cluster1, cluster2, noise])
        
        clusters = detector.cluster_points(points)
        
        # Should find 2 clusters (noise point should be filtered out)
        assert len(clusters) == 2
        
        # Each cluster should have the expected number of points
        cluster_sizes = [len(cluster) for cluster in clusters]
        assert 7 in cluster_sizes  # Both clusters have 7 points
        
        # Test with insufficient points
        few_points = np.array([[1.0, 1.0], [2.0, 2.0]])
        clusters_few = detector.cluster_points(few_points)
        assert len(clusters_few) == 0  # Not enough points for clustering
    
    def test_clusters_to_obstacles(self, detector):
        """Test conversion of clusters to obstacle detections."""
        # Create a cluster representing a worker-sized obstacle
        cluster = np.array([[2.0, 1.0], [2.1, 1.0], [2.0, 1.1], [2.1, 1.1]])
        clusters = [cluster]
        
        current_time = 0.0
        detections = detector.clusters_to_obstacles(clusters, current_time)
        
        assert len(detections) == 1
        detection = detections[0]
        
        # Check detection properties
        assert isinstance(detection, DetectedObstacle)
        assert len(detection.position) == 2
        assert detection.radius > 0
        assert detection.classification in ["worker", "forklift", "unknown"]
        assert 0.0 <= detection.confidence <= 1.0
        
        # Position should be near cluster center
        expected_center = np.mean(cluster, axis=0)
        assert np.allclose(detection.position, expected_center, atol=0.1)


class TestObstacleTrack:
    """Test cases for the ObstacleTrack class."""
    
    def test_track_creation(self):
        """Test obstacle track creation."""
        detection = DetectedObstacle(
            position=np.array([1.0, 2.0]),
            radius=0.5,
            classification="worker",
            confidence=0.8,
            timestamp=0.0
        )
        
        track = ObstacleTrack(track_id=1, initial_detection=detection, current_time=0.0)
        
        assert track.track_id == 1
        assert np.allclose(track.get_position(), detection.position)
        assert np.allclose(track.get_velocity(), [0.0, 0.0])  # Initial velocity is zero
        assert track.classification == detection.classification
        assert track.confidence == detection.confidence
        assert track.age == 0
        assert track.missed_detections == 0
        assert track.total_detections == 1
    
    def test_track_update(self):
        """Test obstacle track update with Kalman filter."""
        initial_detection = DetectedObstacle(
            position=np.array([1.0, 2.0]),
            radius=0.5,
            classification="worker",
            confidence=0.8,
            timestamp=0.0
        )
        
        track = ObstacleTrack(track_id=1, initial_detection=initial_detection, current_time=0.0)
        
        # Update with new position after 1 second
        new_detection = DetectedObstacle(
            position=np.array([2.0, 3.0]),  # Moved 1m in x, 1m in y
            radius=0.5,
            classification="worker",
            confidence=0.9,
            timestamp=1.0
        )
        
        # First predict, then update (as would happen in real tracking)
        track.predict(1.0)
        track.update(new_detection, current_time=1.0)
        
        # Position should be close to new detection (Kalman filter estimate)
        assert np.allclose(track.get_position(), new_detection.position, atol=0.2)
        
        # Velocity should be estimated by Kalman filter
        velocity = track.get_velocity()
        assert np.linalg.norm(velocity) > 0.5  # Should have some velocity estimate
        
        # Other properties should be updated
        assert track.confidence == new_detection.confidence
        assert track.missed_detections == 0
        assert track.total_detections == 2


@pytest.mark.unit
def test_detected_obstacle_creation():
    """Test that DetectedObstacle can be created correctly."""
    position = np.array([1.5, 2.5])
    radius = 0.6
    classification = "worker"
    confidence = 0.85
    timestamp = 0.0
    
    obstacle = DetectedObstacle(
        position=position,
        radius=radius,
        classification=classification,
        confidence=confidence,
        timestamp=timestamp
    )
    
    assert np.array_equal(obstacle.position, position)
    assert obstacle.radius == radius
    assert obstacle.classification == classification
    assert obstacle.confidence == confidence
    assert obstacle.timestamp == timestamp


if __name__ == '__main__':
    pytest.main([__file__])