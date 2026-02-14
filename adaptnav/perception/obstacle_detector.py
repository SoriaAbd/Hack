#!/usr/bin/env python3
"""
Obstacle Detector Node for AdaptNav

This node implements sensor fusion between LiDAR and depth camera data to detect
and track dynamic obstacles in the warehouse environment. It uses DBSCAN clustering
for obstacle detection and publishes detected obstacles as custom_msgs/ObstacleArray.

Requirements: 4.1, 4.3
"""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import LaserScan, Image, CameraInfo
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Vector3
from std_msgs.msg import Header
from builtin_interfaces.msg import Time
from custom_msgs.msg import Obstacle, ObstacleArray
from visualization_msgs.msg import MarkerArray, Marker
import cv2
from cv_bridge import CvBridge
from sklearn.cluster import DBSCAN
import time
from typing import List, Tuple, Optional, Dict
import threading
from scipy.optimize import linear_sum_assignment


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
    
    def predict(self, dt: Optional[float] = None) -> None:
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


class ObstacleDetector(Node):
    """
    ROS 2 node for detecting and tracking dynamic obstacles using sensor fusion.
    
    Subscribes to:
        - /scan (sensor_msgs/LaserScan): LiDAR point cloud data
        - /camera/depth/image_raw (sensor_msgs/Image): Depth camera images
        - /camera/camera_info (sensor_msgs/CameraInfo): Camera calibration
        - /odom (nav_msgs/Odometry): Robot odometry for coordinate transforms
    
    Publishes:
        - /obstacles/detected (custom_msgs/ObstacleArray): Detected obstacles
        - /obstacles/visualization (visualization_msgs/MarkerArray): Visualization markers
    """
    
    def __init__(self):
        super().__init__('obstacle_detector')
        
        # Initialize cv_bridge for image conversion
        self._bridge = CvBridge()
        
        # QoS profile for sensor data (best effort, volatile)
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )
        
        # Subscribers
        self._scan_sub = self.create_subscription(
            LaserScan, '/scan', self._scan_callback, sensor_qos)
        self._depth_image_sub = self.create_subscription(
            Image, '/camera/depth/image_raw', self._depth_image_callback, sensor_qos)
        self._camera_info_sub = self.create_subscription(
            CameraInfo, '/camera/camera_info', self._camera_info_callback, 10)
        self._odom_sub = self.create_subscription(
            Odometry, '/odom', self._odom_callback, 10)
        
        # Publishers
        self._obstacles_pub = self.create_publisher(
            ObstacleArray, '/obstacles/detected', 10)
        self._visualization_pub = self.create_publisher(
            MarkerArray, '/obstacles/visualization', 10)
        
        # Data storage
        self._latest_scan: Optional[LaserScan] = None
        self._latest_depth_image: Optional[np.ndarray] = None
        self._latest_camera_info: Optional[CameraInfo] = None
        self._latest_odom: Optional[Odometry] = None
        
        # Synchronization
        self._data_lock = threading.Lock()
        
        # Detection parameters
        self._dbscan_eps = 0.5  # Maximum distance between points in a cluster (meters)
        self._dbscan_min_samples = 3  # Minimum points to form a cluster
        self._max_detection_range = 8.0  # Maximum range for obstacle detection (meters)
        self._min_obstacle_size = 0.2  # Minimum obstacle radius (meters)
        self._max_obstacle_size = 2.0  # Maximum obstacle radius (meters)
        
        # Tracking parameters
        self._obstacle_tracks: Dict[int, ObstacleTrack] = {}
        self._next_track_id = 1
        self._max_track_age = 2.0  # Maximum time to keep track without detection (seconds)
        self._association_threshold = 1.0  # Maximum distance for track association (meters)
        
        # Processing timer - run detection at 10 Hz
        self._detection_timer = self.create_timer(0.1, self._detection_callback)
        
        self.get_logger().info('Obstacle detector initialized')
    
    def _scan_callback(self, msg: LaserScan) -> None:
        """Callback for LiDAR scan data."""
        with self._data_lock:
            self._latest_scan = msg
    
    def _depth_image_callback(self, msg: Image) -> None:
        """Callback for depth camera images."""
        try:
            # Convert ROS image to OpenCV format
            depth_image = self._bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
            with self._data_lock:
                self._latest_depth_image = depth_image
        except Exception as e:
            self.get_logger().error(f'Failed to convert depth image: {e}')
    
    def _camera_info_callback(self, msg: CameraInfo) -> None:
        """Callback for camera calibration info."""
        with self._data_lock:
            self._latest_camera_info = msg
    
    def _odom_callback(self, msg: Odometry) -> None:
        """Callback for robot odometry."""
        with self._data_lock:
            self._latest_odom = msg
    
    def _detection_callback(self) -> None:
        """Main detection processing callback."""
        try:
            # Get latest sensor data
            with self._data_lock:
                scan = self._latest_scan
                depth_image = self._latest_depth_image
                camera_info = self._latest_camera_info
                odom = self._latest_odom
            
            # Check if we have all required data
            if scan is None or odom is None:
                return
            
            current_time = self.get_clock().now()
            
            # Extract point clouds from sensors
            lidar_points = self._extract_lidar_points(scan)
            depth_points = []
            
            if depth_image is not None and camera_info is not None:
                depth_points = self._extract_depth_points(depth_image, camera_info)
            
            # Fuse point clouds
            fused_points = self._fuse_point_clouds(lidar_points, depth_points)
            
            if len(fused_points) == 0:
                # No points to process, publish empty obstacle array
                self._publish_obstacles([], current_time)
                return
            
            # Cluster points to detect obstacles
            clusters = self._cluster_points(fused_points)
            
            # Convert clusters to obstacle detections
            detections = self._clusters_to_obstacles(clusters, current_time)
            
            # Update tracking
            tracked_obstacles = self._update_tracking(detections, current_time)
            
            # Publish results
            self._publish_obstacles(tracked_obstacles, current_time)
            self._publish_visualization(tracked_obstacles, current_time)
            
        except Exception as e:
            self.get_logger().error(f'Error in detection callback: {e}')
    
    def _extract_lidar_points(self, scan: LaserScan) -> np.ndarray:
        """
        Extract 2D points from LiDAR scan.
        
        Args:
            scan: LiDAR scan message
            
        Returns:
            Array of shape (N, 2) containing [x, y] points in robot frame
        """
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
    
    def _extract_depth_points(self, depth_image: np.ndarray, camera_info: CameraInfo) -> np.ndarray:
        """
        Extract 3D points from depth camera and project to 2D.
        
        Args:
            depth_image: Depth image as numpy array
            camera_info: Camera calibration information
            
        Returns:
            Array of shape (N, 2) containing [x, y] points in robot frame
        """
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
    
    def _fuse_point_clouds(self, lidar_points: np.ndarray, depth_points: np.ndarray) -> np.ndarray:
        """
        Fuse LiDAR and depth camera point clouds.
        
        Args:
            lidar_points: Points from LiDAR (N, 2)
            depth_points: Points from depth camera (M, 2)
            
        Returns:
            Fused point cloud (N+M, 2)
        """
        if len(lidar_points) == 0 and len(depth_points) == 0:
            return np.empty((0, 2))
        elif len(lidar_points) == 0:
            return depth_points
        elif len(depth_points) == 0:
            return lidar_points
        else:
            # Simple concatenation - could be improved with more sophisticated fusion
            return np.vstack([lidar_points, depth_points])
    
    def _cluster_points(self, points: np.ndarray) -> List[np.ndarray]:
        """
        Cluster points using DBSCAN to identify obstacles.
        
        Args:
            points: Point cloud (N, 2)
            
        Returns:
            List of point clusters, each as (M, 2) array
        """
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
    
    def _clusters_to_obstacles(self, clusters: List[np.ndarray], current_time) -> List['DetectedObstacle']:
        """
        Convert point clusters to obstacle detections with enhanced classification heuristics.
        
        Classification is based on:
        - Size: workers ~0.5m radius, forklifts ~1.0m radius
        - Velocity: workers <2 m/s, forklifts <5 m/s (applied during tracking)
        - Confidence: based on detection consistency and cluster quality
        
        Args:
            clusters: List of point clusters
            current_time: Current ROS time
            
        Returns:
            List of detected obstacles
        """
        detections = []
        
        for i, cluster in enumerate(clusters):
            # Compute obstacle properties
            center = np.mean(cluster, axis=0)
            distances = np.linalg.norm(cluster - center, axis=1)
            radius = np.max(distances)
            
            # Enhanced size-based classification with more precise thresholds
            classification, size_confidence = self._classify_by_size(radius)
            
            # Estimate confidence based on cluster quality
            cluster_confidence = self._compute_cluster_confidence(cluster, center, radius)
            
            # Combine confidences
            overall_confidence = size_confidence * cluster_confidence
            
            detection = DetectedObstacle(
                position=center,
                radius=radius,
                classification=classification,
                confidence=overall_confidence,
                timestamp=current_time
            )
            
            detections.append(detection)
        
        return detections
    
    def _classify_by_size(self, radius: float) -> Tuple[str, float]:
        """
        Classify obstacle based on size with confidence scoring.
        
        Classification thresholds:
        - Worker: 0.3m - 0.7m radius (typical ~0.5m)
        - Forklift: 0.8m - 1.5m radius (typical ~1.0m)
        - Unknown: outside these ranges or in overlap zones
        
        Args:
            radius: Obstacle radius in meters
            
        Returns:
            Tuple of (classification, confidence)
        """
        # Define classification ranges with confidence zones
        worker_min, worker_max = 0.3, 0.7
        worker_optimal = 0.5
        forklift_min, forklift_max = 0.8, 1.5
        forklift_optimal = 1.0
        
        if worker_min <= radius <= worker_max:
            # Worker classification
            # Confidence is highest at optimal size, decreases toward boundaries
            distance_from_optimal = abs(radius - worker_optimal)
            max_distance = max(worker_optimal - worker_min, worker_max - worker_optimal)
            confidence = 1.0 - (distance_from_optimal / max_distance) * 0.3  # Max 30% penalty
            return "worker", max(0.7, confidence)  # Minimum 70% confidence
            
        elif forklift_min <= radius <= forklift_max:
            # Forklift classification
            distance_from_optimal = abs(radius - forklift_optimal)
            max_distance = max(forklift_optimal - forklift_min, forklift_max - forklift_optimal)
            confidence = 1.0 - (distance_from_optimal / max_distance) * 0.3  # Max 30% penalty
            return "forklift", max(0.7, confidence)  # Minimum 70% confidence
            
        elif radius < worker_min:
            # Too small - likely noise or small object
            return "unknown", 0.3
            
        elif worker_max < radius < forklift_min:
            # In between worker and forklift - ambiguous
            # Lean toward worker if closer to worker range
            if radius < (worker_max + forklift_min) / 2:
                return "worker", 0.5  # Low confidence worker
            else:
                return "forklift", 0.5  # Low confidence forklift
                
        else:
            # Too large for typical forklift
            return "unknown", 0.4
    
    def _compute_cluster_confidence(self, cluster: np.ndarray, center: np.ndarray, radius: float) -> float:
        """
        Compute confidence based on cluster quality and consistency.
        
        Factors considered:
        - Number of points (more points = higher confidence)
        - Cluster compactness (tighter clusters = higher confidence)
        - Cluster shape regularity
        
        Args:
            cluster: Point cluster
            center: Cluster center
            radius: Cluster radius
            
        Returns:
            Confidence score [0.0, 1.0]
        """
        num_points = len(cluster)
        
        # Point density confidence (more points = better)
        # Expect at least 3 points, optimal around 8-12 points
        point_confidence = min(1.0, num_points / 8.0)
        
        # Compactness confidence (consistent distances from center)
        distances = np.linalg.norm(cluster - center, axis=1)
        if radius > 0:
            distance_std = np.std(distances)
            compactness = 1.0 - min(1.0, distance_std / radius)  # Lower std = higher compactness
        else:
            compactness = 1.0
        
        # Shape regularity (how circular/regular the cluster is)
        if num_points >= 4:
            # Compute angles from center to each point
            vectors = cluster - center
            angles = np.arctan2(vectors[:, 1], vectors[:, 0])
            
            # Sort angles and compute angular gaps
            angles_sorted = np.sort(angles)
            angular_gaps = np.diff(angles_sorted)
            # Add wrap-around gap
            angular_gaps = np.append(angular_gaps, 2*np.pi - (angles_sorted[-1] - angles_sorted[0]))
            
            # Regular distribution would have similar angular gaps
            expected_gap = 2 * np.pi / num_points
            gap_variance = np.var(angular_gaps)
            regularity = np.exp(-gap_variance / (expected_gap**2))  # Exponential decay with variance
        else:
            regularity = 0.7  # Default for small clusters
        
        # Combine factors with weights
        overall_confidence = (
            0.4 * point_confidence +
            0.4 * compactness +
            0.2 * regularity
        )
        
        return max(0.2, min(1.0, overall_confidence))  # Clamp to [0.2, 1.0]
    
    def _update_tracking(self, detections: List['DetectedObstacle'], current_time) -> List[Obstacle]:
        """
        Update obstacle tracking with new detections using improved data association.
        
        Args:
            detections: New obstacle detections
            current_time: Current ROS time
            
        Returns:
            List of tracked obstacles
        """
        current_time_sec = current_time.nanoseconds / 1e9
        
        # Predict all existing tracks to current time
        for track in self._obstacle_tracks.values():
            track.predict(current_time_sec)
        
        # Remove tracks that should be deleted
        tracks_to_remove = []
        for track_id, track in self._obstacle_tracks.items():
            if track.should_delete():
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self._obstacle_tracks[track_id]
            self.get_logger().debug(f'Removed track {track_id}')
        
        # Data association using Hungarian algorithm
        if len(self._obstacle_tracks) > 0 and len(detections) > 0:
            # Create cost matrix (distance between tracks and detections)
            tracks = list(self._obstacle_tracks.values())
            cost_matrix = np.zeros((len(tracks), len(detections)))
            
            for i, track in enumerate(tracks):
                track_pos = track.get_position()
                for j, detection in enumerate(detections):
                    # Mahalanobis distance considering position uncertainty
                    pos_diff = detection.position - track_pos
                    pos_cov = track.kalman_filter.get_position_covariance()
                    
                    # Add small regularization to avoid singular matrix
                    pos_cov += np.eye(2) * 1e-6
                    
                    try:
                        mahal_dist = np.sqrt(pos_diff.T @ np.linalg.inv(pos_cov) @ pos_diff)
                        cost_matrix[i, j] = mahal_dist
                    except np.linalg.LinAlgError:
                        # Fallback to Euclidean distance if covariance is singular
                        cost_matrix[i, j] = np.linalg.norm(pos_diff)
            
            # Solve assignment problem
            track_indices, detection_indices = linear_sum_assignment(cost_matrix)
            
            # Apply assignments with distance threshold
            max_association_distance = 2.0  # Maximum Mahalanobis distance for association
            associated_detections = set()
            
            for track_idx, det_idx in zip(track_indices, detection_indices):
                if cost_matrix[track_idx, det_idx] <= max_association_distance:
                    # Valid association
                    track = tracks[track_idx]
                    detection = detections[det_idx]
                    track.update(detection, current_time_sec)
                    associated_detections.add(det_idx)
                    self.get_logger().debug(f'Associated detection {det_idx} with track {track.track_id}')
                else:
                    # Association too far, mark track as missed
                    tracks[track_idx].mark_missed()
            
            # Mark unassociated tracks as missed
            for i, track in enumerate(tracks):
                if i not in track_indices or cost_matrix[i, detection_indices[np.where(track_indices == i)[0][0]]] > max_association_distance:
                    track.mark_missed()
            
            # Create new tracks for unassociated detections
            for det_idx, detection in enumerate(detections):
                if det_idx not in associated_detections:
                    self._create_new_track(detection, current_time_sec)
        
        elif len(detections) > 0:
            # No existing tracks, create new tracks for all detections
            for detection in detections:
                self._create_new_track(detection, current_time_sec)
        
        else:
            # No detections, mark all tracks as missed
            for track in self._obstacle_tracks.values():
                track.mark_missed()
        
        # Convert tracks to ROS messages
        obstacles = []
        for track in self._obstacle_tracks.values():
            obstacle = track.to_obstacle_msg(current_time)
            obstacles.append(obstacle)
        
        return obstacles
    
    def _create_new_track(self, detection: DetectedObstacle, current_time: float) -> None:
        """
        Create a new track for an unassociated detection.
        
        Args:
            detection: Obstacle detection
            current_time: Current timestamp
        """
        track = ObstacleTrack(
            track_id=self._next_track_id,
            initial_detection=detection,
            current_time=current_time
        )
        self._obstacle_tracks[self._next_track_id] = track
        self.get_logger().debug(f'Created new track {self._next_track_id} at position ({detection.position[0]:.2f}, {detection.position[1]:.2f})')
        self._next_track_id += 1
    
    def _publish_obstacles(self, obstacles: List[Obstacle], current_time) -> None:
        """Publish detected obstacles."""
        msg = ObstacleArray()
        msg.header = Header()
        msg.header.stamp = current_time.to_msg()
        msg.header.frame_id = 'base_link'  # Robot frame
        msg.obstacles = obstacles
        
        self._obstacles_pub.publish(msg)
    
    def _publish_visualization(self, obstacles: List[Obstacle], current_time) -> None:
        """Publish visualization markers for obstacles."""
        marker_array = MarkerArray()
        
        for i, obstacle in enumerate(obstacles):
            # Position marker
            marker = Marker()
            marker.header.frame_id = 'base_link'
            marker.header.stamp = current_time.to_msg()
            marker.ns = 'obstacles'
            marker.id = int(obstacle.id)
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            
            marker.pose.position = obstacle.position
            marker.pose.position.z = 0.5  # Half height for visualization
            marker.pose.orientation.w = 1.0
            
            # Size based on obstacle classification
            if obstacle.classification == "worker":
                marker.scale.x = marker.scale.y = 1.0  # 0.5m radius
                marker.scale.z = 1.7  # Human height
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
            elif obstacle.classification == "forklift":
                marker.scale.x = marker.scale.y = 2.0  # 1.0m radius
                marker.scale.z = 2.0  # Forklift height
                marker.color.r = 1.0
                marker.color.g = 0.5
                marker.color.b = 0.0
            else:  # unknown
                marker.scale.x = marker.scale.y = 1.5
                marker.scale.z = 1.0
                marker.color.r = 0.5
                marker.color.g = 0.5
                marker.color.b = 0.5
            
            # Color intensity based on confidence
            marker.color.a = 0.4 + 0.4 * obstacle.confidence  # Semi-transparent, brighter with higher confidence
            marker.lifetime.sec = 1  # Marker lifetime
            
            marker_array.markers.append(marker)
            
            # Velocity arrow
            vel_magnitude = np.sqrt(obstacle.velocity.x**2 + obstacle.velocity.y**2)
            if vel_magnitude > 0.1:
                arrow = Marker()
                arrow.header = marker.header
                arrow.ns = 'velocities'
                arrow.id = int(obstacle.id)
                arrow.type = Marker.ARROW
                arrow.action = Marker.ADD
                
                arrow.pose.position = obstacle.position
                arrow.pose.position.z = 1.0  # Above the obstacle
                
                # Arrow direction from velocity
                arrow.scale.x = vel_magnitude  # Arrow length
                arrow.scale.y = 0.1  # Arrow width
                arrow.scale.z = 0.1  # Arrow height
                
                # Arrow orientation
                yaw = np.arctan2(obstacle.velocity.y, obstacle.velocity.x)
                arrow.pose.orientation.z = np.sin(yaw / 2)
                arrow.pose.orientation.w = np.cos(yaw / 2)
                
                arrow.color.r = 1.0
                arrow.color.g = 1.0
                arrow.color.b = 0.0
                arrow.color.a = 0.8
                arrow.lifetime.sec = 1
                
                marker_array.markers.append(arrow)
            
            # Uncertainty ellipse (position covariance visualization)
            if len(obstacle.covariance) >= 4:
                # Extract position covariance (2x2 from 4x4)
                pos_cov = np.array([
                    [obstacle.covariance[0], obstacle.covariance[1]],
                    [obstacle.covariance[4], obstacle.covariance[5]]
                ])
                
                # Compute eigenvalues and eigenvectors for ellipse
                eigenvals, eigenvecs = np.linalg.eigh(pos_cov)
                
                if np.all(eigenvals > 0):  # Valid covariance
                    # Create uncertainty ellipse
                    ellipse = Marker()
                    ellipse.header = marker.header
                    ellipse.ns = 'uncertainty'
                    ellipse.id = int(obstacle.id)
                    ellipse.type = Marker.CYLINDER
                    ellipse.action = Marker.ADD
                    
                    ellipse.pose.position = obstacle.position
                    ellipse.pose.position.z = 0.05  # Just above ground
                    
                    # Ellipse orientation from eigenvectors
                    angle = np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0])
                    ellipse.pose.orientation.z = np.sin(angle / 2)
                    ellipse.pose.orientation.w = np.cos(angle / 2)
                    
                    # Ellipse size from eigenvalues (2-sigma bounds)
                    ellipse.scale.x = 2 * np.sqrt(eigenvals[0]) * 2  # 2-sigma
                    ellipse.scale.y = 2 * np.sqrt(eigenvals[1]) * 2  # 2-sigma
                    ellipse.scale.z = 0.1  # Thin ellipse
                    
                    ellipse.color.r = 1.0
                    ellipse.color.g = 1.0
                    ellipse.color.b = 1.0
                    ellipse.color.a = 0.2  # Very transparent
                    ellipse.lifetime.sec = 1
                    
                    marker_array.markers.append(ellipse)
        
        self._visualization_pub.publish(marker_array)


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
        
        # Refine classification using velocity information
        refined_classification, velocity_confidence = self._refine_classification_with_velocity(
            detection.classification, detection.confidence
        )
        
        # Update track properties with refined classification
        self.classification = refined_classification
        self.confidence = velocity_confidence
        self.last_update_time = current_time
        self.missed_detections = 0
        self.total_detections += 1
    
    def _refine_classification_with_velocity(self, size_classification: str, size_confidence: float) -> Tuple[str, float]:
        """
        Refine obstacle classification using velocity information.
        
        Velocity thresholds:
        - Workers: typically < 2 m/s (walking speed ~1.4 m/s)
        - Forklifts: typically < 5 m/s (max speed ~3-4 m/s in warehouse)
        
        Args:
            size_classification: Classification from size heuristics
            size_confidence: Confidence from size heuristics
            
        Returns:
            Tuple of (refined_classification, refined_confidence)
        """
        velocity_magnitude = self.get_velocity_magnitude()
        
        # Define velocity thresholds
        worker_max_velocity = 2.0  # m/s
        forklift_max_velocity = 5.0  # m/s
        
        # Velocity-based confidence adjustments
        if velocity_magnitude <= worker_max_velocity:
            # Consistent with worker movement
            if size_classification == "worker":
                # Size and velocity both suggest worker - high confidence
                velocity_confidence = min(1.0, size_confidence + 0.2)
                return "worker", velocity_confidence
            elif size_classification == "forklift":
                # Size suggests forklift but velocity suggests worker
                # Could be a slow-moving forklift or misclassified worker
                if velocity_magnitude < 0.5:  # Very slow, likely stationary forklift
                    return "forklift", size_confidence * 0.8  # Slight penalty
                else:  # Moving at worker speed, likely worker
                    return "worker", 0.6  # Medium confidence
            else:  # unknown
                # Use velocity to make a guess
                if velocity_magnitude > 0.1:  # Moving
                    return "worker", 0.5  # Low confidence worker
                else:  # Stationary
                    return "unknown", size_confidence
                    
        elif velocity_magnitude <= forklift_max_velocity:
            # Consistent with forklift movement (too fast for worker)
            if size_classification == "forklift":
                # Size and velocity both suggest forklift - high confidence
                velocity_confidence = min(1.0, size_confidence + 0.2)
                return "forklift", velocity_confidence
            elif size_classification == "worker":
                # Size suggests worker but velocity suggests forklift
                # Likely a misclassified forklift or very fast worker
                if velocity_magnitude > 2.5:  # Very fast, likely forklift
                    return "forklift", 0.7  # Good confidence
                else:  # Moderately fast, could be either
                    return "unknown", 0.4  # Low confidence, ambiguous
            else:  # unknown
                # Use velocity to make a guess
                return "forklift", 0.5  # Low confidence forklift
                
        else:
            # Too fast for typical warehouse obstacles
            # Could be measurement error or unusual object
            if size_classification in ["worker", "forklift"]:
                # Velocity contradicts size classification
                return "unknown", 0.3  # Low confidence due to inconsistency
            else:
                return "unknown", size_confidence * 0.5  # Penalty for high velocity
    
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
    
    def to_obstacle_msg(self, current_time) -> Obstacle:
        """Convert track to ROS Obstacle message with enhanced confidence scoring."""
        obstacle = Obstacle()
        obstacle.id = self.track_id
        
        # Position from Kalman filter
        position = self.get_position()
        obstacle.position = Point()
        obstacle.position.x = float(position[0])
        obstacle.position.y = float(position[1])
        obstacle.position.z = 0.0
        
        # Velocity from Kalman filter
        velocity = self.get_velocity()
        obstacle.velocity = Vector3()
        obstacle.velocity.x = float(velocity[0])
        obstacle.velocity.y = float(velocity[1])
        obstacle.velocity.z = 0.0
        
        # Covariance from Kalman filter (4x4 -> 16 element array)
        full_cov = self.kalman_filter.get_full_covariance()
        obstacle.covariance = full_cov.flatten().tolist()
        
        obstacle.classification = self.classification
        
        # Enhanced confidence scoring based on detection consistency
        obstacle.confidence = self._compute_detection_consistency_confidence()
        
        obstacle.last_seen = current_time.to_msg()
        
        return obstacle
    
    def _compute_detection_consistency_confidence(self) -> float:
        """
        Compute confidence based on detection consistency over time.
        
        Factors considered:
        - Track maturity (more detections = higher confidence)
        - Detection rate (fewer missed detections = higher confidence)
        - Position uncertainty (lower uncertainty = higher confidence)
        - Velocity consistency (stable velocity = higher confidence)
        
        Returns:
            Confidence score [0.0, 1.0]
        """
        # Base confidence from latest classification
        base_confidence = self.confidence
        
        # Track maturity factor (more detections = more reliable)
        # Confidence increases with number of detections, plateaus at 10
        maturity_factor = min(1.0, self.total_detections / 10.0)
        
        # Detection rate factor (consistent detection = more reliable)
        # Penalize tracks with many missed detections
        if self.age > 0:
            detection_rate = self.total_detections / self.age
            detection_factor = min(1.0, detection_rate * 2.0)  # Expect ~50% detection rate
        else:
            detection_factor = 1.0
        
        # Position uncertainty factor (lower uncertainty = higher confidence)
        position_uncertainty = self.get_position_uncertainty()
        # Normalize uncertainty (typical values 0.1-2.0)
        uncertainty_factor = max(0.3, 1.0 - min(1.0, position_uncertainty / 2.0))
        
        # Velocity consistency factor
        velocity_magnitude = self.get_velocity_magnitude()
        if self.total_detections >= 3:
            # For mature tracks, check if velocity is reasonable for classification
            if self.classification == "worker" and velocity_magnitude > 2.0:
                velocity_factor = 0.7  # Penalty for inconsistent velocity
            elif self.classification == "forklift" and velocity_magnitude > 5.0:
                velocity_factor = 0.7  # Penalty for inconsistent velocity
            else:
                velocity_factor = 1.0  # Consistent velocity
        else:
            velocity_factor = 0.9  # Slight penalty for immature tracks
        
        # Recent detection factor (penalize tracks not seen recently)
        recent_factor = max(0.5, 1.0 - self.missed_detections * 0.1)
        
        # Combine all factors
        overall_confidence = (
            base_confidence * 
            maturity_factor * 
            detection_factor * 
            uncertainty_factor * 
            velocity_factor * 
            recent_factor
        )
        
        return max(0.1, min(1.0, overall_confidence))  # Clamp to [0.1, 1.0]


def main(args=None):
    """Main entry point for the obstacle detector node."""
    rclpy.init(args=args)
    
    try:
        detector = ObstacleDetector()
        rclpy.spin(detector)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error in obstacle detector: {e}')
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()