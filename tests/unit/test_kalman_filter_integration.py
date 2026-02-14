#!/usr/bin/env python3
"""
Integration tests for Kalman filter obstacle tracking.

Tests the complete Kalman filter implementation including:
- Position and velocity estimation
- Data association with Hungarian algorithm
- Track management (creation, deletion)
"""

import pytest
import numpy as np
import time
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


class DetectedObstacle:
    """Represents a detected obstacle before tracking."""
    
    def __init__(self, position: np.ndarray, radius: float, classification: str, 
                 confidence: float, timestamp: float):
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


class TestKalmanFilterIntegration:
    """Integration tests for Kalman filter obstacle tracking."""
    
    def test_single_obstacle_tracking(self):
        """Test tracking a single moving obstacle."""
        # Create initial detection
        initial_pos = np.array([1.0, 2.0])
        detection = DetectedObstacle(
            position=initial_pos,
            radius=0.5,
            classification="worker",
            confidence=0.8,
            timestamp=0.0
        )
        
        # Create track
        track = ObstacleTrack(track_id=1, initial_detection=detection, current_time=0.0)
        
        # Simulate movement at 1 m/s in x direction
        dt = 0.1
        true_velocity = np.array([1.0, 0.0])
        
        for i in range(10):
            current_time = (i + 1) * dt
            
            # Predict
            track.predict(current_time)
            
            # Create new detection (true position + small noise)
            true_pos = initial_pos + true_velocity * current_time
            noise = np.random.normal(0, 0.05, 2)  # 5cm noise
            measured_pos = true_pos + noise
            
            new_detection = DetectedObstacle(
                position=measured_pos,
                radius=0.5,
                classification="worker",
                confidence=0.8,
                timestamp=current_time
            )
            
            # Update
            track.update(new_detection, current_time)
        
        # Check final estimates
        final_pos = track.get_position()
        final_vel = track.get_velocity()
        
        # Position should be close to true position
        true_final_pos = initial_pos + true_velocity * 1.0  # After 1 second
        pos_error = np.linalg.norm(final_pos - true_final_pos)
        assert pos_error < 0.2, f"Position error too large: {pos_error:.3f}m"
        
        # Velocity should be close to true velocity
        vel_error = np.linalg.norm(final_vel - true_velocity)
        assert vel_error < 0.3, f"Velocity error too large: {vel_error:.3f}m/s"
        
        # Track should have correct properties
        assert track.total_detections == 11  # Initial + 10 updates
        assert track.missed_detections == 0
        assert not track.is_stationary()
    
    def test_data_association_hungarian(self):
        """Test data association using Hungarian algorithm."""
        # Create two tracks
        track1 = ObstacleTrack(
            track_id=1,
            initial_detection=DetectedObstacle(
                position=np.array([1.0, 1.0]),
                radius=0.5,
                classification="worker",
                confidence=0.8,
                timestamp=0.0
            ),
            current_time=0.0
        )
        
        track2 = ObstacleTrack(
            track_id=2,
            initial_detection=DetectedObstacle(
                position=np.array([3.0, 3.0]),
                radius=0.5,
                classification="worker",
                confidence=0.8,
                timestamp=0.0
            ),
            current_time=0.0
        )
        
        tracks = [track1, track2]
        
        # Create detections (slightly moved)
        detections = [
            DetectedObstacle(
                position=np.array([1.1, 1.1]),  # Close to track1
                radius=0.5,
                classification="worker",
                confidence=0.8,
                timestamp=0.1
            ),
            DetectedObstacle(
                position=np.array([3.1, 3.1]),  # Close to track2
                radius=0.5,
                classification="worker",
                confidence=0.8,
                timestamp=0.1
            )
        ]
        
        # Create cost matrix
        cost_matrix = np.zeros((len(tracks), len(detections)))
        
        for i, track in enumerate(tracks):
            track_pos = track.get_position()
            for j, detection in enumerate(detections):
                # Euclidean distance
                cost_matrix[i, j] = np.linalg.norm(detection.position - track_pos)
        
        # Solve assignment problem
        track_indices, detection_indices = linear_sum_assignment(cost_matrix)
        
        # Check assignments
        assert len(track_indices) == 2
        assert len(detection_indices) == 2
        
        # Track 1 should be assigned to detection 1 (both at ~[1,1])
        # Track 2 should be assigned to detection 2 (both at ~[3,3])
        for track_idx, det_idx in zip(track_indices, detection_indices):
            if track_idx == 0:  # track1
                assert det_idx == 0, "Track1 should be assigned to detection1"
            else:  # track2
                assert det_idx == 1, "Track2 should be assigned to detection2"
    
    def test_track_deletion_criteria(self):
        """Test track deletion based on missed detections."""
        # Create track
        detection = DetectedObstacle(
            position=np.array([1.0, 2.0]),
            radius=0.5,
            classification="worker",
            confidence=0.8,
            timestamp=0.0
        )
        
        track = ObstacleTrack(track_id=1, initial_detection=detection, current_time=0.0)
        
        # Initially should not be deleted
        assert not track.should_delete()
        
        # Mark as missed several times
        for i in range(6):  # More than max_missed (5)
            track.mark_missed()
        
        # Should be deleted due to too many missed detections
        assert track.should_delete()
    
    def test_velocity_estimation_accuracy(self):
        """Test accuracy of velocity estimation."""
        # Create track with known velocity
        initial_pos = np.array([0.0, 0.0])
        detection = DetectedObstacle(
            position=initial_pos,
            radius=0.5,
            classification="worker",
            confidence=0.8,
            timestamp=0.0
        )
        
        track = ObstacleTrack(track_id=1, initial_detection=detection, current_time=0.0)
        
        # Simulate constant velocity movement
        true_velocity = np.array([2.0, 1.5])  # 2 m/s in x, 1.5 m/s in y
        dt = 0.1
        
        # Provide several measurements
        for i in range(20):
            current_time = (i + 1) * dt
            
            # Predict
            track.predict(current_time)
            
            # True position
            true_pos = initial_pos + true_velocity * current_time
            
            # Add small amount of noise
            noise = np.random.normal(0, 0.02, 2)  # 2cm noise
            measured_pos = true_pos + noise
            
            new_detection = DetectedObstacle(
                position=measured_pos,
                radius=0.5,
                classification="worker",
                confidence=0.8,
                timestamp=current_time
            )
            
            track.update(new_detection, current_time)
        
        # Check velocity estimate
        estimated_velocity = track.get_velocity()
        velocity_error = np.linalg.norm(estimated_velocity - true_velocity)
        
        # Should be quite accurate after 20 measurements
        assert velocity_error < 0.2, f"Velocity estimation error too large: {velocity_error:.3f}m/s"
        
        # Check that track is not considered stationary
        assert not track.is_stationary()
        assert track.get_velocity_magnitude() > 2.0  # Should be close to true magnitude
    
    def test_position_uncertainty_tracking(self):
        """Test that position uncertainty is properly tracked."""
        detection = DetectedObstacle(
            position=np.array([1.0, 2.0]),
            radius=0.5,
            classification="worker",
            confidence=0.8,
            timestamp=0.0
        )
        
        track = ObstacleTrack(track_id=1, initial_detection=detection, current_time=0.0)
        
        # Initial uncertainty should be reasonable
        initial_uncertainty = track.get_position_uncertainty()
        assert 0.1 < initial_uncertainty < 1.0
        
        # After several updates, uncertainty should decrease
        for i in range(5):
            current_time = (i + 1) * 0.1
            track.predict(current_time)
            
            new_detection = DetectedObstacle(
                position=np.array([1.0 + i * 0.1, 2.0]),
                radius=0.5,
                classification="worker",
                confidence=0.8,
                timestamp=current_time
            )
            
            track.update(new_detection, current_time)
        
        # Uncertainty should have decreased with more measurements
        final_uncertainty = track.get_position_uncertainty()
        assert final_uncertainty < initial_uncertainty
        
        # But should still be positive
        assert final_uncertainty > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])