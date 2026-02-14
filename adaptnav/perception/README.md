# Obstacle Detection Module

This module implements dynamic obstacle detection and tracking for the AdaptNav warehouse navigation system.

## Overview

The obstacle detector node performs sensor fusion between LiDAR and depth camera data to detect and track dynamic obstacles (workers, forklifts) in the warehouse environment.

## Features

- **Sensor Fusion**: Combines LiDAR point cloud and depth camera data
- **DBSCAN Clustering**: Groups sensor points into obstacle clusters
- **Object Tracking**: Maintains consistent tracking IDs across frames
- **Velocity Estimation**: Estimates obstacle velocities using position history
- **Classification**: Simple heuristic-based classification (worker/forklift/unknown)
- **ROS 2 Integration**: Standard ROS 2 topics and message types

## Node Information

### Subscribed Topics

- `/scan` (sensor_msgs/LaserScan): LiDAR point cloud data at 20 Hz
- `/camera/depth/image_raw` (sensor_msgs/Image): Depth camera images at 15 Hz  
- `/camera/camera_info` (sensor_msgs/CameraInfo): Camera calibration parameters
- `/odom` (nav_msgs/Odometry): Robot odometry for coordinate transforms

### Published Topics

- `/obstacles/detected` (custom_msgs/ObstacleArray): Detected obstacles with positions, velocities, and classifications
- `/obstacles/visualization` (visualization_msgs/MarkerArray): Visualization markers for RViz

### Parameters

- `dbscan_eps`: Maximum distance between points in a cluster (default: 0.5m)
- `dbscan_min_samples`: Minimum points to form a cluster (default: 3)
- `max_detection_range`: Maximum range for obstacle detection (default: 8.0m)
- `min_obstacle_size`: Minimum obstacle radius (default: 0.2m)
- `max_obstacle_size`: Maximum obstacle radius (default: 2.0m)

## Algorithm Details

### Sensor Fusion

1. **LiDAR Processing**: Converts polar coordinates to Cartesian, filters by range
2. **Depth Camera Processing**: Projects depth pixels to 3D points, transforms to robot frame
3. **Fusion**: Simple concatenation of point clouds (can be enhanced)

### Clustering

Uses DBSCAN algorithm to group nearby points into clusters representing obstacles:
- Groups points within `dbscan_eps` distance
- Requires minimum `dbscan_min_samples` points per cluster
- Filters clusters by size (radius between `min_obstacle_size` and `max_obstacle_size`)

### Tracking

Maintains obstacle tracks across frames:
- Associates new detections with existing tracks using distance threshold
- Creates new tracks for unassociated detections
- Removes tracks that haven't been updated for `max_track_age` seconds
- Estimates velocity using exponential smoothing

### Classification

Simple heuristic based on obstacle size:
- `radius < 0.7m`: "worker"
- `radius > 1.2m`: "forklift"  
- Otherwise: "unknown"

## Usage

### Running the Node

```bash
# With ROS 2 installed and sourced:
ros2 run adaptnav obstacle_detector

# Or using the launch script:
python scripts/run_obstacle_detector.py
```

### Visualization

View detected obstacles in RViz:
1. Add MarkerArray display
2. Set topic to `/obstacles/visualization`
3. Obstacles appear as colored cylinders with velocity arrows

### Integration

The obstacle detector is designed to work with:
- MuJoCo simulation (provides sensor data)
- Safety controller (consumes obstacle data)
- Navigation controller (uses for replanning)
- Dashboard (displays obstacle information)

## Testing

Run unit tests for the core algorithms:

```bash
python -m pytest tests/unit/test_obstacle_detector_algorithms.py -v
```

Tests cover:
- LiDAR point extraction
- Depth camera point extraction  
- Point cloud fusion
- DBSCAN clustering
- Obstacle tracking
- Velocity estimation

## Requirements

Validates the following system requirements:
- **4.1**: Obstacle detection from sensor data
- **4.3**: Sensor fusion between LiDAR and depth camera

## Future Enhancements

- More sophisticated sensor fusion (e.g., weighted by confidence)
- Kalman filter for better velocity estimation
- Machine learning-based classification
- Multi-hypothesis tracking for better data association
- Obstacle trajectory prediction