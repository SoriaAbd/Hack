"""
MuJoCo simulation backend for AdaptNav warehouse environment.

This module implements the BaseSimulation interface using MuJoCo physics engine.
It provides realistic warehouse simulation with differential drive robot,
dynamic obstacles (workers and forklifts), and sensor simulation.
"""

import os
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    import mujoco
    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False
    mujoco = None

try:
    from cv_bridge import CvBridge
    CV_BRIDGE_AVAILABLE = True
except ImportError:
    CV_BRIDGE_AVAILABLE = False
    CvBridge = None

from geometry_msgs.msg import PoseStamped, Twist, Quaternion
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan, Image, CameraInfo
from std_msgs.msg import Header
from builtin_interfaces.msg import Time

from adaptnav.simulation.base_simulation import BaseSimulation
from adaptnav.core.robot_state import RobotState
from adaptnav.core.dynamic_obstacle import DynamicObstacle


class MuJoCoSimulation(BaseSimulation):
    """
    MuJoCo-based warehouse simulation.
    
    This class implements the BaseSimulation interface using MuJoCo physics engine.
    It simulates:
    - Warehouse environment with walls, shelves, and columns
    - Differential drive robot with LiDAR and depth camera
    - Dynamic obstacles (workers and forklifts) with scripted motion
    - Realistic physics including collisions and friction
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 render_mode: Optional[str] = None):
        """
        Initialize MuJoCo simulation.
        
        Args:
            model_path: Path to MJCF model file. If None, uses default warehouse model.
            render_mode: Rendering mode ('human', 'rgb_array', or None for headless)
        """
        if not MUJOCO_AVAILABLE:
            raise ImportError("MuJoCo is not installed. Install with: pip install mujoco")
        
        if not CV_BRIDGE_AVAILABLE:
            raise ImportError("cv_bridge is not installed. Install with: pip install cv_bridge")
        
        super().__init__('mujoco_simulation')
        
        # Initialize cv_bridge for image conversion
        self._bridge = CvBridge()
        
        # Load MuJoCo model
        if model_path is None:
            # Use default warehouse model
            package_dir = os.path.dirname(os.path.dirname(__file__))
            model_path = os.path.join(package_dir, 'simulation', 'models', 'warehouse.xml')
        
        self.get_logger().info(f'Loading MuJoCo model from: {model_path}')
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # Rendering setup
        self.render_mode = render_mode
        self.renderer = None
        if render_mode == 'human':
            try:
                import mujoco.viewer
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            except Exception as e:
                self.get_logger().warn(f'Could not initialize viewer: {e}')
                self.viewer = None
        else:
            self.viewer = None
        
        # Get body and sensor IDs
        self._robot_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'robot')
        self._worker_1_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'worker_1')
        self._worker_2_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'worker_2')
        self._forklift_1_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'forklift_1')
        
        # Get sensor IDs
        self._robot_pos_sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, 'robot_pos')
        self._robot_quat_sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, 'robot_quat')
        self._robot_linvel_sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, 'robot_linvel')
        self._robot_angvel_sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, 'robot_angvel')
        
        # Obstacle motion parameters (for scripted motion)
        self._obstacle_motion_time = 0.0
        self._worker_1_path_center = np.array([-8.0, -5.0])
        self._worker_1_path_radius = 3.0
        self._worker_1_speed = 0.5  # m/s
        
        self._worker_2_path_center = np.array([8.0, 5.0])
        self._worker_2_path_radius = 4.0
        self._worker_2_speed = 0.7  # m/s
        
        self._forklift_1_path_start = np.array([0.0, -15.0])
        self._forklift_1_path_end = np.array([0.0, 15.0])
        self._forklift_1_speed = 1.5  # m/s
        
        # LiDAR parameters
        self._lidar_range_max = 10.0  # meters
        self._lidar_range_min = 0.1  # meters
        self._lidar_num_rays = 360  # 1 degree resolution
        self._lidar_noise_std = 0.01  # meters
        
        # Depth camera parameters
        self._depth_camera_width = 640
        self._depth_camera_height = 480
        self._depth_camera_fov = np.pi / 2  # 90 degrees horizontal FOV
        self._depth_camera_range_min = 0.5  # meters
        self._depth_camera_range_max = 5.0  # meters
        self._depth_camera_noise_std = 0.02  # meters (increases with distance)
        self._depth_camera_publish_period = 1.0 / 15.0  # 15 Hz
        
        # ROS 2 publishers for sensor data
        self._scan_pub = self.create_publisher(LaserScan, '/scan', 10)
        self._odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self._depth_image_pub = self.create_publisher(Image, '/camera/depth/image_raw', 10)
        self._camera_info_pub = self.create_publisher(CameraInfo, '/camera/camera_info', 10)
        
        # Simulation timing
        self._sim_time = 0.0
        self._last_sensor_publish_time = 0.0
        self._last_depth_publish_time = 0.0
        self._sensor_publish_period = 0.05  # 20 Hz for LiDAR
        
        # Collision detection
        self._collision_detected = False
        
        self.set_initialized(True)
        self.get_logger().info('MuJoCo simulation initialized successfully')

    def step(self, dt: float) -> bool:
        """
        Advance the simulation by one time step.
        
        Args:
            dt: Time step in seconds
            
        Returns:
            True if step was successful, False if simulation error occurred
        """
        try:
            # Apply velocity command to robot
            cmd_vel = self.get_current_cmd_vel()
            if cmd_vel is not None:
                self._apply_velocity_command(cmd_vel)
            
            # Update scripted obstacle motion
            self._update_obstacle_motion(dt)
            
            # Step the physics simulation
            mujoco.mj_step(self.model, self.data)
            
            # Update simulation time
            self._sim_time += dt
            self._obstacle_motion_time += dt
            
            # Check for collisions
            self._collision_detected = self._check_collisions()
            
            # Publish sensor data at appropriate rates
            self._publish_sensor_data()
            
            # Publish ground truth data
            self.publish_ground_truth()
            
            # Update viewer if in human render mode
            if self.viewer is not None:
                self.viewer.sync()
            
            # Increment step counter
            self.increment_step_count()
            
            return True
            
        except Exception as e:
            self.get_logger().error(f'Simulation step failed: {e}')
            return False
    
    def reset(self, 
              robot_position: Optional[Tuple[float, float, float]] = None,
              obstacle_configs: Optional[List[Dict]] = None) -> bool:
        """
        Reset the simulation to initial state.
        
        Args:
            robot_position: Optional (x, y, theta) for robot spawn position
            obstacle_configs: Optional list of obstacle configurations
            
        Returns:
            True if reset was successful, False otherwise
        """
        try:
            # Reset MuJoCo simulation
            mujoco.mj_resetData(self.model, self.data)
            
            # Set robot position
            if robot_position is not None:
                x, y, theta = robot_position
                self._set_body_pose(self._robot_body_id, x, y, 0.15, theta)
            else:
                # Default spawn position
                self._set_body_pose(self._robot_body_id, -20.0, -20.0, 0.15, 0.0)
            
            # Reset obstacle positions
            if obstacle_configs is not None:
                # Custom obstacle configuration
                for config in obstacle_configs:
                    body_name = config.get('name', '')
                    pos = config.get('position', [0, 0])
                    vel = config.get('velocity', [0, 0])
                    
                    body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
                    if body_id >= 0:
                        self._set_body_pose(body_id, pos[0], pos[1], 0.85, 0.0)
                        self._set_body_velocity(body_id, vel[0], vel[1], 0.0)
            else:
                # Default obstacle positions
                self._set_body_pose(self._worker_1_body_id, -8.0, -5.0, 0.85, 0.0)
                self._set_body_pose(self._worker_2_body_id, 8.0, 5.0, 0.85, 0.0)
                self._set_body_pose(self._forklift_1_body_id, 0.0, -15.0, 0.5, 0.0)
            
            # Reset simulation state
            self._sim_time = 0.0
            self._obstacle_motion_time = 0.0
            self._last_sensor_publish_time = 0.0
            self._collision_detected = False
            self._step_count = 0
            
            # Forward simulation to stabilize
            for _ in range(10):
                mujoco.mj_step(self.model, self.data)
            
            self.get_logger().info('Simulation reset successfully')
            return True
            
        except Exception as e:
            self.get_logger().error(f'Simulation reset failed: {e}')
            return False
    
    def get_observation(self) -> Dict:
        """
        Get the current observation from the simulation.
        
        Returns:
            Dictionary containing observation data
        """
        obs = {
            'lidar_scan': self._get_lidar_scan(),
            'odometry': self._get_odometry(),
            'timestamp': self._sim_time
        }
        return obs
    
    def get_ground_truth_robot_state(self) -> RobotState:
        """
        Get the ground truth robot state.
        
        Returns:
            RobotState object with true position, orientation, and velocities
        """
        # Get robot position from sensor
        robot_pos = self.data.sensordata[self._robot_pos_sensor_id:self._robot_pos_sensor_id+3]
        
        # Get robot quaternion and convert to yaw angle
        robot_quat = self.data.sensordata[self._robot_quat_sensor_id:self._robot_quat_sensor_id+4]
        yaw = self._quat_to_yaw(robot_quat)
        
        # Get robot velocities
        robot_linvel = self.data.sensordata[self._robot_linvel_sensor_id:self._robot_linvel_sensor_id+3]
        robot_angvel = self.data.sensordata[self._robot_angvel_sensor_id:self._robot_angvel_sensor_id+3]
        
        # Create RobotState
        robot_state = RobotState(
            position=np.array([robot_pos[0], robot_pos[1]]),
            orientation=yaw,
            linear_velocity=float(np.linalg.norm(robot_linvel[:2])),
            angular_velocity=float(robot_angvel[2]),
            timestamp=self._sim_time
        )
        
        return robot_state
    
    def get_ground_truth_obstacles(self) -> List[DynamicObstacle]:
        """
        Get the ground truth positions and velocities of all dynamic obstacles.
        
        Returns:
            List of DynamicObstacle objects
        """
        obstacles = []
        
        # Worker 1
        worker_1_pos_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, 'worker_1_pos')
        worker_1_vel_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, 'worker_1_vel')
        worker_1_pos = self.data.sensordata[worker_1_pos_id:worker_1_pos_id+3]
        worker_1_vel = self.data.sensordata[worker_1_vel_id:worker_1_vel_id+3]
        
        obstacles.append(DynamicObstacle(
            id=1,
            position=np.array([worker_1_pos[0], worker_1_pos[1]]),
            velocity=np.array([worker_1_vel[0], worker_1_vel[1]]),
            radius=0.4,
            classification='worker'
        ))
        
        # Worker 2
        worker_2_pos_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, 'worker_2_pos')
        worker_2_vel_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, 'worker_2_vel')
        worker_2_pos = self.data.sensordata[worker_2_pos_id:worker_2_pos_id+3]
        worker_2_vel = self.data.sensordata[worker_2_vel_id:worker_2_vel_id+3]
        
        obstacles.append(DynamicObstacle(
            id=2,
            position=np.array([worker_2_pos[0], worker_2_pos[1]]),
            velocity=np.array([worker_2_vel[0], worker_2_vel[1]]),
            radius=0.4,
            classification='worker'
        ))
        
        # Forklift 1
        forklift_1_pos_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, 'forklift_1_pos')
        forklift_1_vel_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, 'forklift_1_vel')
        forklift_1_pos = self.data.sensordata[forklift_1_pos_id:forklift_1_pos_id+3]
        forklift_1_vel = self.data.sensordata[forklift_1_vel_id:forklift_1_vel_id+3]
        
        obstacles.append(DynamicObstacle(
            id=3,
            position=np.array([forklift_1_pos[0], forklift_1_pos[1]]),
            velocity=np.array([forklift_1_vel[0], forklift_1_vel[1]]),
            radius=1.2,
            classification='forklift'
        ))
        
        return obstacles
    
    def check_collision(self) -> bool:
        """
        Check if the robot is currently in collision.
        
        Returns:
            True if collision detected, False otherwise
        """
        return self._collision_detected

    # Helper methods
    
    def _apply_velocity_command(self, cmd_vel: Twist) -> None:
        """
        Apply velocity command to robot using differential drive model.
        
        Args:
            cmd_vel: Twist message with linear and angular velocity commands
        """
        # Get current robot state
        robot_pos = self.data.sensordata[self._robot_pos_sensor_id:self._robot_pos_sensor_id+3]
        robot_quat = self.data.sensordata[self._robot_quat_sensor_id:self._robot_quat_sensor_id+4]
        yaw = self._quat_to_yaw(robot_quat)
        
        # Convert velocity command to forces
        # Linear velocity in robot frame
        vx_robot = cmd_vel.linear.x
        vy_robot = cmd_vel.linear.y
        
        # Transform to world frame
        vx_world = vx_robot * np.cos(yaw) - vy_robot * np.sin(yaw)
        vy_world = vx_robot * np.sin(yaw) + vy_robot * np.cos(yaw)
        
        # Apply forces to achieve desired velocities (simple proportional control)
        current_vel = self.data.sensordata[self._robot_linvel_sensor_id:self._robot_linvel_sensor_id+3]
        
        kp_linear = 50.0  # Proportional gain for linear velocity
        kp_angular = 20.0  # Proportional gain for angular velocity
        
        force_x = kp_linear * (vx_world - current_vel[0])
        force_y = kp_linear * (vy_world - current_vel[1])
        
        current_angvel = self.data.sensordata[self._robot_angvel_sensor_id:self._robot_angvel_sensor_id+3]
        torque_z = kp_angular * (cmd_vel.angular.z - current_angvel[2])
        
        # Apply actuator controls
        try:
            robot_force_x_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'robot_force_x')
            robot_force_y_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'robot_force_y')
            robot_torque_z_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'robot_torque_z')
            
            self.data.ctrl[robot_force_x_id] = np.clip(force_x, -100, 100)
            self.data.ctrl[robot_force_y_id] = np.clip(force_y, -100, 100)
            self.data.ctrl[robot_torque_z_id] = np.clip(torque_z, -50, 50)
        except Exception as e:
            self.get_logger().warn(f'Failed to apply velocity command: {e}')
    
    def _update_obstacle_motion(self, dt: float) -> None:
        """
        Update scripted motion for dynamic obstacles.
        
        Args:
            dt: Time step in seconds
        """
        # Worker 1: Circular motion
        angle_1 = (self._obstacle_motion_time * self._worker_1_speed / self._worker_1_path_radius)
        target_pos_1 = self._worker_1_path_center + self._worker_1_path_radius * np.array([
            np.cos(angle_1), np.sin(angle_1)
        ])
        target_vel_1 = self._worker_1_speed * np.array([
            -np.sin(angle_1), np.cos(angle_1)
        ])
        
        self._set_body_pose(self._worker_1_body_id, target_pos_1[0], target_pos_1[1], 0.85, 0.0)
        self._set_body_velocity(self._worker_1_body_id, target_vel_1[0], target_vel_1[1], 0.0)
        
        # Worker 2: Circular motion (opposite direction)
        angle_2 = -(self._obstacle_motion_time * self._worker_2_speed / self._worker_2_path_radius)
        target_pos_2 = self._worker_2_path_center + self._worker_2_path_radius * np.array([
            np.cos(angle_2), np.sin(angle_2)
        ])
        target_vel_2 = self._worker_2_speed * np.array([
            np.sin(angle_2), -np.cos(angle_2)
        ])
        
        self._set_body_pose(self._worker_2_body_id, target_pos_2[0], target_pos_2[1], 0.85, 0.0)
        self._set_body_velocity(self._worker_2_body_id, target_vel_2[0], target_vel_2[1], 0.0)
        
        # Forklift: Back and forth motion
        path_length = np.linalg.norm(self._forklift_1_path_end - self._forklift_1_path_start)
        cycle_time = 2 * path_length / self._forklift_1_speed
        t_normalized = (self._obstacle_motion_time % cycle_time) / cycle_time
        
        if t_normalized < 0.5:
            # Moving forward
            progress = t_normalized * 2
            target_pos_fork = self._forklift_1_path_start + progress * (self._forklift_1_path_end - self._forklift_1_path_start)
            target_vel_fork = self._forklift_1_speed * (self._forklift_1_path_end - self._forklift_1_path_start) / path_length
        else:
            # Moving backward
            progress = (t_normalized - 0.5) * 2
            target_pos_fork = self._forklift_1_path_end + progress * (self._forklift_1_path_start - self._forklift_1_path_end)
            target_vel_fork = self._forklift_1_speed * (self._forklift_1_path_start - self._forklift_1_path_end) / path_length
        
        self._set_body_pose(self._forklift_1_body_id, target_pos_fork[0], target_pos_fork[1], 0.5, 0.0)
        self._set_body_velocity(self._forklift_1_body_id, target_vel_fork[0], target_vel_fork[1], 0.0)
    
    def _check_collisions(self) -> bool:
        """
        Check for collisions between robot and obstacles.
        
        Returns:
            True if collision detected, False otherwise
        """
        # Check MuJoCo contact data
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            
            # Get body IDs involved in contact
            geom1 = contact.geom1
            geom2 = contact.geom2
            
            body1 = self.model.geom_bodyid[geom1]
            body2 = self.model.geom_bodyid[geom2]
            
            # Check if robot is involved in contact with obstacles
            if (body1 == self._robot_body_id and body2 in [self._worker_1_body_id, self._worker_2_body_id, self._forklift_1_body_id]) or \
               (body2 == self._robot_body_id and body1 in [self._worker_1_body_id, self._worker_2_body_id, self._forklift_1_body_id]):
                return True
        
        return False
    
    def _set_body_pose(self, body_id: int, x: float, y: float, z: float, yaw: float) -> None:
        """
        Set the pose of a body in the simulation.
        
        Args:
            body_id: MuJoCo body ID
            x, y, z: Position coordinates
            yaw: Orientation angle in radians
        """
        # Set position
        self.data.qpos[body_id*7:body_id*7+3] = [x, y, z]
        
        # Set orientation (quaternion from yaw)
        quat = self._yaw_to_quat(yaw)
        self.data.qpos[body_id*7+3:body_id*7+7] = quat
    
    def _set_body_velocity(self, body_id: int, vx: float, vy: float, vz: float) -> None:
        """
        Set the velocity of a body in the simulation.
        
        Args:
            body_id: MuJoCo body ID
            vx, vy, vz: Velocity components
        """
        self.data.qvel[body_id*6:body_id*6+3] = [vx, vy, vz]
    
    def _quat_to_yaw(self, quat: np.ndarray) -> float:
        """
        Convert quaternion to yaw angle.
        
        Args:
            quat: Quaternion [w, x, y, z]
            
        Returns:
            Yaw angle in radians
        """
        w, x, y, z = quat
        yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return yaw
    
    def _yaw_to_quat(self, yaw: float) -> np.ndarray:
        """
        Convert yaw angle to quaternion.
        
        Args:
            yaw: Yaw angle in radians
            
        Returns:
            Quaternion [w, x, y, z]
        """
        return np.array([
            np.cos(yaw / 2.0),
            0.0,
            0.0,
            np.sin(yaw / 2.0)
        ])
    
    def _get_lidar_scan(self) -> LaserScan:
            """
            Simulate LiDAR scan using ray casting with proper occlusion.

            Implements:
            - 360° scan with 1° resolution (360 rays)
            - Realistic Gaussian noise on range measurements
            - Range limitations (0.1-10m)
            - Proper occlusion effects using MuJoCo ray casting
            - Detects both static obstacles (walls, shelves) and dynamic obstacles

            Returns:
                LaserScan message with ranges and metadata
            """
            scan = LaserScan()
            scan.header.stamp = self.get_clock().now().to_msg()
            scan.header.frame_id = 'laser'

            scan.angle_min = 0.0
            scan.angle_max = 2 * np.pi
            scan.angle_increment = 2 * np.pi / self._lidar_num_rays
            scan.time_increment = 0.0
            scan.scan_time = self._sensor_publish_period
            scan.range_min = self._lidar_range_min
            scan.range_max = self._lidar_range_max

            # Get robot pose
            robot_pos = self.data.sensordata[self._robot_pos_sensor_id:self._robot_pos_sensor_id+3]
            robot_quat = self.data.sensordata[self._robot_quat_sensor_id:self._robot_quat_sensor_id+4]
            robot_yaw = self._quat_to_yaw(robot_quat)

            # LiDAR is mounted at a fixed height on the robot
            lidar_height = 0.35  # 0.15 (robot base) + 0.2 (lidar offset)
            lidar_pos = robot_pos.copy()
            lidar_pos[2] = lidar_height

            # Perform ray casting for each angle
            ranges = []
            for i in range(self._lidar_num_rays):
                angle = i * scan.angle_increment + robot_yaw

                # Ray direction (horizontal plane)
                ray_dir = np.array([np.cos(angle), np.sin(angle), 0.0])

                # Ray end point at maximum range
                ray_end = lidar_pos + ray_dir * self._lidar_range_max

                # Use MuJoCo's ray casting to find first intersection
                # mj_ray returns distance to first geom hit, or -1 if no hit
                geom_id = np.array([-1], dtype=np.int32)
                distance = mujoco.mj_ray(
                    self.model,
                    self.data,
                    lidar_pos,
                    ray_dir,
                    None,  # geomgroup (None = all geoms)
                    1,     # flg_static (1 = include static geoms)
                    self._robot_body_id,  # bodyexclude (exclude robot itself)
                    geom_id
                )

                # Process the ray cast result
                if distance >= 0 and distance <= self._lidar_range_max:
                    # Hit detected within range
                    measured_range = max(distance, self._lidar_range_min)
                else:
                    # No hit or hit beyond max range
                    measured_range = self._lidar_range_max

                # Add realistic Gaussian noise to the measurement
                noisy_range = measured_range + np.random.normal(0, self._lidar_noise_std)

                # Clamp to valid range
                noisy_range = np.clip(noisy_range, self._lidar_range_min, self._lidar_range_max)

                ranges.append(float(noisy_range))

            scan.ranges = ranges
            scan.intensities = []  # Not simulating intensity values

            return scan
    def _get_depth_image(self) -> Tuple[Image, CameraInfo]:
        """
        Simulate depth camera using ray casting with 90° FOV and 640x480 resolution.

        Implements:
        - 90° horizontal field of view
        - 640x480 resolution depth image
        - Range limitations (0.5-5m)
        - Realistic noise model (depth accuracy degradation with distance)
        - Proper occlusion effects using MuJoCo ray casting

        Returns:
            Tuple of (depth Image message, CameraInfo message)
        """
        # Get robot pose
        robot_pos = self.data.sensordata[self._robot_pos_sensor_id:self._robot_pos_sensor_id+3]
        robot_quat = self.data.sensordata[self._robot_quat_sensor_id:self._robot_quat_sensor_id+4]
        robot_yaw = self._quat_to_yaw(robot_quat)

        # Camera is mounted at a fixed height and offset on the robot
        camera_height = 0.4  # 0.15 (robot base) + 0.25 (camera offset)
        camera_forward_offset = 0.1  # 10cm forward from robot center
        camera_pos = robot_pos.copy()
        camera_pos[2] = camera_height
        camera_pos[0] += camera_forward_offset * np.cos(robot_yaw)
        camera_pos[1] += camera_forward_offset * np.sin(robot_yaw)

        # Create depth image array
        depth_array = np.zeros((self._depth_camera_height, self._depth_camera_width), dtype=np.float32)

        # Camera intrinsics for 90° FOV
        fx = self._depth_camera_width / (2.0 * np.tan(self._depth_camera_fov / 2.0))
        fy = fx  # Assume square pixels
        cx = self._depth_camera_width / 2.0
        cy = self._depth_camera_height / 2.0

        # Generate rays for each pixel
        for v in range(self._depth_camera_height):
            for u in range(self._depth_camera_width):
                # Convert pixel coordinates to normalized camera coordinates
                x_norm = (u - cx) / fx
                y_norm = (v - cy) / fy

                # Ray direction in camera frame (z forward, x right, y down)
                ray_dir_camera = np.array([x_norm, -y_norm, 1.0])  # -y because image y is down
                ray_dir_camera = ray_dir_camera / np.linalg.norm(ray_dir_camera)

                # Transform ray direction to world frame
                # Camera frame: x=right, y=down, z=forward
                # World frame: x=forward, y=left, z=up
                cos_yaw = np.cos(robot_yaw)
                sin_yaw = np.sin(robot_yaw)

                # Rotation matrix from camera frame to world frame
                # Camera z (forward) -> World x*cos + y*sin (robot forward)
                # Camera x (right) -> World x*sin - y*cos (robot right)
                # Camera y (down) -> World -z (down)
                ray_dir_world = np.array([
                    ray_dir_camera[2] * cos_yaw + ray_dir_camera[0] * sin_yaw,
                    ray_dir_camera[2] * sin_yaw - ray_dir_camera[0] * cos_yaw,
                    -ray_dir_camera[1]  # Camera y (down) -> World -z
                ])

                # Perform ray casting
                geom_id = np.array([-1], dtype=np.int32)
                distance = mujoco.mj_ray(
                    self.model,
                    self.data,
                    camera_pos,
                    ray_dir_world,
                    None,  # geomgroup (None = all geoms)
                    1,     # flg_static (1 = include static geoms)
                    self._robot_body_id,  # bodyexclude (exclude robot itself)
                    geom_id
                )

                # Process the ray cast result
                if distance >= 0 and distance <= self._depth_camera_range_max:
                    # Hit detected within range
                    measured_depth = max(distance, self._depth_camera_range_min)
                else:
                    # No hit or hit beyond max range - set to max range
                    measured_depth = self._depth_camera_range_max

                # Add realistic noise that increases with distance
                # Noise model: std = base_noise + distance_factor * distance
                noise_std = self._depth_camera_noise_std * (1.0 + 0.5 * measured_depth / self._depth_camera_range_max)
                noisy_depth = measured_depth + np.random.normal(0, noise_std)

                # Clamp to valid range
                noisy_depth = np.clip(noisy_depth, self._depth_camera_range_min, self._depth_camera_range_max)

                depth_array[v, u] = noisy_depth

        # Create ROS Image message
        depth_image = Image()
        depth_image.header.stamp = self.get_clock().now().to_msg()
        depth_image.header.frame_id = 'camera_link'
        depth_image.height = self._depth_camera_height
        depth_image.width = self._depth_camera_width
        depth_image.encoding = '32FC1'  # 32-bit float, single channel
        depth_image.is_bigendian = False
        depth_image.step = self._depth_camera_width * 4  # 4 bytes per float32

        # Convert numpy array to bytes
        depth_image.data = depth_array.astype(np.float32).tobytes()

        # Create CameraInfo message
        camera_info = CameraInfo()
        camera_info.header = depth_image.header
        camera_info.height = self._depth_camera_height
        camera_info.width = self._depth_camera_width
        camera_info.distortion_model = 'plumb_bob'

        # Camera matrix K = [fx  0 cx]
        #                   [ 0 fy cy]
        #                   [ 0  0  1]
        camera_info.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]

        # Distortion coefficients (assuming no distortion)
        camera_info.d = [0.0, 0.0, 0.0, 0.0, 0.0]

        # Rectification matrix (identity for monocular camera)
        camera_info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]

        # Projection matrix P = [fx  0 cx  0]
        #                       [ 0 fy cy  0]
        #                       [ 0  0  1  0]
        camera_info.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]

        return depth_image, camera_info

    
    def _get_odometry(self) -> Odometry:
        """
        Get robot odometry with noise.
        
        Returns:
            Odometry message
        """
        odom = Odometry()
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_link'
        
        # Get true robot state
        robot_state = self.get_ground_truth_robot_state()
        
        # Add noise to simulate odometry drift
        pos_noise = np.random.normal(0, 0.01, 2)  # 1cm standard deviation
        angle_noise = np.random.normal(0, 0.01)  # ~0.5 degree standard deviation
        
        noisy_pos = robot_state.position + pos_noise
        noisy_yaw = robot_state.orientation + angle_noise
        
        # Set pose
        odom.pose.pose.position.x = float(noisy_pos[0])
        odom.pose.pose.position.y = float(noisy_pos[1])
        odom.pose.pose.position.z = 0.0
        
        quat = self._yaw_to_quat(noisy_yaw)
        odom.pose.pose.orientation.w = float(quat[0])
        odom.pose.pose.orientation.x = float(quat[1])
        odom.pose.pose.orientation.y = float(quat[2])
        odom.pose.pose.orientation.z = float(quat[3])
        
        # Set velocity
        odom.twist.twist.linear.x = float(robot_state.linear_velocity)
        odom.twist.twist.angular.z = float(robot_state.angular_velocity)
        
        return odom
    
    def _publish_sensor_data(self) -> None:
        """Publish sensor data to ROS 2 topics."""
        current_time = self._sim_time
        
        # Publish LiDAR scan at 20 Hz
        if current_time - self._last_sensor_publish_time >= self._sensor_publish_period:
            scan = self._get_lidar_scan()
            self._scan_pub.publish(scan)
            
            # Publish odometry
            odom = self._get_odometry()
            self._odom_pub.publish(odom)
            
            self._last_sensor_publish_time = current_time
        
        # Publish depth camera at 15 Hz
        if current_time - self._last_depth_publish_time >= self._depth_camera_publish_period:
            depth_image, camera_info = self._get_depth_image()
            self._depth_image_pub.publish(depth_image)
            self._camera_info_pub.publish(camera_info)
            
            self._last_depth_publish_time = current_time
    
    def __del__(self):
        """Cleanup on destruction."""
        if hasattr(self, 'viewer') and self.viewer is not None:
            try:
                self.viewer.close()
            except:
                pass
