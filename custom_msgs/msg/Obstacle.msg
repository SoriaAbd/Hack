# Obstacle.msg
# Represents a detected dynamic obstacle in the warehouse environment

# Unique tracking ID for this obstacle
uint32 id

# Position of the obstacle center in the map frame
geometry_msgs/Point position

# Velocity of the obstacle in the map frame
geometry_msgs/Vector3 velocity

# Position and velocity covariance matrix (4x4 flattened to 16 elements)
# Order: [x, y, vx, vy] covariance
float32[16] covariance

# Classification of the obstacle type
# Valid values: "worker", "forklift", "unknown"
string classification

# Detection confidence score [0.0, 1.0]
float32 confidence

# Timestamp when this obstacle was last detected
builtin_interfaces/Time last_seen
