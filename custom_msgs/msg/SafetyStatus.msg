# SafetyStatus.msg
# Current safety state of the robot and safety controller

# Header with timestamp
std_msgs/Header header

# Current safety state
# Valid values: "SAFE", "CAUTION", "EMERGENCY_STOP"
string state

# Distance to the closest detected obstacle (meters)
float32 closest_obstacle_distance

# Velocity scale factor applied by safety controller [0.0, 1.0]
# 1.0 = no scaling, 0.0 = full stop
float32 velocity_scale

# Whether safety controller is actively overriding commanded velocity
bool override_active

# Estimated time until collision zone is clear (seconds)
# 0.0 if already clear, -1.0 if unknown
float32 time_until_clear
