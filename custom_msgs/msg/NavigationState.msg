# NavigationState.msg
# Current state of the navigation controller for dashboard visualization

# Header with timestamp
std_msgs/Header header

# Current navigation state
# Valid values: "IDLE", "PLANNING", "FOLLOWING_PATH", "AVOIDING_OBSTACLE", 
#               "EMERGENCY_STOP", "GOAL_REACHED"
string state

# Current robot pose in the map frame
geometry_msgs/Pose current_pose

# Goal pose in the map frame
geometry_msgs/Pose goal_pose

# Euclidean distance to goal (meters)
float32 distance_to_goal

# Progress toward goal as percentage [0.0, 100.0]
float32 progress_percentage

# Human-readable reasoning for current decision
# Example: "Following path, no obstacles detected"
# Example: "Avoiding worker at 2.3m ahead"
# Example: "Emergency stop - obstacle in collision zone"
string reasoning
