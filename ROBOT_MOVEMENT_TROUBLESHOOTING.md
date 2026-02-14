# Robot Movement Troubleshooting Guide

If your robot isn't moving in the AdaptNav demo, follow these steps to diagnose and fix the issue:

## Quick Diagnosis

1. **Run the diagnostic script first:**
   ```bash
   python debug_robot_movement.py
   ```
   This will check all major components and identify the most likely issue.

## Common Issues and Solutions

### 1. Path Planning Failure
**Symptoms:** Robot doesn't move at all, console shows "Could not plan initial path"

**Causes:**
- Start or goal position is inside an obstacle
- No valid path exists due to obstacle placement
- Warehouse map has issues

**Solutions:**
- Check that start position (2.0, 2.0) and goal position (17.0, 17.0) are valid
- Reduce obstacle sizes or move them
- Try different start/goal positions

### 2. Robot Stuck at Start
**Symptoms:** Robot appears but doesn't move, velocities are 0.0

**Causes:**
- Navigation controller not generating commands
- Safety controller blocking all movement
- Robot thinks it's already at goal

**Solutions:**
- Check console output for navigation status
- Verify goal distance is > 0.5m
- Check if obstacles are too close to robot

### 3. Animation Not Running
**Symptoms:** Window opens but nothing moves or updates

**Causes:**
- Matplotlib animation not started
- Simulation loop not running
- Window focus issues

**Solutions:**
- Click on the demo window to give it focus
- Check for error messages in console
- Try closing and restarting the demo

### 4. Safety Controller Blocking Movement
**Symptoms:** Robot detects obstacles everywhere and stops

**Causes:**
- Sensor simulation creating false obstacles
- Safety zones too large
- Obstacle detection too sensitive

**Solutions:**
- Check obstacle detection parameters
- Reduce safety zone radius
- Verify sensor simulation is working correctly

## Step-by-Step Debugging

### Step 1: Check Basic Setup
```python
# In the demo, verify these values:
warehouse_size = (20, 20)  # Should be reasonable
robot_start = (2.0, 2.0)   # Should be in free space
goal_position = (17.0, 17.0)  # Should be reachable
```

### Step 2: Verify Path Planning
Look for these messages in console:
- ✅ "Initial path planned with X waypoints"
- ❌ "Warning: Could not plan initial path"

### Step 3: Check Movement Commands
Look for debug output every 5 seconds:
```
Step 50: Robot at (2.00, 2.00), Goal dist: 21.21m, Vel: 0.500 m/s, 0.200 rad/s
```

If velocities are always 0.0, the navigation controller has an issue.

### Step 4: Monitor Robot Position
The robot position should change over time. If it stays at (2.00, 2.00), check:
- Are velocity commands being generated?
- Is the safety controller blocking movement?
- Is the simulation step running?

## Manual Fixes

### Fix 1: Simplify Obstacles
If path planning fails, try removing obstacles:
```python
# In setup_warehouse(), comment out obstacle lines:
# self.warehouse_map.set_obstacle(6, 6, 1.5, 3)
# self.warehouse_map.set_obstacle(12, 4, 3, 1)
# self.warehouse_map.set_obstacle(16, 12, 1.5, 3)
```

### Fix 2: Adjust Goal Position
Try a closer, easier goal:
```python
# In setup_robot(), change:
self.goal_position = (10.0, 10.0)  # Instead of (17.0, 17.0)
```

### Fix 3: Increase Movement Speed
If robot moves too slowly:
```python
# In simple_navigation_control(), increase velocities:
linear_velocity = min(1.5, max(0.3, target_distance * 0.6))  # Faster
```

### Fix 4: Disable Safety Controller
For testing, temporarily disable safety:
```python
# In apply_safety_control(), return original commands:
return linear_vel, angular_vel, "DISABLED"
```

## Expected Behavior

When working correctly, you should see:
1. Demo window opens with warehouse layout
2. Blue robot circle at bottom-left (2, 2)
3. Green goal circle at top-right (17, 17)
4. Green dashed line showing planned path
5. Robot moves along the path, avoiding orange/red obstacles
6. Console shows periodic position updates
7. Robot reaches goal and stops

## Getting Help

If none of these solutions work:

1. **Check Python version:** Requires Python 3.7+
2. **Check dependencies:** Run `pip install numpy matplotlib`
3. **Check file permissions:** Ensure all files are readable
4. **Check console output:** Look for error messages or stack traces
5. **Try the diagnostic script:** `python debug_robot_movement.py`

## Advanced Debugging

For developers, you can add more debug output:

```python
# In simulation_step(), add:
print(f"Navigation command: linear={linear_vel:.3f}, angular={angular_vel:.3f}")
print(f"Safety filtered: linear={safe_linear_vel:.3f}, angular={safe_angular_vel:.3f}")
print(f"Robot position: ({robot_pos[0]:.3f}, {robot_pos[1]:.3f})")
```

This will show exactly what's happening at each simulation step.