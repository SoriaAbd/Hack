#!/usr/bin/env python3
"""
Demonstration of the WarehouseNavigationEnv Gymnasium environment.

This script shows how to use the environment for training and testing
PPO agents. It includes examples of:
- Environment initialization
- Random action sampling
- Episode execution
- Basic training loop structure

Note: This demo uses mock simulation when ROS 2 is not available.
"""

import numpy as np
import random
from typing import Dict, Any

# Try to import the environment
try:
    from adaptnav.rl.warehouse_env import WarehouseNavigationEnv
    ENV_AVAILABLE = True
except ImportError as e:
    print(f"Environment not available: {e}")
    ENV_AVAILABLE = False


def demo_random_agent():
    """Demonstrate the environment with a random agent."""
    if not ENV_AVAILABLE:
        print("Environment not available for demo")
        return
    
    print("=== WarehouseNavigationEnv Random Agent Demo ===")
    
    try:
        # Create environment
        env = WarehouseNavigationEnv(
            max_episode_steps=100,  # Short episodes for demo
            goal_tolerance=1.0,     # Larger tolerance for easier success
            render_mode="human"
        )
        
        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")
        
        # Run multiple episodes
        num_episodes = 3
        
        for episode in range(num_episodes):
            print(f"\n--- Episode {episode + 1} ---")
            
            # Reset environment
            observation, info = env.reset(seed=42 + episode)
            print(f"Initial distance to goal: {info['initial_distance']:.2f}m")
            print(f"Number of obstacles: {info['obstacle_count']}")
            
            episode_reward = 0.0
            step = 0
            
            while True:
                # Sample random action
                action = env.action_space.sample()
                
                # Take step
                observation, reward, terminated, truncated, info = env.step(action)
                
                episode_reward += reward
                step += 1
                
                # Print step info
                if step % 20 == 0 or terminated or truncated:
                    print(f"Step {step}: reward={reward:.3f}, "
                          f"distance={info['distance_to_goal']:.2f}m, "
                          f"collision={info['collision']}, "
                          f"goal_reached={info['goal_reached']}")
                
                # Check if episode is done
                if terminated or truncated:
                    break
            
            # Episode summary
            print(f"Episode {episode + 1} finished:")
            print(f"  Total steps: {step}")
            print(f"  Total reward: {episode_reward:.2f}")
            print(f"  Goal reached: {info['goal_reached']}")
            print(f"  Collision occurred: {info['collision']}")
            print(f"  Final distance: {info['distance_to_goal']:.2f}m")
            
            if terminated:
                if info['goal_reached']:
                    print("  Result: SUCCESS!")
                elif info['collision']:
                    print("  Result: COLLISION")
            else:
                print("  Result: TIMEOUT")
        
        env.close()
        
    except Exception as e:
        print(f"Demo failed: {e}")
        print("This is expected when ROS 2 is not available.")


def demo_simple_policy():
    """Demonstrate the environment with a simple policy."""
    if not ENV_AVAILABLE:
        print("Environment not available for demo")
        return
    
    print("\n=== WarehouseNavigationEnv Simple Policy Demo ===")
    
    try:
        # Create environment
        env = WarehouseNavigationEnv(
            max_episode_steps=200,
            goal_tolerance=0.5,
        )
        
        # Simple policy: move toward goal with obstacle avoidance
        def simple_policy(observation: np.ndarray) -> np.ndarray:
            """
            Simple policy that moves toward goal while avoiding obstacles.
            
            Args:
                observation: 372-dim observation vector
                
            Returns:
                2-dim action vector [linear_vel, angular_vel]
            """
            # Extract components from observation
            lidar_scan = observation[:360]
            goal_direction = observation[360:362]
            current_velocity = observation[362:364]
            obstacle_proximity = observation[364:372]
            
            # Check for nearby obstacles
            min_obstacle_distance = np.min(obstacle_proximity)
            
            if min_obstacle_distance < 0.3:  # Very close obstacle
                # Emergency stop
                return np.array([0.0, 0.0])
            elif min_obstacle_distance < 0.5:  # Close obstacle
                # Slow down and turn away
                obstacle_sector = np.argmin(obstacle_proximity)
                turn_direction = 1.0 if obstacle_sector < 4 else -1.0
                return np.array([0.2, 0.3 * turn_direction])
            else:
                # Move toward goal
                goal_distance = np.linalg.norm(goal_direction)
                if goal_distance > 0.1:
                    # Calculate desired heading
                    desired_angle = np.arctan2(goal_direction[1], goal_direction[0])
                    
                    # Simple proportional control
                    linear_vel = min(0.8, goal_distance)
                    angular_vel = np.clip(desired_angle * 0.5, -0.4, 0.4)
                    
                    return np.array([linear_vel, angular_vel])
                else:
                    # Close to goal, stop
                    return np.array([0.0, 0.0])
        
        # Run episode with simple policy
        print("Running episode with simple policy...")
        
        observation, info = env.reset(seed=123)
        print(f"Initial distance to goal: {info['initial_distance']:.2f}m")
        
        episode_reward = 0.0
        step = 0
        
        while True:
            # Use simple policy
            action = simple_policy(observation)
            
            # Take step
            observation, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            step += 1
            
            # Print progress
            if step % 25 == 0 or terminated or truncated:
                print(f"Step {step}: action=[{action[0]:.2f}, {action[1]:.2f}], "
                      f"reward={reward:.3f}, distance={info['distance_to_goal']:.2f}m")
            
            if terminated or truncated:
                break
        
        # Episode summary
        print(f"\nSimple policy episode finished:")
        print(f"  Total steps: {step}")
        print(f"  Total reward: {episode_reward:.2f}")
        print(f"  Goal reached: {info['goal_reached']}")
        print(f"  Final distance: {info['distance_to_goal']:.2f}m")
        
        env.close()
        
    except Exception as e:
        print(f"Simple policy demo failed: {e}")
        print("This is expected when ROS 2 is not available.")


def demo_training_setup():
    """Demonstrate how to set up the environment for training."""
    print("\n=== Training Setup Demo ===")
    
    print("Example training setup with Stable Baselines3:")
    print("""
# Install dependencies
pip install stable-baselines3[extra]

# Training script example:
from stable_baselines3 import PPO
from adaptnav.rl.warehouse_env import WarehouseNavigationEnv

# Create environment
env = WarehouseNavigationEnv(
    max_episode_steps=1000,
    goal_tolerance=0.5,
    collision_penalty=10.0,
    goal_bonus=100.0
)

# Create PPO agent
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    verbose=1
)

# Train the agent
model.learn(total_timesteps=1_000_000)

# Save the model
model.save("warehouse_navigation_ppo")

# Test the trained agent
obs, info = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
""")


def main():
    """Run all demos."""
    print("AdaptNav Warehouse Navigation Environment Demo")
    print("=" * 50)
    
    # Run demos
    demo_random_agent()
    demo_simple_policy()
    demo_training_setup()
    
    print("\n" + "=" * 50)
    print("Demo completed!")
    
    if not ENV_AVAILABLE:
        print("\nNote: Full functionality requires ROS 2 and MuJoCo simulation.")
        print("Install ROS 2 Humble and run in a proper ROS environment for full demo.")


if __name__ == "__main__":
    main()