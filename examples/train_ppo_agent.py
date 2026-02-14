#!/usr/bin/env python3
"""
Training script for PPO agent in WarehouseNavigationEnv.

This script demonstrates how to train a PPO agent using Stable Baselines3
for warehouse navigation. It includes:
- Environment setup with custom parameters
- PPO agent configuration
- Training loop with logging
- Model saving and evaluation

Requirements:
- ROS 2 Humble
- MuJoCo simulation
- stable-baselines3[extra]
"""

import os
import numpy as np
from typing import Dict, Any
import argparse

# Try to import required packages
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
    from stable_baselines3.common.monitor import Monitor
    SB3_AVAILABLE = True
except ImportError:
    print("Stable Baselines3 not available. Install with: pip install stable-baselines3[extra]")
    SB3_AVAILABLE = False

try:
    from adaptnav.rl.warehouse_env import WarehouseNavigationEnv
    ENV_AVAILABLE = True
except ImportError as e:
    print(f"Environment not available: {e}")
    ENV_AVAILABLE = False


def create_training_env(env_kwargs: Dict[str, Any] = None) -> WarehouseNavigationEnv:
    """
    Create a training environment with appropriate parameters.
    
    Args:
        env_kwargs: Additional environment parameters
        
    Returns:
        Configured WarehouseNavigationEnv
    """
    default_kwargs = {
        'max_episode_steps': 1000,
        'goal_tolerance': 0.5,
        'collision_penalty': 10.0,
        'goal_bonus': 100.0,
        'progress_weight': 1.0,
        'distance_weight': 0.1,
        'action_weight': 0.01,
    }
    
    if env_kwargs:
        default_kwargs.update(env_kwargs)
    
    return WarehouseNavigationEnv(**default_kwargs)


def create_evaluation_env(env_kwargs: Dict[str, Any] = None) -> WarehouseNavigationEnv:
    """
    Create an evaluation environment (typically with different parameters).
    
    Args:
        env_kwargs: Additional environment parameters
        
    Returns:
        Configured WarehouseNavigationEnv for evaluation
    """
    default_kwargs = {
        'max_episode_steps': 1000,
        'goal_tolerance': 0.5,
        'collision_penalty': 10.0,
        'goal_bonus': 100.0,
        'progress_weight': 1.0,
        'distance_weight': 0.1,
        'action_weight': 0.01,
    }
    
    if env_kwargs:
        default_kwargs.update(env_kwargs)
    
    return WarehouseNavigationEnv(**default_kwargs)


def train_ppo_agent(
    total_timesteps: int = 1_000_000,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    save_path: str = "models/warehouse_navigation_ppo",
    log_dir: str = "logs/",
    eval_freq: int = 10000,
    n_eval_episodes: int = 10,
    target_reward: float = 80.0
):
    """
    Train a PPO agent for warehouse navigation.
    
    Args:
        total_timesteps: Total training timesteps
        learning_rate: Learning rate for PPO
        n_steps: Number of steps per rollout
        batch_size: Batch size for training
        n_epochs: Number of epochs per update
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        clip_range: PPO clip range
        save_path: Path to save the trained model
        log_dir: Directory for tensorboard logs
        eval_freq: Frequency of evaluation (in timesteps)
        n_eval_episodes: Number of episodes for evaluation
        target_reward: Target reward for early stopping
    """
    if not ENV_AVAILABLE or not SB3_AVAILABLE:
        print("Required packages not available for training")
        return
    
    print("=== PPO Training for Warehouse Navigation ===")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Learning rate: {learning_rate}")
    print(f"Batch size: {batch_size}")
    print(f"Target reward: {target_reward}")
    
    # Create directories
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    try:
        # Create training environment
        print("Creating training environment...")
        train_env = create_training_env()
        train_env = Monitor(train_env, log_dir + "train")
        
        # Create evaluation environment
        print("Creating evaluation environment...")
        eval_env = create_evaluation_env()
        eval_env = Monitor(eval_env, log_dir + "eval")
        
        # Create PPO agent
        print("Creating PPO agent...")
        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            verbose=1,
            tensorboard_log=log_dir
        )
        
        # Create callbacks
        stop_callback = StopTrainingOnRewardThreshold(
            reward_threshold=target_reward,
            verbose=1
        )
        
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=save_path + "_best",
            log_path=log_dir + "eval",
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
            render=False,
            callback_on_new_best=stop_callback,
            verbose=1
        )
        
        # Train the agent
        print("Starting training...")
        model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback,
            tb_log_name="PPO_warehouse_navigation"
        )
        
        # Save final model
        print(f"Saving final model to {save_path}")
        model.save(save_path)
        
        print("Training completed!")
        
        # Clean up
        train_env.close()
        eval_env.close()
        
    except Exception as e:
        print(f"Training failed: {e}")
        print("Make sure ROS 2 and MuJoCo simulation are properly set up.")


def evaluate_agent(
    model_path: str,
    n_episodes: int = 10,
    render: bool = False,
    deterministic: bool = True
):
    """
    Evaluate a trained PPO agent.
    
    Args:
        model_path: Path to the trained model
        n_episodes: Number of episodes to evaluate
        render: Whether to render the environment
        deterministic: Whether to use deterministic actions
    """
    if not ENV_AVAILABLE or not SB3_AVAILABLE:
        print("Required packages not available for evaluation")
        return
    
    print(f"=== Evaluating Agent: {model_path} ===")
    
    try:
        # Load model
        model = PPO.load(model_path)
        
        # Create environment
        env = create_evaluation_env()
        if render:
            env.render_mode = "human"
        
        # Run evaluation episodes
        episode_rewards = []
        episode_lengths = []
        success_count = 0
        collision_count = 0
        
        for episode in range(n_episodes):
            obs, info = env.reset()
            episode_reward = 0.0
            episode_length = 0
            
            print(f"\nEpisode {episode + 1}:")
            print(f"  Initial distance: {info['initial_distance']:.2f}m")
            print(f"  Obstacles: {info['obstacle_count']}")
            
            while True:
                action, _states = model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                
                episode_reward += reward
                episode_length += 1
                
                if terminated or truncated:
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            # Episode results
            if info['goal_reached']:
                success_count += 1
                result = "SUCCESS"
            elif info['collision']:
                collision_count += 1
                result = "COLLISION"
            else:
                result = "TIMEOUT"
            
            print(f"  Result: {result}")
            print(f"  Steps: {episode_length}")
            print(f"  Reward: {episode_reward:.2f}")
            print(f"  Final distance: {info['distance_to_goal']:.2f}m")
        
        # Summary statistics
        print(f"\n=== Evaluation Summary ===")
        print(f"Episodes: {n_episodes}")
        print(f"Success rate: {success_count / n_episodes * 100:.1f}%")
        print(f"Collision rate: {collision_count / n_episodes * 100:.1f}%")
        print(f"Average reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
        print(f"Average episode length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
        
        env.close()
        
    except Exception as e:
        print(f"Evaluation failed: {e}")


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Train or evaluate PPO agent for warehouse navigation")
    parser.add_argument("--mode", choices=["train", "eval"], default="train",
                       help="Mode: train or evaluate")
    parser.add_argument("--timesteps", type=int, default=1_000_000,
                       help="Total training timesteps")
    parser.add_argument("--model-path", type=str, default="models/warehouse_navigation_ppo",
                       help="Path to save/load model")
    parser.add_argument("--eval-episodes", type=int, default=10,
                       help="Number of evaluation episodes")
    parser.add_argument("--render", action="store_true",
                       help="Render environment during evaluation")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        train_ppo_agent(
            total_timesteps=args.timesteps,
            save_path=args.model_path
        )
    elif args.mode == "eval":
        evaluate_agent(
            model_path=args.model_path,
            n_episodes=args.eval_episodes,
            render=args.render
        )


if __name__ == "__main__":
    main()