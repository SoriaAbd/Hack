#!/usr/bin/env python3
"""
PPO Training Script for AdaptNav warehouse navigation.

This script trains a PPO agent using the WarehouseNavigationEnv environment.
It includes:
- Hyperparameter configuration
- Training loop with periodic evaluation
- Model checkpointing
- Metrics logging to TensorBoard
- Early stopping based on success rate
"""

import os
import argparse
import time
from typing import Dict, List, Optional, Tuple
import numpy as np

# Stable Baselines3 imports
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import (
        BaseCallback, EvalCallback, CheckpointCallback, CallbackList
    )
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("Warning: Stable Baselines3 not available. Training will not work.")

# AdaptNav imports
from adaptnav.rl.warehouse_env import WarehouseNavigationEnv


class TrainingMetricsCallback(BaseCallback):
    """
    Custom callback for logging training metrics.
    
    Logs additional metrics like success rate, collision rate,
    and path efficiency to TensorBoard.
    """
    
    def __init__(self, eval_freq: int = 10000, verbose: int = 0):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.success_count = 0
        self.collision_count = 0
        self.total_episodes = 0
        
    def _on_step(self) -> bool:
        """Called at each step."""
        # Check if episode ended
        for i, done in enumerate(self.locals.get('dones', [])):
            if done:
                # Get episode info
                info = self.locals.get('infos', [{}])[i]
                
                if 'episode' in info:
                    episode_reward = info['episode']['r']
                    episode_length = info['episode']['l']
                    
                    self.episode_rewards.append(episode_reward)
                    self.episode_lengths.append(episode_length)
                    self.total_episodes += 1
                    
                    # Track success and collision rates
                    if info.get('goal_reached', False):
                        self.success_count += 1
                    if info.get('collision', False):
                        self.collision_count += 1
        
        # Log metrics periodically
        if self.num_timesteps % self.eval_freq == 0 and self.total_episodes > 0:
            self.log_metrics()
        
        return True
    
    def log_metrics(self) -> None:
        """Log training metrics to TensorBoard."""
        if self.total_episodes == 0:
            return
        
        # Calculate metrics
        success_rate = self.success_count / self.total_episodes
        collision_rate = self.collision_count / self.total_episodes
        avg_reward = np.mean(self.episode_rewards[-100:])  # Last 100 episodes
        avg_length = np.mean(self.episode_lengths[-100:])
        
        # Log to TensorBoard
        self.logger.record('train/success_rate', success_rate)
        self.logger.record('train/collision_rate', collision_rate)
        self.logger.record('train/avg_episode_reward', avg_reward)
        self.logger.record('train/avg_episode_length', avg_length)
        self.logger.record('train/total_episodes', self.total_episodes)
        
        if self.verbose > 0:
            print(f"Step {self.num_timesteps}: "
                  f"Success rate: {success_rate:.3f}, "
                  f"Collision rate: {collision_rate:.3f}, "
                  f"Avg reward: {avg_reward:.1f}")


class EarlyStoppingCallback(BaseCallback):
    """
    Callback for early stopping based on success rate.
    
    Stops training when the success rate reaches a target threshold
    and remains stable for a specified number of evaluations.
    """
    
    def __init__(self, 
                 target_success_rate: float = 0.8,
                 min_evaluations: int = 5,
                 eval_freq: int = 50000,
                 verbose: int = 0):
        super().__init__(verbose)
        self.target_success_rate = target_success_rate
        self.min_evaluations = min_evaluations
        self.eval_freq = eval_freq
        self.success_rates: List[float] = []
        
    def _on_step(self) -> bool:
        """Called at each step."""
        # Early stopping is typically triggered by evaluation callback
        # This callback would receive success rate from evaluation
        return True
    
    def on_evaluation_end(self, success_rate: float) -> bool:
        """
        Called after evaluation to check early stopping condition.
        
        Args:
            success_rate: Success rate from evaluation
            
        Returns:
            bool: True to continue training, False to stop
        """
        self.success_rates.append(success_rate)
        
        if self.verbose > 0:
            print(f"Evaluation success rate: {success_rate:.3f}")
        
        # Check if we have enough evaluations
        if len(self.success_rates) < self.min_evaluations:
            return True
        
        # Check if recent success rates meet target
        recent_rates = self.success_rates[-self.min_evaluations:]
        if all(rate >= self.target_success_rate for rate in recent_rates):
            if self.verbose > 0:
                print(f"Early stopping triggered! Success rate {success_rate:.3f} "
                      f"maintained for {self.min_evaluations} evaluations.")
            return False
        
        return True


def create_training_config() -> Dict[str, any]:
    """
    Create default training configuration.
    
    Returns:
        Dict: Training configuration parameters
    """
    return {
        # Environment settings
        'env_id': 'WarehouseNavigation-v0',
        'n_envs': 8,  # Number of parallel environments
        'env_kwargs': {
            'warehouse_size': (20, 20),
            'num_obstacles': 5,
            'max_episode_steps': 1000,
        },
        
        # PPO hyperparameters
        'learning_rate': 3e-4,
        'n_steps': 2048,
        'batch_size': 64,
        'n_epochs': 10,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'ent_coef': 0.01,
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
        
        # Training settings
        'total_timesteps': 2_000_000,
        'eval_freq': 50_000,
        'eval_episodes': 20,
        'checkpoint_freq': 100_000,
        
        # Early stopping
        'target_success_rate': 0.8,
        'min_evaluations': 5,
        
        # Logging
        'log_dir': './logs/ppo_training',
        'model_save_path': './models/ppo_warehouse_nav',
        'tensorboard_log': './logs/tensorboard',
    }


def setup_environment(config: Dict[str, any]) -> DummyVecEnv:
    """
    Set up the training environment.
    
    Args:
        config: Training configuration
        
    Returns:
        DummyVecEnv: Vectorized environment for training
    """
    def make_env():
        env = WarehouseNavigationEnv(**config['env_kwargs'])
        env = Monitor(env)
        return env
    
    # Create vectorized environment
    if config['n_envs'] == 1:
        env = DummyVecEnv([make_env])
    else:
        env = make_vec_env(
            make_env,
            n_envs=config['n_envs'],
            vec_env_cls=DummyVecEnv
        )
    
    return env


def setup_callbacks(config: Dict[str, any], eval_env: DummyVecEnv) -> CallbackList:
    """
    Set up training callbacks.
    
    Args:
        config: Training configuration
        eval_env: Environment for evaluation
        
    Returns:
        CallbackList: List of callbacks for training
    """
    callbacks = []
    
    # Training metrics callback
    metrics_callback = TrainingMetricsCallback(
        eval_freq=config['eval_freq'],
        verbose=1
    )
    callbacks.append(metrics_callback)
    
    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=config['model_save_path'],
        log_path=config['log_dir'],
        eval_freq=config['eval_freq'],
        n_eval_episodes=config['eval_episodes'],
        deterministic=True,
        render=False,
        verbose=1
    )
    callbacks.append(eval_callback)
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=config['checkpoint_freq'],
        save_path=config['model_save_path'],
        name_prefix='ppo_checkpoint'
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping callback
    early_stopping_callback = EarlyStoppingCallback(
        target_success_rate=config['target_success_rate'],
        min_evaluations=config['min_evaluations'],
        eval_freq=config['eval_freq'],
        verbose=1
    )
    callbacks.append(early_stopping_callback)
    
    return CallbackList(callbacks)


def train_ppo_agent(config: Dict[str, any]) -> PPO:
    """
    Train PPO agent with given configuration.
    
    Args:
        config: Training configuration
        
    Returns:
        PPO: Trained PPO model
    """
    if not SB3_AVAILABLE:
        raise ImportError("Stable Baselines3 is required for training")
    
    print("Setting up training environment...")
    
    # Create directories
    os.makedirs(config['log_dir'], exist_ok=True)
    os.makedirs(config['model_save_path'], exist_ok=True)
    os.makedirs(config['tensorboard_log'], exist_ok=True)
    
    # Set up environments
    train_env = setup_environment(config)
    eval_env = setup_environment({**config, 'n_envs': 1})
    
    # Set up logging
    logger = configure(config['log_dir'], ["stdout", "csv", "tensorboard"])
    
    # Create PPO model
    print("Creating PPO model...")
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=config['learning_rate'],
        n_steps=config['n_steps'],
        batch_size=config['batch_size'],
        n_epochs=config['n_epochs'],
        gamma=config['gamma'],
        gae_lambda=config['gae_lambda'],
        clip_range=config['clip_range'],
        ent_coef=config['ent_coef'],
        vf_coef=config['vf_coef'],
        max_grad_norm=config['max_grad_norm'],
        tensorboard_log=config['tensorboard_log'],
        verbose=1
    )
    
    # Set logger
    model.set_logger(logger)
    
    # Set up callbacks
    callbacks = setup_callbacks(config, eval_env)
    
    # Train the model
    print(f"Starting training for {config['total_timesteps']} timesteps...")
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=config['total_timesteps'],
            callback=callbacks,
            progress_bar=True
        )
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.1f} seconds")
        
        # Save final model
        final_model_path = os.path.join(config['model_save_path'], 'final_model')
        model.save(final_model_path)
        print(f"Final model saved to: {final_model_path}")
        
        return model
        
    except KeyboardInterrupt:
        print("Training interrupted by user")
        # Save current model
        interrupt_model_path = os.path.join(config['model_save_path'], 'interrupted_model')
        model.save(interrupt_model_path)
        print(f"Model saved to: {interrupt_model_path}")
        return model
    
    finally:
        # Clean up environments
        train_env.close()
        eval_env.close()


def main():
    """Main training script entry point."""
    parser = argparse.ArgumentParser(description='Train PPO agent for warehouse navigation')
    
    parser.add_argument('--config', type=str, help='Path to config file (optional)')
    parser.add_argument('--total-timesteps', type=int, help='Total training timesteps')
    parser.add_argument('--learning-rate', type=float, help='Learning rate')
    parser.add_argument('--n-envs', type=int, help='Number of parallel environments')
    parser.add_argument('--model-save-path', type=str, help='Path to save trained model')
    parser.add_argument('--log-dir', type=str, help='Directory for logs')
    
    args = parser.parse_args()
    
    # Load configuration
    config = create_training_config()
    
    # Override config with command line arguments
    if args.total_timesteps:
        config['total_timesteps'] = args.total_timesteps
    if args.learning_rate:
        config['learning_rate'] = args.learning_rate
    if args.n_envs:
        config['n_envs'] = args.n_envs
    if args.model_save_path:
        config['model_save_path'] = args.model_save_path
    if args.log_dir:
        config['log_dir'] = args.log_dir
    
    # Print configuration
    print("Training Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    try:
        # Train the model
        model = train_ppo_agent(config)
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Training failed: {str(e)}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())