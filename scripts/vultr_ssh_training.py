#!/usr/bin/env python3
"""
VULTR SSH-Based Training Script for AdaptNav PPO Agent

This script uses SSH to connect to manually created VULTR instances
instead of using the API to create them automatically.
"""

import os
import sys
import yaml
import logging
import asyncio
import subprocess
from typing import Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

@dataclass
class VultrSSHInstance:
    """VULTR instance configuration for SSH access."""
    name: str
    ip_address: str
    ssh_key_path: str
    username: str = "root"
    role: str = "worker"  # 'master', 'worker', 'simulation'

class VultrSSHManager:
    """Manages VULTR instances via SSH instead of API."""
    
    def __init__(self, config_path: str = "config/vultr_ssh_config.yaml"):
        self.config = self._load_config(config_path)
        self.instances: List[VultrSSHInstance] = []
        self.logger = self._setup_logging()
        self._load_instances()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load VULTR SSH configuration."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            self.logger.warning(f"Config file {config_path} not found, using defaults")
            return self._default_config()
            
    def _default_config(self) -> Dict:
        """Default configuration when no config file exists."""
        return {
            'instances': [],
            'ssh_settings': {
                'timeout': 30,
                'retries': 3
            },
            'training': {
                'sync_code': True,
                'install_dependencies': True
            }
        }
            
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for SSH operations."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
        
    def _load_instances(self):
        """Load instance configurations from config."""
        for instance_config in self.config.get('instances', []):
            instance = VultrSSHInstance(
                name=instance_config['name'],
                ip_address=instance_config['ip_address'],
                ssh_key_path=instance_config['ssh_key_path'],
                username=instance_config.get('username', 'root'),
                role=instance_config.get('role', 'worker')
            )
            self.instances.append(instance)
            
    def test_ssh_connection(self, instance: VultrSSHInstance) -> bool:
        """Test SSH connection to an instance."""
        try:
            cmd = [
                'ssh', 
                '-i', instance.ssh_key_path,
                '-o', 'ConnectTimeout=10',
                '-o', 'StrictHostKeyChecking=no',
                f'{instance.username}@{instance.ip_address}',
                'echo "SSH connection successful"'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            
            if result.returncode == 0:
                self.logger.info(f"✓ SSH connection to {instance.name} successful")
                return True
            else:
                self.logger.error(f"✗ SSH connection to {instance.name} failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.error(f"✗ SSH connection to {instance.name} timed out")
            return False
        except Exception as e:
            self.logger.error(f"✗ SSH connection to {instance.name} error: {e}")
            return False
            
    def execute_remote_command(self, instance: VultrSSHInstance, command: str) -> bool:
        """Execute a command on a remote instance via SSH."""
        try:
            cmd = [
                'ssh', 
                '-i', instance.ssh_key_path,
                '-o', 'ConnectTimeout=10',
                '-o', 'StrictHostKeyChecking=no',
                f'{instance.username}@{instance.ip_address}',
                command
            ]
            
            self.logger.info(f"Executing on {instance.name}: {command}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                self.logger.info(f"✓ Command successful on {instance.name}")
                if result.stdout:
                    self.logger.info(f"Output: {result.stdout.strip()}")
                return True
            else:
                self.logger.error(f"✗ Command failed on {instance.name}: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.error(f"✗ Command timed out on {instance.name}")
            return False
        except Exception as e:
            self.logger.error(f"✗ Command error on {instance.name}: {e}")
            return False
            
    def sync_code_to_instance(self, instance: VultrSSHInstance) -> bool:
        """Sync local code to remote instance using rsync."""
        try:
            # Create remote directory
            self.execute_remote_command(instance, "mkdir -p ~/adaptnav")
            
            # Sync code using rsync
            cmd = [
                'rsync', '-avz', '--delete',
                '-e', f'ssh -i {instance.ssh_key_path} -o StrictHostKeyChecking=no',
                './',  # Current directory
                f'{instance.username}@{instance.ip_address}:~/adaptnav/'
            ]
            
            self.logger.info(f"Syncing code to {instance.name}...")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                self.logger.info(f"✓ Code sync to {instance.name} successful")
                return True
            else:
                self.logger.error(f"✗ Code sync to {instance.name} failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.error(f"✗ Code sync to {instance.name} timed out")
            return False
        except Exception as e:
            self.logger.error(f"✗ Code sync to {instance.name} error: {e}")
            return False
            
    def setup_instance(self, instance: VultrSSHInstance) -> bool:
        """Setup an instance for training."""
        self.logger.info(f"Setting up instance {instance.name}...")
        
        # Update system
        if not self.execute_remote_command(instance, "apt update && apt upgrade -y"):
            return False
            
        # Install Python and dependencies
        if not self.execute_remote_command(instance, "apt install -y python3 python3-pip git"):
            return False
            
        # Install Python packages
        if not self.execute_remote_command(instance, "cd ~/adaptnav && pip3 install -r requirements.txt"):
            return False
            
        self.logger.info(f"✓ Instance {instance.name} setup complete")
        return True
        
    def start_training(self, instance: VultrSSHInstance) -> bool:
        """Start training on an instance."""
        self.logger.info(f"Starting training on {instance.name}...")
        
        # Start training in background
        command = "cd ~/adaptnav && nohup python3 adaptnav/rl/train_ppo.py > training.log 2>&1 &"
        
        if self.execute_remote_command(instance, command):
            self.logger.info(f"✓ Training started on {instance.name}")
            return True
        else:
            self.logger.error(f"✗ Failed to start training on {instance.name}")
            return False
            
    def check_training_status(self, instance: VultrSSHInstance) -> str:
        """Check training status on an instance."""
        # Check if training process is running
        cmd = [
            'ssh', 
            '-i', instance.ssh_key_path,
            '-o', 'ConnectTimeout=10',
            '-o', 'StrictHostKeyChecking=no',
            f'{instance.username}@{instance.ip_address}',
            'pgrep -f "train_ppo.py" && echo "RUNNING" || echo "STOPPED"'
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            return result.stdout.strip()
        except:
            return "UNKNOWN"
            
    def run_distributed_training(self):
        """Run distributed training across all configured instances."""
        self.logger.info("Starting distributed training setup...")
        
        # Test all connections first
        failed_instances = []
        for instance in self.instances:
            if not self.test_ssh_connection(instance):
                failed_instances.append(instance.name)
                
        if failed_instances:
            self.logger.error(f"Failed to connect to instances: {failed_instances}")
            return False
            
        # Sync code to all instances
        for instance in self.instances:
            if not self.sync_code_to_instance(instance):
                self.logger.error(f"Failed to sync code to {instance.name}")
                return False
                
        # Setup instances if needed
        if self.config.get('training', {}).get('install_dependencies', True):
            for instance in self.instances:
                if not self.setup_instance(instance):
                    self.logger.error(f"Failed to setup {instance.name}")
                    return False
                    
        # Start training on all instances
        for instance in self.instances:
            if not self.start_training(instance):
                self.logger.error(f"Failed to start training on {instance.name}")
                return False
                
        self.logger.info("✓ Distributed training started on all instances")
        return True
        
    def monitor_training(self):
        """Monitor training progress on all instances."""
        self.logger.info("Monitoring training progress...")
        
        while True:
            try:
                for instance in self.instances:
                    status = self.check_training_status(instance)
                    self.logger.info(f"{instance.name}: {status}")
                    
                # Wait before next check
                import time
                time.sleep(30)
                
            except KeyboardInterrupt:
                self.logger.info("Monitoring stopped by user")
                break

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='VULTR SSH-based distributed training')
    parser.add_argument('--config', default='config/vultr_ssh_config.yaml',
                       help='Path to SSH configuration file')
    parser.add_argument('--action', choices=['test', 'setup', 'train', 'monitor'],
                       default='train', help='Action to perform')
    
    args = parser.parse_args()
    
    manager = VultrSSHManager(args.config)
    
    if args.action == 'test':
        # Test all SSH connections
        for instance in manager.instances:
            manager.test_ssh_connection(instance)
            
    elif args.action == 'setup':
        # Setup all instances
        for instance in manager.instances:
            manager.setup_instance(instance)
            
    elif args.action == 'train':
        # Run distributed training
        manager.run_distributed_training()
        
    elif args.action == 'monitor':
        # Monitor training
        manager.monitor_training()

if __name__ == "__main__":
    main()