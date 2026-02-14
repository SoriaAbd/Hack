# VULTR SSH Setup Guide for AdaptNav Training

This guide shows you how to use SSH keys with VULTR instances instead of API keys for distributed training.

## Overview

Instead of using the VULTR API to automatically create instances, this approach:
1. You manually create instances in the VULTR dashboard
2. You configure SSH key access
3. The script connects via SSH to run training

## Step 1: Generate SSH Key Pair

### On Windows:
```cmd
# Using Git Bash or WSL
ssh-keygen -t rsa -b 4096 -f ~/.ssh/vultr_key
```

### On Linux/macOS:
```bash
ssh-keygen -t rsa -b 4096 -f ~/.ssh/vultr_key
```

This creates:
- `~/.ssh/vultr_key` (private key - keep secret)
- `~/.ssh/vultr_key.pub` (public key - upload to VULTR)

## Step 2: Add SSH Key to VULTR

1. **Login to VULTR Dashboard**
2. **Go to Account → SSH Keys**
3. **Click "Add SSH Key"**
4. **Copy your public key content:**
   ```bash
   cat ~/.ssh/vultr_key.pub
   ```
5. **Paste it in VULTR and give it a name like "AdaptNav Training Key"**

## Step 3: Create VULTR Instances

### Recommended Instance Types:

#### For Master Node:
- **Type**: vc2-8c-32gb (8 CPU, 32GB RAM)
- **OS**: Ubuntu 22.04 LTS
- **Location**: Choose closest to you
- **SSH Key**: Select your "AdaptNav Training Key"

#### For Worker Nodes:
- **Type**: vc2-4c-16gb (4 CPU, 16GB RAM) 
- **OS**: Ubuntu 22.04 LTS
- **Location**: Same as master
- **SSH Key**: Select your "AdaptNav Training Key"

### Create Instances:
1. **Go to Products → Compute**
2. **Click "Deploy New Server"**
3. **Choose server type and configuration**
4. **Select your SSH key**
5. **Deploy and wait for instances to start**
6. **Note down the IP addresses**

## Step 4: Configure SSH Training

1. **Edit the configuration file:**
   ```bash
   # Edit config/vultr_ssh_config.yaml
   ```

2. **Update with your instance IPs:**
   ```yaml
   instances:
     - name: "adaptnav-master"
       ip_address: "YOUR_MASTER_IP"  # Replace with actual IP
       ssh_key_path: "~/.ssh/vultr_key"
       username: "root"
       role: "master"
       
     - name: "adaptnav-worker-1"
       ip_address: "YOUR_WORKER_1_IP"  # Replace with actual IP
       ssh_key_path: "~/.ssh/vultr_key"
       username: "root"
       role: "worker"
   ```

## Step 5: Test SSH Connections

```bash
# Test SSH connections to all instances
python scripts/vultr_ssh_training.py --action test
```

Expected output:
```
✓ SSH connection to adaptnav-master successful
✓ SSH connection to adaptnav-worker-1 successful
```

## Step 6: Setup Instances

```bash
# Install dependencies on all instances
python scripts/vultr_ssh_training.py --action setup
```

This will:
- Update system packages
- Install Python and pip
- Install AdaptNav dependencies
- Sync your code to the instances

## Step 7: Start Distributed Training

```bash
# Start training on all instances
python scripts/vultr_ssh_training.py --action train
```

## Step 8: Monitor Training

```bash
# Monitor training progress
python scripts/vultr_ssh_training.py --action monitor
```

## Manual SSH Access

You can also manually connect to any instance:

```bash
# Connect to master
ssh -i ~/.ssh/vultr_key root@YOUR_MASTER_IP

# Connect to worker
ssh -i ~/.ssh/vultr_key root@YOUR_WORKER_IP
```

## Troubleshooting

### SSH Connection Issues

1. **Permission denied:**
   ```bash
   chmod 600 ~/.ssh/vultr_key
   ```

2. **Host key verification failed:**
   ```bash
   ssh-keyscan -H YOUR_INSTANCE_IP >> ~/.ssh/known_hosts
   ```

3. **Connection timeout:**
   - Check instance is running in VULTR dashboard
   - Verify IP address is correct
   - Check firewall settings

### Instance Setup Issues

1. **Package installation fails:**
   ```bash
   # SSH to instance and run manually:
   sudo apt update && sudo apt upgrade -y
   sudo apt install -y python3 python3-pip git
   ```

2. **Code sync fails:**
   ```bash
   # Check rsync is available:
   which rsync
   # Install if missing:
   sudo apt install rsync
   ```

## Cost Optimization

### Instance Sizing:
- **Start small**: Use vc2-2c-4gb for testing
- **Scale up**: Move to vc2-8c-32gb for serious training
- **GPU instances**: Use if available in your region

### Cost Management:
- **Destroy instances** when not training
- **Use snapshots** to save configured instances
- **Monitor usage** in VULTR dashboard

## Security Best Practices

1. **SSH Key Security:**
   - Never share your private key
   - Use strong passphrase for SSH key
   - Rotate keys periodically

2. **Instance Security:**
   - Keep systems updated
   - Use firewall rules
   - Monitor access logs

3. **Network Security:**
   - Use VPC if available
   - Restrict SSH access to your IP
   - Use non-standard SSH ports if needed

## Advantages of SSH Approach

✅ **No API key needed** - just SSH keys
✅ **Manual control** - you create instances yourself  
✅ **Flexible** - works with any cloud provider
✅ **Secure** - SSH key authentication
✅ **Simple** - easier to understand and debug

## Disadvantages

❌ **Manual setup** - no automatic instance creation
❌ **No auto-scaling** - fixed number of instances
❌ **More steps** - requires manual instance management

## Next Steps

Once you have this working:
1. **Optimize training parameters** for your instance sizes
2. **Add more worker nodes** for faster training
3. **Implement model checkpointing** for reliability
4. **Set up monitoring dashboards** for better visibility

This SSH-based approach gives you full control over your VULTR instances while avoiding the need for API keys.