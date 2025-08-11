# Backup Functionality for PPO Training

## Overview

The PPO training system now includes comprehensive backup functionality to prevent loss of progress during long training runs. This system automatically saves intermediate model states and provides easy recovery options.

## Key Features

### 1. Dual Save System

- **Backup Saves**: Frequent intermediate saves (default: every 10,000 timesteps)
- **Main Checkpoints**: Less frequent comprehensive saves (default: every 25,000 timesteps)
- **Final Model**: Complete model save at the end of training

### 2. Automatic Cleanup

- Keeps only the 3 most recent backup files
- Prevents disk space issues during long training runs
- Automatically removes old backup files after training completes

### 3. Easy Recovery

- Resume training from any backup point
- Maintains all training statistics and progress
- No loss of learning progress

## File Naming Convention

```
ppo_balatro_backup_<timesteps>.pth    # Backup files
ppo_balatro_<timesteps>.pth           # Main checkpoints  
ppo_balatro_final.pth                 # Final model
training_curves_<timesteps>.png       # Training plots
```

## Usage Examples

### Basic Training with Default Settings

```bash
python train_ppo.py --timesteps 100000
```

**Creates:**
- Backup every 10,000 timesteps: `ppo_balatro_backup_10000.pth`, `ppo_balatro_backup_20000.pth`, etc.
- Main checkpoints every 25,000 timesteps: `ppo_balatro_25000.pth`, `ppo_balatro_50000.pth`, etc.
- Final model: `ppo_balatro_final.pth`

### Custom Backup Intervals

```bash
python train_ppo.py --timesteps 100000 --backup-interval 5000 --save-interval 15000
```

**Creates:**
- Backup every 5,000 timesteps: `ppo_balatro_backup_5000.pth`, `ppo_balatro_backup_10000.pth`, etc.
- Main checkpoints every 15,000 timesteps: `ppo_balatro_15000.pth`, `ppo_balatro_30000.pth`, etc.

### Resume Training from Backup

```bash
python train_ppo.py --load-model ppo_balatro_backup_20000.pth --timesteps 50000
```

**Continues training from the 20,000 timestep backup for an additional 50,000 timesteps.**

## Configuration Options

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--backup-interval` | 10000 | Timesteps between backup saves |
| `--save-interval` | 25000 | Timesteps between main checkpoints |
| `--load-model` | None | Path to model file to continue from |

### Training Parameters

```python
trainer.train(
    total_timesteps=100000,
    backup_interval=10000,    # Backup every 10k steps
    save_interval=25000,      # Main checkpoints every 25k steps
    # ... other parameters
)
```

## Recovery Scenarios

### 1. Training Interruption

If training is interrupted (power loss, crash, etc.):

```bash
# Find the most recent backup
ls -la ppo_balatro_backup_*.pth

# Resume from the latest backup
python train_ppo.py --load-model ppo_balatro_backup_30000.pth --timesteps 70000
```

### 2. Experiment Comparison

Compare different training stages:

```bash
# Test model at 20k timesteps
python test_ppo.py --model ppo_balatro_backup_20000.pth

# Test model at 40k timesteps  
python test_ppo.py --model ppo_balatro_backup_40000.pth

# Test final model
python test_ppo.py --model ppo_balatro_final.pth
```

### 3. Fine-tuning

Continue training with different parameters:

```bash
# Train for 50k steps with original settings
python train_ppo.py --timesteps 50000

# Continue with different learning rate
python train_ppo.py --load-model ppo_balatro_backup_50000.pth --lr 1e-4 --timesteps 30000
```

## Implementation Details

### Backup Save Process

```python
# Save backup model (more frequent)
if timesteps_so_far % backup_interval == 0:
    backup_path = f"ppo_balatro_backup_{timesteps_so_far}.pth"
    self.agent.save_model(backup_path)
    print(f"💾 Backup model saved to {backup_path}")

# Save main checkpoint (less frequent)
if timesteps_so_far % save_interval == 0:
    model_path = f"ppo_balatro_{timesteps_so_far}.pth"
    self.agent.save_model(model_path)
    print(f"💾 Model saved to {model_path}")
```

### Automatic Cleanup

```python
def _cleanup_old_backups(self, keep_count: int = 3):
    """Clean up old backup files, keeping only the most recent ones"""
    backup_files = glob.glob("ppo_balatro_backup_*.pth")
    
    if len(backup_files) > keep_count:
        # Sort by modification time (newest first)
        backup_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        
        # Remove old backup files
        files_to_remove = backup_files[keep_count:]
        for file_path in files_to_remove:
            os.remove(file_path)
```

## Best Practices

### 1. Choose Appropriate Intervals

- **Short training runs** (< 50k timesteps): Use smaller intervals
  ```bash
  --backup-interval 5000 --save-interval 10000
  ```

- **Long training runs** (> 100k timesteps): Use larger intervals
  ```bash
  --backup-interval 15000 --save-interval 30000
  ```

### 2. Monitor Disk Space

- Backup files can be large (several MB each)
- Use automatic cleanup to prevent disk space issues
- Consider external storage for very long training runs

### 3. Test Recovery

- Periodically test that you can resume from backups
- Verify that training statistics are preserved
- Check that model performance is maintained

### 4. Version Control

- Keep track of which backup corresponds to which training configuration
- Document any parameter changes between training sessions
- Use descriptive file names for important checkpoints

## Troubleshooting

### Common Issues

1. **"No backup files found"**
   - Check if training completed the backup interval
   - Verify file permissions in the working directory

2. **"Cannot load model"**
   - Ensure the backup file exists and is not corrupted
   - Check that the model architecture matches

3. **"Training stats not preserved"**
   - Backup files include training statistics
   - If stats are missing, check the `load_model` method

### Debug Commands

```bash
# List all backup files with details
ls -la ppo_balatro_backup_*.pth

# Check file sizes
du -h ppo_balatro_backup_*.pth

# Verify file integrity
python -c "import torch; torch.load('ppo_balatro_backup_10000.pth')"
```

## Benefits

1. **Data Safety**: Never lose training progress due to interruptions
2. **Flexibility**: Resume training from any point with different parameters
3. **Experimentation**: Compare models at different training stages
4. **Efficiency**: Avoid restarting long training runs from scratch
5. **Storage Management**: Automatic cleanup prevents disk space issues

The backup system ensures that your training investment is protected and provides maximum flexibility for experimentation and recovery. 