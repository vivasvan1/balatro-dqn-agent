# Folder Structure for Organized Training Outputs

## Overview

The training script now automatically organizes all outputs into separate folders for better file management and organization.

## Folder Structure

```
balatro_agent/
├── checkpoints/          # Model weights and checkpoints
│   ├── ppo_balatro_25000.pth
│   ├── ppo_balatro_50000.pth
│   └── ppo_balatro_final.pth
├── plots/               # Training curves and visualizations
│   ├── training_curves_25000.png
│   ├── training_curves_50000.png
│   └── final_training_curves.png
├── backups/             # Intermediate backup models
│   ├── ppo_balatro_backup_10000.pth
│   ├── ppo_balatro_backup_20000.pth
│   └── ppo_balatro_backup_30000.pth
└── train_ppo.py         # Main training script
```

## Folder Details

### `checkpoints/`
- **Purpose**: Store final and intermediate model weights
- **Contents**: 
  - `ppo_balatro_{timesteps}.pth` - Checkpoint models at save intervals
  - `ppo_balatro_final.pth` - Final trained model
- **Usage**: Load models for continued training or evaluation

### `plots/`
- **Purpose**: Store training visualizations and curves
- **Contents**:
  - `training_curves_{timesteps}.png` - Training progress plots
  - `final_training_curves.png` - Final comprehensive training analysis
- **Features**: 12-panel plots showing rewards, losses, win rates, and debugging info

### `backups/`
- **Purpose**: Store frequent backup models for safety
- **Contents**:
  - `ppo_balatro_backup_{timesteps}.pth` - Backup models at backup intervals
- **Auto-cleanup**: Keeps only the 3 most recent backup files

## Automatic Folder Creation

The training script automatically creates these folders when training starts:

```python
# Create output directories
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("plots", exist_ok=True)
os.makedirs("backups", exist_ok=True)
```

## File Naming Convention

### Models
- **Checkpoints**: `ppo_balatro_{timesteps}.pth`
- **Final Model**: `ppo_balatro_final.pth`
- **Backups**: `ppo_balatro_backup_{timesteps}.pth`

### Plots
- **Training Curves**: `training_curves_{timesteps}.png`
- **Final Analysis**: `final_training_curves.png`

## Usage Examples

### Start Training with Organized Output
```bash
python train_ppo.py --timesteps 1000000 --curriculum-learning
```

### Continue Training from Checkpoint
```bash
python train_ppo.py --timesteps 1000000 --load-model checkpoints/ppo_balatro_final.pth
```

### Load Specific Checkpoint
```python
from train_ppo import ComprehensivePPOTrainer

trainer = ComprehensivePPOTrainer()
trainer.agent.load_model("checkpoints/ppo_balatro_50000.pth")
```

## Benefits

1. **Clean Organization**: No more cluttered main directory
2. **Easy Navigation**: Clear separation of different file types
3. **Automatic Management**: Folders created automatically
4. **Backup Safety**: Frequent backups with auto-cleanup
5. **Version Control**: Easy to track different training runs

## Customization

You can modify the folder names by editing the training script:

```python
# In train_ppo.py, modify these lines:
os.makedirs("checkpoints", exist_ok=True)  # Change "checkpoints" to your preferred name
os.makedirs("plots", exist_ok=True)        # Change "plots" to your preferred name
os.makedirs("backups", exist_ok=True)      # Change "backups" to your preferred name
```

## Monitoring Training Progress

During training, you'll see output like:
```
💾 Backup model saved to backups/ppo_balatro_backup_10000.pth
💾 Model saved to checkpoints/ppo_balatro_25000.pth
📊 Training curves saved to plots/training_curves_25000.png
```

This makes it easy to track where your files are being saved and monitor training progress. 