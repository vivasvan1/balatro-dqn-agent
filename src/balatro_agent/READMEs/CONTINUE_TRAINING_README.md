# Model Loading and Continuation Training

This document explains how to use the new model loading and continuation training functionality in the PPO training script.

## Overview

The `train_ppo.py` script now supports loading a previously trained model and continuing training from where it left off. This is useful for:

- Resuming interrupted training sessions
- Continuing training with different hyperparameters
- Fine-tuning pre-trained models
- Experimenting with longer training runs

## Usage

### Basic Usage

```bash
# Start fresh training
python train_ppo.py --timesteps 100000

# Continue training from a saved model
python train_ppo.py --timesteps 100000 --load-model ppo_balatro_50000.pth
```

### Command Line Arguments

The new `--load-model` argument accepts a path to a previously saved model file:

```bash
python train_ppo.py \
    --timesteps 100000 \
    --batch-size 2048 \
    --lr 3e-4 \
    --load-model ppo_balatro_final.pth \
    --eval-interval 5000
```

### Available Arguments

- `--load-model`: Path to model file to continue training from
- `--timesteps`: Total timesteps for training
- `--batch-size`: Batch size for training
- `--lr`: Learning rate
- `--blind-score`: Target score to win
- `--hidden-dim`: Hidden dimension for neural network
- `--clip-ratio`: PPO clip ratio
- `--device`: Device to use (auto/cpu/cuda)
- `--eval-interval`: Evaluation interval
- `--demo-interval`: Demo episode interval
- `--debug-interval`: Debug monitoring interval

## Features

### Automatic Training Stats Continuation

When loading a model, the training script automatically:

1. Loads the model weights and optimizer state
2. Restores all previous training statistics
3. Continues plotting from the previous training curves
4. Skips initial evaluation if previous data exists

### Graceful Error Handling

- If the model file doesn't exist, training starts fresh with a warning
- Compatible with models saved by older versions (missing stats are initialized as empty)
- Preserves all training history for continuous plotting

### Training Statistics Preserved

The following statistics are preserved when loading a model:

- Episode rewards
- Episode lengths
- Win rates
- Policy losses
- Value losses
- Entropy losses
- KL divergences
- Action distributions
- And more...

## Examples

### Example 1: Resume Interrupted Training

```bash
# Original training was interrupted at 50k timesteps
python train_ppo.py --timesteps 100000 --load-model ppo_balatro_50000.pth
```

### Example 2: Continue with Different Hyperparameters

```bash
# Continue training with a lower learning rate
python train_ppo.py \
    --timesteps 50000 \
    --load-model ppo_balatro_final.pth \
    --lr 1e-4 \
    --batch-size 1024
```

### Example 3: Fine-tune on Different Target Score

```bash
# Continue training but change the target score
python train_ppo.py \
    --timesteps 50000 \
    --load-model ppo_balatro_final.pth \
    --blind-score 400
```

## File Structure

When you save a model, it creates a `.pth` file containing:

```python
{
    'actor_critic_state_dict': model_weights,
    'optimizer_state_dict': optimizer_state,
    'training_stats': {
        'episode_rewards': [...],
        'policy_losses': [...],
        'value_losses': [...],
        # ... all training statistics
    }
}
```

## Testing

Run the test script to verify the functionality:

```bash
python test_continue_training.py
```

This will:
1. Start a short initial training run
2. Continue training from the saved model
3. Test error handling with non-existent files

## Tips

1. **Save frequently**: Use `--save-interval` to save models regularly
2. **Check file existence**: Verify the model file exists before continuing
3. **Monitor training curves**: The plots will show the complete training history
4. **Experiment with hyperparameters**: Try different learning rates or batch sizes when continuing
5. **Backup important models**: Keep copies of your best performing models

## Troubleshooting

### Model file not found
```
⚠️  Warning: Model file model.pth not found. Starting fresh training.
```
This is normal if the file doesn't exist. Training will start fresh.

### Incompatible model format
If you get errors loading older models, they may have a different format. The script handles this gracefully by initializing missing statistics as empty lists.

### Memory issues
If you encounter memory issues when loading large models, try:
- Reducing batch size
- Using CPU instead of GPU
- Loading on a machine with more RAM 