# Balatro Gym Environment

This directory contains a custom Gymnasium environment for training AI agents to play Balatro, specifically focused on the first stage of the game with hand evaluation and discard decisions.

## Overview

The Balatro Gym Environment simulates the core mechanics of Balatro's first stage:
- **8-card hands** from a standard 52-card deck
- **Discard decisions**: Choose which cards to keep or discard
- **Best 5-card selection**: Automatically finds the best 5-card poker hand from 8 cards
- **Balatro-style scoring**: Base score × multiplier system
- **Strategic gameplay**: Balance between keeping good cards and drawing for better hands

## Files

- `balatro_gym.py` - The main gym environment implementation
- `train_balatro_gym.py` - Training script using the existing DQN agent
- `test_gym.py` - Test script to verify the environment works correctly

## Environment Details

### State Space (21 dimensions)
- **Cards**: 8 cards × 2 values each (rank + suit) = 16 values
- **Chips**: Current chip count = 1 value
- **Hand evaluation**: Hand type, base score, multiplier = 3 values
- **Step counter**: Current step in episode = 1 value

### Action Space (256 actions)
- Binary representation for 8-card hand
- Each bit represents keep (1) or discard (0) for that card position
- Example: Action 255 (11111111) = keep all cards
- Example: Action 128 (10000000) = keep only first card

### Reward System
- **Hand strength**: Base score × multiplier
- **Hand type bonuses**: Extra rewards for better poker hands
- **Discard penalty**: Small penalty for discarding too many cards
- **No-keep penalty**: Large penalty for discarding all cards

## Usage

### 1. Test the Environment

First, test that everything works correctly:

```bash
cd src/balatro_agent
python test_gym.py
```

This will run comprehensive tests on:
- Hand evaluation accuracy
- Environment functionality
- Action encoding/decoding
- State representation

### 2. Train the Agent

Start training the DQN agent:

```bash
python train_balatro_gym.py
```

The training script will:
- Train for 5000 episodes by default
- Save checkpoints every 1000 episodes
- Save the best model based on performance
- Generate training plots
- Evaluate the final model
- Log metrics to MLflow

### 3. Monitor Training

The training script provides:
- Progress updates every 100 episodes
- MLflow tracking for metrics
- Automatic model saving on improvement
- Final evaluation results

## Training Parameters

You can modify the training parameters in `train_balatro_gym.py`:

```python
# Training parameters
LEARNING_RATE = 0.001
N_EPISODES = 5000
START_EPSILON = 1.0
EPSILON_DECAY = START_EPSILON / (N_EPISODES / 2)
FINAL_EPSILON = 0.1

# Model parameters
STATE_SIZE = 21
ACTION_SIZE = 256
BUFFER_SIZE = int(1e5)
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 1e-3
UPDATE_EVERY = 4
```

## Expected Learning Outcomes

The agent should learn to:
1. **Recognize strong hands**: Keep cards that form good poker hands
2. **Strategic discarding**: Discard cards that don't contribute to potential hands
3. **Risk assessment**: Balance between keeping safe cards and drawing for better hands
4. **Hand improvement**: Understand which cards to keep for drawing to straights/flushes

## Example Agent Behavior

After training, the agent should make decisions like:
- **Keep pairs**: Hold cards that form pairs or better
- **Keep suited cards**: Hold cards of the same suit for flush potential
- **Keep connected cards**: Hold cards that could form straights
- **Discard high cards**: Sometimes discard high cards that don't fit the hand

## Integration with Balatro Mod

The trained model can be integrated into the Balatro mod by:
1. Loading the trained weights
2. Converting the state representation to match the game
3. Using the agent's action predictions for discard decisions

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure you're in the correct directory and have all dependencies installed
2. **Memory issues**: Reduce `BUFFER_SIZE` or `BATCH_SIZE` if you run out of memory
3. **Slow training**: Reduce `N_EPISODES` for faster testing

### Dependencies

Make sure you have:
- `gymnasium`
- `torch`
- `numpy`
- `matplotlib`
- `tqdm`
- `mlflow` (for tracking)

## Future Enhancements

Potential improvements to the environment:
- **Multi-stage gameplay**: Include multiple rounds and blinds
- **Jokers and modifiers**: Add Balatro's unique card modifiers
- **Deck building**: Include deck construction elements
- **Multiplayer simulation**: Simulate playing against other agents
- **More complex scoring**: Include chips, multipliers, and special effects

## Performance Expectations

With the current setup, you should expect:
- **Initial performance**: Random-like behavior (low rewards)
- **Mid-training**: Basic pattern recognition (moderate rewards)
- **Final performance**: Strategic decision making (high rewards)

The agent should achieve significantly better performance than random play after 1000-2000 episodes of training. 