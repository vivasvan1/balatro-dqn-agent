# Strategic Improvements for Balatro PPO Agent

## Problem Analysis

Your agent was making basic plays but not developing strategic thinking for building better hands like flushes and three-of-a-kinds. After analyzing the demo episode, I identified several key issues:

1. **Weak reward incentives** - The agent wasn't getting strong enough signals to build better hands
2. **Limited state information** - The agent lacked strategic context about hand potential
3. **No curriculum learning** - The agent was trying to learn everything at once
4. **Suboptimal hyperparameters** - Learning rate and network size could be improved

## Key Improvements Made

### 1. Enhanced Reward Function

**Before:**
- Basic hand bonuses (Flush: 20, Three of a Kind: 10)
- Weak penalties for playing bad hands (-25)

**After:**
- Much stronger hand bonuses (Flush: 60, Three of a Kind: 30)
- Stronger penalties for playing weak hands (-50)
- Bonus for using more cards (encourages building bigger hands)
- Strategic discard bonuses based on hand potential

### 2. Strategic State Features

**New features added to state representation:**
- **Flush potential** (0-1 scale) - How close to a flush
- **Three-of-a-kind potential** (0-1 scale) - How close to three of a kind
- **High card ratio** (0-1 scale) - Proportion of high-value cards
- **Current best hand score** (normalized) - Best possible hand from current cards
- **Discard urgency** (0-1 scale) - When to discard bad hands
- **Play urgency** (0-1 scale) - When to play good hands
- **Hand diversity** (0-1 scale) - How diverse the hand is
- **Strategic value** (0-1 scale) - Overall hand potential

**State size increased from 23 to 31 features**

### 3. Curriculum Learning

**Progressive difficulty stages:**
1. **Stage 1**: Target score 100 (easy)
2. **Stage 2**: Target score 200 (medium)
3. **Stage 3**: Target score 250 (hard)
4. **Stage 4**: Target score 300 (final)

The agent learns basic strategies on easy targets, then gradually applies them to harder challenges.

### 4. Improved Hyperparameters

**Before:**
- Learning rate: 3e-4
- Hidden dimension: 256
- Entropy coefficient: 0.01

**After:**
- Learning rate: 1e-4 (more stable learning)
- Hidden dimension: 512 (larger network capacity)
- Entropy coefficient: 0.02 (more exploration)

## How to Use the Improvements

### Quick Start

```bash
# Test the improvements
python test_improved_training.py

# Start training with improvements
python train_ppo.py --timesteps 1000000 --curriculum-learning

# Train without curriculum learning
python train_ppo.py --timesteps 1000000 --no-curriculum

# Continue from existing model
python train_ppo.py --timesteps 1000000 --load-model ppo_balatro_final.pth
```

### Advanced Options

```bash
# Custom hyperparameters
python train_ppo.py \
    --timesteps 2000000 \
    --lr 5e-5 \
    --hidden-dim 1024 \
    --batch-size 4096 \
    --curriculum-learning

# Different target score
python train_ppo.py --blind-score 400 --timesteps 1000000
```

## Expected Improvements

With these changes, you should see:

1. **Better strategic play** - Agent will learn to discard weak hands and build better ones
2. **Flush awareness** - Agent will recognize and preserve flush opportunities
3. **Three-of-a-kind building** - Agent will collect cards of the same rank
4. **Faster learning** - Curriculum learning helps the agent learn step by step
5. **More stable training** - Better hyperparameters reduce training instability

## Training Recommendations

### For Quick Results (1-2 million timesteps)
```bash
python train_ppo.py --timesteps 1000000 --curriculum-learning
```

### For Best Performance (5+ million timesteps)
```bash
python train_ppo.py --timesteps 5000000 --lr 5e-5 --hidden-dim 1024 --curriculum-learning
```

### For Continued Training
```bash
python train_ppo.py --timesteps 2000000 --load-model ppo_balatro_final.pth --curriculum-learning
```

## Monitoring Progress

The training script provides comprehensive monitoring:

- **Evaluation metrics** every 5000 timesteps
- **Demo episodes** every 10000 timesteps
- **Training curves** with strategic indicators
- **Curriculum stage progression**
- **Debug analysis** for training issues

## Troubleshooting

### If the agent still plays weak hands:
1. Increase training time (try 2-3 million timesteps)
2. Reduce learning rate further (try 5e-5)
3. Increase entropy coefficient (try 0.03)

### If training is unstable:
1. Reduce learning rate
2. Increase batch size
3. Reduce clip ratio (try 0.1)

### If the agent doesn't learn strategic play:
1. Enable curriculum learning
2. Increase training time
3. Check that strategic features are working (run test script)

## Technical Details

### Reward Function Changes

The new reward function provides much stronger signals:

```python
# Stronger hand bonuses
hand_bonuses = {
    "Royal Flush": 200, "Straight Flush": 150, "Four of a Kind": 100,
    "Full House": 80, "Flush": 60, "Straight": 40,
    "Three of a Kind": 30, "Two Pair": 15, "Pair": 5, "High Card": -10
}

# Much stronger penalty for playing weak hands
if hand_type in ["High Card", "Pair"] and score_gained < 30 and self.discards_left > 0:
    reward -= 50.0  # Was -25.0 before

# Bonus for using more cards
if cards_used >= 4:
    reward += 10.0
elif cards_used >= 5:
    reward += 20.0
```

### State Representation

The new state includes strategic features that help the agent understand hand potential:

```python
strategic_features = [
    flush_potential,      # How close to a flush
    three_kind_potential, # How close to three of a kind
    high_card_ratio,      # Proportion of high cards
    normalized_best_score, # Best possible hand score
    discard_urgency,      # When to discard
    play_urgency,         # When to play
    hand_diversity,       # Hand variety
    strategic_value       # Overall potential
]
```

## Conclusion

These improvements should significantly help your agent learn strategic play. The key is the combination of:

1. **Stronger rewards** that clearly signal good vs bad decisions
2. **Better state information** that helps the agent understand hand potential
3. **Curriculum learning** that builds skills progressively
4. **Optimized hyperparameters** for stable learning

Start with 1-2 million timesteps and curriculum learning enabled. You should see much better strategic play within the first 500k timesteps! 