# PPO Agent for Balatro

This directory contains a complete implementation of Proximal Policy Optimization (PPO) for the Balatro gym environment. PPO is a state-of-the-art reinforcement learning algorithm that's particularly well-suited for policy optimization in sequential decision-making games like Balatro.

## 🎯 Why PPO for Balatro?

**PPO is ideal for Balatro because:**

1. **Policy Optimization**: Balatro requires learning optimal card-playing strategies, which PPO excels at
2. **Stable Training**: PPO's clipped objective prevents large policy updates that could destabilize training
3. **Sample Efficiency**: PPO can learn effective policies with relatively few samples
4. **Continuous Learning**: PPO can adapt to different game states and scoring targets

## 📁 Files Overview

- `ppo_agent.py` - Complete PPO implementation with actor-critic architecture
- `train_ppo.py` - Training script with command-line interface
- `balatro_gym_v2_simple.py` - Simplified Balatro environment (23-dimensional state space)

## 🚀 Quick Start

### 1. Basic Training

```bash
# Train a PPO agent with default settings
python train_ppo.py --mode train

# Train with custom parameters
python train_ppo.py --mode train \
    --blind-score 500 \
    --timesteps 1000000 \
    --batch-size 4096 \
    --lr 1e-4
```

### 2. Evaluate a Trained Model

```bash
# Evaluate the trained model
python train_ppo.py --mode evaluate \
    --model-path ppo_balatro_model.pth \
    --eval-episodes 100
```

### 3. Watch a Demo Episode

```bash
# Run a demonstration episode
python train_ppo.py --mode demo \
    --model-path ppo_balatro_model.pth
```

## 🏗️ Architecture

### Actor-Critic Network

The PPO agent uses a shared actor-critic architecture:

```
Input (23-dim state) 
    ↓
Shared Layers (256 units, ReLU)
    ↓
┌─────────────┬─────────────┐
│   Actor     │   Critic    │
│ (Policy)    │  (Value)    │
│ 128 units   │  128 units  │
│   ReLU      │   ReLU      │
│             │             │
│ Action Dim  │     1       │
│ (Logits)    │  (Value)    │
└─────────────┴─────────────┘
```

### State Representation (23 dimensions)

1. **Hand Encoding** (16 values): 8 cards × (rank + suit)
2. **Game State** (7 values):
   - Plays left (1)
   - Discards left (1) 
   - Current score (1)
   - Blind score (1)
   - Game over flag (1)
   - Progress to target (1)
   - Hand quality score (1)

## ⚙️ Hyperparameters

### Default PPO Settings

```python
learning_rate = 3e-4
gamma = 0.99              # Discount factor
gae_lambda = 0.95         # GAE parameter
clip_ratio = 0.2          # PPO clipping parameter
value_loss_coef = 0.5     # Value function loss coefficient
entropy_coef = 0.01       # Entropy bonus coefficient
max_grad_norm = 0.5       # Gradient clipping
target_kl = 0.01          # Early stopping KL threshold
hidden_dim = 256          # Neural network hidden dimension
```

### Training Settings

```python
total_timesteps = 500000   # Total training timesteps
batch_size = 2048         # Timesteps per batch
update_epochs = 10        # PPO update epochs per batch
eval_interval = 10000     # Evaluation frequency
save_interval = 50000     # Model save frequency
```

## 📊 Training Process

### 1. Data Collection
- Collect episodes using current policy
- Store states, actions, rewards, values, and log probabilities
- Compute Generalized Advantage Estimation (GAE)

### 2. Policy Update
- Perform multiple epochs of PPO updates on collected data
- Use clipped surrogate objective to prevent large policy changes
- Early stopping based on KL divergence

### 3. Evaluation
- Regular evaluation of current policy
- Track win rates, average rewards, and episode lengths
- Generate training curves

## 🎮 Action Space

The action space includes all valid card combinations:

- **Play Actions**: All combinations of 1-5 cards to play
- **Discard Actions**: All combinations of 1-5 cards to discard
- **Total Actions**: ~254 possible actions

## 💡 Key Features

### 1. Reward Shaping
- Base reward from score gained
- Hand quality bonuses for good poker hands
- Progress bonuses for approaching target score
- Efficiency bonuses for using fewer cards effectively

### 2. Exploration
- Entropy regularization encourages exploration
- Stochastic policy sampling during training
- Deterministic action selection during evaluation

### 3. Training Stability
- Gradient clipping prevents exploding gradients
- KL divergence early stopping
- Advantage normalization
- Orthogonal weight initialization

## 📈 Expected Performance

### Training Progress
- **Early Training**: Random actions, low win rates (~5-10%)
- **Mid Training**: Learning basic strategies, improving win rates (~20-40%)
- **Late Training**: Optimized strategies, high win rates (~60-80%)

### Performance Metrics
- **Win Rate**: Percentage of games won
- **Average Reward**: Mean reward per episode
- **Episode Length**: Average steps per episode
- **Score Efficiency**: Points gained per action

## 🔧 Customization

### Environment Modifications

```python
# Create environment with different target score
env = BalatroGymEnvSimple(blind_score=500)

# Modify reward function in balatro_gym_v2_simple.py
def _calculate_reward(self, action_type: str, result: Any) -> float:
    # Custom reward shaping
    pass
```

### Network Architecture

```python
# Modify hidden dimensions
agent = PPOAgent(
    env=env,
    hidden_dim=512,  # Larger network
    device="cuda"
)
```

### Training Parameters

```python
# Custom training schedule
agent.train(
    total_timesteps=1000000,
    batch_size=4096,
    update_epochs=15,
    eval_interval=5000
)
```

## 🐛 Troubleshooting

### Common Issues

1. **Low Win Rates**
   - Increase training time
   - Adjust reward shaping
   - Try different learning rates

2. **Training Instability**
   - Reduce learning rate
   - Increase batch size
   - Adjust gradient clipping

3. **Slow Training**
   - Use GPU acceleration
   - Reduce batch size
   - Simplify network architecture

### Debugging Tips

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Monitor training stats
print(f"Policy Loss: {update_stats['policy_loss']:.4f}")
print(f"Value Loss: {update_stats['value_loss']:.4f}")
print(f"KL Divergence: {update_stats['kl_div']:.4f}")
```

## 📚 Advanced Usage

### Curriculum Learning

```python
# Start with easier targets and gradually increase difficulty
targets = [100, 200, 300, 400, 500]
for target in targets:
    env = BalatroGymEnvSimple(blind_score=target)
    agent = PPOAgent(env=env)
    agent.train(total_timesteps=200000)
```

### Multi-Objective Training

```python
# Train for multiple objectives (score + efficiency)
def custom_reward(score_gained, cards_used, hand_type):
    efficiency_bonus = score_gained / max(cards_used, 1)
    return score_gained + efficiency_bonus
```

### Ensemble Methods

```python
# Train multiple agents and ensemble their predictions
agents = [PPOAgent(env) for _ in range(3)]
# Train each agent
# Ensemble predictions during evaluation
```

## 🎯 Next Steps

1. **Experiment with different reward functions**
2. **Try curriculum learning approaches**
3. **Implement more sophisticated exploration strategies**
4. **Add attention mechanisms for card relationships**
5. **Explore multi-agent training scenarios**

## 📖 References

- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [GAE Paper](https://arxiv.org/abs/1506.02438)
- [Balatro Game](https://store.steampowered.com/app/2379780/Balatro/)

---

**Happy Training! 🎰🤖** 