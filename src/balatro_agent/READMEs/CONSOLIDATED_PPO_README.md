# Consolidated PPO Training System for Balatro

This directory contains a comprehensive PPO (Proximal Policy Optimization) implementation for training agents to play Balatro, a poker-like card game.

## 🎯 Overview

The system consists of two main components:
1. **`train_ppo.py`** - Comprehensive training script with debugging and monitoring
2. **`test_ppo.py`** - Evaluation and demo script for analyzing trained models

## 📁 Files

### Core Files
- **`train_ppo.py`** - Main training script with debugging features
- **`test_ppo.py`** - Evaluation and demo script
- **`ppo_agent.py`** - PPO agent implementation with actor-critic architecture
- **`balatro_gym_v2_simple.py`** - Simplified Balatro environment
- **`test_consolidated.py`** - Test script to verify system functionality

### Training Artifacts
- **`ppo_balatro_final.pth`** - Trained model weights
- **`final_training_curves.png`** - Training progress visualization
- **`TRAINING_SUMMARY.md`** - Detailed training analysis

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install torch numpy matplotlib tqdm gymnasium
```

### 2. Run Training
```bash
# Basic training (100k timesteps)
python train_ppo.py

# Custom parameters
python train_ppo.py --blind-score 400 --timesteps 200000 --lr 1e-4 --batch-size 4096
```

### 3. Evaluate Trained Model
```bash
# Comprehensive evaluation
python test_ppo.py --model-path ppo_balatro_final.pth --mode all

# Just demo episodes
python test_ppo.py --model-path ppo_balatro_final.pth --mode demo --num-demos 5

# Policy analysis only
python test_ppo.py --model-path ppo_balatro_final.pth --mode analyze
```

### 4. Test System
```bash
# Run comprehensive tests
python test_consolidated.py
```

## 🔧 Training Parameters

### Command Line Options
```bash
python train_ppo.py [OPTIONS]

Options:
  --blind-score INT      Target score to win (default: 300)
  --timesteps INT        Total timesteps for training (default: 100000)
  --batch-size INT       Batch size for training (default: 2048)
  --lr FLOAT            Learning rate (default: 3e-4)
  --hidden-dim INT      Hidden dimension (default: 256)
  --clip-ratio FLOAT    PPO clip ratio (default: 0.2)
  --device STR          Device to use (default: "auto")
  --eval-interval INT   Evaluation interval (default: 5000)
  --demo-interval INT   Demo interval (default: 10000)
  --debug-interval INT  Debug interval (default: 2000)
```

### Advanced Configuration
You can also create a custom trainer instance:

```python
from train_ppo import ComprehensivePPOTrainer

trainer = ComprehensivePPOTrainer(
    blind_score=300,
    learning_rate=3e-4,
    clip_ratio=0.2,
    hidden_dim=256,
    device="cuda"
)

agent = trainer.train(
    total_timesteps=100000,
    batch_size=2048,
    eval_interval=5000,
    demo_interval=10000
)
```

## 📊 Evaluation Features

### Evaluation Modes
- **`evaluate`** - Run comprehensive evaluation over many episodes
- **`demo`** - Show detailed step-by-step gameplay
- **`analyze`** - Analyze policy preferences and value predictions
- **`all`** - Run all evaluation modes

### Evaluation Metrics
- **Win Rate** - Percentage of games won
- **Average Reward** - Mean reward per episode
- **Action Distribution** - Ratio of play vs discard actions
- **Hand Type Distribution** - Types of poker hands played
- **Policy Confidence** - How certain the agent is about its actions

### Example Evaluation Output
```
📊 Evaluation Results:
  Win Rate: 45.0%
  Average Reward: 12.3 ± 8.7
  Action Distribution: {'play': '65.2%', 'discard': '34.8%'}
  Hand Type Distribution: {'pair': 45, 'three_of_a_kind': 12, 'straight': 8}
```

## 🔍 Debugging Features

The training system includes comprehensive debugging to help diagnose training issues:

### Debugging Indicators
- **Policy Loss** - Should be non-zero and changing
- **KL Divergence** - Should be between 0.001 and 0.05
- **Entropy Loss** - Should be negative (encouraging exploration)
- **Value Loss** - Should be decreasing over time

### Common Issues and Solutions

| Symptom | Possible Cause | Solution |
|---------|----------------|----------|
| Agent only discards | Bad reward function | Adjust reward shaping |
| Agent only plays | Over-penalized discarding | Reduce discard penalties |
| Policy loss near zero | Learning rate too small | Increase learning rate |
| High KL divergence | Learning rate too high | Decrease learning rate |
| Low entropy | Policy too confident | Increase entropy coefficient |

### Debug Monitoring
The training script automatically monitors:
- Policy confidence trends
- Action distribution changes
- Reward signal quality
- Advantage estimation accuracy

## 📈 Training Visualization

The system generates comprehensive training plots including:
- Episode rewards over time
- Win rate progression
- Policy and value losses
- KL divergence trends
- Action distribution evolution
- Training diagnostics and recommendations

## 🎮 Demo Episodes

The demo feature shows detailed gameplay with:
- Step-by-step action analysis
- Action probabilities for top 3 choices
- Reward breakdown
- Game state information
- Hand type identification

Example demo output:
```
Step 1: PLAY ['AS', 'AD', 'AH']
  Probability: 0.847
  Reward: 15.20
  Hand type: three_of_a_kind
  Score: 45/300
```

## 🏗️ Architecture

### PPO Agent
- **Actor-Critic Network** - Shared feature layers with separate policy and value heads
- **GAE (Generalized Advantage Estimation)** - For stable advantage computation
- **PPO Clipping** - Prevents too large policy updates
- **Entropy Regularization** - Encourages exploration

### Environment
- **Simplified Balatro** - Focused on core decision-making
- **State Representation** - 23-dimensional observation space
- **Action Space** - 436 possible actions (play/discard combinations)
- **Reward Shaping** - Balanced rewards for strategic play

## 🔧 Customization

### Adding New Features
1. **Custom Reward Functions** - Modify `balatro_gym_v2_simple.py`
2. **Network Architecture** - Update `ppo_agent.py` ActorCritic class
3. **Training Algorithms** - Extend `ComprehensivePPOTrainer` class
4. **Evaluation Metrics** - Add new metrics to `PPOEvaluator`

### Hyperparameter Tuning
The system is designed for easy hyperparameter experimentation:
- Learning rate scheduling
- Curriculum learning
- Reward function tuning
- Network architecture changes

## 📚 Usage Examples

### Basic Training
```python
from train_ppo import ComprehensivePPOTrainer

trainer = ComprehensivePPOTrainer(blind_score=300)
agent = trainer.train(total_timesteps=100000)
```

### Custom Evaluation
```python
from test_ppo import PPOEvaluator

evaluator = PPOEvaluator("ppo_balatro_final.pth", blind_score=300)
eval_stats = evaluator.evaluate(num_episodes=100)
evaluator.demo_episodes(num_episodes=3)
```

### Policy Analysis
```python
action_prefs, value_preds = evaluator.analyze_policy(num_samples=1000)
print(f"Action preferences: {action_prefs}")
print(f"Value predictions: {np.mean(value_preds):.2f}")
```

## 🎯 Performance Expectations

### Training Time
- **10k timesteps**: ~20 seconds (CPU)
- **100k timesteps**: ~3 minutes (CPU)
- **1M timesteps**: ~30 minutes (CPU)

### Expected Performance
- **Win Rate**: 40-60% (depending on blind score)
- **Average Reward**: 10-50 (depending on reward function)
- **Action Balance**: 60-70% play, 30-40% discard

## 🐛 Troubleshooting

### Common Errors
1. **CUDA out of memory** - Reduce batch size or use CPU
2. **Model loading errors** - Ensure hidden_dim matches training
3. **Import errors** - Install missing dependencies
4. **Training divergence** - Check reward function and learning rate

### Getting Help
- Check the debug output during training
- Review training curves for anomalies
- Test with smaller parameters first
- Verify environment behavior independently

## 📄 License

This implementation is part of the Balatro Agent project. See the main project README for license information.

---

**Happy Training! 🎰🤖** 