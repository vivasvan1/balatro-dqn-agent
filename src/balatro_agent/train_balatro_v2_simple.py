#!/usr/bin/env python3
"""
Training script for Simplified Balatro Gym Environment
Uses reduced state space and improved hyperparameters for better training stability
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random
from typing import List, Tuple, Dict, Any
import matplotlib.pyplot as plt
from datetime import datetime
import mlflow
import mlflow.pytorch

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from balatro_gym_v2_simple import BalatroGymEnvSimple
from mlflow_tracker import MLflowTracker
from training_plots import TrainingPlotter

# Hyperparameters optimized for simplified environment
LEARNING_RATE = 0.0001  # Further reduced for stability
BATCH_SIZE = 64
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.05  # Higher minimum exploration
EPSILON_DECAY = 0.999  # Much slower decay for stability
MEMORY_SIZE = 10000
TARGET_UPDATE = 100
TRAINING_EPISODES = 100000
EVAL_INTERVAL = 1000
PLOT_INTERVAL = 500
DIAGNOSTICS_INTERVAL = 1000

# Simplified DQN Network for 23-dimensional state
class SimpleDQN(nn.Module):
    def __init__(self, state_size: int, action_size: int):
        super(SimpleDQN, self).__init__()
        
        # Smaller network for simplified state space
        self.fc1 = nn.Linear(state_size, 128)  # Reduced from 256
        self.fc2 = nn.Linear(128, 128)         # Reduced from 256
        self.fc3 = nn.Linear(128, 64)          # New layer for better representation
        self.fc4 = nn.Linear(64, action_size)
        
        self.dropout = nn.Dropout(0.1)  # Add dropout for regularization
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class SimpleDQNAgent:
    def __init__(self, state_size: int, action_size: int, device: str = 'cpu'):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        
        # Networks
        self.q_network = SimpleDQN(state_size, action_size).to(device)
        self.target_network = SimpleDQN(state_size, action_size).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer with reduced learning rate
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=LEARNING_RATE)
        
        # Training parameters
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = EPSILON_START
        self.steps = 0
        
        # Experience tuple
        self.Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append(self.Experience(state, action, reward, next_state, done))
    
    def act(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        return q_values.argmax().item()
    
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([e.state for e in batch]).to(self.device)
        actions = torch.LongTensor([e.action for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in batch]).to(self.device)
        dones = torch.BoolTensor([e.done for e in batch]).to(self.device)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (GAMMA * next_q_values * ~dones)
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=0.5)  # Reduced for stability
        
        self.optimizer.step()
        
        # Update epsilon
        if self.epsilon > EPSILON_END:
            self.epsilon *= EPSILON_DECAY
        
        self.steps += 1
        
        # Update target network
        if self.steps % TARGET_UPDATE == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return loss.item()
    
    def save(self, filepath):
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps
        }, filepath)
    
    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']

def evaluate_agent(agent, env, episodes=100):
    """Evaluate agent performance"""
    wins = 0
    total_scores = []
    total_rewards = []
    
    for _ in range(episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.act(obs, training=False)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
        
        if info.get('won', False):
            wins += 1
        total_scores.append(info.get('total_score', 0))
        total_rewards.append(episode_reward)
    
    win_rate = wins / episodes
    avg_score = np.mean(total_scores)
    avg_reward = np.mean(total_rewards)
    
    return win_rate, avg_score, avg_reward

def main():
    print("🎰 Training Simplified Balatro DQN Agent")
    print("=" * 50)
    
    # Setup MLflow
    mlflow_tracker = MLflowTracker(
        experiment_name="balatro_simple_dqn"
    )
    mlflow_tracker.start_run(run_name=f"simple_dqn_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    # Setup environment and agent
    env = BalatroGymEnvSimple(blind_score=300)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    print(f"State size: {state_size}")
    print(f"Action size: {action_size}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Epsilon decay: {EPSILON_DECAY}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    agent = SimpleDQNAgent(state_size, action_size, device)
    
    # Try to load best model from MLflow artifacts if it exists
    best_model_path = os.path.join("weights", "best_simple_model.pth")
    if os.path.exists(best_model_path):
        try:
            agent.load(best_model_path)
            print(f"📂 Loaded existing best model from: {best_model_path}")
            print(f"   Epsilon: {agent.epsilon:.3f}")
            print(f"   Steps: {agent.steps}")
        except Exception as e:
            print(f"⚠️ Failed to load existing model: {e}")
            print("   Starting with fresh model")
    else:
        print("🆕 Starting with fresh model (no existing weights found)")
    
    # Setup plotting
    plotter = TrainingPlotter(
        save_dir="training_plots_simple",
        mlflow_tracker=mlflow_tracker
    )
    
    # Training loop
    episode_rewards = []
    episode_scores = []
    episode_wins = []
    episode_lengths = []
    losses = []
    
    best_win_rate = 0.0
    
    print(f"\n🚀 Starting training for {TRAINING_EPISODES} episodes...")
    print("-" * 50)
    
    for episode in range(1, TRAINING_EPISODES + 1):
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            action = agent.act(obs, training=True)
            next_obs, reward, done, truncated, info = env.step(action)
            
            agent.remember(obs, action, reward, next_obs, done)
            obs = next_obs
            episode_reward += reward
            episode_length += 1
            
            # Train on batch
            if len(agent.memory) > BATCH_SIZE:
                loss = agent.replay(BATCH_SIZE)
                if loss is not None:
                    losses.append(loss)
        
        # Record episode data
        episode_rewards.append(episode_reward)
        episode_scores.append(info.get('total_score', 0))
        episode_wins.append(1 if info.get('won', False) else 0)
        episode_lengths.append(episode_length)
        
        # Log to MLflow
        mlflow_tracker.log_custom_metric('episode_reward', episode_reward, step=episode)
        mlflow_tracker.log_custom_metric('episode_score', info.get('total_score', 0), step=episode)
        mlflow_tracker.log_custom_metric('episode_won', 1 if info.get('won', False) else 0, step=episode)
        mlflow_tracker.log_custom_metric('episode_length', episode_length, step=episode)
        mlflow_tracker.log_custom_metric('epsilon', agent.epsilon, step=episode)
        mlflow_tracker.log_custom_metric('memory_size', len(agent.memory), step=episode)
        
        # Print progress
        if episode % 100 == 0:
            recent_rewards = episode_rewards[-100:]
            recent_wins = episode_wins[-100:]
            recent_scores = episode_scores[-100:]
            
            avg_reward = np.mean(recent_rewards)
            win_rate = np.mean(recent_wins)
            avg_score = np.mean(recent_scores)
            
            print(f"Episode {episode:5d} | "
                  f"Avg Reward: {avg_reward:6.2f} | "
                  f"Win Rate: {win_rate:5.1%} | "
                  f"Avg Score: {avg_score:6.1f} | "
                  f"Epsilon: {agent.epsilon:.3f}")
        
        # Evaluation
        if episode % EVAL_INTERVAL == 0:
            win_rate, avg_score, avg_reward = evaluate_agent(agent, env, episodes=50)
            
            print(f"\n📊 Evaluation at Episode {episode}:")
            print(f"   Win Rate: {win_rate:.1%}")
            print(f"   Avg Score: {avg_score:.1f}")
            print(f"   Avg Reward: {avg_reward:.2f}")
            
            # Log evaluation metrics
            mlflow_tracker.log_custom_metric('eval_win_rate', win_rate, step=episode)
            mlflow_tracker.log_custom_metric('eval_avg_score', avg_score, step=episode)
            mlflow_tracker.log_custom_metric('eval_avg_reward', avg_reward, step=episode)
            
            # Save best model
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                model_path = os.path.join("weights", "best_simple_model.pth")
                os.makedirs("weights", exist_ok=True)
                agent.save(model_path)
                mlflow_tracker.log_artifact(model_path, "models")
                print(f"   🏆 New best model saved! Win rate: {win_rate:.1%}")
        
        # Plotting
        if episode % PLOT_INTERVAL == 0:
            plotter.add_data_point(episode, {
                'episode_rewards': episode_reward,
                'episode_scores': info.get('total_score', 0),
                'win_rates': 1 if info.get('won', False) else 0,
                'episode_lengths': episode_length,
                'epsilon_values': agent.epsilon,
                'buffer_sizes': len(agent.memory),
                # 'losses': losses[-1] if losses else 0  # Optionally add latest loss
            })
            
            plotter.plot_training_progress(episode)
        
        # Diagnostics
        if episode % DIAGNOSTICS_INTERVAL == 0:
            # Run diagnostics
            recent_episodes = 1000
            if len(episode_rewards) >= recent_episodes:
                recent_rewards = episode_rewards[-recent_episodes:]
                recent_scores = episode_scores[-recent_episodes:]
                recent_wins = episode_wins[-recent_episodes:]
                
                # Calculate trends
                reward_trend = np.polyfit(range(len(recent_rewards)), recent_rewards, 1)[0]
                score_trend = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]
                
                print(f"\n🔍 Training Diagnostics Report - Episode {episode}")
                print("=" * 60)
                print(f"📈 Learning Progress:")
                print(f"   Reward trend: {reward_trend:.4f} {'✅' if reward_trend > 0 else '❌'}")
                print(f"   Score trend: {score_trend:.4f} {'✅' if score_trend > 0 else '❌'}")
                print(f"   Exploration decreasing: {'✅' if agent.epsilon < 0.5 else '❌'}")
                print(f"   Average reward: {np.mean(recent_rewards):.2f}")
                print(f"   Average score: {np.mean(recent_scores):.2f}")
                
                print(f"\n🎯 Performance Analysis:")
                print(f"   Win rate: {np.mean(recent_wins):.1%}")
                print(f"   Score range: {min(recent_scores)} - {max(recent_scores)}")
                print(f"   Score variance: {np.var(recent_scores):.2f}")
                
                # Recommendations
                print(f"\n💡 Recommendations:")
                if reward_trend < 0:
                    print("   1. ⚠️ Rewards decreasing - consider reducing learning rate further")
                if np.mean(recent_scores) < 200:
                    print("   2. 🎯 Low scores - consider improving reward shaping")
                if agent.epsilon > 0.3:
                    print("   3. 🔍 High exploration - training may need more episodes")
                if np.var(recent_scores) > 5000:
                    print("   4. 📊 High variance - consider stabilizing training")
    
    # Final evaluation and save
    print(f"\n🎯 Final Evaluation:")
    win_rate, avg_score, avg_reward = evaluate_agent(agent, env, episodes=100)
    print(f"   Final Win Rate: {win_rate:.1%}")
    print(f"   Final Avg Score: {avg_score:.1f}")
    print(f"   Final Avg Reward: {avg_reward:.2f}")
    
    # Save final model
    final_model_path = os.path.join("weights", "final_simple_model.pth")
    agent.save(final_model_path)
    mlflow_tracker.log_artifact(final_model_path, "models")
    
    # Final plots
    plotter.add_data_point(TRAINING_EPISODES, {
        'episode_rewards': episode_rewards[-1] if episode_rewards else 0,
        'episode_scores': episode_scores[-1] if episode_scores else 0,
        'win_rates': episode_wins[-1] if episode_wins else 0,
        'episode_lengths': episode_lengths[-1] if episode_lengths else 0,
        'epsilon_values': agent.epsilon,
        'buffer_sizes': len(agent.memory),
        # 'losses': losses[-1] if losses else 0
    })
    
    plotter.plot_final_summary(TRAINING_EPISODES)
    
    print(f"\n✅ Training completed!")
    print(f"📁 Models saved to: weights/")
    print(f"📊 Plots saved to: training_plots_simple/")
    print(f"📈 MLflow experiment: {mlflow_tracker.experiment_name}")

if __name__ == "__main__":
    main() 