#!/usr/bin/env python3
"""
Curriculum Learning Training Script for Simplified Balatro Gym Environment
Starts with easier targets and gradually increases difficulty for better learning
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

# Curriculum Learning Parameters
CURRICULUM_TARGETS = [100, 150, 200, 250, 300]  # Progressive difficulty
CURRICULUM_EPISODES = 5000  # Episodes per curriculum level
CURRICULUM_WIN_RATE_THRESHOLD = 0.15  # Win rate needed to advance

# Hyperparameters optimized for curriculum learning
LEARNING_RATE = 0.0001  # Conservative learning rate
BATCH_SIZE = 64
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.999  # Very slow decay
MEMORY_SIZE = 10000
TARGET_UPDATE = 100
TRAINING_EPISODES = 25000  # Total episodes across all curriculum levels
EVAL_INTERVAL = 500
PLOT_INTERVAL = 250
DIAGNOSTICS_INTERVAL = 1000

# Simplified DQN Network for 23-dimensional state
class SimpleDQN(nn.Module):
    def __init__(self, state_size: int, action_size: int):
        super(SimpleDQN, self).__init__()
        
        # Smaller network for simplified state space
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, action_size)
        
        self.dropout = nn.Dropout(0.1)
        
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
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=0.5)
        
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
    print("🎓 Curriculum Learning for Simplified Balatro DQN Agent")
    print("=" * 60)
    
    # Setup MLflow
    mlflow_tracker = MLflowTracker(
        experiment_name="balatro_curriculum_dqn"
    )
    mlflow_tracker.start_run(run_name=f"curriculum_dqn_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    # Setup environment and agent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Curriculum learning loop
    current_curriculum_level = 0
    total_episodes = 0
    best_win_rate = 0.0
    
    # Setup plotting
    plotter = TrainingPlotter(
        save_dir="training_plots_curriculum",
        mlflow_tracker=mlflow_tracker
    )
    
    while current_curriculum_level < len(CURRICULUM_TARGETS) and total_episodes < TRAINING_EPISODES:
        target_score = CURRICULUM_TARGETS[current_curriculum_level]
        
        print(f"\n📚 Curriculum Level {current_curriculum_level + 1}/{len(CURRICULUM_TARGETS)}")
        print(f"🎯 Target Score: {target_score}")
        print(f"📊 Total Episodes: {total_episodes}")
        print("-" * 50)
        
        # Create environment with current target
        env = BalatroGymEnvSimple(blind_score=target_score)
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n
        
        # Create agent (or reuse if not first level)
        if current_curriculum_level == 0:
            agent = SimpleDQNAgent(state_size, action_size, device)
        else:
            # Load previous best model and continue training
            best_model_path = os.path.join("weights", f"best_curriculum_level_{current_curriculum_level-1}.pth")
            if os.path.exists(best_model_path):
                agent = SimpleDQNAgent(state_size, action_size, device)
                agent.load(best_model_path)
                print(f"📂 Loaded model from previous curriculum level")
            else:
                agent = SimpleDQNAgent(state_size, action_size, device)
        
        # Training loop for current curriculum level
        episode_rewards = []
        episode_scores = []
        episode_wins = []
        episode_lengths = []
        losses = []
        
        level_episodes = 0
        level_best_win_rate = 0.0
        
        while level_episodes < CURRICULUM_EPISODES:
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
            mlflow_tracker.log_custom_metric('episode_reward', episode_reward, step=total_episodes)
            mlflow_tracker.log_custom_metric('episode_score', info.get('total_score', 0), step=total_episodes)
            mlflow_tracker.log_custom_metric('episode_won', 1 if info.get('won', False) else 0, step=total_episodes)
            mlflow_tracker.log_custom_metric('episode_length', episode_length, step=total_episodes)
            mlflow_tracker.log_custom_metric('epsilon', agent.epsilon, step=total_episodes)
            mlflow_tracker.log_custom_metric('memory_size', len(agent.memory), step=total_episodes)
            mlflow_tracker.log_custom_metric('curriculum_level', current_curriculum_level, step=total_episodes)
            mlflow_tracker.log_custom_metric('target_score', target_score, step=total_episodes)
            
            level_episodes += 1
            total_episodes += 1
            
            # Print progress
            if level_episodes % 100 == 0:
                recent_rewards = episode_rewards[-100:]
                recent_wins = episode_wins[-100:]
                recent_scores = episode_scores[-100:]
                
                avg_reward = np.mean(recent_rewards)
                win_rate = np.mean(recent_wins)
                avg_score = np.mean(recent_scores)
                
                print(f"Level {current_curriculum_level + 1} Episode {level_episodes:4d} | "
                      f"Avg Reward: {avg_reward:6.2f} | "
                      f"Win Rate: {win_rate:5.1%} | "
                      f"Avg Score: {avg_score:6.1f} | "
                      f"Epsilon: {agent.epsilon:.3f}")
            
            # Evaluation
            if level_episodes % EVAL_INTERVAL == 0:
                win_rate, avg_score, avg_reward = evaluate_agent(agent, env, episodes=50)
                
                print(f"\n📊 Evaluation at Level {current_curriculum_level + 1}, Episode {level_episodes}:")
                print(f"   Win Rate: {win_rate:.1%}")
                print(f"   Avg Score: {avg_score:.1f}")
                print(f"   Avg Reward: {avg_reward:.2f}")
                
                # Log evaluation metrics
                mlflow_tracker.log_custom_metric('eval_win_rate', win_rate, step=total_episodes)
                mlflow_tracker.log_custom_metric('eval_avg_score', avg_score, step=total_episodes)
                mlflow_tracker.log_custom_metric('eval_avg_reward', avg_reward, step=total_episodes)
                
                # Save best model for current level
                if win_rate > level_best_win_rate:
                    level_best_win_rate = win_rate
                    model_path = os.path.join("weights", f"best_curriculum_level_{current_curriculum_level}.pth")
                    os.makedirs("weights", exist_ok=True)
                    agent.save(model_path)
                    mlflow_tracker.log_artifact(model_path, "models")
                    print(f"   🏆 New best model for level {current_curriculum_level + 1}! Win rate: {win_rate:.1%}")
                
                # Check if ready to advance to next curriculum level
                if win_rate >= CURRICULUM_WIN_RATE_THRESHOLD:
                    print(f"   🎓 Ready to advance to next curriculum level! Win rate: {win_rate:.1%}")
                    break
            
            # Plotting
            if level_episodes % PLOT_INTERVAL == 0:
                plotter.add_data_point(total_episodes, {
                    'episode_rewards': episode_reward,
                    'episode_scores': info.get('total_score', 0),
                    'win_rates': 1 if info.get('won', False) else 0,
                    'episode_lengths': episode_length,
                    'epsilon_values': agent.epsilon,
                    'buffer_sizes': len(agent.memory),
                })
                
                plotter.plot_training_progress(total_episodes)
        
        # End of curriculum level
        print(f"\n📚 Completed Curriculum Level {current_curriculum_level + 1}")
        print(f"   Final Win Rate: {level_best_win_rate:.1%}")
        print(f"   Episodes Trained: {level_episodes}")
        
        # Advance to next curriculum level
        current_curriculum_level += 1
        
        # Save final model for this level
        final_model_path = os.path.join("weights", f"final_curriculum_level_{current_curriculum_level-1}.pth")
        agent.save(final_model_path)
        mlflow_tracker.log_artifact(final_model_path, "models")
    
    # Final evaluation
    print(f"\n🎯 Final Evaluation:")
    final_env = BalatroGymEnvSimple(blind_score=300)  # Final target
    win_rate, avg_score, avg_reward = evaluate_agent(agent, final_env, episodes=100)
    print(f"   Final Win Rate: {win_rate:.1%}")
    print(f"   Final Avg Score: {avg_score:.1f}")
    print(f"   Final Avg Reward: {avg_reward:.2f}")
    
    # Save final model
    final_model_path = os.path.join("weights", "final_curriculum_model.pth")
    agent.save(final_model_path)
    mlflow_tracker.log_artifact(final_model_path, "models")
    
    # Final plots
    plotter.add_data_point(total_episodes, {
        'episode_rewards': episode_rewards[-1] if episode_rewards else 0,
        'episode_scores': episode_scores[-1] if episode_scores else 0,
        'win_rates': episode_wins[-1] if episode_wins else 0,
        'episode_lengths': episode_lengths[-1] if episode_lengths else 0,
        'epsilon_values': agent.epsilon,
        'buffer_sizes': len(agent.memory),
    })
    
    plotter.plot_final_summary(total_episodes)
    
    print(f"\n✅ Curriculum Learning completed!")
    print(f"📁 Models saved to: weights/")
    print(f"📊 Plots saved to: training_plots_curriculum/")
    print(f"📈 MLflow experiment: {mlflow_tracker.experiment_name}")

if __name__ == "__main__":
    main() 