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
import time

# Check for TPU availability
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    TPU_AVAILABLE = True
except ImportError:
    TPU_AVAILABLE = False

# Check for GPU availability
GPU_AVAILABLE = torch.cuda.is_available()

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from balatro_gym_v2_simple import BalatroGymEnvSimple
from training_plots import TrainingPlotter

# Hyperparameters optimized for simplified environment
LEARNING_RATE = 0.00005  # Further reduced for stability
BATCH_SIZE = 128 if GPU_AVAILABLE or TPU_AVAILABLE else 64  # Larger batch size for GPU/TPU
GAMMA = 0.99

# Epsilon decay strategies
EPSILON_START = 1.0
EPSILON_END = 0.1  # Increased minimum exploration from 0.01 to 0.1
EPSILON_DECAY_TYPE = "curriculum"  # Changed to curriculum for staged exploration

# Manual epsilon override for aggressive exploration (set to None to use decay)
MANUAL_EPSILON = 0.15  # Reduced to 0.15 to reduce random High Card plays

# For exponential decay
EPSILON_DECAY_RATE = 0.9999  # Even slower decay for more exploration

# For adaptive decay (based on performance)
ADAPTIVE_DECAY_MIN_EPISODES = 1000  # Minimum episodes before adaptive decay
ADAPTIVE_DECAY_PERFORMANCE_THRESHOLD = 0.1  # Win rate threshold for faster decay

# For curriculum decay (staged learning)
CURRICULUM_STAGES = [
    {"episodes": 10000, "epsilon": 0.9},   # Very high exploration initially
    {"episodes": 25000, "epsilon": 0.7},   # High exploration
    {"episodes": 50000, "epsilon": 0.4},   # Medium exploration
    {"episodes": 75000, "epsilon": 0.2},   # Lower exploration
    {"episodes": float('inf'), "epsilon": 0.1}  # Minimum exploration (increased)
]

MEMORY_SIZE = 50000 if GPU_AVAILABLE or TPU_AVAILABLE else 10000  # Larger memory for GPU/TPU
TARGET_UPDATE = 100
TRAINING_EPISODES = 100000
EVAL_INTERVAL = 5000  # Further reduced evaluation frequency to avoid slowdowns
PLOT_INTERVAL = 500

# GPU/TPU optimization settings
MIXED_PRECISION = True  # Use mixed precision training
GRADIENT_ACCUMULATION_STEPS = 2 if GPU_AVAILABLE or TPU_AVAILABLE else 1  # Gradient accumulation

class SmartEpsilonDecay:
    """Advanced epsilon decay strategies for better exploration-exploitation balance"""
    
    def __init__(self, decay_type="exponential", **kwargs):
        self.decay_type = decay_type
        self.steps = 0
        self.episodes = 0
        self.current_epsilon = EPSILON_START
        
        # Performance tracking for adaptive decay
        self.recent_performances = deque(maxlen=100)
        self.performance_history = []
        
        # Curriculum stage tracking
        self.curriculum_stage = 0
        
        # Decay parameters
        if decay_type == "linear":
            self.decay_rate = kwargs.get('decay_rate', 0.0001)
        elif decay_type == "exponential":
            self.decay_rate = kwargs.get('decay_rate', EPSILON_DECAY_RATE)
        elif decay_type == "adaptive":
            self.min_episodes = kwargs.get('min_episodes', ADAPTIVE_DECAY_MIN_EPISODES)
            self.performance_threshold = kwargs.get('performance_threshold', ADAPTIVE_DECAY_PERFORMANCE_THRESHOLD)
        elif decay_type == "curriculum":
            self.stages = kwargs.get('stages', CURRICULUM_STAGES)
        
        # Ensure decay_rate is always initialized
        if not hasattr(self, 'decay_rate'):
            self.decay_rate = EPSILON_DECAY_RATE
    
    def update(self, episode_performance=None):
        """Update epsilon based on the chosen decay strategy"""
        self.steps += 1
        
        if self.decay_type == "linear":
            self.current_epsilon = max(EPSILON_END, EPSILON_START - self.decay_rate * self.steps)
            
        elif self.decay_type == "exponential":
            if self.current_epsilon > EPSILON_END:
                self.current_epsilon *= self.decay_rate
                self.current_epsilon = max(EPSILON_END, self.current_epsilon)
                
        elif self.decay_type == "adaptive":
            if episode_performance is not None:
                self.recent_performances.append(episode_performance)
                self.performance_history.append(episode_performance)
            
            if self.episodes >= self.min_episodes and len(self.recent_performances) >= 50:
                avg_performance = np.mean(list(self.recent_performances))
                
                # Faster decay if performing well
                if avg_performance > self.performance_threshold:
                    decay_rate = 0.999  # Fast decay
                else:
                    decay_rate = 0.9995  # Slow decay
                
                if self.current_epsilon > EPSILON_END:
                    self.current_epsilon *= decay_rate
                    self.current_epsilon = max(EPSILON_END, self.current_epsilon)
                    
        elif self.decay_type == "curriculum":
            # Find current stage
            current_stage = None
            for stage in self.stages:
                if self.episodes <= stage["episodes"]:
                    current_stage = stage
                    break
            
            if current_stage:
                target_epsilon = current_stage["epsilon"]
                # Smooth transition to target epsilon
                if abs(self.current_epsilon - target_epsilon) > 0.01:
                    self.current_epsilon += (target_epsilon - self.current_epsilon) * 0.01
                else:
                    self.current_epsilon = target_epsilon
    
    def get_epsilon(self):
        return self.current_epsilon
    
    def get_decay_info(self):
        """Get information about current decay state"""
        if self.decay_type == "adaptive" and len(self.performance_history) > 0:
            recent_avg = np.mean(list(self.recent_performances)) if self.recent_performances else 0
            return {
                "type": self.decay_type,
                "epsilon": self.current_epsilon,
                "steps": self.steps,
                "episodes": self.episodes,
                "recent_performance": recent_avg,
                "performance_threshold": self.performance_threshold
            }
        elif self.decay_type == "curriculum":
            current_stage = None
            for stage in self.stages:
                if self.episodes <= stage["episodes"]:
                    current_stage = stage
                    break
            return {
                "type": self.decay_type,
                "epsilon": self.current_epsilon,
                "episodes": self.episodes,
                "current_stage": current_stage["episodes"] if current_stage else "final"
            }
        else:
            return {
                "type": self.decay_type,
                "epsilon": self.current_epsilon,
                "steps": self.steps
            }

# Simplified DQN Network for 17-dimensional state
class SimpleDQN(nn.Module):
    def __init__(self, state_size: int, action_size: int):
        super(SimpleDQN, self).__init__()
        
        # Optimized network for GPU/TPU
        self.fc1 = nn.Linear(state_size, 128)  # Reduced for smaller state
        self.fc2 = nn.Linear(128, 128)         # Reduced for smaller state
        self.fc3 = nn.Linear(128, 64)          # Reduced for smaller state
        self.fc4 = nn.Linear(64, action_size)
        
        # Batch normalization for faster training
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(64)
        
        self.dropout = nn.Dropout(0.1)  # Add dropout for regularization
        
        # Initialize weights for better convergence
        self._init_weights()
        
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        
    def forward(self, x):
        # Handle both single samples and batches
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        x = self.fc1(x)
        if x.size(0) > 1:  # Only apply batch norm for batches
            x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        if x.size(0) > 1:
            x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        if x.size(0) > 1:
            x = self.bn3(x)
        x = torch.relu(x)
        x = self.fc4(x)
        
        return x.squeeze(0) if x.size(0) == 1 else x

class SimpleDQNAgent:
    def __init__(self, state_size: int, action_size: int, device: str = 'cpu'):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        
        # Networks
        self.q_network = SimpleDQN(state_size, action_size).to(device)
        self.target_network = SimpleDQN(state_size, action_size).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer with reduced learning rate and weight decay
        self.optimizer = optim.AdamW(
            self.q_network.parameters(), 
            lr=LEARNING_RATE, 
            weight_decay=1e-3,  # Increased weight decay for regularization
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.7, patience=500, verbose=True, min_lr=1e-6
        )
        
        # Mixed precision training
        if MIXED_PRECISION and (GPU_AVAILABLE or TPU_AVAILABLE):
            self.scaler = torch.cuda.amp.GradScaler() if GPU_AVAILABLE else None
        else:
            self.scaler = None
        
        # Training parameters
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon_decay = SmartEpsilonDecay(decay_type=EPSILON_DECAY_TYPE)
        self.epsilon = self.epsilon_decay.get_epsilon()
        self.steps = 0
        self.gradient_accumulation_counter = 0
        
        # Experience tuple
        self.Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append(self.Experience(state, action, reward, next_state, done))
    
    def act(self, state, training=True):
        # Use manual epsilon if set, otherwise use decayed epsilon
        current_epsilon = MANUAL_EPSILON if MANUAL_EPSILON is not None else self.epsilon
        
        if training and random.random() < current_epsilon:
            # During exploration, avoid High Card actions
            valid_actions = list(range(self.action_size))
            # Filter out actions that would result in High Card (this will be handled by environment)
            return random.choice(valid_actions)
        
        with torch.no_grad():
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
        
        # Use mixed precision if available
        if self.scaler is not None:
            with torch.cuda.amp.autocast():
                current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
                next_q_values = self.target_network(next_states).max(1)[0].detach()
                target_q_values = rewards + (GAMMA * next_q_values * ~dones)
                loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
            
            # Scale loss and backward pass
            self.scaler.scale(loss).backward()
            
            # Gradient accumulation
            self.gradient_accumulation_counter += 1
            if self.gradient_accumulation_counter % GRADIENT_ACCUMULATION_STEPS == 0:
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=0.5)  # More conservative clipping
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
        else:
            # Standard training without mixed precision
            current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
            next_q_values = self.target_network(next_states).max(1)[0].detach()
            target_q_values = rewards + (GAMMA * next_q_values * ~dones)
            loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
            
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=0.5)  # More conservative clipping
            self.optimizer.step()
        
        # Update epsilon using smart decay (will be called with performance in training loop)
        # Don't update here for adaptive decay - it's updated in the main training loop
        if self.epsilon_decay.decay_type != "adaptive":
            self.epsilon_decay.update()
            self.epsilon = self.epsilon_decay.get_epsilon()
        
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
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epsilon': self.epsilon,
            'epsilon_decay_state': {
                'decay_type': self.epsilon_decay.decay_type,
                'steps': self.epsilon_decay.steps,
                'episodes': self.epsilon_decay.episodes,
                'current_epsilon': self.epsilon_decay.current_epsilon,
                'recent_performances': list(self.epsilon_decay.recent_performances),
                'performance_history': self.epsilon_decay.performance_history
            },
            'steps': self.steps,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None
        }, filepath)
    
    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
        
        # Load epsilon decay state if available
        if 'epsilon_decay_state' in checkpoint:
            decay_state = checkpoint['epsilon_decay_state']
            self.epsilon_decay.decay_type = decay_state['decay_type']
            self.epsilon_decay.steps = decay_state['steps']
            self.epsilon_decay.episodes = decay_state['episodes']
            self.epsilon_decay.current_epsilon = decay_state['current_epsilon']
            self.epsilon_decay.recent_performances = deque(decay_state['recent_performances'], maxlen=100)
            self.epsilon_decay.performance_history = decay_state['performance_history']
        
        if 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict'] and self.scaler:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

def evaluate_agent(agent, env, episodes=100):
    """Evaluate agent performance (with timeout protection)"""
    wins = 0
    total_scores = []
    total_rewards = []
    
    for episode in range(episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        step_count = 0
        max_steps = 50  # Prevent infinite loops
        
        while not done and step_count < max_steps:
            action = agent.act(obs, training=False)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1
        
        if info.get('won', False):
            wins += 1
        total_scores.append(info.get('total_score', 0))
        total_rewards.append(episode_reward)
    
    win_rate = wins / episodes
    avg_score = np.mean(total_scores)
    avg_reward = np.mean(total_rewards)
    
    return win_rate, avg_score, avg_reward

def main():
    print("🎰 Training Simplified Balatro DQN Agent (GPU/TPU Optimized)")
    print("=" * 50)
    
    # Setup environment and agent
    env = BalatroGymEnvSimple()  # Increased score requirement for 10 plays
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    print(f"State size: {state_size}")
    print(f"Action size: {action_size}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Epsilon decay type: {EPSILON_DECAY_TYPE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Memory size: {MEMORY_SIZE}")
    
    # Device setup
    if TPU_AVAILABLE:
        device = xm.xla_device()
        print(f"🚀 Using TPU: {device}")
    elif GPU_AVAILABLE:
        device = torch.device("cuda")
        print(f"🎮 Using GPU: {torch.cuda.get_device_name()}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        print(f"💻 Using CPU")
    
    print(f"Mixed precision: {MIXED_PRECISION and (GPU_AVAILABLE or TPU_AVAILABLE)}")
    print(f"Gradient accumulation steps: {GRADIENT_ACCUMULATION_STEPS}")
    
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
        save_dir="training_plots_simple"
    )
    
    # Training loop
    episode_rewards = []
    episode_scores = []
    episode_wins = []
    episode_lengths = []
    losses = []
    
    # Track hand types for analysis
    episode_hand_types = []  # Store hand types for each episode
    hand_type_counts = {
        "High Card": 0, "Pair": 0, "Two Pair": 0, "Three of a Kind": 0,
        "Straight": 0, "Flush": 0, "Full House": 0, "Four of a Kind": 0,
        "Straight Flush": 0, "Royal Flush": 0
    }
    
    best_win_rate = 0.0
    start_time = time.time()
    
    print(f"\n🚀 Starting training for {TRAINING_EPISODES} episodes...")
    print("-" * 50)
    
    for episode in range(1, TRAINING_EPISODES + 1):
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        episode_hands = []  # Track hands played in this episode
        done = False
        
        step_count = 0
        max_steps_per_episode = 50  # Prevent infinite loops
        
        while not done and step_count < max_steps_per_episode:
            action = agent.act(obs, training=True)
            next_obs, reward, done, truncated, info = env.step(action)
            
            agent.remember(obs, action, reward, next_obs, done)
            obs = next_obs
            episode_reward += reward
            episode_length += 1
            step_count += 1
            
            # Force episode end if too many steps
            if step_count >= max_steps_per_episode:
                done = True
                reward -= 10.0  # Penalty for timeout
            
            # Train on batch
            if len(agent.memory) > BATCH_SIZE:
                loss = agent.replay(BATCH_SIZE)
                if loss is not None:
                    losses.append(loss)
            
            # Track hand type if this was a play action
            if info.get('action_type') == 'play' and 'hand_type' in info:
                episode_hands.append(info['hand_type'])
        
        # Record episode data
        episode_rewards.append(episode_reward)
        episode_scores.append(info.get('total_score', 0))
        episode_wins.append(1 if info.get('won', False) else 0)
        episode_lengths.append(episode_length)
        episode_hand_types.append(episode_hands)
        
        # Update global hand type counts
        for hand_type in episode_hands:
            if hand_type in hand_type_counts:
                hand_type_counts[hand_type] += 1
        
        # Update epsilon decay with episode performance (for adaptive decay)
        # Only update if not using manual epsilon
        if MANUAL_EPSILON is None:
            episode_performance = 1 if info.get('won', False) else 0  # Win rate as performance metric
            agent.epsilon_decay.episodes = episode
            agent.epsilon_decay.update(episode_performance=episode_performance)
            agent.epsilon = agent.epsilon_decay.get_epsilon()
        else:
            # Use manual epsilon
            agent.epsilon = MANUAL_EPSILON
        
        # Print progress
        if episode % 100 == 0:
            # Add debugging info for stuck episodes
            if step_count >= max_steps_per_episode:
                print(f"⚠️ Episode {episode} hit step limit ({max_steps_per_episode}) - possible infinite loop")
            recent_rewards = episode_rewards[-100:]
            recent_wins = episode_wins[-100:]
            recent_scores = episode_scores[-100:]
            recent_hand_types = episode_hand_types[-100:] if episode_hand_types else []
            
            avg_reward = np.mean(recent_rewards)
            win_rate = np.mean(recent_wins)
            avg_score = np.mean(recent_scores)
            
            # Calculate hand type statistics for recent episodes
            recent_hand_counts = {}
            for hand_list in recent_hand_types:
                for hand_type in hand_list:
                    recent_hand_counts[hand_type] = recent_hand_counts.get(hand_type, 0) + 1
            
            # Get most common hand types
            if recent_hand_counts:
                sorted_hands = sorted(recent_hand_counts.items(), key=lambda x: x[1], reverse=True)
                top_hands = [f"{hand}:{count}" for hand, count in sorted_hands[:3]]
                hand_summary = ", ".join(top_hands)
            else:
                hand_summary = "No hands played"
            
            elapsed_time = time.time() - start_time
            episodes_per_sec = episode / elapsed_time
            
            # Get current epsilon (manual or decayed)
            current_epsilon = MANUAL_EPSILON if MANUAL_EPSILON is not None else agent.epsilon
            
            print(f"Episode {episode:5d} | "
                  f"Avg Reward: {avg_reward:6.2f} | "
                  f"Win Rate: {win_rate:5.1%} | "
                  f"Avg Score: {avg_score:6.1f} | "
                  f"Epsilon: {current_epsilon:.3f} | "
                  f"Speed: {episodes_per_sec:.1f} ep/s")
            print(f"  Recent Hands: {hand_summary}")
        
        # Evaluation
        if episode % EVAL_INTERVAL == 0:
            win_rate, avg_score, avg_reward = evaluate_agent(agent, env, episodes=25)  # Reduced evaluation episodes
            
            print(f"\n📊 Evaluation at Episode {episode}:")
            print(f"   Win Rate: {win_rate:.1%}")
            print(f"   Avg Score: {avg_score:.1f}")
            print(f"   Avg Reward: {avg_reward:.2f}")
            

            
            # Save best model
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                model_path = os.path.join("weights", "best_simple_model.pth")
                os.makedirs("weights", exist_ok=True)
                agent.save(model_path)
                print(f"   🏆 New best model saved! Win rate: {win_rate:.1%}")
                
                # Update learning rate scheduler
                agent.scheduler.step(win_rate)
        
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
        
        # Diagnostics removed for faster training
    
    # Final evaluation and save
    total_time = time.time() - start_time
    print(f"\n🎯 Final Evaluation:")
    win_rate, avg_score, avg_reward = evaluate_agent(agent, env, episodes=50)  # Reduced final evaluation episodes
    print(f"   Final Win Rate: {win_rate:.1%}")
    print(f"   Final Avg Score: {avg_score:.1f}")
    print(f"   Final Avg Reward: {avg_reward:.2f}")
    print(f"   Total Training Time: {total_time/3600:.1f} hours")
    print(f"   Average Speed: {TRAINING_EPISODES/total_time:.1f} episodes/second")
    
    # Print hand type statistics
    print(f"\n🃏 Hand Type Statistics (All Training):")
    total_hands = sum(hand_type_counts.values())
    if total_hands > 0:
        sorted_hands = sorted(hand_type_counts.items(), key=lambda x: x[1], reverse=True)
        for hand_type, count in sorted_hands:
            percentage = (count / total_hands) * 100
            print(f"   {hand_type:15s}: {count:4d} ({percentage:5.1f}%)")
    else:
        print("   No hands played during training")
    
    # Save final model
    final_model_path = os.path.join("weights", "final_simple_model.pth")
    agent.save(final_model_path)
    
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
    print(f"⚡ Performance: {TRAINING_EPISODES/total_time:.1f} episodes/second")
    if GPU_AVAILABLE:
        print(f"🎮 GPU Memory Used: {torch.cuda.max_memory_allocated()/1e9:.1f} GB")

if __name__ == "__main__":
    main() 