#!/usr/bin/env python3
"""
PPO (Proximal Policy Optimization) Agent for Balatro Gym Environment
Implements a complete PPO algorithm with actor-critic architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import random
from collections import deque
import math
from tqdm import tqdm
import matplotlib.pyplot as plt

from balatro_gym_v2_simple import BalatroGymEnvSimple

class ActorCritic(nn.Module):
    """
    Actor-Critic neural network for PPO
    Actor: outputs action probabilities
    Critic: outputs state value
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(ActorCritic, self).__init__()
        
        # Shared layers
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Actor (policy) head
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )
        
        # Critic (value) head
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            module.bias.data.zero_()
    
    def forward(self, state):
        shared_features = self.shared_layers(state)
        
        # Actor: logits for action probabilities
        action_logits = self.actor(shared_features)
        
        # Critic: state value
        value = self.critic(shared_features)
        
        return action_logits, value
    
    def get_action_probs(self, state):
        """Get action probabilities from state"""
        action_logits, _ = self.forward(state)
        return F.softmax(action_logits, dim=-1)
    
    def get_value(self, state):
        """Get state value"""
        _, value = self.forward(state)
        return value

class PPOBuffer:
    """
    Buffer for storing PPO training data
    """
    
    def __init__(self, buffer_size: int, state_dim: int, device: str = "cpu"):
        self.buffer_size = buffer_size
        self.device = device
        
        # Storage
        self.states = torch.zeros((buffer_size, state_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros(buffer_size, dtype=torch.long, device=device)
        self.rewards = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.values = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.log_probs = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.dones = torch.zeros(buffer_size, dtype=torch.bool, device=device)
        
        self.ptr = 0
        self.size = 0
    
    def add(self, state, action, reward, value, log_prob, done):
        """Add a transition to the buffer"""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)
    
    def get_all(self):
        """Get all data from buffer"""
        return (
            self.states[:self.size],
            self.actions[:self.size],
            self.rewards[:self.size],
            self.values[:self.size],
            self.log_probs[:self.size],
            self.dones[:self.size]
        )
    
    def clear(self):
        """Clear the buffer"""
        self.ptr = 0
        self.size = 0

class PPOAgent:
    """
    PPO Agent for Balatro environment
    """
    
    def __init__(
        self,
        env: BalatroGymEnvSimple,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.02,
        max_grad_norm: float = 0.5,
        target_kl: float = 0.01,
        hidden_dim: int = 256,
        device: str = "cpu"
    ):
        self.env = env
        self.device = device
        
        # PPO hyperparameters
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        
        # Networks
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        
        self.actor_critic = ActorCritic(state_dim, action_dim, hidden_dim).to(device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        
        # Training stats
        self.training_stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'win_rates': [],
            'policy_losses': [],
            'value_losses': [],
            'entropy_losses': [],
            'kl_divergences': [],
            'advantages': [],
            'policy_ratios': [],
            'entropies': [],
            'action_distributions': [],
            'avg_advantages': [],
            'policy_confidence': []
        }
    
    def compute_gae(self, rewards, values, dones, next_value):
        """Compute Generalized Advantage Estimation"""
        advantages = torch.zeros_like(rewards)
        last_advantage = 0
        
        # Convert dones to float for arithmetic operations
        dones_float = dones.float()
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value_t = next_value
            else:
                next_value_t = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value_t * (1 - dones_float[t]) - values[t]
            advantages[t] = delta + self.gamma * self.gae_lambda * (1 - dones_float[t]) * last_advantage
            last_advantage = advantages[t]
        
        returns = advantages + values
        return advantages, returns
    
    def play_demo_episode(self, max_steps: int = 10):
        """Play a demonstration episode to show current policy"""
        obs, _ = self.env.reset()
        obs = torch.FloatTensor(obs).to(self.device)
        total_reward = 0
        step = 0
        
        print(f"  Initial hand: {[str(card) for card in self.env.hand]}")
        print(f"  Target score: {self.env.blind_score}")
        print()
        
        while step < max_steps:
            step += 1
            
            # Show current hand before action
            print(f"  Step {step} - Hand: {[str(card) for card in self.env.hand]}")
            
            # Get action from policy
            with torch.no_grad():
                action_logits, value = self.actor_critic(obs.unsqueeze(0))
                action_probs = torch.softmax(action_logits, dim=-1)
                action = torch.argmax(action_probs).item()
                action_prob = action_probs[0, action].item()
            
            # Decode action
            action_type, card_indices = self.env._decode_action(action)
            
            # Capture cards that will be played BEFORE taking the action
            cards_to_play = [str(self.env.hand[i]) for i in card_indices if i < len(self.env.hand)]
            
            # Take action
            obs, reward, done, truncated, info = self.env.step(action)
            obs = torch.FloatTensor(obs).to(self.device)
            total_reward += reward
            
            # Show action details
            print(f"    Action: {action_type.upper()} {cards_to_play} (prob: {action_prob:.3f}, reward: {reward:.2f})")
            print(f"    Score: {self.env.current_score}/{self.env.blind_score}, Plays: {self.env.plays_left}, Discards: {self.env.discards_left}")
            
            # Show new hand after the action
            print(f"    New hand: {[str(card) for card in self.env.hand]}")
            print()
            
            if done or truncated:
                break
        
        print(f"  Final result: {'WIN' if self.env.won else 'LOSS'} (Total reward: {total_reward:.2f})")
        print()
    
    def compute_loss(self, states, actions, old_log_probs, advantages, returns):
        """Compute PPO loss"""
        action_logits, values = self.actor_critic(states)
        
        # Policy loss
        action_probs = F.softmax(action_logits, dim=-1)
        dist = torch.distributions.Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        value_loss = F.mse_loss(values.squeeze(), returns)
        
        # Entropy loss (for exploration)
        entropy_loss = -dist.entropy().mean()
        
        # Total loss
        total_loss = (
            policy_loss + 
            self.value_loss_coef * value_loss + 
            self.entropy_coef * entropy_loss
        )
        
        # KL divergence for early stopping
        kl_div = (old_log_probs - log_probs).mean()
        
        return total_loss, policy_loss, value_loss, entropy_loss, kl_div
    
    def collect_episode(self, max_steps: int = 1000) -> Tuple[List, List, List, List, List, List]:
        """Collect a single episode"""
        states, actions, rewards, values, log_probs, dones = [], [], [], [], [], []
        
        obs, _ = self.env.reset()
        obs = torch.FloatTensor(obs).to(self.device)
        
        for step in range(max_steps):
            # Get action from policy
            action_logits, value = self.actor_critic(obs.unsqueeze(0))
            action_probs = F.softmax(action_logits, dim=-1)
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            # Take action
            next_obs, reward, done, truncated, info = self.env.step(action.item())
            next_obs = torch.FloatTensor(next_obs).to(self.device)
            
            # Store transition
            states.append(obs)
            actions.append(action)
            rewards.append(reward)
            values.append(value.squeeze())
            log_probs.append(log_prob)
            dones.append(done or truncated)
            
            obs = next_obs
            
            if done or truncated:
                break
        
        return states, actions, rewards, values, log_probs, dones
    
    def collect_batch(self, batch_size: int, max_steps_per_episode: int = 1000) -> PPOBuffer:
        """Collect a batch of episodes"""
        buffer = PPOBuffer(batch_size, self.env.observation_space.shape[0], self.device)
        
        total_steps = 0
        while total_steps < batch_size:
            states, actions, rewards, values, log_probs, dones = self.collect_episode(max_steps_per_episode)
            
            # Convert to tensors
            states = torch.stack(states)
            actions = torch.stack(actions)
            rewards = torch.FloatTensor(rewards).to(self.device)
            values = torch.stack(values)
            log_probs = torch.stack(log_probs)
            dones = torch.BoolTensor(dones).to(self.device)
            
            # Compute GAE
            if dones[-1].item():  # Convert boolean tensor to Python bool
                next_value = 0.0
            else:
                with torch.no_grad():
                    next_obs = states[-1]
                    _, next_value = self.actor_critic(next_obs.unsqueeze(0))
                    next_value = next_value.squeeze()
            
            advantages, returns = self.compute_gae(rewards, values, dones, next_value)
            
            # Add to buffer
            for i in range(len(states)):
                if total_steps + i < batch_size:
                    buffer.add(
                        states[i].detach(), actions[i].detach(), rewards[i], 
                        values[i].detach(), log_probs[i].detach(), dones[i]
                    )
            
            total_steps += len(states)
        
        return buffer
    
    def update(self, buffer: PPOBuffer, epochs: int = 10) -> Dict[str, float]:
        """Update policy using PPO"""
        states, actions, rewards, values, log_probs, dones = buffer.get_all()
        
        # Compute GAE for the entire buffer
        if dones[-1].item():  # Convert boolean tensor to Python bool
            next_value = 0.0
        else:
            with torch.no_grad():
                next_obs = states[-1]
                _, next_value = self.actor_critic(next_obs.unsqueeze(0))
                next_value = next_value.squeeze()
        
        advantages, returns = self.compute_gae(rewards, values, dones, next_value)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        update_stats = {
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy_loss': 0.0,
            'kl_div': 0.0
        }
        
        num_updates = 0
        
        for epoch in range(epochs):
            # Shuffle data
            indices = torch.randperm(len(states))
            
            for start_idx in range(0, len(states), 64):  # Mini-batch size of 64
                end_idx = min(start_idx + 64, len(states))
                batch_indices = indices[start_idx:end_idx]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Compute loss
                total_loss, policy_loss, value_loss, entropy_loss, kl_div = self.compute_loss(
                    batch_states, batch_actions, batch_old_log_probs, 
                    batch_advantages, batch_returns
                )
                
                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Accumulate stats
                update_stats['policy_loss'] += policy_loss.item()
                update_stats['value_loss'] += value_loss.item()
                update_stats['entropy_loss'] += entropy_loss.item()
                update_stats['kl_div'] += kl_div.item()
                num_updates += 1
            
            # Early stopping if KL divergence is too high
            avg_kl = update_stats['kl_div'] / num_updates if num_updates > 0 else 0
            if avg_kl > self.target_kl:
                break
        
        # Average stats
        if num_updates > 0:
            for key in update_stats:
                update_stats[key] /= num_updates
        
        return update_stats
    
    def train(
        self, 
        total_timesteps: int = 1000000,
        batch_size: int = 2048,
        update_epochs: int = 10,
        eval_interval: int = 10000,
        num_eval_episodes: int = 10,
        save_interval: int = 50000,
        demo_interval: int = 20000
    ):
        """Train the PPO agent"""
        timesteps_so_far = 0
        episode_count = 0
        
        print(f"Starting PPO training for {total_timesteps} timesteps")
        print(f"Batch size: {batch_size}, Update epochs: {update_epochs}")
        print("=" * 60)
        
        with tqdm(total=total_timesteps, desc="Training Progress") as pbar:
            while timesteps_so_far < total_timesteps:
                # Collect batch
                buffer = self.collect_batch(batch_size)
                timesteps_so_far += buffer.size
                
                # Update policy
                update_stats = self.update(buffer, update_epochs)
                
                # Store training stats (every update, not just at eval intervals)
                self.training_stats['value_losses'].append(update_stats['value_loss'])
                self.training_stats['policy_losses'].append(update_stats['policy_loss'])
                self.training_stats['entropy_losses'].append(update_stats['entropy_loss'])
                self.training_stats['kl_divergences'].append(update_stats['kl_div'])
                
                # Evaluation
                if timesteps_so_far % eval_interval == 0:
                    eval_stats = self.evaluate(num_eval_episodes)
                    self.training_stats['episode_rewards'].append(eval_stats['avg_reward'])
                    self.training_stats['episode_lengths'].append(eval_stats['avg_length'])
                    self.training_stats['win_rates'].append(eval_stats['win_rate'])
                    
                    print(f"\n📊 Evaluation at {timesteps_so_far} timesteps:")
                    print(f"  Average Reward: {eval_stats['avg_reward']:.2f}")
                    print(f"  Win Rate: {eval_stats['win_rate']:.2%}")
                    print(f"  Average Length: {eval_stats['avg_length']:.1f}")
                    print(f"  Policy Loss: {update_stats['policy_loss']:.4f}")
                    print(f"  Value Loss: {update_stats['value_loss']:.4f}")
                    print(f"  Entropy Loss: {update_stats['entropy_loss']:.4f}")
                    print(f"  KL Divergence: {update_stats['kl_div']:.4f}")
                    print("-" * 40)
                
                # Demo episode
                if timesteps_so_far % demo_interval == 0:
                    print(f"\n🎮 Demo Episode at {timesteps_so_far} timesteps:")
                    self.play_demo_episode()
                
                # Save model and plot
                if timesteps_so_far % save_interval == 0:
                    model_path = f"ppo_balatro_{timesteps_so_far}.pth"
                    self.save_model(model_path)
                    print(f"\n💾 Model saved to {model_path}")
                    
                    # Plot current training curves
                    self.plot_training_curves(save_path=f"training_curves_{timesteps_so_far}.png")
                
                pbar.update(buffer.size)
                pbar.set_postfix({
                    'Policy Loss': f"{update_stats['policy_loss']:.4f}",
                    'Value Loss': f"{update_stats['value_loss']:.4f}",
                    'KL Div': f"{update_stats['kl_div']:.4f}",
                    'Timesteps': f"{timesteps_so_far}"
                })
        
        print("Training completed!")
        self.save_model("ppo_balatro_final.pth")
        self.plot_training_curves(save_path="final_training_curves.png")
    
    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """Evaluate the current policy"""
        rewards = []
        lengths = []
        wins = 0
        
        for _ in range(num_episodes):
            obs, _ = self.env.reset()
            obs = torch.FloatTensor(obs).to(self.device)
            episode_reward = 0
            episode_length = 0
            
            while True:
                with torch.no_grad():
                    action_logits, _ = self.actor_critic(obs.unsqueeze(0))
                    action_probs = F.softmax(action_logits, dim=-1)
                    action = torch.argmax(action_probs).item()
                
                obs, reward, done, truncated, info = self.env.step(action)
                obs = torch.FloatTensor(obs).to(self.device)
                
                episode_reward += reward
                episode_length += 1
                
                if done or truncated:
                    if info.get('won', False):
                        wins += 1
                    break
            
            rewards.append(episode_reward)
            lengths.append(episode_length)
        
        return {
            'avg_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards),
            'avg_length': np.mean(lengths),
            'win_rate': wins / num_episodes,
            'rewards': rewards,
            'lengths': lengths
        }
    
    def save_model(self, filename: str):
        """Save the model"""
        torch.save({
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_stats': self.training_stats
        }, filename)
        print(f"Model saved to {filename}")
    
    def load_model(self, filename: str):
        """Load the model"""
        checkpoint = torch.load(filename, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load training stats with compatibility handling
        loaded_stats = checkpoint.get('training_stats', {})
        for key in self.training_stats:
            if key in loaded_stats:
                self.training_stats[key] = loaded_stats[key]
            # If key doesn't exist in loaded stats, keep the default empty list
        
        print(f"Model loaded from {filename}")
    
    def plot_training_curves(self, save_path: str = "ppo_training_curves.png"):
        """Plot training curves with detailed statistics"""
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        
        # Episode rewards
        if self.training_stats['episode_rewards']:
            axes[0, 0].plot(self.training_stats['episode_rewards'], 'b-', linewidth=2)
            axes[0, 0].set_title('Episode Rewards', fontsize=14, fontweight='bold')
            axes[0, 0].set_xlabel('Evaluation Step')
            axes[0, 0].set_ylabel('Average Reward')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Win rate
        if self.training_stats['win_rates']:
            axes[0, 1].plot(self.training_stats['win_rates'], 'g-', linewidth=2)
            axes[0, 1].set_title('Win Rate', fontsize=14, fontweight='bold')
            axes[0, 1].set_xlabel('Evaluation Step')
            axes[0, 1].set_ylabel('Win Rate')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].set_ylim(0, 1)
        
        # Episode lengths
        if self.training_stats['episode_lengths']:
            axes[0, 2].plot(self.training_stats['episode_lengths'], 'r-', linewidth=2)
            axes[0, 2].set_title('Episode Lengths', fontsize=14, fontweight='bold')
            axes[0, 2].set_xlabel('Evaluation Step')
            axes[0, 2].set_ylabel('Average Length')
            axes[0, 2].grid(True, alpha=0.3)
        
        # Policy loss
        if self.training_stats['policy_losses']:
            axes[1, 0].plot(self.training_stats['policy_losses'], 'purple', linewidth=2, label='Policy Loss')
            axes[1, 0].set_title('Policy Loss', fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('Update Step')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].legend()
        
        # Value loss
        if self.training_stats['value_losses']:
            axes[1, 1].plot(self.training_stats['value_losses'], 'orange', linewidth=2, label='Value Loss')
            axes[1, 1].set_title('Value Loss', fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel('Update Step')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].legend()
        
        # Entropy loss
        if self.training_stats['entropy_losses']:
            axes[1, 2].plot(self.training_stats['entropy_losses'], 'brown', linewidth=2, label='Entropy Loss')
            axes[1, 2].set_title('Entropy Loss', fontsize=14, fontweight='bold')
            axes[1, 2].set_xlabel('Update Step')
            axes[1, 2].set_ylabel('Loss')
            axes[1, 2].grid(True, alpha=0.3)
            axes[1, 2].legend()
        
        # KL divergence
        if self.training_stats['kl_divergences']:
            axes[2, 0].plot(self.training_stats['kl_divergences'], 'teal', linewidth=2, label='KL Divergence')
            axes[2, 0].set_title('KL Divergence', fontsize=14, fontweight='bold')
            axes[2, 0].set_xlabel('Update Step')
            axes[2, 0].set_ylabel('KL Div')
            axes[2, 0].grid(True, alpha=0.3)
            axes[2, 0].legend()
        
        # Combined losses
        if (self.training_stats['policy_losses'] and 
            self.training_stats['value_losses'] and 
            self.training_stats['entropy_losses']):
            axes[2, 1].plot(self.training_stats['policy_losses'], 'purple', linewidth=2, label='Policy')
            axes[2, 1].plot(self.training_stats['value_losses'], 'orange', linewidth=2, label='Value')
            axes[2, 1].plot(self.training_stats['entropy_losses'], 'brown', linewidth=2, label='Entropy')
            axes[2, 1].set_title('All Training Losses', fontsize=14, fontweight='bold')
            axes[2, 1].set_xlabel('Update Step')
            axes[2, 1].set_ylabel('Loss')
            axes[2, 1].grid(True, alpha=0.3)
            axes[2, 1].legend()
        
        # Training summary
        if self.training_stats['policy_losses']:
            total_updates = len(self.training_stats['policy_losses'])
            avg_policy_loss = np.mean(self.training_stats['policy_losses'][-10:]) if total_updates >= 10 else np.mean(self.training_stats['policy_losses'])
            avg_value_loss = np.mean(self.training_stats['value_losses'][-10:]) if total_updates >= 10 else np.mean(self.training_stats['value_losses'])
            
            axes[2, 2].text(0.1, 0.8, f'Total Updates: {total_updates}', fontsize=12, transform=axes[2, 2].transAxes)
            axes[2, 2].text(0.1, 0.7, f'Avg Policy Loss: {avg_policy_loss:.4f}', fontsize=12, transform=axes[2, 2].transAxes)
            axes[2, 2].text(0.1, 0.6, f'Avg Value Loss: {avg_value_loss:.4f}', fontsize=12, transform=axes[2, 2].transAxes)
            
            if self.training_stats['win_rates']:
                latest_win_rate = self.training_stats['win_rates'][-1]
                axes[2, 2].text(0.1, 0.5, f'Latest Win Rate: {latest_win_rate:.2%}', fontsize=12, transform=axes[2, 2].transAxes)
            
            if self.training_stats['episode_rewards']:
                latest_reward = self.training_stats['episode_rewards'][-1]
                axes[2, 2].text(0.1, 0.4, f'Latest Avg Reward: {latest_reward:.2f}', fontsize=12, transform=axes[2, 2].transAxes)
            
            axes[2, 2].set_title('Training Summary', fontsize=14, fontweight='bold')
            axes[2, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Training curves saved to {save_path}")
        plt.show()

def main():
    """Main training function"""
    # Create environment
    env = BalatroGymEnvSimple(blind_score=300)
    
    # Create PPO agent
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    agent = PPOAgent(
        env=env,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        value_loss_coef=0.5,
        entropy_coef=0.02,
        max_grad_norm=0.5,
        target_kl=0.01,
        hidden_dim=256,
        device=device
    )
    
    # Train the agent
    agent.train(
        total_timesteps=500000,  # Adjust based on your needs
        batch_size=2048,
        update_epochs=10,
        eval_interval=10000,
        num_eval_episodes=10,
        save_interval=50000
    )
    
    # Plot training curves
    agent.plot_training_curves()
    
    # Final evaluation
    print("\nFinal Evaluation:")
    eval_stats = agent.evaluate(100)
    print(f"Average Reward: {eval_stats['avg_reward']:.2f} ± {eval_stats['std_reward']:.2f}")
    print(f"Win Rate: {eval_stats['win_rate']:.2%}")
    print(f"Min/Max Reward: {eval_stats['min_reward']:.2f}/{eval_stats['max_reward']:.2f}")

if __name__ == "__main__":
    main() 