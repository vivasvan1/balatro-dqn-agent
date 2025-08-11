#!/usr/bin/env python3
"""
Simple PPO Test using CartPole environment
This script tests the PPO implementation on a standard Gym environment
to ensure the algorithm is working correctly before applying it to Balatro.
"""

import torch
import torch.nn.functional as F
import numpy as np
import argparse
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, List, Tuple, Any
import os
import gymnasium as gym

from ppo_agent import ActorCritic, PPOBuffer

class SimpleGymEnv:
    """Wrapper to make standard gym environment compatible with our PPO agent"""
    
    def __init__(self, env_name: str = "CartPole-v1"):
        self.env = gym.make(env_name)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.env_name = env_name
        
    def reset(self):
        obs, info = self.env.reset()
        return obs, info
    
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        return obs, reward, done, truncated, info
    
    def close(self):
        self.env.close()

class SimplePPOTrainer:
    """Simple PPO trainer for standard gym environments"""
    
    def __init__(
        self,
        env_name: str = "CartPole-v1",
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        target_kl: float = 0.01,
        hidden_dim: int = 64,
        device: str = "auto"
    ):
        self.env_name = env_name
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create environment
        self.env = SimpleGymEnv(env_name)
        
        # Create agent (we'll create a simplified version)
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n
        
        self.actor_critic = ActorCritic(state_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        
        # PPO hyperparameters
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        
        # Training stats
        self.training_stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'policy_losses': [],
            'value_losses': [],
            'entropy_losses': [],
            'kl_divergences': [],
        }
        
        print(f"🎯 Simple PPO Trainer Initialized")
        print(f"  Environment: {env_name}")
        print(f"  Device: {self.device}")
        print(f"  State Dim: {state_dim}")
        print(f"  Action Dim: {action_dim}")
        print(f"  Network Params: {sum(p.numel() for p in self.actor_critic.parameters()):,}")
        print("=" * 50)
    
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
    
    def collect_batch(self, batch_size: int) -> PPOBuffer:
        """Collect a batch of experiences"""
        buffer = PPOBuffer(batch_size, self.env.observation_space.shape[0], self.device)
        
        timesteps = 0
        while timesteps < batch_size:
            obs, _ = self.env.reset()
            obs = torch.FloatTensor(obs).to(self.device)
            episode_reward = 0
            episode_length = 0
            
            while timesteps < batch_size:
                # Get action from policy
                with torch.no_grad():
                    action_logits, value = self.actor_critic(obs.unsqueeze(0))
                    action_probs = F.softmax(action_logits, dim=-1)
                    action = torch.multinomial(action_probs, 1).item()
                    log_prob = F.log_softmax(action_logits, dim=-1)[0, action].item()
                
                # Take action
                next_obs, reward, done, truncated, _ = self.env.step(action)
                next_obs = torch.FloatTensor(next_obs).to(self.device)
                
                # Store transition
                buffer.add(obs, action, reward, value.item(), log_prob, done or truncated)
                
                obs = next_obs
                episode_reward += reward
                episode_length += 1
                timesteps += 1
                
                if done or truncated:
                    break
        
        return buffer
    
    def compute_loss(self, states, actions, old_log_probs, advantages, returns):
        """Compute PPO loss"""
        # Get current policy and value
        action_logits, values = self.actor_critic(states)
        action_probs = F.softmax(action_logits, dim=-1)
        log_probs = F.log_softmax(action_logits, dim=-1)
        
        # Get log probs for taken actions
        action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute policy ratio
        ratio = torch.exp(action_log_probs - old_log_probs)
        
        # Compute clipped policy loss
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Compute value loss
        value_loss = F.mse_loss(values.squeeze(), returns)
        
        # Compute entropy loss
        entropy = -(action_probs * log_probs).sum(dim=1).mean()
        entropy_loss = -entropy
        
        # Total loss
        total_loss = policy_loss + self.value_loss_coef * value_loss + self.entropy_coef * entropy_loss
        
        # Compute KL divergence
        kl_div = (old_log_probs - action_log_probs).mean().item()
        
        return {
            'total_loss': total_loss,
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'kl_div': kl_div,
            'entropy': entropy.item()
        }
    
    def update(self, buffer: PPOBuffer, epochs: int = 10) -> Dict[str, float]:
        """Update policy using PPO"""
        states, actions, rewards, values, old_log_probs, dones = buffer.get_all()
        
        # Compute advantages and returns
        with torch.no_grad():
            # Get final value for GAE computation
            final_obs = states[-1].unsqueeze(0)
            final_value = self.actor_critic(final_obs)[1].item()
            
            advantages, returns = self.compute_gae(rewards, values, dones, final_value)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Multiple epochs of updates
        for epoch in range(epochs):
            # Compute loss
            loss_dict = self.compute_loss(states, actions, old_log_probs, advantages, returns)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss_dict['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            # Early stopping if KL divergence is too high
            if loss_dict['kl_div'] > 1.5 * self.target_kl:
                break
        
        return loss_dict
    
    def evaluate(self, num_episodes: int = 10) -> Dict[str, Any]:
        """Evaluate current policy"""
        rewards = []
        lengths = []
        
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
                
                obs, reward, done, truncated, _ = self.env.step(action)
                obs = torch.FloatTensor(obs).to(self.device)
                
                episode_reward += reward
                episode_length += 1
                
                if done or truncated:
                    break
            
            rewards.append(episode_reward)
            lengths.append(episode_length)
        
        return {
            'avg_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'avg_length': np.mean(lengths),
            'rewards': rewards,
            'lengths': lengths
        }
    
    def train(
        self,
        total_timesteps: int = 100000,
        batch_size: int = 2048,
        update_epochs: int = 10,
        eval_interval: int = 5000,
        num_eval_episodes: int = 10,
        save_interval: int = 25000
    ):
        """Train the agent"""
        
        # Create output directories
        os.makedirs("checkpoints", exist_ok=True)
        os.makedirs("plots", exist_ok=True)
        
        print(f"🚀 Starting Simple PPO Training on {self.env_name}")
        print(f"  Total Timesteps: {total_timesteps:,}")
        print(f"  Batch Size: {batch_size}")
        print(f"  Update Epochs: {update_epochs}")
        print(f"  Eval Interval: {eval_interval}")
        print("=" * 50)
        
        timesteps_so_far = 0
        start_time = time.time()
        
        # Initial evaluation
        print("\n📊 Initial Evaluation:")
        initial_eval_stats = self.evaluate(num_eval_episodes)
        self.training_stats['episode_rewards'].append(initial_eval_stats['avg_reward'])
        self.training_stats['episode_lengths'].append(initial_eval_stats['avg_length'])
        print(f"  Average Reward: {initial_eval_stats['avg_reward']:.2f} ± {initial_eval_stats['std_reward']:.2f}")
        print(f"  Average Length: {initial_eval_stats['avg_length']:.1f}")
        print("-" * 40)
        
        with tqdm(total=total_timesteps, desc="Training Progress") as pbar:
            while timesteps_so_far < total_timesteps:
                # Collect batch
                buffer = self.collect_batch(batch_size)
                timesteps_so_far += buffer.size
                
                # Update policy
                update_stats = self.update(buffer, update_epochs)
                
                # Store training stats
                self.training_stats['policy_losses'].append(update_stats['policy_loss'])
                self.training_stats['value_losses'].append(update_stats['value_loss'])
                self.training_stats['entropy_losses'].append(update_stats['entropy_loss'])
                self.training_stats['kl_divergences'].append(update_stats['kl_div'])
                
                # Evaluation
                if timesteps_so_far % eval_interval == 0:
                    eval_stats = self.evaluate(num_eval_episodes)
                    self.training_stats['episode_rewards'].append(eval_stats['avg_reward'])
                    self.training_stats['episode_lengths'].append(eval_stats['avg_length'])
                    
                    print(f"\n📊 Evaluation at {timesteps_so_far:,} timesteps:")
                    print(f"  Average Reward: {eval_stats['avg_reward']:.2f} ± {eval_stats['std_reward']:.2f}")
                    print(f"  Average Length: {eval_stats['avg_length']:.1f}")
                    print(f"  Policy Loss: {update_stats['policy_loss']:.4f}")
                    print(f"  Value Loss: {update_stats['value_loss']:.4f}")
                    print(f"  KL Divergence: {update_stats['kl_div']:.4f}")
                    print("-" * 40)
                
                # Save model
                if timesteps_so_far % save_interval == 0:
                    model_path = f"checkpoints/ppo_{self.env_name}_{timesteps_so_far}.pth"
                    torch.save({
                        'actor_critic_state_dict': self.actor_critic.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'training_stats': self.training_stats,
                        'timesteps': timesteps_so_far
                    }, model_path)
                    print(f"\n💾 Model saved to {model_path}")
                    
                    # Plot training curves
                    self.plot_training_curves(save_path=f"plots/training_curves_{self.env_name}_{timesteps_so_far}.png")
                
                pbar.update(buffer.size)
                pbar.set_postfix({
                    'Policy Loss': f"{update_stats['policy_loss']:.4f}",
                    'Value Loss': f"{update_stats['value_loss']:.4f}",
                    'KL Div': f"{update_stats['kl_div']:.4f}",
                    'Timesteps': f"{timesteps_so_far:,}"
                })
        
        training_time = time.time() - start_time
        
        # Final evaluation
        print("\n📊 Final Evaluation:")
        final_eval_stats = self.evaluate(num_eval_episodes)
        self.training_stats['episode_rewards'].append(final_eval_stats['avg_reward'])
        self.training_stats['episode_lengths'].append(final_eval_stats['avg_length'])
        print(f"  Average Reward: {final_eval_stats['avg_reward']:.2f} ± {final_eval_stats['std_reward']:.2f}")
        print(f"  Average Length: {final_eval_stats['avg_length']:.1f}")
        print("-" * 40)
        
        # Final save
        final_model_path = f"checkpoints/ppo_{self.env_name}_final.pth"
        torch.save({
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_stats': self.training_stats,
            'timesteps': timesteps_so_far
        }, final_model_path)
        
        self.plot_training_curves(save_path=f"plots/final_training_curves_{self.env_name}.png")
        
        print(f"\n🎉 Training completed in {training_time:.1f} seconds!")
        print(f"Final model saved to: {final_model_path}")
        print(f"Training curves saved to: plots/final_training_curves_{self.env_name}.png")
        
        return self.actor_critic
    
    def plot_training_curves(self, save_path: str = "training_curves.png"):
        """Plot training curves"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Episode rewards
        if self.training_stats['episode_rewards']:
            axes[0, 0].plot(self.training_stats['episode_rewards'], 'b-', linewidth=2, marker='o')
            axes[0, 0].set_title('Episode Rewards', fontsize=14, fontweight='bold')
            axes[0, 0].set_xlabel('Evaluation Step')
            axes[0, 0].set_ylabel('Average Reward')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Episode lengths
        if self.training_stats['episode_lengths']:
            axes[0, 1].plot(self.training_stats['episode_lengths'], 'r-', linewidth=2, marker='o')
            axes[0, 1].set_title('Episode Lengths', fontsize=14, fontweight='bold')
            axes[0, 1].set_xlabel('Evaluation Step')
            axes[0, 1].set_ylabel('Average Length')
            axes[0, 1].grid(True, alpha=0.3)
        
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
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Training curves saved to {save_path}")
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Simple PPO Test on Gym Environment")
    parser.add_argument("--env", type=str, default="CartPole-v1", help="Gym environment name")
    parser.add_argument("--timesteps", type=int, default=100000, help="Total timesteps for training")
    parser.add_argument("--batch-size", type=int, default=2048, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden dimension")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    parser.add_argument("--eval-interval", type=int, default=5000, help="Evaluation interval")
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = SimplePPOTrainer(
        env_name=args.env,
        learning_rate=args.lr,
        hidden_dim=args.hidden_dim,
        device=args.device
    )
    
    # Train
    trainer.train(
        total_timesteps=args.timesteps,
        batch_size=args.batch_size,
        eval_interval=args.eval_interval
    )

if __name__ == "__main__":
    main() 