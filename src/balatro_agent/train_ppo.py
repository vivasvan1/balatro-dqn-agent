#!/usr/bin/env python3
"""
Comprehensive PPO Training for Balatro
Consolidated training script with debugging and monitoring features
"""

import torch
import torch.nn.functional as F
import numpy as np
import argparse
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, List, Tuple, Any

from ppo_agent import PPOAgent
from balatro_gym_v2_simple import BalatroGymEnvSimple

class ComprehensivePPOTrainer:
    """Comprehensive PPO trainer with debugging and monitoring"""
    
    def __init__(
        self,
        blind_score: int = 300,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        target_kl: float = 0.01,
        hidden_dim: int = 256,
        device: str = "auto"
    ):
        self.blind_score = blind_score
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create environment and agent
        self.env = BalatroGymEnvSimple(blind_score=blind_score)
        self.agent = PPOAgent(
            env=self.env,
            learning_rate=learning_rate,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_ratio=clip_ratio,
            value_loss_coef=value_loss_coef,
            entropy_coef=entropy_coef,
            max_grad_norm=max_grad_norm,
            target_kl=target_kl,
            hidden_dim=hidden_dim,
            device=self.device
        )
        
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
        
        print(f"🎰 Comprehensive PPO Trainer Initialized")
        print(f"  Target Score: {blind_score}")
        print(f"  Device: {self.device}")
        print(f"  State Dim: {self.env.observation_space.shape[0]}")
        print(f"  Action Dim: {self.env.action_space.n}")
        print(f"  Network Params: {sum(p.numel() for p in self.agent.actor_critic.parameters()):,}")
        print("=" * 60)
    
    def train(
        self,
        total_timesteps: int = 100000,
        batch_size: int = 2048,
        update_epochs: int = 10,
        eval_interval: int = 5000,
        num_eval_episodes: int = 10,
        save_interval: int = 25000,
        demo_interval: int = 10000,
        debug_interval: int = 2000
    ):
        """Train with comprehensive monitoring and debugging"""
        
        print(f"🚀 Starting Comprehensive PPO Training")
        print(f"  Total Timesteps: {total_timesteps:,}")
        print(f"  Batch Size: {batch_size}")
        print(f"  Update Epochs: {update_epochs}")
        print(f"  Eval Interval: {eval_interval}")
        print(f"  Debug Interval: {debug_interval}")
        print("=" * 60)
        
        timesteps_so_far = 0
        start_time = time.time()
        
        with tqdm(total=total_timesteps, desc="Training Progress") as pbar:
            while timesteps_so_far < total_timesteps:
                # Collect batch
                buffer = self.agent.collect_batch(batch_size)
                timesteps_so_far += buffer.size
                
                # Update policy
                update_stats = self.agent.update(buffer, update_epochs)
                
                # Store training stats
                self.training_stats['policy_losses'].append(update_stats['policy_loss'])
                self.training_stats['value_losses'].append(update_stats['value_loss'])
                self.training_stats['entropy_losses'].append(update_stats['entropy_loss'])
                self.training_stats['kl_divergences'].append(update_stats['kl_div'])
                
                # Debug monitoring
                if timesteps_so_far % debug_interval == 0:
                    self._debug_training_step(update_stats, buffer)
                
                # Evaluation
                if timesteps_so_far % eval_interval == 0:
                    eval_stats = self.evaluate(num_eval_episodes)
                    self.training_stats['episode_rewards'].append(eval_stats['avg_reward'])
                    self.training_stats['episode_lengths'].append(eval_stats['avg_length'])
                    self.training_stats['win_rates'].append(eval_stats['win_rate'])
                    self.training_stats['action_distributions'].append(eval_stats['action_distribution'])
                    
                    print(f"\n📊 Evaluation at {timesteps_so_far:,} timesteps:")
                    print(f"  Average Reward: {eval_stats['avg_reward']:.2f} ± {eval_stats['std_reward']:.2f}")
                    print(f"  Win Rate: {eval_stats['win_rate']:.2%}")
                    print(f"  Average Length: {eval_stats['avg_length']:.1f}")
                    print(f"  Action Distribution: {eval_stats['action_distribution']}")
                    print(f"  Policy Loss: {update_stats['policy_loss']:.4f}")
                    print(f"  Value Loss: {update_stats['value_loss']:.4f}")
                    print(f"  Entropy Loss: {update_stats['entropy_loss']:.4f}")
                    print(f"  KL Divergence: {update_stats['kl_div']:.4f}")
                    print("-" * 40)
                
                # Demo episode
                if timesteps_so_far % demo_interval == 0:
                    print(f"\n🎮 Demo Episode at {timesteps_so_far:,} timesteps:")
                    self._play_demo_episode()
                
                # Save model and plots
                if timesteps_so_far % save_interval == 0:
                    model_path = f"ppo_balatro_{timesteps_so_far}.pth"
                    self.agent.save_model(model_path)
                    print(f"\n💾 Model saved to {model_path}")
                    
                    # Plot current training curves
                    self.plot_training_curves(save_path=f"training_curves_{timesteps_so_far}.png")
                
                pbar.update(buffer.size)
                pbar.set_postfix({
                    'Policy Loss': f"{update_stats['policy_loss']:.4f}",
                    'Value Loss': f"{update_stats['value_loss']:.4f}",
                    'KL Div': f"{update_stats['kl_div']:.4f}",
                    'Timesteps': f"{timesteps_so_far:,}"
                })
        
        training_time = time.time() - start_time
        
        # Final save and evaluation
        self.agent.save_model("ppo_balatro_final.pth")
        self.plot_training_curves(save_path="final_training_curves.png")
        
        print(f"\n🎉 Training completed in {training_time:.1f} seconds!")
        print(f"Final model saved to: ppo_balatro_final.pth")
        print(f"Training curves saved to: final_training_curves.png")
        
        return self.agent
    
    def _debug_training_step(self, update_stats: Dict, buffer):
        """Debug training step with detailed analysis"""
        print(f"\n🔍 Debug Analysis at {len(self.training_stats['policy_losses'])} updates:")
        
        # Check if policy is becoming too confident
        if len(self.training_stats['policy_losses']) > 10:
            recent_entropy = np.mean(self.training_stats['entropy_losses'][-10:])
            print(f"  Recent Avg Entropy Loss: {recent_entropy:.4f}")
            
            if recent_entropy > -0.001:  # Very low entropy
                print(f"  ⚠️  Policy becoming too confident (low entropy)")
        
        # Check KL divergence
        kl_div = update_stats['kl_div']
        print(f"  KL Divergence: {kl_div:.4f}")
        if kl_div > 0.05:
            print(f"  ⚠️  High KL divergence - policy changing too fast")
        elif kl_div < 0.001:
            print(f"  ⚠️  Very low KL divergence - policy not learning")
        
        # Check policy loss
        policy_loss = update_stats['policy_loss']
        print(f"  Policy Loss: {policy_loss:.4f}")
        if abs(policy_loss) < 0.001:
            print(f"  ⚠️  Very low policy loss - might be stuck")
        
        # Check value loss
        value_loss = update_stats['value_loss']
        print(f"  Value Loss: {value_loss:.4f}")
        if value_loss > 1000:
            print(f"  ⚠️  High value loss - value function struggling")
    
    def evaluate(self, num_episodes: int = 10) -> Dict[str, Any]:
        """Evaluate current policy with detailed statistics"""
        rewards = []
        lengths = []
        wins = 0
        action_counts = {"play": 0, "discard": 0}
        
        for _ in range(num_episodes):
            obs, _ = self.env.reset()
            obs = torch.FloatTensor(obs).to(self.device)
            episode_reward = 0
            episode_length = 0
            episode_actions = {"play": 0, "discard": 0}
            
            while True:
                with torch.no_grad():
                    action_logits, _ = self.agent.actor_critic(obs.unsqueeze(0))
                    action_probs = F.softmax(action_logits, dim=-1)
                    action = torch.argmax(action_probs).item()
                
                action_type, _ = self.env._decode_action(action)
                episode_actions[action_type] += 1
                
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
            action_counts["play"] += episode_actions["play"]
            action_counts["discard"] += episode_actions["discard"]
        
        total_actions = action_counts["play"] + action_counts["discard"]
        action_distribution = {
            "play": f"{action_counts['play']/total_actions:.1%}" if total_actions > 0 else "0%",
            "discard": f"{action_counts['discard']/total_actions:.1%}" if total_actions > 0 else "0%"
        }
        
        return {
            'avg_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards),
            'avg_length': np.mean(lengths),
            'win_rate': wins / num_episodes,
            'action_distribution': action_distribution,
            'rewards': rewards,
            'lengths': lengths
        }
    
    def _play_demo_episode(self, max_steps: int = 10):
        """Play a demonstration episode"""
        obs, _ = self.env.reset()
        obs = torch.FloatTensor(obs).to(self.device)
        total_reward = 0
        step = 0
        
        print(f"  Initial hand: {[str(card) for card in self.env.hand]}")
        print(f"  Target score: {self.env.blind_score}")
        
        while step < max_steps:
            step += 1
            
            # Get action from policy
            with torch.no_grad():
                action_logits, value = self.agent.actor_critic(obs.unsqueeze(0))
                action_probs = F.softmax(action_logits, dim=-1)
                action = torch.argmax(action_probs).item()
                action_prob = action_probs[0, action].item()
            
            # Decode action
            action_type, card_indices = self.env._decode_action(action)
            
            # Take action
            obs, reward, done, truncated, info = self.env.step(action)
            obs = torch.FloatTensor(obs).to(self.device)
            total_reward += reward
            
            # Show action details
            cards_str = [str(self.env.hand[i]) for i in card_indices if i < len(self.env.hand)]
            print(f"  Step {step}: {action_type.upper()} {cards_str} (prob: {action_prob:.3f}, reward: {reward:.2f})")
            print(f"    Score: {self.env.current_score}/{self.env.blind_score}, Plays: {self.env.plays_left}, Discards: {self.env.discards_left}")
            
            if done or truncated:
                break
        
        print(f"  Final result: {'WIN' if self.env.won else 'LOSS'} (Total reward: {total_reward:.2f})")
        print()
    
    def plot_training_curves(self, save_path: str = "training_curves.png"):
        """Plot comprehensive training curves with debugging info"""
        fig, axes = plt.subplots(4, 3, figsize=(20, 20))
        
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
        
        # Action distribution over time
        if self.training_stats['action_distributions']:
            play_ratios = [float(dist['play'].rstrip('%'))/100 for dist in self.training_stats['action_distributions']]
            discard_ratios = [float(dist['discard'].rstrip('%'))/100 for dist in self.training_stats['action_distributions']]
            
            axes[2, 2].plot(play_ratios, 'blue', linewidth=2, label='Play Ratio')
            axes[2, 2].plot(discard_ratios, 'red', linewidth=2, label='Discard Ratio')
            axes[2, 2].set_title('Action Distribution Over Time', fontsize=14, fontweight='bold')
            axes[2, 2].set_xlabel('Evaluation Step')
            axes[2, 2].set_ylabel('Ratio')
            axes[2, 2].grid(True, alpha=0.3)
            axes[2, 2].legend()
            axes[2, 2].set_ylim(0, 1)
        
        # Training summary
        if self.training_stats['policy_losses']:
            total_updates = len(self.training_stats['policy_losses'])
            avg_policy_loss = np.mean(self.training_stats['policy_losses'][-10:]) if total_updates >= 10 else np.mean(self.training_stats['policy_losses'])
            avg_value_loss = np.mean(self.training_stats['value_losses'][-10:]) if total_updates >= 10 else np.mean(self.training_stats['value_losses'])
            avg_kl = np.mean(self.training_stats['kl_divergences'][-10:]) if total_updates >= 10 else np.mean(self.training_stats['kl_divergences'])
            
            axes[3, 0].text(0.1, 0.8, f'Total Updates: {total_updates}', fontsize=12, transform=axes[3, 0].transAxes)
            axes[3, 0].text(0.1, 0.7, f'Avg Policy Loss: {avg_policy_loss:.4f}', fontsize=12, transform=axes[3, 0].transAxes)
            axes[3, 0].text(0.1, 0.6, f'Avg Value Loss: {avg_value_loss:.4f}', fontsize=12, transform=axes[3, 0].transAxes)
            axes[3, 0].text(0.1, 0.5, f'Avg KL Div: {avg_kl:.4f}', fontsize=12, transform=axes[3, 0].transAxes)
            
            if self.training_stats['win_rates']:
                latest_win_rate = self.training_stats['win_rates'][-1]
                axes[3, 0].text(0.1, 0.4, f'Latest Win Rate: {latest_win_rate:.2%}', fontsize=12, transform=axes[3, 0].transAxes)
            
            if self.training_stats['episode_rewards']:
                latest_reward = self.training_stats['episode_rewards'][-1]
                axes[3, 0].text(0.1, 0.3, f'Latest Avg Reward: {latest_reward:.2f}', fontsize=12, transform=axes[3, 0].transAxes)
            
            axes[3, 0].set_title('Training Summary', fontsize=14, fontweight='bold')
            axes[3, 0].axis('off')
        
        # Debugging indicators
        axes[3, 1].text(0.1, 0.8, 'Debugging Indicators:', fontsize=14, fontweight='bold', transform=axes[3, 1].transAxes)
        
        if self.training_stats['policy_losses']:
            recent_policy_loss = np.mean(self.training_stats['policy_losses'][-5:])
            if abs(recent_policy_loss) < 0.001:
                axes[3, 1].text(0.1, 0.7, '⚠️ Low Policy Loss', fontsize=12, color='red', transform=axes[3, 1].transAxes)
            else:
                axes[3, 1].text(0.1, 0.7, '✅ Policy Learning', fontsize=12, color='green', transform=axes[3, 1].transAxes)
        
        if self.training_stats['kl_divergences']:
            recent_kl = np.mean(self.training_stats['kl_divergences'][-5:])
            if recent_kl > 0.05:
                axes[3, 1].text(0.1, 0.6, '⚠️ High KL Div', fontsize=12, color='red', transform=axes[3, 1].transAxes)
            elif recent_kl < 0.001:
                axes[3, 1].text(0.1, 0.6, '⚠️ Low KL Div', fontsize=12, color='orange', transform=axes[3, 1].transAxes)
            else:
                axes[3, 1].text(0.1, 0.6, '✅ Good KL Div', fontsize=12, color='green', transform=axes[3, 1].transAxes)
        
        if self.training_stats['entropy_losses']:
            recent_entropy = np.mean(self.training_stats['entropy_losses'][-5:])
            if recent_entropy > -0.001:
                axes[3, 1].text(0.1, 0.5, '⚠️ Low Entropy', fontsize=12, color='red', transform=axes[3, 1].transAxes)
            else:
                axes[3, 1].text(0.1, 0.5, '✅ Good Entropy', fontsize=12, color='green', transform=axes[3, 1].transAxes)
        
        axes[3, 1].axis('off')
        
        # Recommendations
        axes[3, 2].text(0.1, 0.8, 'Recommendations:', fontsize=14, fontweight='bold', transform=axes[3, 2].transAxes)
        
        if self.training_stats['policy_losses'] and self.training_stats['kl_divergences']:
            recent_policy_loss = np.mean(self.training_stats['policy_losses'][-5:])
            recent_kl = np.mean(self.training_stats['kl_divergences'][-5:])
            
            if abs(recent_policy_loss) < 0.001:
                axes[3, 2].text(0.1, 0.7, '• Increase learning rate', fontsize=10, transform=axes[3, 2].transAxes)
                axes[3, 2].text(0.1, 0.6, '• Check reward function', fontsize=10, transform=axes[3, 2].transAxes)
            elif recent_kl > 0.05:
                axes[3, 2].text(0.1, 0.7, '• Reduce learning rate', fontsize=10, transform=axes[3, 2].transAxes)
                axes[3, 2].text(0.1, 0.6, '• Increase clip ratio', fontsize=10, transform=axes[3, 2].transAxes)
            elif recent_kl < 0.001:
                axes[3, 2].text(0.1, 0.7, '• Increase learning rate', fontsize=10, transform=axes[3, 2].transAxes)
                axes[3, 2].text(0.1, 0.6, '• Check advantage estimates', fontsize=10, transform=axes[3, 2].transAxes)
            else:
                axes[3, 2].text(0.1, 0.7, '• Training looks good!', fontsize=10, color='green', transform=axes[3, 2].transAxes)
                axes[3, 2].text(0.1, 0.6, '• Continue training', fontsize=10, transform=axes[3, 2].transAxes)
        
        axes[3, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Training curves saved to {save_path}")
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Comprehensive PPO Training for Balatro")
    parser.add_argument("--blind-score", type=int, default=300, help="Target score to win")
    parser.add_argument("--timesteps", type=int, default=100000, help="Total timesteps for training")
    parser.add_argument("--batch-size", type=int, default=2048, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--clip-ratio", type=float, default=0.2, help="PPO clip ratio")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    parser.add_argument("--eval-interval", type=int, default=5000, help="Evaluation interval")
    parser.add_argument("--demo-interval", type=int, default=10000, help="Demo interval")
    parser.add_argument("--debug-interval", type=int, default=2000, help="Debug interval")
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = ComprehensivePPOTrainer(
        blind_score=args.blind_score,
        learning_rate=args.lr,
        clip_ratio=args.clip_ratio,
        hidden_dim=args.hidden_dim,
        device=args.device
    )
    
    # Train
    trainer.train(
        total_timesteps=args.timesteps,
        batch_size=args.batch_size,
        eval_interval=args.eval_interval,
        demo_interval=args.demo_interval,
        debug_interval=args.debug_interval
    )

if __name__ == "__main__":
    main() 