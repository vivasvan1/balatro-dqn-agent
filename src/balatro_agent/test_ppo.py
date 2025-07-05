#!/usr/bin/env python3
"""
PPO Evaluation and Demo Script
Load trained models and analyze agent behavior
"""

import torch
import torch.nn.functional as F
import numpy as np
import argparse
import time
from typing import Dict, List, Any
import matplotlib.pyplot as plt

from ppo_agent import PPOAgent
from balatro_gym_v2_simple import BalatroGymEnvSimple

class PPOEvaluator:
    """Comprehensive PPO evaluation and demo tool"""
    
    def __init__(self, model_path: str, blind_score: int = 300, device: str = "auto", hidden_dim: int = 256):
        self.model_path = model_path
        self.blind_score = blind_score
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create environment and agent
        self.env = BalatroGymEnvSimple(blind_score=blind_score)
        self.agent = PPOAgent(env=self.env, device=self.device, hidden_dim=hidden_dim)
        
        # Load model
        self.agent.load_model(model_path)
        print(f"🎯 PPO Evaluator Initialized")
        print(f"  Model: {model_path}")
        print(f"  Target Score: {blind_score}")
        print(f"  Device: {self.device}")
        print("=" * 50)
    
    def evaluate(self, num_episodes: int = 100) -> Dict[str, Any]:
        """Comprehensive evaluation of the trained agent"""
        print(f"📊 Evaluating agent over {num_episodes} episodes...")
        
        rewards = []
        lengths = []
        wins = 0
        action_counts = {"play": 0, "discard": 0}
        hand_types = []
        final_scores = []
        
        for episode in range(num_episodes):
            obs, _ = self.env.reset()
            obs = torch.FloatTensor(obs).to(self.device)
            episode_reward = 0
            episode_length = 0
            episode_actions = {"play": 0, "discard": 0}
            episode_hand_types = []
            
            while True:
                with torch.no_grad():
                    action_logits, value = self.agent.actor_critic(obs.unsqueeze(0))
                    action_probs = F.softmax(action_logits, dim=-1)
                    action = torch.argmax(action_probs).item()
                    action_prob = action_probs[0, action].item()
                
                action_type, card_indices = self.env._decode_action(action)
                episode_actions[action_type] += 1
                
                obs, reward, done, truncated, info = self.env.step(action)
                obs = torch.FloatTensor(obs).to(self.device)
                
                episode_reward += reward
                episode_length += 1
                
                # Record hand type if it was a play action
                if action_type == "play" and "hand_type" in info:
                    episode_hand_types.append(info["hand_type"])
                
                if done or truncated:
                    if info.get('won', False):
                        wins += 1
                    break
            
            rewards.append(episode_reward)
            lengths.append(episode_length)
            final_scores.append(self.env.current_score)
            action_counts["play"] += episode_actions["play"]
            action_counts["discard"] += episode_actions["discard"]
            hand_types.extend(episode_hand_types)
            
            if (episode + 1) % 20 == 0:
                print(f"  Completed {episode + 1}/{num_episodes} episodes...")
        
        total_actions = action_counts["play"] + action_counts["discard"]
        action_distribution = {
            "play": f"{action_counts['play']/total_actions:.1%}" if total_actions > 0 else "0%",
            "discard": f"{action_counts['discard']/total_actions:.1%}" if total_actions > 0 else "0%"
        }
        
        # Calculate hand type distribution
        hand_type_counts = {}
        for hand_type in hand_types:
            hand_type_counts[hand_type] = hand_type_counts.get(hand_type, 0) + 1
        
        return {
            'avg_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards),
            'avg_length': np.mean(lengths),
            'win_rate': wins / num_episodes,
            'action_distribution': action_distribution,
            'hand_type_distribution': hand_type_counts,
            'avg_final_score': np.mean(final_scores),
            'rewards': rewards,
            'lengths': lengths,
            'final_scores': final_scores
        }
    
    def demo_episodes(self, num_episodes: int = 3, max_steps: int = 15):
        """Run detailed demo episodes with step-by-step analysis"""
        print(f"🎮 Running {num_episodes} demo episodes...")
        
        for episode in range(num_episodes):
            print(f"\n{'='*20} Demo Episode {episode + 1} {'='*20}")
            
            obs, _ = self.env.reset()
            obs = torch.FloatTensor(obs).to(self.device)
            total_reward = 0
            step = 0
            actions_taken = {"play": 0, "discard": 0}
            
            print(f"Initial hand: {[str(card) for card in self.env.hand]}")
            print(f"Target score: {self.env.blind_score}")
            print()
            
            while step < max_steps:
                step += 1
                
                # Get action from policy
                with torch.no_grad():
                    action_logits, value = self.agent.actor_critic(obs.unsqueeze(0))
                    action_probs = F.softmax(action_logits, dim=-1)
                    action = torch.argmax(action_probs).item()
                    action_prob = action_probs[0, action].item()
                    
                    # Get top 3 actions for analysis
                    top_actions = torch.topk(action_probs, 3)
                    top_action_indices = top_actions.indices[0].cpu().numpy()
                    top_action_probs = top_actions.values[0].cpu().numpy()
                
                # Decode action
                action_type, card_indices = self.env._decode_action(action)
                actions_taken[action_type] += 1
                
                # Take action
                obs, reward, done, truncated, info = self.env.step(action)
                obs = torch.FloatTensor(obs).to(self.device)
                total_reward += reward
                
                # Show detailed action analysis
                cards_str = [str(self.env.hand[i]) for i in card_indices if i < len(self.env.hand)]
                print(f"Step {step}: {action_type.upper()} {cards_str}")
                print(f"  Probability: {action_prob:.3f}")
                print(f"  Reward: {reward:.2f}")
                print(f"  Score: {self.env.current_score}/{self.env.blind_score}")
                print(f"  Plays left: {self.env.plays_left}, Discards left: {self.env.discards_left}")
                
                if action_type == "play" and "hand_type" in info:
                    print(f"  Hand type: {info['hand_type']}")
                    if "score_gained" in info:
                        print(f"  Score gained: {info['score_gained']}")
                
                # Show top 3 actions for comparison
                print(f"  Top 3 actions:")
                for i in range(3):
                    top_action_type, top_card_indices = self.env._decode_action(top_action_indices[i])
                    top_cards_str = [str(self.env.hand[j]) for j in top_card_indices if j < len(self.env.hand)]
                    print(f"    {i+1}. {top_action_type.upper()} {top_cards_str} (prob: {top_action_probs[i]:.3f})")
                
                print()
                
                if done or truncated:
                    break
            
            print(f"Final result: {'WIN' if self.env.won else 'LOSS'}")
            print(f"Total reward: {total_reward:.2f}")
            print(f"Actions taken: {actions_taken}")
            print(f"Final score: {self.env.current_score}/{self.env.blind_score}")
            print()
    
    def analyze_policy(self, num_samples: int = 1000):
        """Analyze the policy's action preferences"""
        print(f"🔍 Analyzing policy preferences...")
        
        action_preferences = {}
        value_predictions = []
        
        for _ in range(num_samples):
            obs, _ = self.env.reset()
            obs = torch.FloatTensor(obs).to(self.device)
            
            with torch.no_grad():
                action_logits, value = self.agent.actor_critic(obs.unsqueeze(0))
                action_probs = F.softmax(action_logits, dim=-1)
                
                # Get top action
                action = torch.argmax(action_probs).item()
                action_type, _ = self.env._decode_action(action)
                
                action_preferences[action_type] = action_preferences.get(action_type, 0) + 1
                value_predictions.append(value.item())
        
        # Calculate percentages
        total_samples = sum(action_preferences.values())
        action_percentages = {k: v/total_samples*100 for k, v in action_preferences.items()}
        
        print(f"Policy Analysis Results:")
        print(f"  Action Preferences:")
        for action_type, percentage in action_percentages.items():
            print(f"    {action_type}: {percentage:.1f}%")
        
        print(f"  Value Predictions:")
        print(f"    Average: {np.mean(value_predictions):.2f}")
        print(f"    Std: {np.std(value_predictions):.2f}")
        print(f"    Min: {np.min(value_predictions):.2f}")
        print(f"    Max: {np.max(value_predictions):.2f}")
        
        return action_percentages, value_predictions
    
    def plot_evaluation_results(self, eval_stats: Dict[str, Any], save_path: str = "evaluation_results.png"):
        """Plot comprehensive evaluation results"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Reward distribution
        axes[0, 0].hist(eval_stats['rewards'], bins=20, alpha=0.7, color='blue')
        axes[0, 0].set_title('Reward Distribution', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Reward')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Final score distribution
        axes[0, 1].hist(eval_stats['final_scores'], bins=20, alpha=0.7, color='green')
        axes[0, 1].set_title('Final Score Distribution', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Final Score')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Episode length distribution
        axes[0, 2].hist(eval_stats['lengths'], bins=20, alpha=0.7, color='red')
        axes[0, 2].set_title('Episode Length Distribution', fontsize=14, fontweight='bold')
        axes[0, 2].set_xlabel('Episode Length')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Action distribution pie chart
        action_dist = eval_stats['action_distribution']
        play_pct = float(action_dist['play'].rstrip('%'))
        discard_pct = float(action_dist['discard'].rstrip('%'))
        
        axes[1, 0].pie([play_pct, discard_pct], labels=['Play', 'Discard'], autopct='%1.1f%%', startangle=90)
        axes[1, 0].set_title('Action Distribution', fontsize=14, fontweight='bold')
        
        # Hand type distribution
        if eval_stats['hand_type_distribution']:
            hand_types = list(eval_stats['hand_type_distribution'].keys())
            hand_counts = list(eval_stats['hand_type_distribution'].values())
            
            axes[1, 1].bar(hand_types, hand_counts, alpha=0.7, color='purple')
            axes[1, 1].set_title('Hand Type Distribution', fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel('Hand Type')
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].grid(True, alpha=0.3)
        
        # Summary statistics
        axes[1, 2].text(0.1, 0.9, 'Evaluation Summary', fontsize=16, fontweight='bold', transform=axes[1, 2].transAxes)
        axes[1, 2].text(0.1, 0.8, f'Win Rate: {eval_stats["win_rate"]*100:.2f}%', fontsize=12, transform=axes[1, 2].transAxes)
        axes[1, 2].text(0.1, 0.7, f'Avg Reward: {eval_stats["avg_reward"]:.2f} ± {eval_stats["std_reward"]:.2f}', fontsize=12, transform=axes[1, 2].transAxes)
        axes[1, 2].text(0.1, 0.6, f'Avg Length: {eval_stats["avg_length"]:.1f}', fontsize=12, transform=axes[1, 2].transAxes)
        axes[1, 2].text(0.1, 0.5, f'Avg Final Score: {eval_stats["avg_final_score"]:.1f}', fontsize=12, transform=axes[1, 2].transAxes)
        axes[1, 2].text(0.1, 0.4, f'Play Ratio: {action_dist["play"]}', fontsize=12, transform=axes[1, 2].transAxes)
        axes[1, 2].text(0.1, 0.3, f'Discard Ratio: {action_dist["discard"]}', fontsize=12, transform=axes[1, 2].transAxes)
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Evaluation results saved to {save_path}")
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="PPO Evaluation and Demo")
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--blind-score", type=int, default=300, help="Target score for evaluation")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension for model")
    parser.add_argument("--mode", choices=["evaluate", "demo", "analyze", "all"], default="all", help="Evaluation mode")
    parser.add_argument("--num-episodes", type=int, default=100, help="Number of episodes for evaluation")
    parser.add_argument("--num-demos", type=int, default=3, help="Number of demo episodes")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = PPOEvaluator(
        model_path=args.model_path,
        blind_score=args.blind_score,
        device=args.device,
        hidden_dim=args.hidden_dim
    )
    
    if args.mode in ["evaluate", "all"]:
        print("📊 Running evaluation...")
        eval_stats = evaluator.evaluate(args.num_episodes)
        
        print(f"\n🏆 Evaluation Results:")
        print(f"  Win Rate: {eval_stats['win_rate']*100:.2f}%")
        print(f"  Average Reward: {eval_stats['avg_reward']:.2f} ± {eval_stats['std_reward']:.2f}")
        print(f"  Average Length: {eval_stats['avg_length']:.1f}")
        print(f"  Action Distribution: {eval_stats['action_distribution']}")
        print(f"  Hand Type Distribution: {eval_stats['hand_type_distribution']}")
        
        # Plot results
        evaluator.plot_evaluation_results(eval_stats)
    
    if args.mode in ["demo", "all"]:
        print("\n🎮 Running demo episodes...")
        evaluator.demo_episodes(args.num_demos)
    
    if args.mode in ["analyze", "all"]:
        print("\n🔍 Analyzing policy...")
        evaluator.analyze_policy()
    
    print(f"\n✅ Evaluation completed!")

if __name__ == "__main__":
    main() 