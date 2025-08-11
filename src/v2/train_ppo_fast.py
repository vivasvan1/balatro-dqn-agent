#!/usr/bin/env python3
"""
Fast PPO Training for Balatro - Optimized for Speed
Uses larger batch sizes and optimized settings for faster training
"""

import torch
import torch.nn.functional as F
import numpy as np
import argparse
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, List, Tuple, Any, Optional
import os
import json
from datetime import datetime

# Optional MLflow import with shadowing guard
try:
    import mlflow  # type: ignore
    # Guard against a local directory named "mlflow" shadowing the real package
    if not hasattr(mlflow, "set_tracking_uri") or not hasattr(mlflow, "start_run"):
        print("⚠️  A local 'mlflow' directory is shadowing the real MLflow package. Disabling MLflow logging.")
        mlflow = None
except Exception:  # pragma: no cover
    mlflow = None

from ppo_agent import PPOAgent
from balatro_gym_v2_simple import BalatroGymEnvSimple

class FastPPOTrainer:
    """Fast PPO trainer optimized for speed with larger batch sizes"""
    
    def __init__(
        self,
        blind_score: int = 300,
        learning_rate: float = 1e-4,  # Lower LR for stability
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,  # Lower entropy for faster convergence
        max_grad_norm: float = 0.5,
        target_kl: float = 0.01,
        hidden_dim: int = 256,  # Smaller network for speed
        device: str = "auto",
        load_model_path: str = None,
        curriculum_learning: bool = True
    ):
        self.blind_score = blind_score
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model_path = load_model_path
        self.curriculum_learning = curriculum_learning
        
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
        
        # Load model if specified
        if load_model_path and os.path.exists(load_model_path):
            print(f"🔄 Loading model from {load_model_path}")
            self.agent.load_model(load_model_path)
            self.training_stats = self.agent.training_stats
            print(f"✅ Model loaded successfully!")
        else:
            # Initialize fresh training stats
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
            if load_model_path and not os.path.exists(load_model_path):
                print(f"⚠️  Warning: Model file {load_model_path} not found. Starting fresh training.")
        
        print(f"⚡ Fast PPO Trainer Initialized")
        print(f"  Target Score: {blind_score}")
        print(f"  Device: {self.device}")
        print(f"  State Dim: {self.env.observation_space.shape[0]}")
        print(f"  Action Space: {self.env.action_space}")
        print(f"  Network Params: {sum(p.numel() for p in self.agent.actor_critic.parameters()):,}")
        print(f"  Learning Rate: {learning_rate}")
        print(f"  Hidden Dim: {hidden_dim}")
        if load_model_path:
            print(f"  Loaded Model: {load_model_path}")
        print("=" * 60)
    
    def train(
        self,
        total_timesteps: int = 100000,
        batch_size: int = 4096,  # Smaller batch size for stability
        update_epochs: int = 4,   # Fewer epochs for speed
        eval_interval: int = 10000,  # Interval for on-screen eval/metrics
        eval_log_interval: int = 500,  # Interval to save 5-episode eval logs to file
        num_eval_episodes: int = 5,   # Number of eval episodes per evaluation
        save_interval: int = 50000,
        backup_interval: int = 25000,
        demo_interval: int = 25000,   # Less frequent demos
        debug_interval: int = 5000,    # Less frequent debug
        mlflow_logging: bool = False,
        mlflow_experiment: str = "BalatroV2",
        mlflow_tracking_uri: Optional[str] = None,
    ):
        """Train with optimized settings for speed"""
        
        # Create output directories
        os.makedirs("checkpoints", exist_ok=True)
        os.makedirs("plots", exist_ok=True)
        os.makedirs("backups", exist_ok=True)
        # os.makedirs("logs", exist_ok=True)
        
        print(f"🚀 Starting Fast PPO Training")
        print(f"  Total Timesteps: {total_timesteps:,}")
        print(f"  Batch Size: {batch_size:,}")
        print(f"  Update Epochs: {update_epochs}")
        print(f"  Eval Interval: {eval_interval}")
        print(f"  Eval Log Interval: {eval_log_interval}")
        print(f"  Save Interval: {save_interval}")
        print(f"  Backup Interval: {backup_interval}")
        print(f"  Demo Interval: {demo_interval}")
        print(f"  Debug Interval: {debug_interval}")
        print(f"  Curriculum Learning: {self.curriculum_learning}")
        print(f"  Output folders: checkpoints/, plots/, backups/")
        if self.load_model_path:
            print(f"  Continuing from: {self.load_model_path}")
        print("=" * 60)
        
        timesteps_so_far = 0
        start_time = time.time()
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"fast_ppo_{self.run_timestamp}_S{self.blind_score}"

        # Setup MLflow if enabled
        active_mlflow_run = None
        if mlflow_logging:
            if mlflow is None:
                print("⚠️  MLflow is not installed. Disable --mlflow or install mlflow to enable tracking.")
            else:
                if mlflow_tracking_uri:
                    mlflow.set_tracking_uri(mlflow_tracking_uri)
                mlflow.set_experiment(mlflow_experiment)
                active_mlflow_run = mlflow.start_run(run_name=run_name)
                # Log trainer/agent/env params
                mlflow.log_params({
                    "blind_score": self.blind_score,
                    "device": self.device,
                    "learning_rate": self.agent.learning_rate,
                    "gamma": self.agent.gamma,
                    "gae_lambda": self.agent.gae_lambda,
                    "clip_ratio": self.agent.clip_ratio,
                    "value_loss_coef": self.agent.value_loss_coef,
                    "entropy_coef": self.agent.entropy_coef,
                    "max_grad_norm": self.agent.max_grad_norm,
                    "target_kl": self.agent.target_kl,
                    "hidden_dim": self.agent.actor_critic.shared_layers[0].out_features if hasattr(self.agent.actor_critic, 'shared_layers') else 256,
                    "total_timesteps": total_timesteps,
                    "batch_size": batch_size,
                    "update_epochs": update_epochs,
                    "eval_interval": eval_interval,
                    "eval_log_interval": eval_log_interval,
                    "num_eval_episodes": num_eval_episodes,
                    "save_interval": save_interval,
                    "backup_interval": backup_interval,
                    "demo_interval": demo_interval,
                    "debug_interval": debug_interval,
                    "curriculum_learning": self.curriculum_learning,
                })
        
        # Curriculum learning setup
        if self.curriculum_learning:
            curriculum_stages = [
                {"blind_score": 100, "timesteps": total_timesteps // 4},  # Easy target
                {"blind_score": 200, "timesteps": total_timesteps // 4},  # Medium target
                {"blind_score": 250, "timesteps": total_timesteps // 4},  # Hard target
                {"blind_score": self.blind_score, "timesteps": total_timesteps // 4}  # Final target
            ]
            current_stage = 0
            stage_timesteps = 0
            
            # Start with easier target
            self.env.blind_score = curriculum_stages[current_stage]["blind_score"]
            print(f"📚 Curriculum Stage {current_stage + 1}: Target Score = {self.env.blind_score}")
        
        # Initial evaluation (skip if we have previous data)
        if not self.training_stats['episode_rewards']:
            print("\n📊 Initial Evaluation:")
            initial_eval_stats = self.evaluate(num_eval_episodes)
            self.training_stats['episode_rewards'].append(initial_eval_stats['avg_reward'])
            self.training_stats['episode_lengths'].append(initial_eval_stats['avg_length'])
            self.training_stats['win_rates'].append(initial_eval_stats['win_rate'])
            self.training_stats['action_distributions'].append(initial_eval_stats['action_distribution'])
            print(f"  Average Reward: {initial_eval_stats['avg_reward']:.2f} ± {initial_eval_stats['std_reward']:.2f}")
            print(f"  Win Rate: {initial_eval_stats['win_rate']:.2%}")
            print(f"  Average Length: {initial_eval_stats['avg_length']:.1f}")
            print(f"  Action Distribution: {initial_eval_stats['action_distribution']}")
            print("-" * 40)
            # Log initial metrics
            if active_mlflow_run is not None:
                mlflow.log_metrics({
                    "eval/avg_reward": float(initial_eval_stats['avg_reward']),
                    "eval/win_rate": float(initial_eval_stats['win_rate']),
                    "eval/avg_length": float(initial_eval_stats['avg_length']),
                }, step=timesteps_so_far)
            # Save initial eval to file
            self._write_eval_log(initial_eval_stats, timesteps_so_far)
        else:
            print(f"\n📊 Continuing from previous training with {len(self.training_stats['episode_rewards'])} evaluation points")
            latest_reward = self.training_stats['episode_rewards'][-1]
            latest_win_rate = self.training_stats['win_rates'][-1]
            print(f"  Latest Average Reward: {latest_reward:.2f}")
            print(f"  Latest Win Rate: {latest_win_rate:.2%}")
            print("-" * 40)
        
        # Initialize eval_stats for curriculum learning
        eval_stats = None
        
        # Schedule thresholds to avoid skipping events when buffer steps over multiples
        next_debug_step = debug_interval if debug_interval and debug_interval > 0 else float('inf')
        next_eval_step = eval_interval if eval_interval and eval_interval > 0 else float('inf')
        next_eval_log_step = eval_log_interval if eval_log_interval and eval_log_interval > 0 else float('inf')
        next_demo_step = demo_interval if demo_interval and demo_interval > 0 else float('inf')
        next_backup_step = backup_interval if backup_interval and backup_interval > 0 else float('inf')
        next_save_step = save_interval if save_interval and save_interval > 0 else float('inf')
        
        with tqdm(total=total_timesteps, desc="Fast Training Progress") as pbar:
            while timesteps_so_far < total_timesteps:
                # Curriculum learning: check if we should advance to next stage
                if (self.curriculum_learning and 
                    eval_stats is not None and
                    stage_timesteps >= curriculum_stages[current_stage]["timesteps"] and 
                    eval_stats['win_rate'] > 0.6):
                    if current_stage < len(curriculum_stages) - 1:
                        current_stage += 1
                        stage_timesteps = 0
                        self.env.blind_score = curriculum_stages[current_stage]["blind_score"]
                        print(f"\n📚 Advancing to Curriculum Stage {current_stage + 1}: Target Score = {self.env.blind_score}")
                        
                        # Quick evaluation on new difficulty
                        eval_stats = self.evaluate(num_eval_episodes)
                        print(f"  Performance on new difficulty:")
                        print(f"    Average Reward: {eval_stats['avg_reward']:.2f}")
                        print(f"    Win Rate: {eval_stats['win_rate']:.2%}")
                        print("-" * 40)
                
                # Collect batch
                buffer = self.agent.collect_batch(batch_size)
                timesteps_so_far += buffer.size
                if self.curriculum_learning:
                    stage_timesteps += buffer.size
                
                # Update policy
                update_stats = self.agent.update(buffer, update_epochs)
                
                # Store training stats
                self.training_stats['policy_losses'].append(update_stats['policy_loss'])
                self.training_stats['value_losses'].append(update_stats['value_loss'])
                self.training_stats['entropy_losses'].append(update_stats['entropy_loss'])
                self.training_stats['kl_divergences'].append(update_stats['kl_div'])
                
                # Debug monitoring (threshold-based)
                while timesteps_so_far >= next_debug_step:
                    self._debug_training_step(update_stats, buffer)
                    next_debug_step += debug_interval
                
                # Evaluation and logging (threshold-based; ensure no skips if we stepped over multiples)
                need_console_eval = False
                need_log_eval = False
                while timesteps_so_far >= next_eval_step:
                    need_console_eval = True
                    next_eval_step += eval_interval
                while timesteps_so_far >= next_eval_log_step:
                    need_log_eval = True
                    next_eval_log_step += eval_log_interval

                if need_console_eval or need_log_eval:
                    eval_stats = self.evaluate(num_eval_episodes)
                    # Update tracked stats once per eval
                    self.training_stats['episode_rewards'].append(eval_stats['avg_reward'])
                    self.training_stats['episode_lengths'].append(eval_stats['avg_length'])
                    self.training_stats['win_rates'].append(eval_stats['win_rate'])
                    self.training_stats['action_distributions'].append(eval_stats['action_distribution'])

                    if need_console_eval:
                        print(f"\n📊 Evaluation at {timesteps_so_far:,} timesteps:")
                        if self.curriculum_learning:
                            print(f"  Curriculum Stage: {current_stage + 1}/{len(curriculum_stages)}")
                        print(f"  Average Reward: {eval_stats['avg_reward']:.2f} ± {eval_stats['std_reward']:.2f}")
                        print(f"  Win Rate: {eval_stats['win_rate']:.2%}")
                        print(f"  Average Length: {eval_stats['avg_length']:.1f}")
                        print(f"  Policy Loss: {update_stats['policy_loss']:.4f}")
                        print(f"  Value Loss: {update_stats['value_loss']:.4f}")
                        print(f"  KL Divergence: {update_stats['kl_div']:.4f}")
                        print("-" * 40)

                    if need_log_eval:
                        self._write_eval_log(eval_stats, timesteps_so_far)

                    # MLflow metrics
                    if active_mlflow_run is not None:
                        mlflow.log_metrics({
                            "train/policy_loss": float(update_stats['policy_loss']),
                            "train/value_loss": float(update_stats['value_loss']),
                            "train/entropy_loss": float(update_stats['entropy_loss']),
                            "train/kl_div": float(update_stats['kl_div']),
                            "eval/avg_reward": float(eval_stats['avg_reward']),
                            "eval/win_rate": float(eval_stats['win_rate']),
                            "eval/avg_length": float(eval_stats['avg_length']),
                            "eval/avg_hands_per_episode": float(eval_stats.get('avg_plays_per_episode', 0.0)),
                            "eval/avg_discards_per_episode": float(eval_stats.get('avg_discards_per_episode', 0.0)),
                        }, step=timesteps_so_far)
                
                # Demo episode (threshold-based)
                while timesteps_so_far >= next_demo_step:
                    print(f"\n🎮 Demo Episode at {timesteps_so_far:,} timesteps:")
                    if self.curriculum_learning:
                        print(f"  Curriculum Stage: {current_stage + 1}/{len(curriculum_stages)}")
                    self._play_demo_episode(max_steps=5)  # Shorter demo
                    next_demo_step += demo_interval
                
                # Save backup model (threshold-based)
                while timesteps_so_far >= next_backup_step:
                    backup_path = f"backups/ppo_balatro_fast_backup_{timesteps_so_far}.pth"
                    self.agent.save_model(backup_path)
                    print(f"\n💾 Backup model saved to {backup_path}")
                    next_backup_step += backup_interval
                
                # Save model and plots (threshold-based)
                while timesteps_so_far >= next_save_step:
                    model_path = f"checkpoints/ppo_balatro_fast_{self.run_timestamp}_{timesteps_so_far}.pth"
                    self.agent.save_model(model_path)
                    print(f"\n💾 Model saved to {model_path}")
                    
                    # Plot current training curves
                    plot_path = f"plots/fast_training_curves_{self.run_timestamp}_{timesteps_so_far}.png"
                    self.plot_training_curves(save_path=plot_path)
                    # Log artifacts to MLflow
                    if active_mlflow_run is not None:
                        mlflow.log_artifact(model_path)
                        mlflow.log_artifact(plot_path)
                    next_save_step += save_interval
                
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
        self.training_stats['win_rates'].append(final_eval_stats['win_rate'])
        self.training_stats['action_distributions'].append(final_eval_stats['action_distribution'])
        print(f"  Average Reward: {final_eval_stats['avg_reward']:.2f} ± {final_eval_stats['std_reward']:.2f}")
        print(f"  Win Rate: {final_eval_stats['win_rate']:.2%}")
        print(f"  Average Length: {final_eval_stats['avg_length']:.1f}")
        print(f"  Action Distribution: {final_eval_stats['action_distribution']}")
        print("-" * 40)
        # Save final eval log
        self._write_eval_log(final_eval_stats, timesteps_so_far, final=True)
        if active_mlflow_run is not None:
            mlflow.log_metrics({
                "final/avg_reward": float(final_eval_stats['avg_reward']),
                "final/win_rate": float(final_eval_stats['win_rate']),
                "final/avg_length": float(final_eval_stats['avg_length']),
            }, step=timesteps_so_far)
        
        # Final save and evaluation
        final_model_path = f"checkpoints/ppo_balatro_fast_final_{self.run_timestamp}.pth"
        final_plot_path = f"plots/fast_final_training_curves_{self.run_timestamp}.png"
        self.agent.save_model(final_model_path)
        self.plot_training_curves(save_path=final_plot_path)
        if active_mlflow_run is not None:
            mlflow.log_artifact(final_model_path)
            # Do not log final plot to MLflow per request
            mlflow.end_run()
        
        print(f"\n🎉 Fast training completed in {training_time:.1f} seconds!")
        print(f"Final model saved to: {final_model_path}")
        print(f"Training curves saved to: {final_plot_path}")
        
        return self.agent

    def _write_eval_log(self, eval_stats: Dict[str, Any], timesteps: int, final: bool = False) -> None:
        """No-op log writer to keep compatibility when file logs are disabled."""
        try:
            # Metrics are already logged to MLflow elsewhere; keep this as a stub.
            return
        except Exception:
            return
    
    def _debug_training_step(self, update_stats: Dict, buffer):
        """Quick debug training step"""
        print(f"\n🔍 Quick Debug at {len(self.training_stats['policy_losses'])} updates:")
        print(f"  Policy Loss: {update_stats['policy_loss']:.4f}")
        print(f"  Value Loss: {update_stats['value_loss']:.4f}")
        print(f"  KL Divergence: {update_stats['kl_div']:.4f}")
        print(f"  Buffer Size: {buffer.size}")
    
    def evaluate(self, num_episodes: int = 5) -> Dict[str, Any]:
        """Evaluate current policy with fewer episodes for speed"""
        rewards = []
        lengths = []
        wins = 0
        action_counts = {"play": 0, "discard": 0, "pass": 0}
        plays_per_episode: List[int] = []
        discards_per_episode: List[int] = []
        
        for _ in range(num_episodes):
            obs, _ = self.env.reset()
            obs = torch.FloatTensor(obs).to(self.device)
            episode_reward = 0
            episode_length = 0
            episode_actions = {"play": 0, "discard": 0, "pass": 0}
            
            while True:
                with torch.no_grad():
                    action_logits, _ = self.agent.actor_critic(obs.unsqueeze(0))
                    action_probs = [F.softmax(logits, dim=-1) for logits in action_logits]
                    action = [torch.argmax(probs).item() for probs in action_probs]
                
                action_type, _ = self.env._decode_multi_head_action(np.array(action))
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
            action_counts["pass"] += episode_actions["pass"]
            plays_per_episode.append(episode_actions["play"])
            discards_per_episode.append(episode_actions["discard"])
        
        total_actions = action_counts["play"] + action_counts["discard"] + action_counts["pass"]
        action_distribution = {
            "play": f"{action_counts['play']/total_actions:.1%}" if total_actions > 0 else "0%",
            "discard": f"{action_counts['discard']/total_actions:.1%}" if total_actions > 0 else "0%",
            "pass": f"{action_counts['pass']/total_actions:.1%}" if total_actions > 0 else "0%"
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
            'lengths': lengths,
            'plays_per_episode': plays_per_episode,
            'discards_per_episode': discards_per_episode,
            'avg_plays_per_episode': float(np.mean(plays_per_episode)) if len(plays_per_episode) > 0 else 0.0,
            'avg_discards_per_episode': float(np.mean(discards_per_episode)) if len(discards_per_episode) > 0 else 0.0,
        }
    
    def _play_demo_episode(self, max_steps: int = 5):
        """Play a short demonstration episode"""
        obs, _ = self.env.reset()
        obs = torch.FloatTensor(obs).to(self.device)
        total_reward = 0
        step = 0
        
        print(f"  Initial hand: {[str(card) for card in self.env.hand]}")
        print(f"  Target score: {self.env.blind_score}")
        print()
        
        while step < max_steps:
            step += 1
            
            # Get action from policy
            with torch.no_grad():
                action_logits, value = self.agent.actor_critic(obs.unsqueeze(0))
                action_probs = [F.softmax(logits, dim=-1) for logits in action_logits]
                action = [torch.argmax(probs).item() for probs in action_probs]
            
            # Decode action
            action_type, card_indices = self.env._decode_multi_head_action(np.array(action))
            cards_to_play = [str(self.env.hand[i]) for i in card_indices if i < len(self.env.hand)]
            
            # Take action
            obs, reward, done, truncated, info = self.env.step(action)
            obs = torch.FloatTensor(obs).to(self.device)
            total_reward += reward
            
            # Show action details
            print(f"  Step {step}: {action_type.upper()} {cards_to_play} (reward: {reward:.2f})")
            print(f"    Score: {self.env.current_score}/{self.env.blind_score}, Plays: {self.env.plays_left}, Discards: {self.env.discards_left}")
            
            if done or truncated:
                break
        
        print(f"  Final result: {'WIN' if self.env.won else 'LOSS'} (Total reward: {total_reward:.2f})")
        print()
    
    def plot_training_curves(self, save_path: str = "plots/fast_training_curves.png"):
        """Plot training curves"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Keep plots compact to limit file size and on-screen footprint
        fig, axes = plt.subplots(2, 2, figsize=(10, 6))
        
        # Episode rewards
        if self.training_stats['episode_rewards']:
            axes[0, 0].plot(self.training_stats['episode_rewards'], 'b-', linewidth=2, marker='o')
            axes[0, 0].set_title('Episode Rewards (Fast Training)', fontsize=14, fontweight='bold')
            axes[0, 0].set_xlabel('Evaluation Step')
            axes[0, 0].set_ylabel('Average Reward')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Win rate
        if self.training_stats['win_rates']:
            axes[0, 1].plot(self.training_stats['win_rates'], 'g-', linewidth=2, marker='o')
            axes[0, 1].set_title('Win Rate (Fast Training)', fontsize=14, fontweight='bold')
            axes[0, 1].set_xlabel('Evaluation Step')
            axes[0, 1].set_ylabel('Win Rate')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].set_ylim(0, 1)
        
        # Policy loss
        if self.training_stats['policy_losses']:
            axes[1, 0].plot(self.training_stats['policy_losses'], 'purple', linewidth=2, label='Policy Loss')
            axes[1, 0].set_title('Policy Loss (Fast Training)', fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('Update Step')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].legend()
        
        # Value loss
        if self.training_stats['value_losses']:
            axes[1, 1].plot(self.training_stats['value_losses'], 'orange', linewidth=2, label='Value Loss')
            axes[1, 1].set_title('Value Loss (Fast Training)', fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel('Update Step')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].legend()
        
        plt.tight_layout()
        # Cap DPI to keep file size manageable
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Fast training curves saved to {save_path}")
        # plt.show()

def main():
    parser = argparse.ArgumentParser(description="Fast PPO Training for Balatro")
    parser.add_argument("--blind-score", type=int, default=300, help="Target score to win")
    parser.add_argument("--timesteps", type=int, default=100000, help="Total timesteps for training")
    parser.add_argument("--batch-size", type=int, default=4096, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    parser.add_argument("--eval-interval", type=int, default=10000, help="Evaluation interval for on-screen metrics")
    parser.add_argument("--eval-log-interval", type=int, default=1000, help="Interval to save 5-episode evaluation logs to file")
    parser.add_argument("--save-interval", type=int, default=50000, help="Save interval")
    parser.add_argument("--load-model", type=str, help="Path to model file to continue training from")
    parser.add_argument("--no-curriculum", action="store_true", help="Disable curriculum learning")
    # PPO hyperparameters
    parser.add_argument("--entropy-coef", type=float, default=0.01, help="Entropy coefficient for exploration")
    parser.add_argument("--clip-ratio", type=float, default=0.2, help="PPO clip ratio")
    parser.add_argument("--target-kl", type=float, default=0.01, help="Target KL divergence for early stopping")
    # MLflow options
    parser.add_argument("--mlflow", action="store_true", help="Enable MLflow experiment tracking")
    parser.add_argument("--mlflow-experiment", type=str, default="BalatroV2", help="MLflow experiment name")
    parser.add_argument("--mlflow-tracking-uri", type=str, default=None, help="MLflow tracking URI (optional)")
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = FastPPOTrainer(
        blind_score=args.blind_score,
        learning_rate=args.lr,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=args.clip_ratio,
        value_loss_coef=0.5,
        entropy_coef=args.entropy_coef,
        max_grad_norm=0.5,
        target_kl=args.target_kl,
        hidden_dim=args.hidden_dim,
        device=args.device,
        load_model_path=args.load_model,
        curriculum_learning=not args.no_curriculum
    )
    
    # Train
    trainer.train(
        total_timesteps=args.timesteps,
        batch_size=args.batch_size,
        eval_interval=args.eval_interval,
        eval_log_interval=args.eval_log_interval,
        save_interval=args.save_interval,
        mlflow_logging=args.mlflow,
        mlflow_experiment=args.mlflow_experiment,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
    )

if __name__ == "__main__":
    main() 