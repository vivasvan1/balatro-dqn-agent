#!/usr/bin/env python3
"""
PPO (Proximal Policy Optimization) Agent for Balatro Gym Environment

This module contains a complete, self-contained PPO implementation tailored to
the simplified Balatro environment that exposes an OpenAI Five-style
multi-head action space. The code is organized into three primary parts:

- MultiHeadActorCritic: a neural network with a shared torso and multiple
  independent policy heads (one per action head), plus a value head.
- PPOBuffer: a fixed-size, contiguous buffer used to store a single batch of
  transitions for on-policy PPO updates.
- PPOAgent: the orchestration layer that collects batches, computes advantages
  and returns, performs PPO updates, evaluates, and plots training progress.

Where helpful, comments explain the intent behind design decisions (the "why"),
not just the mechanics (the "how").
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

from balatro_gym_v2_simple import BalatroGymEnvSimple  # Environment interface


class MultiHeadActorCritic(nn.Module):
    """
    Multi-head Actor-Critic neural network for PPO with OpenAI Five-style actions
    Actor: outputs action probabilities for each action head
    Critic: outputs state value
    """

    def __init__(self, state_dim: int, action_dims: List[int], hidden_dim: int = 256):
        super(MultiHeadActorCritic, self).__init__()

        # Shared layers: encode state into a compact latent representation
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Multi-head actor (policy) heads: one categorical distribution per action head
        self.actor_heads = nn.ModuleList()
        for action_dim in action_dims:
            actor_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, action_dim),
            )
            self.actor_heads.append(actor_head)

        # Critic (value) head: predicts V(s) to compute advantages
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Initialize weights for stability (orthogonal init)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            module.bias.data.zero_()
        # Initialize value head with smaller weights for stability
        if isinstance(module, nn.Linear) and module.out_features == 1:
            torch.nn.init.orthogonal_(module.weight, gain=0.01)
            module.bias.data.zero_()

    def forward(self, state):
        shared_features = self.shared_layers(state)

        # Actor: logits for each action head
        action_logits = []
        for actor_head in self.actor_heads:
            logits = actor_head(shared_features)
            action_logits.append(logits)

        # Critic: state value
        value = self.critic(shared_features)

        return action_logits, value

    def get_action_probs(self, state):
        """Get action probabilities from state for each head"""
        action_logits, _ = self.forward(state)
        action_probs = []
        for logits in action_logits:
            probs = F.softmax(logits, dim=-1)
            action_probs.append(probs)
        return action_probs

    def get_value(self, state):
        """Get state value"""
        _, value = self.forward(state)
        return value


class PPOBuffer:
    """
    Buffer for storing PPO training data with multi-head actions
    """

    def __init__(
        self,
        buffer_size: int,
        state_dim: int,
        action_dims: List[int],
        device: str = "cpu",
    ):
        self.buffer_size = buffer_size
        self.device = device
        self.action_dims = action_dims

        # Storage tensors (preallocated for performance)
        self.states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self.actions = [  # list-of-tensors, aligned with multi-head discrete actions
            torch.zeros(buffer_size, dtype=torch.long, device=device)
            for _ in action_dims
        ]
        self.rewards = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.values = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.log_probs = [  # per-head log-probs of sampled actions
            torch.zeros(buffer_size, dtype=torch.float32, device=device)
            for _ in action_dims
        ]
        self.dones = torch.zeros(buffer_size, dtype=torch.bool, device=device)

        self.ptr = 0
        self.size = 0

    def add(self, state, actions, reward, value, log_probs, done):
        """Add a transition to the buffer"""
        self.states[self.ptr] = state
        for i, action in enumerate(actions):
            self.actions[i][self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        for i, log_prob in enumerate(log_probs):
            self.log_probs[i][self.ptr] = log_prob
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def get_all(self):
        """Get all data from buffer"""
        return (
            self.states[: self.size],
            [action[: self.size] for action in self.actions],
            self.rewards[: self.size],
            self.values[: self.size],
            [log_prob[: self.size] for log_prob in self.log_probs],
            self.dones[: self.size],
        )

    def clear(self):
        """Clear the buffer"""
        self.ptr = 0
        self.size = 0


class PPOAgent:
    """
    PPO Agent for Balatro environment with multi-head actions
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
        device: str = "cpu",
    ):
        self.env = env
        self.device = device

        # PPO hyperparameters (exposed for clarity/tuning)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl

        # Networks: infer dimensions from env spaces
        state_dim = env.observation_space.shape[0]
        action_dims = env.action_space.nvec.tolist()  # Multi-head action dimensions

        self.actor_critic = MultiHeadActorCritic(state_dim, action_dims, hidden_dim).to(
            device
        )
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)

        # Training stats
        self.training_stats = {
            "episode_rewards": [],
            "episode_lengths": [],
            "win_rates": [],
            "policy_losses": [],
            "value_losses": [],
            "entropy_losses": [],
            "kl_divergences": [],
            "advantages": [],
            "policy_ratios": [],
            "entropies": [],
            "action_distributions": [],
            "avg_advantages": [],
            "policy_confidence": [],
        }

    def compute_gae(self, rewards, values, dones, next_value):
        """Compute Generalized Advantage Estimation"""
        # Advantages buffer (same shape as rewards)
        advantages = torch.zeros_like(rewards)
        last_advantage = 0

        # Convert dones to float for arithmetic operations
        dones_float = dones.float()

        # Walk backward through the rollout computing temporal-difference residuals (delta)
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value_t = next_value
            else:
                next_value_t = values[t + 1]

            delta = (
                rewards[t]
                + self.gamma * next_value_t * (1 - dones_float[t])
                - values[t]
            )
            # Standard GAE recurrence
            advantages[t] = (
                delta
                + self.gamma * self.gae_lambda * (1 - dones_float[t]) * last_advantage
            )
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
                action_probs = [torch.softmax(logits, dim=-1) for logits in action_logits]
                action = [torch.argmax(probs).item() for probs in action_probs]

            # Decode action
            action_type, card_indices = self.env._decode_multi_head_action(np.array(action))

            # Capture cards that will be played BEFORE taking the action
            cards_to_play = [
                str(self.env.hand[i]) for i in card_indices if i < len(self.env.hand)
            ]

            # Take action
            obs, reward, done, truncated, info = self.env.step(action)
            obs = torch.FloatTensor(obs).to(self.device)
            total_reward += reward

            # Show step outcome only (omit action details)
            print(
                f"    Score: {self.env.current_score}/{self.env.blind_score}, Plays: {self.env.plays_left}, Discards: {self.env.discards_left}"
            )

            # Show new hand after the action
            print(f"    New hand: {[str(card) for card in self.env.hand]}")
            print()

            if done or truncated:
                break

        print(
            f"  Final result: {'WIN' if self.env.won else 'LOSS'} (Total reward: {total_reward:.2f})"
        )
        print()

    def compute_loss(self, states, actions, old_log_probs, advantages, returns):
        """Compute PPO loss with multi-head actions"""
        action_logits, values = self.actor_critic(states)

        # Policy loss for each action head
        policy_loss = 0.0
        entropy_loss = 0.0
        kl_div = 0.0

        for i, (logits, action, old_log_prob) in enumerate(
            zip(action_logits, actions, old_log_probs)
        ):
            action_probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(action_probs)
            log_probs = dist.log_prob(action)

            ratio = torch.exp(log_probs - old_log_prob)
            surr1 = ratio * advantages
            surr2 = (
                torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
                * advantages
            )
            policy_loss += -torch.min(surr1, surr2).mean()

            # Entropy loss (for exploration)
            entropy_loss += -dist.entropy().mean()

            # KL divergence for early stopping
            kl_div += (old_log_prob - log_probs).mean()

        # Average over action heads
        num_heads = len(action_logits)
        policy_loss /= num_heads
        entropy_loss /= num_heads
        kl_div /= num_heads

        # Value loss
        value_loss = F.mse_loss(values.squeeze(), returns)

        # Total loss
        total_loss = (
            policy_loss
            + self.value_loss_coef * value_loss
            + self.entropy_coef * entropy_loss
        )

        return total_loss, policy_loss, value_loss, entropy_loss, kl_div

    def collect_batch(
        self, batch_size: int, max_steps_per_episode: int = 1000
    ) -> PPOBuffer:
        """Collect a batch of experiences with multi-head actions"""
        action_dims = self.env.action_space.nvec.tolist()
        buffer = PPOBuffer(
            batch_size, self.env.observation_space.shape[0], action_dims, self.device
        )
        timesteps = 0

        while timesteps < batch_size:
            obs, _ = self.env.reset()
            obs = torch.FloatTensor(obs).to(self.device)
            episode_reward = 0
            episode_length = 0

            while timesteps < batch_size:
                with torch.no_grad():
                    action_logits, value = self.actor_critic(obs.unsqueeze(0))

                    # Sample actions for each head
                    actions = []
                    log_probs = []

                    for i, logits in enumerate(action_logits):
                        # Apply action masking if available; use same dist for sampling and log_prob
                        use_logits = logits
                        if hasattr(self.env, "get_action_mask"):
                            masks = self.env.get_action_mask()
                            if i < len(masks):
                                mask = torch.tensor(
                                    masks[i], dtype=torch.bool, device=self.device
                                )
                                masked_logits = logits.clone()
                                masked_logits[0][~mask] = -1e9
                                use_logits = masked_logits
                        action_probs = F.softmax(use_logits, dim=-1)

                        action = torch.multinomial(action_probs, 1).item()
                        log_prob = F.log_softmax(use_logits, dim=-1)[0, action].item()

                        actions.append(action)
                        log_probs.append(log_prob)

                    # Convert to numpy array for environment
                    action_array = np.array(actions)

                next_obs, reward, done, truncated, _ = self.env.step(action_array)
                next_obs = torch.FloatTensor(next_obs).to(self.device)

                buffer.add(
                    obs, actions, reward, value.item(), log_probs, done or truncated
                )
                timesteps += 1
                obs = next_obs
                episode_reward += reward
                episode_length += 1

                if done or truncated:
                    break

                if timesteps >= batch_size:
                    break

        return buffer

    def update(self, buffer: PPOBuffer, epochs: int = 10) -> Dict[str, float]:
        """Update policy using PPO with multi-head actions"""
        states, actions, rewards, values, log_probs, dones = buffer.get_all()

        # Compute GAE for the entire buffer
        if dones[-1].item():
            next_value = 0.0
        else:
            with torch.no_grad():
                next_obs = states[-1]
                _, next_value = self.actor_critic(next_obs.unsqueeze(0))
                next_value = next_value.squeeze()

        advantages, returns = self.compute_gae(rewards, values, dones, next_value)

        # Normalize advantages; add robust fallback if advantages collapse to ~0
        if len(advantages) > 0:
            adv_std = advantages.std().item()
            adv_max_abs = advantages.abs().max().item()
            if adv_std > 1e-6:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            else:
                # If advantages are numerically near-constant or zero, try a simple reward baseline
                approx_adv = rewards - rewards.mean()
                approx_std = approx_adv.std().item()
                if approx_std > 1e-6:
                    advantages = (approx_adv - approx_adv.mean()) / (approx_adv.std() + 1e-8)
                elif adv_max_abs <= 1e-8:
                    # As a last resort, set a small non-zero advantage to encourage learning
                    advantages = torch.full_like(advantages, 0.1)

        # PPO update
        update_stats = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy_loss": 0.0,
            "kl_div": 0.0,
        }

        num_updates = 0

        for epoch in range(epochs):
            # Shuffle data
            indices = torch.randperm(len(states))

            for start_idx in range(0, len(states), 64):  # Mini-batch size of 64
                end_idx = min(start_idx + 64, len(states))
                batch_indices = indices[start_idx:end_idx]

                batch_states = states[batch_indices]
                batch_actions = [action[batch_indices] for action in actions]
                batch_old_log_probs = [
                    log_prob[batch_indices] for log_prob in log_probs
                ]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # Compute loss
                total_loss, policy_loss, value_loss, entropy_loss, kl_div = (
                    self.compute_loss(
                        batch_states,
                        batch_actions,
                        batch_old_log_probs,
                        batch_advantages,
                        batch_returns,
                    )
                )

                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.actor_critic.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                # Accumulate stats
                update_stats["policy_loss"] += policy_loss.item()
                update_stats["value_loss"] += value_loss.item()
                update_stats["entropy_loss"] += entropy_loss.item()
                update_stats["kl_div"] += kl_div.item()
                num_updates += 1

            # Early stopping if KL divergence is too high
            avg_kl = update_stats["kl_div"] / num_updates if num_updates > 0 else 0
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
        update_epochs: int = 8,
        eval_interval: int = 10000,
        num_eval_episodes: int = 10,
        save_interval: int = 50000,
        demo_interval: int = 20000,
    ):
        """Train the PPO agent"""
        timesteps_so_far = 0
        episode_count = 0

        print(f"Starting PPO training for {total_timesteps} timesteps")
        print(f"Batch size: {batch_size}, Update epochs: {update_epochs}")
        print(f"Action space: {self.env.action_space}")
        print("=" * 60)

        with tqdm(total=total_timesteps, desc="Training Progress") as pbar:
            while timesteps_so_far < total_timesteps:
                # Collect batch
                buffer = self.collect_batch(batch_size)
                timesteps_so_far += buffer.size

                # Update policy
                update_stats = self.update(buffer, update_epochs)

                # Store training stats (every update, not just at eval intervals)
                self.training_stats["value_losses"].append(update_stats["value_loss"])
                self.training_stats["policy_losses"].append(update_stats["policy_loss"])
                self.training_stats["entropy_losses"].append(
                    update_stats["entropy_loss"]
                )
                self.training_stats["kl_divergences"].append(update_stats["kl_div"])

                # Evaluation
                if timesteps_so_far % eval_interval == 0:
                    eval_stats = self.evaluate(num_eval_episodes)
                    self.training_stats["episode_rewards"].append(
                        eval_stats["avg_reward"]
                    )
                    self.training_stats["episode_lengths"].append(
                        eval_stats["avg_length"]
                    )
                    self.training_stats["win_rates"].append(eval_stats["win_rate"])

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
                    self.plot_training_curves(
                        save_path=f"training_curves_{timesteps_so_far}.png"
                    )

                pbar.update(buffer.size)
                pbar.set_postfix(
                    {
                        "Policy Loss": f"{update_stats['policy_loss']:.4f}",
                        "Value Loss": f"{update_stats['value_loss']:.4f}",
                        "KL Div": f"{update_stats['kl_div']:.4f}",
                        "Timesteps": f"{timesteps_so_far}",
                    }
                )

        print("Training completed!")
        self.save_model("ppo_balatro_final.pth")
        self.plot_training_curves(save_path="final_training_curves.png")

    def evaluate(self, num_episodes: int = 10) -> Dict[str, Any]:
        """Evaluate current policy with multi-head actions"""
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

                    # Get best actions for each head
                    actions = []
                    for i, logits in enumerate(action_logits):
                        # Apply action masking if available
                        if hasattr(self.env, "get_action_mask"):
                            masks = self.env.get_action_mask()
                            if i < len(masks):
                                mask = torch.tensor(
                                    masks[i], dtype=torch.bool, device=self.device
                                )
                                masked_logits = logits.clone()
                                masked_logits[0][~mask] = -1e9
                                action_probs = F.softmax(masked_logits, dim=-1)
                            else:
                                action_probs = F.softmax(logits, dim=-1)
                        else:
                            action_probs = F.softmax(logits, dim=-1)

                        action = torch.argmax(action_probs).item()
                        actions.append(action)

                    # Convert to numpy array for environment
                    action_array = np.array(actions)

                # Step environment with the selected actions for all heads
                obs, reward, done, truncated, info = self.env.step(action_array)
                obs = torch.FloatTensor(obs).to(self.device)
                episode_reward += reward
                episode_length += 1

                if done or truncated:
                    if info.get("won", False):
                        wins += 1
                    break

            rewards.append(episode_reward)
            lengths.append(episode_length)

        return {
            "avg_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "min_reward": np.min(rewards) if len(rewards) > 0 else 0.0,
            "max_reward": np.max(rewards) if len(rewards) > 0 else 0.0,
            "avg_length": np.mean(lengths),
            "win_rate": wins / num_episodes,
            "rewards": rewards,
            "lengths": lengths,
        }

    def save_model(self, filename: str):
        """Save the model"""
        torch.save(
            {
                "actor_critic_state_dict": self.actor_critic.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "training_stats": self.training_stats,
            },
            filename,
        )
        print(f"Model saved to {filename}")

    def load_model(self, filename: str):
        """Load the model"""
        checkpoint = torch.load(filename, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint["actor_critic_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Load training stats with compatibility handling
        loaded_stats = checkpoint.get("training_stats", {})
        for key in self.training_stats:
            if key in loaded_stats:
                self.training_stats[key] = loaded_stats[key]
            # If key doesn't exist in loaded stats, keep the default empty list

        print(f"Model loaded from {filename}")

    def plot_training_curves(self, save_path: str = "ppo_training_curves.png"):
        """Plot training curves with detailed statistics"""
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))

        # Episode rewards
        if self.training_stats["episode_rewards"]:
            axes[0, 0].plot(self.training_stats["episode_rewards"], "b-", linewidth=2)
            axes[0, 0].set_title("Episode Rewards", fontsize=14, fontweight="bold")
            axes[0, 0].set_xlabel("Evaluation Step")
            axes[0, 0].set_ylabel("Average Reward")
            axes[0, 0].grid(True, alpha=0.3)

        # Win rate
        if self.training_stats["win_rates"]:
            axes[0, 1].plot(self.training_stats["win_rates"], "g-", linewidth=2)
            axes[0, 1].set_title("Win Rate", fontsize=14, fontweight="bold")
            axes[0, 1].set_xlabel("Evaluation Step")
            axes[0, 1].set_ylabel("Win Rate")
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].set_ylim(0, 1)

        # Episode lengths
        if self.training_stats["episode_lengths"]:
            axes[0, 2].plot(self.training_stats["episode_lengths"], "r-", linewidth=2)
            axes[0, 2].set_title("Episode Lengths", fontsize=14, fontweight="bold")
            axes[0, 2].set_xlabel("Evaluation Step")
            axes[0, 2].set_ylabel("Average Length")
            axes[0, 2].grid(True, alpha=0.3)

        # Policy loss
        if self.training_stats["policy_losses"]:
            axes[1, 0].plot(
                self.training_stats["policy_losses"],
                "purple",
                linewidth=2,
                label="Policy Loss",
            )
            axes[1, 0].set_title("Policy Loss", fontsize=14, fontweight="bold")
            axes[1, 0].set_xlabel("Update Step")
            axes[1, 0].set_ylabel("Loss")
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].legend()

        # Value loss
        if self.training_stats["value_losses"]:
            axes[1, 1].plot(
                self.training_stats["value_losses"],
                "orange",
                linewidth=2,
                label="Value Loss",
            )
            axes[1, 1].set_title("Value Loss", fontsize=14, fontweight="bold")
            axes[1, 1].set_xlabel("Update Step")
            axes[1, 1].set_ylabel("Loss")
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].legend()

        # Entropy loss
        if self.training_stats["entropy_losses"]:
            axes[1, 2].plot(
                self.training_stats["entropy_losses"],
                "brown",
                linewidth=2,
                label="Entropy Loss",
            )
            axes[1, 2].set_title("Entropy Loss", fontsize=14, fontweight="bold")
            axes[1, 2].set_xlabel("Update Step")
            axes[1, 2].set_ylabel("Loss")
            axes[1, 2].grid(True, alpha=0.3)
            axes[1, 2].legend()

        # KL divergence
        if self.training_stats["kl_divergences"]:
            axes[2, 0].plot(
                self.training_stats["kl_divergences"],
                "teal",
                linewidth=2,
                label="KL Divergence",
            )
            axes[2, 0].set_title("KL Divergence", fontsize=14, fontweight="bold")
            axes[2, 0].set_xlabel("Update Step")
            axes[2, 0].set_ylabel("KL Div")
            axes[2, 0].grid(True, alpha=0.3)
            axes[2, 0].legend()

        # Combined losses
        if (
            self.training_stats["policy_losses"]
            and self.training_stats["value_losses"]
            and self.training_stats["entropy_losses"]
        ):
            axes[2, 1].plot(
                self.training_stats["policy_losses"],
                "purple",
                linewidth=2,
                label="Policy",
            )
            axes[2, 1].plot(
                self.training_stats["value_losses"],
                "orange",
                linewidth=2,
                label="Value",
            )
            axes[2, 1].plot(
                self.training_stats["entropy_losses"],
                "brown",
                linewidth=2,
                label="Entropy",
            )
            axes[2, 1].set_title("All Training Losses", fontsize=14, fontweight="bold")
            axes[2, 1].set_xlabel("Update Step")
            axes[2, 1].set_ylabel("Loss")
            axes[2, 1].grid(True, alpha=0.3)
            axes[2, 1].legend()

        # Training summary
        if self.training_stats["policy_losses"]:
            total_updates = len(self.training_stats["policy_losses"])
            avg_policy_loss = (
                np.mean(self.training_stats["policy_losses"][-10:])
                if total_updates >= 10
                else np.mean(self.training_stats["policy_losses"])
            )
            avg_value_loss = (
                np.mean(self.training_stats["value_losses"][-10:])
                if total_updates >= 10
                else np.mean(self.training_stats["value_losses"])
            )

            axes[2, 2].text(
                0.1,
                0.8,
                f"Total Updates: {total_updates}",
                fontsize=12,
                transform=axes[2, 2].transAxes,
            )
            axes[2, 2].text(
                0.1,
                0.7,
                f"Avg Policy Loss: {avg_policy_loss:.4f}",
                fontsize=12,
                transform=axes[2, 2].transAxes,
            )
            axes[2, 2].text(
                0.1,
                0.6,
                f"Avg Value Loss: {avg_value_loss:.4f}",
                fontsize=12,
                transform=axes[2, 2].transAxes,
            )

            if self.training_stats["win_rates"]:
                latest_win_rate = self.training_stats["win_rates"][-1]
                axes[2, 2].text(
                    0.1,
                    0.5,
                    f"Latest Win Rate: {latest_win_rate:.2%}",
                    fontsize=12,
                    transform=axes[2, 2].transAxes,
                )

            if self.training_stats["episode_rewards"]:
                latest_reward = self.training_stats["episode_rewards"][-1]
                axes[2, 2].text(
                    0.1,
                    0.4,
                    f"Latest Avg Reward: {latest_reward:.2f}",
                    fontsize=12,
                    transform=axes[2, 2].transAxes,
                )

            axes[2, 2].set_title("Training Summary", fontsize=14, fontweight="bold")
            axes[2, 2].axis("off")

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
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
        device=device,
    )

    # Train the agent
    agent.train(
        total_timesteps=500000,  # Adjust based on your needs
        batch_size=2048,
        update_epochs=8,
        eval_interval=10000,
        num_eval_episodes=10,
        save_interval=50000,
    )

    # Plot training curves
    agent.plot_training_curves()

    # Final evaluation
    print("\nFinal Evaluation:")
    eval_stats = agent.evaluate(100)
    print(
        f"Average Reward: {eval_stats['avg_reward']:.2f} ± {eval_stats['std_reward']:.2f}"
    )
    print(f"Win Rate: {eval_stats['win_rate']:.2%}")
    print(
        f"Min/Max Reward: {eval_stats['min_reward']:.2f}/{eval_stats['max_reward']:.2f}"
    )


if __name__ == "__main__":
    main()
