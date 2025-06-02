from datetime import datetime
import os

# from datetime import datetime

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(SCRIPT_DIR, "logs")

# Ensure logs directory exists at startup
os.makedirs(LOGS_DIR, exist_ok=True)
print(f"📁 Logs directory: {LOGS_DIR}")

from typing import Dict, Any, Optional, List, Union, Tuple
import traceback
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
from collections import deque
import json
import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from models.dqn_agent import DQNAgent
from mlflow_tracker import MLflowTracker
from config.settings import (
    STATE_SIZE,
    ACTION_SIZE,
    BUFFER_SIZE,
    BATCH_SIZE,
    GAMMA,
    LEARNING_RATE,
    TAU,
    UPDATE_EVERY,
    INITIAL_EPSILON,
    EPSILON_DECAY,
    FINAL_EPSILON,
    MODEL_WEIGHTS_PATH,
)

app = FastAPI(
    title="Balatro DQN Agent API",
    description="API for interacting with the Balatro DQN agent",
    version="1.0.0",
)

# Initialize the DQN agent
agent = DQNAgent(
    state_size=STATE_SIZE,
    action_size=ACTION_SIZE,
    buffer_size=BUFFER_SIZE,
    batch_size=BATCH_SIZE,
    gamma=GAMMA,
    lr=LEARNING_RATE,
    tau=TAU,
    update_every=UPDATE_EVERY,
    initial_epsilon=INITIAL_EPSILON,
    epsilon_decay=EPSILON_DECAY,
    final_epsilon=FINAL_EPSILON,
)

# DEPRECATED: Custom Performance tracking - replaced by MLflow
# class PerformanceTracker:
    # def __init__(self, max_history=1000):
    #     self.max_history = max_history

    #     # Training metrics
    #     self.training_losses = deque(maxlen=max_history)
    #     self.training_steps = deque(maxlen=max_history)

    #     # Episode/Run metrics
    #     self.episode_rewards = deque(maxlen=max_history)
    #     self.episode_lengths = deque(maxlen=max_history)

    #     # Agent metrics
    #     self.epsilon_values = deque(maxlen=max_history)
    #     self.q_values = deque(maxlen=max_history)
    #     self.buffer_sizes = deque(maxlen=max_history)

    #     # Timing
    #     self.timestamps = deque(maxlen=max_history)

    #     # Current episode tracking
    #     self.current_episode_reward = 0
    #     self.current_episode_length = 0
    #     self.current_episode_q_values = []

    #     self.episode_count = 0
    #     self.total_steps = 0

    #     # Performance-based saving metrics
    #     self.best_episode_reward = float("-inf")
    #     self.best_average_reward = float("-inf")
    #     self.episodes_since_last_save = 0
    #     self.last_save_episode = 0
    #     self.save_frequency = 5  # Save every 5 episodes as backup (reduced from 25)
    #     self.min_episodes_before_save = 1  # Allow saves after first episode
        
    #     # Best weights tracking
    #     self.best_episode_weights_path = None
    #     self.best_average_weights_path = None

    # def should_save_model(self):
    #     """Determine if the model should be saved based on performance"""
    #     # Allow saving from episode 1 instead of 5 for better debugging
    #     if self.episode_count < 1:
    #         return False, "No episodes completed yet"

    #     reasons = []

    #     # Check for new best episode reward
    #     current_reward = self.episode_rewards[-1]
    #     if current_reward > self.best_episode_reward:
    #         self.best_episode_reward = current_reward
    #         reasons.append(f"New best episode reward: {current_reward:.2f}")

    #     # Check for improved average reward (last 10 episodes)
    #     if len(self.episode_rewards) >= 10:
    #         recent_avg = np.mean(list(self.episode_rewards)[-10:])
    #         if recent_avg > self.best_average_reward:
    #             self.best_average_reward = recent_avg
    #             reasons.append(f"New best 10-episode average: {recent_avg:.2f}")

    #     # Reduced periodic save frequency for better logging
    #     # if self.episodes_since_last_save >= 5:  # Save every 5 episodes instead of 25
    #     #     reasons.append(f"Periodic save (every 5 episodes)")

    #     should_save = len(reasons) > 0
    #     reason_text = "; ".join(reasons) if reasons else "No improvement"

    #     return should_save, reason_text

    # def save_model_automatically(self):
    #     """Save model automatically with performance-based logic"""
    #     should_save, reason = self.should_save_model()

    #     if should_save:
    #         try:
    #             timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #             saved_paths = []
                
    #             # Save regular model weights
    #             agent.save(MODEL_WEIGHTS_PATH)
    #             saved_paths.append(MODEL_WEIGHTS_PATH)
                
    #             # Check if this is a new best episode reward
    #             current_reward = self.episode_rewards[-1]
    #             if current_reward > self.best_episode_reward:
    #                 best_episode_path = os.path.join(
    #                     LOGS_DIR, 
    #                     f"best_episode_weights_{timestamp}_reward_{current_reward:.2f}.pth"
    #                 )
    #                 agent.save(best_episode_path)
    #                 self.best_episode_weights_path = best_episode_path
    #                 saved_paths.append(best_episode_path)
    #                 print(f"🏆 NEW BEST EPISODE REWARD: {current_reward:.2f}")
                
    #             # Check if this is a new best average reward (last 10 episodes)
    #             if len(self.episode_rewards) >= 10:
    #                 recent_avg = np.mean(list(self.episode_rewards)[-10:])
    #                 if recent_avg > self.best_average_reward:
    #                     best_avg_path = os.path.join(
    #                         LOGS_DIR, 
    #                         f"best_average_weights_{timestamp}_avg_{recent_avg:.2f}.pth"
    #                     )
    #                     agent.save(best_avg_path)
    #                     self.best_average_weights_path = best_avg_path
    #                     saved_paths.append(best_avg_path)
    #                     print(f"📈 NEW BEST 10-EPISODE AVERAGE: {recent_avg:.2f}")

    #             self.episodes_since_last_save = 0
    #             self.last_save_episode = self.episode_count

    #             # Save latest plots
    #             latest_plot_path = create_performance_plots(
    #                 os.path.join(LOGS_DIR, "latest_performance_plots.png")
    #             )

    #             # Save metrics JSON
    #             self.save_metrics_json()

    #             print(f"\n🔄 AUTO-SAVE TRIGGERED - Episode {self.episode_count}")
    #             print(f"📊 Reason: {reason}")
    #             for path in saved_paths:
    #                 print(f"💾 Model saved to: {path}")
    #             if latest_plot_path:
    #                 print(f"📈 Plots saved to: {latest_plot_path}")
    #             print(f"📋 Best episode reward: {self.best_episode_reward:.2f}")
    #             print(f"📋 Best 10-episode average: {self.best_average_reward:.2f}")
    #             if self.best_episode_weights_path:
    #                 print(f"🏆 Best episode weights: {self.best_episode_weights_path}")
    #             if self.best_average_weights_path:
    #                 print(f"📈 Best average weights: {self.best_average_weights_path}")
    #             print("-" * 60)

    #             return True, reason
    #         except Exception as e:
    #             print(f"❌ Error during auto-save: {e}")
    #             return False, f"Save failed: {str(e)}"
    #     else:
    #         self.episodes_since_last_save += 1
    #         return False, reason

    # def log_training_step(self, loss, step):
    #     """Log training loss and step"""
    #     self.training_losses.append(loss)
    #     self.training_steps.append(step)

    # def log_action(self, q_value, epsilon):
    #     """Log action-related metrics"""
    #     self.current_episode_q_values.append(q_value)
    #     self.current_episode_length += 1
    #     self.total_steps += 1

    #     # Log current metrics
    #     self.epsilon_values.append(epsilon)
    #     self.buffer_sizes.append(len(agent.memory))
    #     self.timestamps.append(datetime.now().timestamp())

    # def log_step_reward(self, reward):
    #     """Log reward for current step"""
    #     self.current_episode_reward += reward

    # def log_episode_end(self):
    #     """Log end of episode"""
    #     self.episode_rewards.append(self.current_episode_reward)
    #     self.episode_lengths.append(self.current_episode_length)

    #     # Average Q-value for this episode
    #     if self.current_episode_q_values:
    #         avg_q = np.mean(self.current_episode_q_values)
    #         self.q_values.append(avg_q)

    #     self.episode_count += 1

    #     # Reset episode tracking
    #     self.current_episode_reward = 0
    #     self.current_episode_length = 0
    #     self.current_episode_q_values = []

    #     print(
    #         f"Episode {self.episode_count} completed - Reward: {self.episode_rewards[-1]:.2f}, "
    #         f"Length: {self.episode_lengths[-1]}"
    #     )

    #     # Auto-save plots every 5 episodes (reduced from 10 for better logging)
    #     if self.episode_count % 5 == 0:
    #         self.save_plots_automatically()

    #     # Check for performance-based model saving
    #     self.save_model_automatically()

    #     # Always save basic metrics JSON after each episode for debugging
    #     try:
    #         self.save_basic_metrics_json()
    #     except Exception as e:
    #         print(f"❌ Error saving basic metrics: {e}")

    # def save_plots_automatically(self):
    #     """Save plots automatically with timestamped filenames"""
    #     try:
    #         # save latest version
    #         latest_path = create_performance_plots(
    #             os.path.join(LOGS_DIR, "latest_performance_plots.png")
    #         )
    #         if latest_path:
    #             print(f"📈 Latest performance plots updated: {latest_path}")

    #     except Exception as e:
    #         print(f"❌ Error auto-saving plots: {e}")
    #         import traceback

    #         traceback.print_exc()

    # def save_metrics_json(self):
    #     """Save current metrics to JSON file"""
    #     try:
    #         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #         metrics = {
    #             "timestamp": timestamp,
    #             "episode_count": self.episode_count,
    #             "total_steps": self.total_steps,
    #             "episode_rewards": list(self.episode_rewards),
    #             "episode_lengths": list(self.episode_lengths),
    #             "training_losses": list(self.training_losses),
    #             "epsilon_values": list(self.epsilon_values),
    #             "q_values": list(self.q_values),
    #             "buffer_sizes": list(self.buffer_sizes),
    #             "best_episode_reward": self.best_episode_reward,
    #             "best_average_reward": self.best_average_reward,
    #             "last_save_episode": self.last_save_episode,
    #             "best_episode_weights_path": self.best_episode_weights_path,
    #             "best_average_weights_path": self.best_average_weights_path,
    #         }

    #         filename = os.path.join(LOGS_DIR, f"training_metrics.json")
    #         with open(filename, "w") as f:
    #             json.dump(metrics, f, indent=2)
    #         print(f"Training metrics saved: {filename}")

    #     except Exception as e:
    #         print(f"Error saving metrics: {e}")

    # def save_basic_metrics_json(self):
    #     """Save basic metrics after each episode for debugging"""
    #     try:
    #         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #         basic_metrics = {
    #             "timestamp": timestamp,
    #             "episode_count": self.episode_count,
    #             "total_steps": self.total_steps,
    #             "last_episode_reward": (
    #                 self.episode_rewards[-1] if self.episode_rewards else 0
    #             ),
    #             "current_epsilon": agent.epsilon if "agent" in globals() else "N/A",
    #             "buffer_size": (
    #                 len(agent.memory)
    #                 if "agent" in globals() and hasattr(agent, "memory")
    #                 else 0
    #             ),
    #         }

    #         latest_file = os.path.join(LOGS_DIR, "latest_episode_metrics.json")

    #         for filename in [latest_file]:
    #             with open(filename, "w") as f:
    #                 json.dump(basic_metrics, f, indent=2)

    #         print(f"📊 Episode {self.episode_count} metrics saved: {latest_file}")

    #     except Exception as e:
    #         print(f"Error saving basic metrics: {e}")
    #         import traceback

    #         traceback.print_exc()

    # def save_step_logs(self):
    #     """Save step-level logs after every training call"""
    #     try:
    #         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #         step_metrics = {
    #             "timestamp": timestamp,
    #             "episode_count": self.episode_count,
    #             "total_steps": self.total_steps,
    #             "current_episode_reward": self.current_episode_reward,
    #             "current_episode_length": self.current_episode_length,
    #             "current_epsilon": agent.epsilon if "agent" in globals() else "N/A",
    #             "buffer_size": (
    #                 len(agent.memory)
    #                 if "agent" in globals() and hasattr(agent, "memory")
    #                 else 0
    #             ),
    #             "latest_training_loss": (
    #                 self.training_losses[-1] if self.training_losses else "N/A"
    #             ),
    #             "latest_q_value": (
    #                 self.current_episode_q_values[-1]
    #                 if self.current_episode_q_values
    #                 else "N/A"
    #             ),
    #         }

    #         # Save to both timestamped and latest files
    #         # timestamped_file = os.path.join(
    #         #     LOGS_DIR, f"step_metrics_{self.total_steps}_{timestamp}.json"
    #         # )
    #         latest_file = os.path.join(LOGS_DIR, "latest_step_metrics.json")

    #         for filename in [
    #             # timestamped_file,
    #             latest_file
    #         ]:
    #             with open(filename, "w") as f:
    #                 json.dump(step_metrics, f, indent=2)

    #     except Exception as e:
    #         print(f"Error saving step logs: {e}")
    #         import traceback

    #         traceback.print_exc()

    # def save_step_plots(self):
    #     """Save plots during training (not just at episode end)"""
    #     try:
    #         # Only create plots if we have some data
    #         if self.total_steps < 1:
    #             return None

    #         # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #         # filename = os.path.join(LOGS_DIR, f"step_plots_{self.total_steps}_{timestamp}.png")

    #         # # Create simplified plots for step-level monitoring
    #         # plot_path = create_step_performance_plots(filename)

    #         # if plot_path:
    #         # Also save latest version
    #         latest_path = create_step_performance_plots(
    #             os.path.join(LOGS_DIR, "latest_step_plots.png")
    #         )
    #         return latest_path

    #     except Exception as e:
    #         print(f"Error saving step plots: {e}")
    #         import traceback

    #         traceback.print_exc()
    #         return None


# Initialize MLflow tracker (replaces PerformanceTracker)
tracker = MLflowTracker()

# Start MLflow run for this session
tracker.start_run()

# Try to auto-load best model from previous runs
print("🔍 Checking for existing models to continue training...")
try:
    # Try to auto-load the best model from MLflow
    result = tracker.load_best_model_from_mlflow(agent, "average")  # Prefer average for stability
    if result["success"]:
        print(f"✅ Auto-loaded best average model from MLflow!")
        print(f"📊 Performance: {result['performance']:.2f}")
        print(f"🚀 Continuing training from checkpoint")
    else:
        # Try best episode model as fallback
        result = tracker.load_best_model_from_mlflow(agent, "episode")
        if result["success"]:
            print(f"✅ Auto-loaded best episode model from MLflow!")
            print(f"📊 Performance: {result['performance']:.2f}")
            print(f"🚀 Continuing training from checkpoint")
        else:
            print("💡 No existing models found. Starting fresh training.")
            print("   Models will be automatically saved when performance improves.")
except Exception as e:
    print(f"⚠️  Error during model auto-loading: {e}")
    print("🚀 Starting with fresh model.")
    exit()


def decode_action(action_value: int) -> Tuple[List[int], str]:
    """
    Decodes an integer action_value into selected card indices and action type.
    """
    # Last bit (0 or 1) determines play/discard
    action_type = "discard" if action_value & 1 else "play"

    # The other 8 bits determine card selection (up to 8 cards, we will cap at 5)
    card_selection_bits = action_value >> 1
    selected_indices = []
    for i in range(8):  # Iterate through bits for cards 1 to 8
        if (card_selection_bits >> i) & 1:  # Check if the i-th bit is set
            selected_indices.append(i + 1)  # Add 1-based card index
            if len(selected_indices) == 5:  # Max 5 cards
                break
    return selected_indices, action_type


def encode_action(selected_indices: List[int], action_type: str) -> int:
    """
    Encodes selected card indices and action type into an integer action_value.
    """
    action_value = 0
    if action_type == "discard":
        action_value = 1

    card_selection_bits = 0
    for idx in selected_indices:
        if 1 <= idx <= 8:  # Ensure index is valid
            card_selection_bits |= 1 << (idx - 1)  # Set the (idx-1)-th bit

    action_value = (card_selection_bits << 1) | action_value
    return action_value


@app.post("/predict")
async def predict(request: dict):
    """
    Get action predictions from the DQN agent.

    Args:
        request: the current game state.
        Example: {'state': [0(current chips), 2(hands left), 3(discards left), 1, 'QC', 2, '10C', 3, '10D', 4, '7C', 5, '6C', 6, '6D', 7, '4H', 8, '2S']}

    Returns:
        ActionResponse containing:
        - indices: List of up to 5 card indices (values between 1-8), guaranteed non-empty.
        - action: "play" or "discard"
    """
    print("Received state:", request)
    try:
        # Convert the state to a numeric representation
        numeric_state = []
        for item in request["state"]:
            if isinstance(item, int):
                numeric_state.append(item)
            else:
                # Convert card strings to numeric values
                numeric_state.append(
                    hash(item) % 100
                )  # Simple hash-based encoding for now

        state_array = np.array(numeric_state, dtype=np.float32)
        action_value = agent.get_action(
            state_array
        )  # This is the raw action from the agent

        # Get Q-value and epsilon for tracking
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state_array).unsqueeze(0)
            q_values = agent.qnetwork_local(state_tensor)
            max_q_value = torch.max(q_values).item()

        current_epsilon = agent.epsilon

        # Log metrics
        tracker.log_action(max_q_value, current_epsilon)

        selected_indices, action_type = decode_action(action_value)

        # Enforce game rule: selected_indices cannot be empty
        if not selected_indices:
            selected_indices.append(1)  # Default to selecting the first card
            print(f"Override: No cards selected by agent. Forcing selection of card 1.")

        # Enforce game rule: if discards_left is 0, action must be "play"
        discards_left = request["state"][
            2
        ]  # Index 2 is discards_left as per your example
        if discards_left == 0 and action_type == "discard":
            action_type = "play"
            print(f"Override: Discards left is 0. Forcing action to 'play'.")

        print(f"Decoded - Selected indices: {selected_indices}, Action: {action_type}")
        print(f"Q-value: {max_q_value:.4f}, Epsilon: {current_epsilon:.4f}")

        return {"indices": selected_indices, "action": action_type}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/train")
async def train(request: dict):
    """
    Train the agent with new experiences.

    Args:
        request: the experience tuple
        Example: {
            'state': [0(current chips), 2(hands left), 3(discards left), 1, 'QC', 2, '10C', 3, '10D', 4, '7C', 5, '6C', 6, '6D', 7, '4H', 8, '2S'],
            'action': {'indices': [1, 2, 3, 4, 5], 'action': 'play'},
            'reward': 1.0,
            'next_state': [0(current chips), 2(hands left), 3(discards left), 1, 'QC', 2, '10C', 3, '10D', 4, '7C', 5, '6C', 6, '6D', 7, '4H', 8, '2S'],
            'done': false
        }

    Returns:
        TrainResponse confirming the experience was added
    """
    try:
        print("Received training request:", request)

        # Convert states to numeric representations
        def convert_state(state_list):
            numeric_state = []
            for item in state_list:
                if isinstance(item, (int, float)):
                    numeric_state.append(float(item))
                elif isinstance(item, str):
                    if item == "":  # Handle empty string padding
                        numeric_state.append(0.0)
                    else:
                        numeric_state.append(float(hash(item) % 100))
                else:
                    numeric_state.append(float(hash(str(item)) % 100))
            return np.array(numeric_state, dtype=np.float32)

        state = convert_state(request["state"])
        print("Converted state:", state)

        next_state = convert_state(request["next_state"])
        print("Converted next_state:", next_state)

        action = encode_action(
            request["action"]["indices"], request["action"]["action"]
        )
        print("Encoded action:", action)

        reward = request["reward"]
        done = request["done"]
        print(f"Reward: {reward}, Done: {done}")

        # Log step reward
        tracker.log_step_reward(reward)

        # Store experience in replay buffer
        agent.step(state, action, reward, next_state, done)
        print("Experience added to replay buffer")

        # Check if training occurred and log training loss
        if hasattr(agent, "training_loss") and agent.training_loss:
            latest_loss = agent.training_loss[-1]
            tracker.log_training_step(latest_loss, tracker.total_steps)
            print(f"Training loss: {latest_loss:.6f}")

        # Save step-level logs after every training call
        try:
            tracker.save_step_logs()
            print(f"📊 Step logs saved (Step {tracker.total_steps})")
        except Exception as e:
            print(f"❌ Error saving step logs: {e}")

        # Auto-save model every 20 steps for long episodes (safety backup)
        if tracker.total_steps % 20 == 0:
            try:
                # agent.save(MODEL_WEIGHTS_PATH)

                # Save only latest step metrics and plot
                tracker.save_step_logs()  # This saves latest_step_metrics.json
                tracker.save_step_plots()  # This saves latest_step_plots.png

                print(f"🔄 STEP-BASED AUTO-SAVE - Step {tracker.total_steps}")
                print(f"💾 Model saved to: {MODEL_WEIGHTS_PATH}")
                print(f"📊 Latest metrics and plot updated")
                print("-" * 50)
            except Exception as e:
                print(f"❌ Error during step-based auto-save: {e}")

        # Log episode completion
        if done:
            tracker.log_episode_end(agent)

            # Decay epsilon after each episode
            agent.decay_epsilon()

            print(f"Episode ended. Epsilon decayed to: {agent.epsilon:.4f}")
        else:
            # Save plots even when episode hasn't ended (save every step)
            if tracker.total_steps % 1 == 0:
                try:
                    tracker.save_step_plots()
                    print(f"📈 Step plots saved (Step {tracker.total_steps})")
                except Exception as e:
                    print(f"❌ Error saving step plots: {e}")

        return {"message": "Experience added to replay buffer"}
    except Exception as e:
        print(f"Error in training: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/save")
async def save_model():
    """
    Manual save endpoint - DISABLED

    Model saving is now handled automatically based on performance metrics.
    The model will save when:
    - New best episode reward is achieved
    - New best 10-episode average is achieved
    - Every 5 episodes as backup

    Check /metrics endpoint to see current performance and save status.
    """
    raise HTTPException(
        status_code=403,
        detail="Manual save disabled. Model saving is now automatic based on performance metrics. Check /metrics for save status.",
    )
    """
    Save the current model weights, performance plots, and metrics.

    Returns:
        SaveResponse confirming the save operation
    """
    try:
        # Save model weights
        agent.save(MODEL_WEIGHTS_PATH)
        print(f"Model saved successfully to {MODEL_WEIGHTS_PATH}")

        # Save performance plots with model save timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = os.path.join(
            LOGS_DIR, f"performance_plots_model_save_{timestamp}.png"
        )
        plot_path = create_performance_plots(plot_filename)

        # Save latest plots
        latest_plot_path = create_performance_plots(
            os.path.join(LOGS_DIR, "latest_performance_plots.png")
        )

        # Save metrics JSON
        tracker.save_metrics_json()

        print(f"Performance plots saved: {os.path.abspath(plot_path)}")
        print(f"Latest plots updated: {os.path.abspath(latest_plot_path)}")

        return {
            "status": "success",
            "message": f"Model saved to {MODEL_WEIGHTS_PATH}",
            "plot_saved": plot_filename,
            "episode_count": tracker.episode_count,
        }
    except Exception as e:
        print(f"Error saving model: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


def create_performance_plots(filename="performance_plots.png"):
    """Create comprehensive performance plots"""
    try:
        if tracker.episode_count < 2:
            print("Not enough episodes to create plots (minimum 2 required)")
            return None

        # Create directory if it doesn't exist
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            print(f"Created directory: {directory}")

        # Create a figure with multiple subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(
            f"DQN Agent Performance - Episode {tracker.episode_count}", fontsize=16
        )

        # 1. Episode Rewards
        if tracker.episode_rewards:
            axes[0, 0].plot(
                list(tracker.episode_rewards), "b-", alpha=0.6, label="Episode Reward"
            )
            if len(tracker.episode_rewards) > 10:
                # Moving average
                window = min(10, len(tracker.episode_rewards) // 2)
                moving_avg = np.convolve(
                    tracker.episode_rewards, np.ones(window) / window, mode="valid"
                )
                axes[0, 0].plot(
                    range(window - 1, len(tracker.episode_rewards)),
                    moving_avg,
                    "r-",
                    linewidth=2,
                    label=f"{window}-Episode Moving Avg",
                )
            axes[0, 0].set_title("Episode Rewards")
            axes[0, 0].set_xlabel("Episode")
            axes[0, 0].set_ylabel("Reward")
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

        # 2. Training Loss
        if tracker.training_losses:
            axes[0, 1].plot(
                list(tracker.training_steps),
                list(tracker.training_losses),
                "r-",
                alpha=0.7,
            )
            axes[0, 1].set_title("Training Loss")
            axes[0, 1].set_xlabel("Training Step")
            axes[0, 1].set_ylabel("Loss")
            axes[0, 1].grid(True, alpha=0.3)
            if len(tracker.training_losses) > 1:
                axes[0, 1].set_yscale("log")

        # 3. Epsilon Decay
        if tracker.epsilon_values:
            axes[0, 2].plot(list(tracker.epsilon_values), "purple", linewidth=2)
            axes[0, 2].set_title("Epsilon (Exploration Rate)")
            axes[0, 2].set_xlabel("Step")
            axes[0, 2].set_ylabel("Epsilon")
            axes[0, 2].grid(True, alpha=0.3)

        # 4. Q-Values
        if tracker.q_values:
            axes[1, 0].plot(list(tracker.q_values), "orange", linewidth=2)
            axes[1, 0].set_title("Average Q-Values per Episode")
            axes[1, 0].set_xlabel("Episode")
            axes[1, 0].set_ylabel("Average Q-Value")
            axes[1, 0].grid(True, alpha=0.3)

        # 5. Episode Lengths
        if tracker.episode_lengths:
            axes[1, 1].plot(list(tracker.episode_lengths), "cyan", linewidth=2)
            axes[1, 1].set_title("Episode Lengths")
            axes[1, 1].set_xlabel("Episode")
            axes[1, 1].set_ylabel("Steps per Episode")
            axes[1, 1].grid(True, alpha=0.3)

        # 6. Empty subplot (placeholder)
        axes[1, 2].text(
            0.5,
            0.5,
            "Available for\nFuture Metrics",
            ha="center",
            va="center",
            transform=axes[1, 2].transAxes,
            fontsize=12,
            alpha=0.5,
        )
        axes[1, 2].set_title("Reserved")
        axes[1, 2].set_xticks([])
        axes[1, 2].set_yticks([])

        plt.tight_layout()

        # Save plot with error handling
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close()

        # Verify the file was actually created
        if os.path.exists(filename):
            file_size = os.path.getsize(filename)
            print(f"✅ Plot saved successfully: {filename} (size: {file_size} bytes)")
            return filename
        else:
            print(f"❌ Failed to save plot: {filename}")
            return None

    except Exception as e:
        print(f"❌ Error creating performance plots: {e}")
        import traceback

        traceback.print_exc()
        return None


@app.get("/performance_plot")
async def get_performance_plot():
    """
    Generate and serve performance plots
    """
    try:
        # Create temporary plot for serving
        temp_filename = os.path.join(LOGS_DIR, "temp_performance_plot.png")
        plot_path = create_performance_plots(temp_filename)
        if plot_path and os.path.exists(plot_path):
            return FileResponse(plot_path, media_type="image/png")
        else:
            error_detail = (
                f"Unable to generate plots. Episodes: {tracker.episode_count}, "
                f"Episode rewards: {len(tracker.episode_rewards) if tracker.episode_rewards else 0}"
            )
            raise HTTPException(status_code=404, detail=error_detail)
    except Exception as e:
        print(f"❌ Error in performance_plot endpoint: {e}")
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def get_metrics():
    """
    Get current performance metrics as JSON
    """
    try:
        # Get MLflow-based metrics
        metrics = tracker.get_metrics_summary()
        
        # Add additional agent-specific metrics
        metrics.update({
            "current_epsilon": agent.epsilon,
            "buffer_size": len(agent.memory),
            "max_buffer_size": agent.memory.buffer_size,
        })
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/mlflow/start_run")
async def start_mlflow_run(run_name: str = None):
    """
    Start a new MLflow run
    """
    try:
        run = tracker.start_run(run_name)
        return {
            "status": "success",
            "run_id": run.info.run_id,
            "run_name": run.info.run_name,
            "message": f"Started new MLflow run: {run.info.run_name}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/mlflow/end_run")
async def end_mlflow_run():
    """
    End the current MLflow run
    """
    try:
        if tracker.current_run is None:
            return {"status": "info", "message": "No active MLflow run to end"}
        
        run_name = tracker.current_run.info.run_name
        tracker.end_run()
        return {
            "status": "success",
            "message": f"Ended MLflow run: {run_name}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/mlflow/ui_info")
async def get_mlflow_ui_info():
    """
    Get MLflow UI information
    """
    try:
        from mlflow_tracker import get_mlflow_ui_url
        return {
            "ui_url": get_mlflow_ui_url(),
            "tracking_uri": tracker.setup_mlflow.__globals__.get('mlflow').get_tracking_uri(),
            "experiment_name": tracker.experiment_name,
            "current_run_id": tracker.current_run.info.run_id if tracker.current_run else None,
            "instructions": {
                "start_ui": "Run 'mlflow ui' in your terminal to start the web interface",
                "view_experiments": "Open http://localhost:5000 in your browser"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/list")
async def list_available_models():
    """
    List all available models from previous MLflow runs
    """
    try:
        models_info = tracker.list_available_models()
        return models_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/best_info")
async def get_best_models_info():
    """
    Get information about the best performing models
    """
    try:
        best_info = tracker.get_best_models_info()
        return best_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# if __name__ == "__main__":
#     import uvicorn

#     uvicorn.run(app, host="0.0.0.0", port=5000)
