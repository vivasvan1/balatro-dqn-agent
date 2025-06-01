import os
from datetime import datetime

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

# Load pre-trained weights if they exist
try:
    agent.load(MODEL_WEIGHTS_PATH)
    print(f"Loaded model weights from {MODEL_WEIGHTS_PATH}")
except:
    print("No pre-trained weights found. Starting with fresh model.")


# Performance tracking
class PerformanceTracker:
    def __init__(self, max_history=1000):
        self.max_history = max_history

        # Training metrics
        self.training_losses = deque(maxlen=max_history)
        self.training_steps = deque(maxlen=max_history)

        # Episode/Run metrics
        self.episode_rewards = deque(maxlen=max_history)
        self.episode_lengths = deque(maxlen=max_history)
        self.cumulative_rewards = deque(maxlen=max_history)

        # Agent metrics
        self.epsilon_values = deque(maxlen=max_history)
        self.q_values = deque(maxlen=max_history)
        self.buffer_sizes = deque(maxlen=max_history)

        # Timing
        self.timestamps = deque(maxlen=max_history)

        # Current episode tracking
        self.current_episode_reward = 0
        self.current_episode_length = 0
        self.current_episode_q_values = []

        self.episode_count = 0
        self.total_steps = 0
        
        # Performance-based saving metrics
        self.best_episode_reward = float('-inf')
        self.best_cumulative_reward = float('-inf')
        self.best_average_reward = float('-inf')
        self.episodes_since_last_save = 0
        self.last_save_episode = 0
        self.save_frequency = 5  # Save every 5 episodes as backup (reduced from 25)
        self.min_episodes_before_save = 1  # Allow saves after first episode

    def should_save_model(self):
        """Determine if the model should be saved based on performance"""
        # Allow saving from episode 1 instead of 5 for better debugging
        if self.episode_count < 1:
            return False, "No episodes completed yet"
            
        reasons = []
        
        # Check for new best episode reward
        current_reward = self.episode_rewards[-1]
        if current_reward > self.best_episode_reward:
            self.best_episode_reward = current_reward
            reasons.append(f"New best episode reward: {current_reward:.2f}")
            
        # Check for new best cumulative reward
        current_cumulative = self.cumulative_rewards[-1]
        if current_cumulative > self.best_cumulative_reward:
            self.best_cumulative_reward = current_cumulative
            reasons.append(f"New best cumulative reward: {current_cumulative:.2f}")
            
        # Check for improved average reward (last 10 episodes)
        if len(self.episode_rewards) >= 10:
            recent_avg = np.mean(list(self.episode_rewards)[-10:])
            if recent_avg > self.best_average_reward:
                self.best_average_reward = recent_avg
                reasons.append(f"New best 10-episode average: {recent_avg:.2f}")
        
        # Reduced periodic save frequency for better logging
        if self.episodes_since_last_save >= 5:  # Save every 5 episodes instead of 25
            reasons.append(f"Periodic save (every 5 episodes)")
            
        should_save = len(reasons) > 0
        reason_text = "; ".join(reasons) if reasons else "No improvement"
        
        return should_save, reason_text

    def save_model_automatically(self):
        """Save model automatically with performance-based logic"""
        should_save, reason = self.should_save_model()
        
        if should_save:
            try:
                # Save model weights
                agent.save(MODEL_WEIGHTS_PATH)
                self.episodes_since_last_save = 0
                self.last_save_episode = self.episode_count
                
                # Save performance plots with save reason
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                plot_filename = os.path.join(LOGS_DIR, f"auto_save_episode_{self.episode_count}_{timestamp}.png")
                plot_path = create_performance_plots(plot_filename)
                
                # Save latest plots
                latest_plot_path = create_performance_plots(os.path.join(LOGS_DIR, "latest_performance_plots.png"))
                
                # Save metrics JSON
                self.save_metrics_json()
                
                print(f"\n🔄 AUTO-SAVE TRIGGERED - Episode {self.episode_count}")
                print(f"📊 Reason: {reason}")
                print(f"💾 Model saved to: {MODEL_WEIGHTS_PATH}")
                print(f"📈 Plots saved to: {plot_filename}")
                print(f"📋 Best episode reward: {self.best_episode_reward:.2f}")
                print(f"📋 Best cumulative reward: {self.best_cumulative_reward:.2f}")
                print(f"📋 Best 10-episode average: {self.best_average_reward:.2f}")
                print("-" * 60)
                
                return True, reason
            except Exception as e:
                print(f"❌ Error during auto-save: {e}")
                return False, f"Save failed: {str(e)}"
        else:
            self.episodes_since_last_save += 1
            return False, reason

    def log_training_step(self, loss, step):
        """Log training loss and step"""
        self.training_losses.append(loss)
        self.training_steps.append(step)

    def log_action(self, q_value, epsilon):
        """Log action-related metrics"""
        self.current_episode_q_values.append(q_value)
        self.current_episode_length += 1
        self.total_steps += 1

        # Log current metrics
        self.epsilon_values.append(epsilon)
        self.buffer_sizes.append(len(agent.memory))
        self.timestamps.append(datetime.now().timestamp())

    def log_step_reward(self, reward):
        """Log reward for current step"""
        self.current_episode_reward += reward

    def log_episode_end(self, cumulative_reward):
        """Log end of episode"""
        self.episode_rewards.append(self.current_episode_reward)
        self.episode_lengths.append(self.current_episode_length)
        self.cumulative_rewards.append(cumulative_reward)

        # Average Q-value for this episode
        if self.current_episode_q_values:
            avg_q = np.mean(self.current_episode_q_values)
            self.q_values.append(avg_q)

        self.episode_count += 1

        # Reset episode tracking
        self.current_episode_reward = 0
        self.current_episode_length = 0
        self.current_episode_q_values = []

        print(
            f"Episode {self.episode_count} completed - Reward: {self.episode_rewards[-1]:.2f}, "
            f"Length: {self.episode_lengths[-1]}, Cumulative: {cumulative_reward:.2f}"
        )

        # Auto-save plots every 5 episodes (reduced from 10 for better logging)
        if self.episode_count % 5 == 0:
            self.save_plots_automatically()
            
        # Check for performance-based model saving
        self.save_model_automatically()
        
        # Always save basic metrics JSON after each episode for debugging
        try:
            self.save_basic_metrics_json()
        except Exception as e:
            print(f"❌ Error saving basic metrics: {e}")

    def save_plots_automatically(self):
        """Save plots automatically with timestamped filenames"""
        try:
            # Logs directory already ensured at startup
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(LOGS_DIR, f"performance_plots_episode_{self.episode_count}_{timestamp}.png")

            plot_path = create_performance_plots(filename)
            if plot_path:
                print(f"📈 Performance plots automatically saved: {plot_path}")

                # Also save latest version
                latest_path = create_performance_plots(os.path.join(LOGS_DIR, "latest_performance_plots.png"))
                if latest_path:
                    print(f"📈 Latest performance plots updated: {latest_path}")

        except Exception as e:
            print(f"❌ Error auto-saving plots: {e}")
            import traceback
            traceback.print_exc()

    def save_metrics_json(self):
        """Save current metrics to JSON file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            metrics = {
                "timestamp": timestamp,
                "episode_count": self.episode_count,
                "total_steps": self.total_steps,
                "episode_rewards": list(self.episode_rewards),
                "cumulative_rewards": list(self.cumulative_rewards),
                "training_losses": list(self.training_losses),
                "epsilon_values": list(self.epsilon_values),
                "q_values": list(self.q_values),
                "episode_lengths": list(self.episode_lengths),
                "best_episode_reward": self.best_episode_reward,
                "best_cumulative_reward": self.best_cumulative_reward,
                "best_average_reward": self.best_average_reward,
                "last_save_episode": self.last_save_episode,
            }

            filename = os.path.join(LOGS_DIR, f"training_metrics_{timestamp}.json")
            with open(filename, "w") as f:
                json.dump(metrics, f, indent=2)
            print(f"Training metrics saved: {filename}")

        except Exception as e:
            print(f"Error saving metrics: {e}")

    def save_basic_metrics_json(self):
        """Save basic metrics after each episode for debugging"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            basic_metrics = {
                "timestamp": timestamp,
                "episode_count": self.episode_count,
                "total_steps": self.total_steps,
                "last_episode_reward": self.episode_rewards[-1] if self.episode_rewards else 0,
                "last_cumulative_reward": self.cumulative_rewards[-1] if self.cumulative_rewards else 0,
                "current_epsilon": agent.epsilon if 'agent' in globals() else "N/A",
                "buffer_size": len(agent.memory) if 'agent' in globals() and hasattr(agent, 'memory') else 0,
            }
            
            # Save to both timestamped and latest files
            timestamped_file = os.path.join(LOGS_DIR, f"episode_metrics_{self.episode_count}_{timestamp}.json")
            latest_file = os.path.join(LOGS_DIR, "latest_episode_metrics.json")
            
            for filename in [timestamped_file, latest_file]:
                with open(filename, "w") as f:
                    json.dump(basic_metrics, f, indent=2)
            
            print(f"📊 Episode {self.episode_count} metrics saved: {latest_file}")

        except Exception as e:
            print(f"Error saving basic metrics: {e}")
            import traceback
            traceback.print_exc()

    def save_step_logs(self):
        """Save step-level logs after every training call"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            step_metrics = {
                "timestamp": timestamp,
                "episode_count": self.episode_count,
                "total_steps": self.total_steps,
                "current_episode_reward": self.current_episode_reward,
                "current_episode_length": self.current_episode_length,
                "current_epsilon": agent.epsilon if 'agent' in globals() else "N/A",
                "buffer_size": len(agent.memory) if 'agent' in globals() and hasattr(agent, 'memory') else 0,
                "latest_training_loss": self.training_losses[-1] if self.training_losses else "N/A",
                "latest_q_value": self.current_episode_q_values[-1] if self.current_episode_q_values else "N/A",
            }
            
            # Save to both timestamped and latest files
            timestamped_file = os.path.join(LOGS_DIR, f"step_metrics_{self.total_steps}_{timestamp}.json")
            latest_file = os.path.join(LOGS_DIR, "latest_step_metrics.json")
            
            for filename in [timestamped_file, latest_file]:
                with open(filename, "w") as f:
                    json.dump(step_metrics, f, indent=2)

        except Exception as e:
            print(f"Error saving step logs: {e}")
            import traceback
            traceback.print_exc()

    def save_step_plots(self):
        """Save plots during training (not just at episode end)"""
        try:
            # Only create plots if we have some data
            if self.total_steps < 1:
                return None
                
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(LOGS_DIR, f"step_plots_{self.total_steps}_{timestamp}.png")
            
            # Create simplified plots for step-level monitoring
            plot_path = create_step_performance_plots(filename)
            
            if plot_path:
                # Also save latest version
                latest_path = create_step_performance_plots(os.path.join(LOGS_DIR, "latest_step_plots.png"))
                return plot_path
            
        except Exception as e:
            print(f"Error saving step plots: {e}")
            import traceback
            traceback.print_exc()
            return None


# Initialize performance tracker
tracker = PerformanceTracker()


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

        # Log episode completion
        if done:
            # Calculate cumulative reward from tracker
            cumulative_reward = (
                sum(tracker.episode_rewards) + tracker.current_episode_reward
            )
            print(f"Cumulative reward: {cumulative_reward}")
            print()
            tracker.log_episode_end(cumulative_reward)
            

            # Decay epsilon after each episode
            agent.decay_epsilon()
            
            print(f"Episode ended. Epsilon decayed to: {agent.epsilon:.4f}")
        else:
            # Save plots even when episode hasn't ended (every 10 steps)
            if tracker.total_steps % 10 == 0:
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
    - New best cumulative reward is achieved  
    - New best 10-episode average is achieved
    - Every 25 episodes as backup
    
    Check /metrics endpoint to see current performance and save status.
    """
    raise HTTPException(
        status_code=403, 
        detail="Manual save disabled. Model saving is now automatic based on performance metrics. Check /metrics for save status."
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
        plot_filename = os.path.join(LOGS_DIR, f"performance_plots_model_save_{timestamp}.png")
        plot_path = create_performance_plots(plot_filename)

        # Save latest plots
        latest_plot_path = create_performance_plots(os.path.join(LOGS_DIR, "latest_performance_plots.png"))

        # Save metrics JSON
        tracker.save_metrics_json()

        print(f"Performance plots saved: {os.path.abspath(plot_path)}")
        print(f"Latest plots updated: {os.path.abspath(latest_plot_path)}")

        return {
            "status": "success",
            "message": f"Model saved to {MODEL_WEIGHTS_PATH}",
            "plot_saved": plot_filename,
            "episode_count": tracker.episode_count,
            "cumulative_reward": (
                tracker.cumulative_rewards[-1] if tracker.cumulative_rewards else 0
            ),
        }
    except Exception as e:
        print(f"Error saving model: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


def create_step_performance_plots(filename="step_plots.png"):
    """Create simplified performance plots for step-level monitoring"""
    try:
        # Create directory if it doesn't exist
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        # Create a figure with available metrics
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f"DQN Agent Step-Level Monitoring - Step {tracker.total_steps}", fontsize=14)

        # 1. Training Loss over time
        if tracker.training_losses and tracker.training_steps:
            axes[0, 0].plot(list(tracker.training_steps), list(tracker.training_losses), "r-", alpha=0.7)
            axes[0, 0].set_title("Training Loss")
            axes[0, 0].set_xlabel("Training Step")
            axes[0, 0].set_ylabel("Loss")
            axes[0, 0].grid(True, alpha=0.3)
            if len(tracker.training_losses) > 1:
                axes[0, 0].set_yscale("log")
        else:
            axes[0, 0].text(0.5, 0.5, "No training loss data", ha='center', va='center', transform=axes[0, 0].transAxes)
            axes[0, 0].set_title("Training Loss")

        # 2. Epsilon Decay
        if tracker.epsilon_values:
            axes[0, 1].plot(list(tracker.epsilon_values), "purple", linewidth=2)
            axes[0, 1].set_title("Epsilon (Exploration Rate)")
            axes[0, 1].set_xlabel("Step")
            axes[0, 1].set_ylabel("Epsilon")
            axes[0, 1].grid(True, alpha=0.3)
        else:
            axes[0, 1].text(0.5, 0.5, "No epsilon data", ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title("Epsilon (Exploration Rate)")

        # 3. Buffer Size Growth
        if tracker.buffer_sizes:
            axes[1, 0].plot(list(tracker.buffer_sizes), "cyan", linewidth=2)
            axes[1, 0].set_title("Replay Buffer Size")
            axes[1, 0].set_xlabel("Step")
            axes[1, 0].set_ylabel("Buffer Size")
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, "No buffer data", ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title("Replay Buffer Size")

        # 4. Current Episode Progress
        if tracker.current_episode_q_values:
            axes[1, 1].plot(tracker.current_episode_q_values, "orange", linewidth=2)
            axes[1, 1].set_title(f"Current Episode Q-Values (Ep {tracker.episode_count + 1})")
            axes[1, 1].set_xlabel("Step in Episode")
            axes[1, 1].set_ylabel("Q-Value")
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, "No current Q-values", ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title(f"Current Episode Q-Values (Ep {tracker.episode_count + 1})")

        plt.tight_layout()

        # Save plot with error handling
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close()
        
        # Verify the file was actually created
        if os.path.exists(filename):
            file_size = os.path.getsize(filename)
            return filename
        else:
            return None

    except Exception as e:
        print(f"❌ Error creating step performance plots: {e}")
        import traceback
        traceback.print_exc()
        return None


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

        # 2. Cumulative Rewards
        if tracker.cumulative_rewards:
            axes[0, 1].plot(list(tracker.cumulative_rewards), "g-", linewidth=2)
            axes[0, 1].set_title("Cumulative Rewards")
            axes[0, 1].set_xlabel("Episode")
            axes[0, 1].set_ylabel("Cumulative Reward")
            axes[0, 1].grid(True, alpha=0.3)

        # 3. Training Loss
        if tracker.training_losses:
            axes[0, 2].plot(
                list(tracker.training_steps), list(tracker.training_losses), "r-", alpha=0.7
            )
            axes[0, 2].set_title("Training Loss")
            axes[0, 2].set_xlabel("Training Step")
            axes[0, 2].set_ylabel("Loss")
            axes[0, 2].grid(True, alpha=0.3)
            if len(tracker.training_losses) > 1:
                axes[0, 2].set_yscale("log")

        # 4. Epsilon Decay
        if tracker.epsilon_values:
            axes[1, 0].plot(list(tracker.epsilon_values), "purple", linewidth=2)
            axes[1, 0].set_title("Epsilon (Exploration Rate)")
            axes[1, 0].set_xlabel("Step")
            axes[1, 0].set_ylabel("Epsilon")
            axes[1, 0].grid(True, alpha=0.3)

        # 5. Q-Values
        if tracker.q_values:
            axes[1, 1].plot(list(tracker.q_values), "orange", linewidth=2)
            axes[1, 1].set_title("Average Q-Values per Episode")
            axes[1, 1].set_xlabel("Episode")
            axes[1, 1].set_ylabel("Average Q-Value")
            axes[1, 1].grid(True, alpha=0.3)

        # 6. Episode Lengths
        if tracker.episode_lengths:
            axes[1, 2].plot(list(tracker.episode_lengths), "cyan", linewidth=2)
            axes[1, 2].set_title("Episode Lengths")
            axes[1, 2].set_xlabel("Episode")
            axes[1, 2].set_ylabel("Steps per Episode")
            axes[1, 2].grid(True, alpha=0.3)

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
            error_detail = f"Unable to generate plots. Episodes: {tracker.episode_count}, " \
                          f"Episode rewards: {len(tracker.episode_rewards) if tracker.episode_rewards else 0}"
            raise HTTPException(
                status_code=404, detail=error_detail
            )
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
        metrics = {
            "episode_count": tracker.episode_count,
            "total_steps": tracker.total_steps,
            "current_episode_reward": tracker.current_episode_reward,
            "current_episode_length": tracker.current_episode_length,
            "recent_episode_rewards": (
                list(tracker.episode_rewards)[-10:] if tracker.episode_rewards else []
            ),
            "recent_cumulative_rewards": (
                list(tracker.cumulative_rewards)[-10:]
                if tracker.cumulative_rewards
                else []
            ),
            "recent_training_losses": (
                list(tracker.training_losses)[-10:] if tracker.training_losses else []
            ),
            "current_epsilon": agent.epsilon,
            "buffer_size": len(agent.memory),
            "max_buffer_size": agent.memory.buffer_size,
            "recent_q_values": list(tracker.q_values)[-10:] if tracker.q_values else [],
            "average_episode_reward": (
                np.mean(tracker.episode_rewards) if tracker.episode_rewards else 0
            ),
            "average_episode_length": (
                np.mean(tracker.episode_lengths) if tracker.episode_lengths else 0
            ),
            # Auto-save metrics
            "auto_save_status": {
                "best_episode_reward": tracker.best_episode_reward,
                "best_cumulative_reward": tracker.best_cumulative_reward,
                "best_average_reward": tracker.best_average_reward,
                "last_save_episode": tracker.last_save_episode,
                "episodes_since_last_save": tracker.episodes_since_last_save,
                "save_frequency": tracker.save_frequency,
                "min_episodes_before_save": tracker.min_episodes_before_save,
                "next_periodic_save_in": tracker.save_frequency - tracker.episodes_since_last_save,
            }
        }
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# if __name__ == "__main__":
#     import uvicorn

#     uvicorn.run(app, host="0.0.0.0", port=5000)
