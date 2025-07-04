"""
MLflow-based experiment tracking for Balatro DQN Agent
Replaces the custom PerformanceTracker with professional ML experiment tracking
"""

import os
import mlflow
import mlflow.pytorch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from datetime import datetime
from typing import Optional, Dict, Any, List
import tempfile
import json
import torch


class MLflowTracker:
    """MLflow-based experiment tracker for DQN training"""
    
    def __init__(self, experiment_name: str = "balatro_dqn_training"):
        self.experiment_name = experiment_name
        self.current_run = None
        self.episode_count = 0
        self.total_steps = 0
        
        # Current episode tracking
        self.current_episode_reward = 0
        self.current_episode_length = 0
        self.current_episode_q_values = []
        
        # Best performance tracking
        self.best_episode_reward = float("-inf")
        self.best_average_reward = float("-inf")
        
        # Track episode rewards for average calculation
        self.episode_rewards_history = []
        
        # Running averages tracking
        self.running_averages = {
            "rewards": [],
            "lengths": [],
            "q_values": [],
            "losses": []
        }
        
        # Set up MLflow
        self.setup_mlflow()
        
    def setup_mlflow(self):
        """Initialize MLflow experiment"""
        try:
            # Set tracking URI to local directory
            tracking_uri = "file://" + os.path.abspath("mlflow_tracking")
            mlflow.set_tracking_uri(tracking_uri)
            print(f"📊 MLflow tracking URI: {tracking_uri}")
            
            # Create or get experiment
            try:
                experiment_id = mlflow.create_experiment(self.experiment_name)
                print(f"✅ Created new MLflow experiment: {self.experiment_name}")
            except mlflow.exceptions.MlflowException:
                experiment = mlflow.get_experiment_by_name(self.experiment_name)
                experiment_id = experiment.experiment_id
                print(f"📂 Using existing MLflow experiment: {self.experiment_name}")
            
            mlflow.set_experiment(self.experiment_name)
            
        except Exception as e:
            print(f"❌ Error setting up MLflow: {e}")
            raise
    
    def start_run(self, run_name: Optional[str] = None):
        """Start a new MLflow run"""
        if self.current_run is not None:
            print("⚠️  Ending previous MLflow run before starting new one")
            self.end_run()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if run_name is None:
            run_name = f"dqn_training_{timestamp}"
        
        self.current_run = mlflow.start_run(run_name=run_name)
        print(f"🚀 Started MLflow run: {run_name}")
        
        # Log initial parameters
        self.log_hyperparameters()
        
        return self.current_run
    
    def end_run(self):
        """End the current MLflow run"""
        if self.current_run is not None:
            mlflow.end_run()
            print(f"🏁 Ended MLflow run: {self.current_run.info.run_name}")
            self.current_run = None
    
    def log_hyperparameters(self):
        """Log hyperparameters from config"""
        try:
            from config.settings import (
                STATE_SIZE, ACTION_SIZE, BUFFER_SIZE, BATCH_SIZE, 
                GAMMA, LEARNING_RATE, TAU, UPDATE_EVERY, 
                INITIAL_EPSILON, EPSILON_DECAY, FINAL_EPSILON
            )
            
            params = {
                "state_size": STATE_SIZE,
                "action_size": ACTION_SIZE,
                "buffer_size": BUFFER_SIZE,
                "batch_size": BATCH_SIZE,
                "gamma": GAMMA,
                "learning_rate": LEARNING_RATE,
                "tau": TAU,
                "update_every": UPDATE_EVERY,
                "initial_epsilon": INITIAL_EPSILON,
                "epsilon_decay": EPSILON_DECAY,
                "final_epsilon": FINAL_EPSILON,
            }
            
            mlflow.log_params(params)
            print("📋 Logged hyperparameters to MLflow")
            
        except Exception as e:
            print(f"⚠️  Could not log hyperparameters: {e}")
    
    def log_training_step(self, loss: float, step: int):
        """Log training loss"""
        if self.current_run is None:
            return
        
        mlflow.log_metric("training_loss", loss, step=step)
        
    def log_action(self, q_value: float, epsilon: float):
        """Log action-related metrics"""
        if self.current_run is None:
            return
        
        self.current_episode_q_values.append(q_value)
        self.current_episode_length += 1
        self.total_steps += 1
        
        # Log current step metrics
        mlflow.log_metric("epsilon", epsilon, step=self.total_steps)
        mlflow.log_metric("q_value", q_value, step=self.total_steps)
        
    def log_step_reward(self, reward: float):
        """Log reward for current step"""
        self.current_episode_reward += reward
        
        if self.current_run is not None:
            mlflow.log_metric("step_reward", reward, step=self.total_steps)
    
    def log_episode_end(self, agent):
        """Log end of episode with comprehensive metrics"""
        if self.current_run is None:
            return
        
        self.episode_count += 1
        
        # Calculate episode metrics
        avg_q_value = np.mean(self.current_episode_q_values) if self.current_episode_q_values else 0
        
        # Log episode metrics
        mlflow.log_metric("episode_reward", self.current_episode_reward, step=self.episode_count)
        mlflow.log_metric("episode_length", self.current_episode_length, step=self.episode_count)
        mlflow.log_metric("avg_q_value_per_episode", avg_q_value, step=self.episode_count)
        mlflow.log_metric("buffer_size", len(agent.memory), step=self.episode_count)
        
        # Track episode reward history
        self.episode_rewards_history.append(self.current_episode_reward)
        
        # Log running averages
        self.log_running_averages(agent, avg_q_value, self.current_episode_length)
        
        # Check for new records
        is_best_episode = self.current_episode_reward > self.best_episode_reward
        is_best_average = False
        
        if is_best_episode:
            self.best_episode_reward = self.current_episode_reward
            mlflow.log_metric("best_episode_reward", self.best_episode_reward)
            print(f"🏆 NEW BEST EPISODE REWARD: {self.best_episode_reward:.2f}")
        
        # Calculate 10-episode average
        if len(self.episode_rewards_history) >= 10:
            recent_avg = np.mean(self.episode_rewards_history[-10:])
            mlflow.log_metric("10_episode_average", recent_avg, step=self.episode_count)
            
            if recent_avg > self.best_average_reward:
                self.best_average_reward = recent_avg
                is_best_average = True
                mlflow.log_metric("best_10_episode_average", self.best_average_reward)
                print(f"📈 NEW BEST 10-EPISODE AVERAGE: {self.best_average_reward:.2f}")
        elif len(self.episode_rewards_history) > 0:
            # Log current average for episodes < 10
            current_avg = np.mean(self.episode_rewards_history)
            mlflow.log_metric("current_average", current_avg, step=self.episode_count)
        
        print(f"Episode {self.episode_count} completed - Reward: {self.current_episode_reward:.2f}, Length: {self.current_episode_length}")
        
        # Auto-save model if performance improved
        if is_best_episode or is_best_average:
            self.save_best_model(agent, is_best_episode, is_best_average)
        
        # Reset episode tracking
        self.current_episode_reward = 0
        self.current_episode_length = 0
        self.current_episode_q_values = []
        
        # Create and log plots every 5 episodes
        if self.episode_count % 5 == 0:
            self.create_and_log_plots()
    
    def calculate_running_averages(self, window_sizes: List[int] = [5, 10, 20, 50]):
        """Calculate running averages for different window sizes"""
        if self.current_run is None:
            return
        
        # Calculate running averages for rewards
        for window in window_sizes:
            if len(self.episode_rewards_history) >= window:
                recent_rewards = self.episode_rewards_history[-window:]
                avg_reward = np.mean(recent_rewards)
                mlflow.log_metric(f"running_avg_reward_{window}", avg_reward, step=self.episode_count)
                
                # Store in running averages for plotting
                if f"reward_{window}" not in self.running_averages["rewards"]:
                    self.running_averages["rewards"].append(f"reward_{window}")
                    self.running_averages["rewards"].append([])
                self.running_averages["rewards"][-1].append(avg_reward)
    
    def log_running_averages(self, agent, avg_q_value: float, episode_length: int):
        """Log running averages for current episode"""
        if self.current_run is None:
            return
        
        # Calculate running averages with different window sizes
        self.calculate_running_averages()
        
        # Log episode-specific running averages
        if len(self.episode_rewards_history) > 0:
            # All-time average
            all_time_avg = np.mean(self.episode_rewards_history)
            mlflow.log_metric("all_time_avg_reward", all_time_avg, step=self.episode_count)
            
            # Recent trend (last 5 vs previous 5)
            if len(self.episode_rewards_history) >= 10:
                recent_5 = np.mean(self.episode_rewards_history[-5:])
                previous_5 = np.mean(self.episode_rewards_history[-10:-5])
                trend = recent_5 - previous_5
                mlflow.log_metric("reward_trend_5_episodes", trend, step=self.episode_count)
        
        # Log other running averages
        if len(agent.training_loss) > 0:
            recent_losses = agent.training_loss[-min(50, len(agent.training_loss)):]
            avg_loss = np.mean(recent_losses)
            mlflow.log_metric("running_avg_loss", avg_loss, step=self.episode_count)
        
        # Log episode length trend
        if len(self.running_averages["lengths"]) > 0:
            recent_lengths = self.running_averages["lengths"][-min(10, len(self.running_averages["lengths"])):]
            avg_length = np.mean(recent_lengths)
            mlflow.log_metric("running_avg_episode_length", avg_length, step=self.episode_count)
        
        # Store current episode data for future averages
        self.running_averages["lengths"].append(episode_length)
        self.running_averages["q_values"].append(avg_q_value)
    
    def save_best_model(self, agent, is_best_episode: bool, is_best_average: bool):
        """Save model when best performance is achieved"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if is_best_episode:
                # Save best episode model
                model_name = f"best_episode_model_{timestamp}_reward_{self.best_episode_reward:.2f}"
                mlflow.pytorch.log_model(
                    agent.qnetwork_local,
                    f"models/{model_name}",
                    registered_model_name="balatro_dqn_best_episode"
                )
                print(f"💾 Best episode model saved to MLflow: {model_name}")
            
            if is_best_average:
                # Save best average model  
                model_name = f"best_average_model_{timestamp}_avg_{self.best_average_reward:.2f}"
                mlflow.pytorch.log_model(
                    agent.qnetwork_local,
                    f"models/{model_name}", 
                    registered_model_name="balatro_dqn_best_average"
                )
                print(f"💾 Best average model saved to MLflow: {model_name}")
                
        except Exception as e:
            print(f"❌ Error saving model to MLflow: {e}")
    
    def create_and_log_plots(self):
        """Create performance plots and log to MLflow"""
        try:
            if len(self.episode_rewards_history) < 2:
                print("📊 Not enough episodes for plots yet")
                return
                
            # Create performance plots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f"Performance Overview - Episode {self.episode_count}", fontsize=16)
            
            # Plot 1: Episode Rewards with Running Averages
            episodes = list(range(1, len(self.episode_rewards_history) + 1))
            axes[0, 0].plot(episodes, self.episode_rewards_history, 'b-', alpha=0.7, label='Episode Reward')
            
            # Add multiple running averages if we have enough data
            colors = ['r-', 'g-', 'orange', 'purple']
            window_sizes = [5, 10, 20, 50]
            
            for i, window in enumerate(window_sizes):
                if len(self.episode_rewards_history) >= window:
                    moving_avg = np.convolve(self.episode_rewards_history, np.ones(window)/window, mode='valid')
                    avg_episodes = episodes[window-1:]
                    axes[0, 0].plot(avg_episodes, moving_avg, colors[i], linewidth=2, 
                                   label=f'{window}-Episode Average')
            
            axes[0, 0].set_title("Episode Rewards")
            axes[0, 0].set_xlabel("Episode")
            axes[0, 0].set_ylabel("Reward")
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: Best Performance Tracking
            axes[0, 1].bar(['Best Episode', 'Best 10-Ep Avg'], 
                          [self.best_episode_reward, self.best_average_reward],
                          color=['gold', 'silver'])
            axes[0, 1].set_title("Best Performance")
            axes[0, 1].set_ylabel("Reward")
            
            # Plot 3: Recent Performance (last 10 episodes)
            recent_episodes = episodes[-10:] if len(episodes) >= 10 else episodes
            recent_rewards = self.episode_rewards_history[-10:] if len(self.episode_rewards_history) >= 10 else self.episode_rewards_history
            axes[1, 0].bar(recent_episodes, recent_rewards, color='lightblue')
            axes[1, 0].set_title("Recent Episodes (Last 10)")
            axes[1, 0].set_xlabel("Episode")
            axes[1, 0].set_ylabel("Reward")
            axes[1, 0].grid(True, alpha=0.3)
            
            # Plot 4: Running Averages Comparison
            if len(self.episode_rewards_history) >= 5:
                # Get the most recent running averages
                recent_episodes = episodes[-min(50, len(episodes)):]
                recent_rewards = self.episode_rewards_history[-min(50, len(self.episode_rewards_history)):]
                
                # Calculate running averages for the recent period
                for i, window in enumerate(window_sizes):
                    if len(recent_rewards) >= window:
                        moving_avg = np.convolve(recent_rewards, np.ones(window)/window, mode='valid')
                        avg_episodes = recent_episodes[window-1:]
                        axes[1, 1].plot(avg_episodes, moving_avg, colors[i], linewidth=2, 
                                       label=f'{window}-Episode Avg')
                
                axes[1, 1].set_title("Running Averages Comparison (Recent 50 Episodes)")
                axes[1, 1].set_xlabel("Episode")
                axes[1, 1].set_ylabel("Average Reward")
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
            else:
                # Fallback to statistics if not enough data
                if len(self.episode_rewards_history) > 0:
                    stats_text = f"""
Total Episodes: {self.episode_count}
Current Reward: {self.episode_rewards_history[-1]:.2f}
Average Reward: {np.mean(self.episode_rewards_history):.2f}
Best Reward: {self.best_episode_reward:.2f}
Best 10-Ep Avg: {self.best_average_reward:.2f}
Standard Deviation: {np.std(self.episode_rewards_history):.2f}
                    """
                    axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes, 
                                   fontsize=10, verticalalignment='top', fontfamily='monospace')
                    axes[1, 1].set_title("Performance Statistics")
                    axes[1, 1].set_xticks([])
                    axes[1, 1].set_yticks([])
            
            plt.tight_layout()
            
            # Save to temporary file and log
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                plt.savefig(tmp.name, dpi=150, bbox_inches="tight")
                plt.close()
                
                # Log plot as artifact
                mlflow.log_artifact(tmp.name, f"performance_plots/episode_{self.episode_count}")
                print(f"📈 Performance plot logged to MLflow")
                
                # Clean up
                os.unlink(tmp.name)
                
        except Exception as e:
            print(f"❌ Error creating/logging plots: {e}")
    
    def log_custom_metric(self, key: str, value: float, step: Optional[int] = None):
        """Log any custom metric"""
        if self.current_run is not None:
            mlflow.log_metric(key, value, step=step)
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log any file as an artifact"""
        if self.current_run is not None:
            mlflow.log_artifact(local_path, artifact_path)
    
    def save_step_logs(self):
        """Save step-level logs - MLflow equivalent (compatibility method)"""
        try:
            if self.current_run is None:
                return
            
            # Log current step metrics to MLflow
            mlflow.log_metric("current_episode_reward", self.current_episode_reward, step=self.total_steps)
            mlflow.log_metric("current_episode_length", self.current_episode_length, step=self.total_steps)
            
            if self.current_episode_q_values:
                latest_q_value = self.current_episode_q_values[-1]
                mlflow.log_metric("latest_q_value", latest_q_value, step=self.total_steps)
            
            print(f"📊 Step {self.total_steps} metrics logged to MLflow")
            
        except Exception as e:
            print(f"❌ Error logging step metrics to MLflow: {e}")

    def save_step_plots(self):
        """Save step plots - MLflow equivalent (compatibility method)"""
        try:
            if self.current_run is None or self.total_steps < 1:
                return None
            
            # Create a simple real-time plot
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle(f"Real-time Monitoring - Step {self.total_steps}", fontsize=14)
            
            # Plot 1: Current episode Q-values
            if self.current_episode_q_values:
                axes[0, 0].plot(self.current_episode_q_values, 'orange', linewidth=2)
                axes[0, 0].set_title(f"Current Episode Q-Values (Episode {self.episode_count + 1})")
                axes[0, 0].set_xlabel("Step in Episode")
                axes[0, 0].set_ylabel("Q-Value")
                axes[0, 0].grid(True, alpha=0.3)
            else:
                axes[0, 0].text(0.5, 0.5, "No Q-values yet", ha="center", va="center", transform=axes[0, 0].transAxes)
                axes[0, 0].set_title("Current Episode Q-Values")
            
            # Plot 2: Episode progress
            axes[0, 1].bar(['Reward', 'Length'], [self.current_episode_reward, self.current_episode_length])
            axes[0, 1].set_title("Current Episode Progress")
            axes[0, 1].set_ylabel("Value")
            
            # Plot 3: Steps counter
            axes[1, 0].text(0.5, 0.5, f"Total Steps:\n{self.total_steps}", ha="center", va="center", 
                           transform=axes[1, 0].transAxes, fontsize=16, weight='bold')
            axes[1, 0].set_title("Step Counter")
            axes[1, 0].set_xticks([])
            axes[1, 0].set_yticks([])
            
            # Plot 4: Episode counter
            axes[1, 1].text(0.5, 0.5, f"Episodes:\n{self.episode_count}", ha="center", va="center", 
                           transform=axes[1, 1].transAxes, fontsize=16, weight='bold')
            axes[1, 1].set_title("Episode Counter")
            axes[1, 1].set_xticks([])
            axes[1, 1].set_yticks([])
            
            plt.tight_layout()
            
            # Save to temporary file and log to MLflow
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                plt.savefig(tmp.name, dpi=150, bbox_inches="tight")
                plt.close()
                
                # Log as artifact with step info
                mlflow.log_artifact(tmp.name, f"step_plots/step_{self.total_steps}")
                print(f"📈 Step plot logged to MLflow")
                
                # Clean up
                os.unlink(tmp.name)
                
                return tmp.name
                
        except Exception as e:
            print(f"❌ Error creating/logging step plots: {e}")
            return None

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of current metrics for API"""
        return {
            "episode_count": self.episode_count,
            "total_steps": self.total_steps,
            "current_episode_reward": self.current_episode_reward,
            "current_episode_length": self.current_episode_length,
            "best_episode_reward": self.best_episode_reward,
            "best_average_reward": self.best_average_reward,
            "mlflow_experiment": self.experiment_name,
            "current_run_id": self.current_run.info.run_id if self.current_run else None,
            "mlflow_ui_url": "http://localhost:5000"  # Default MLflow UI URL
        }

    def get_best_models_info(self) -> Dict[str, Any]:
        """Get information about the best models available in MLflow"""
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                return {"error": "Experiment not found"}
            
            # Search for runs with the best models
            best_episode_runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string="metrics.best_episode_reward > 0",
                order_by=["metrics.best_episode_reward DESC"],
                max_results=5
            )
            
            best_average_runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string="metrics.best_10_episode_average > 0",
                order_by=["metrics.best_10_episode_average DESC"],
                max_results=5
            )
            
            return {
                "best_episode_runs": best_episode_runs.to_dict("records") if not best_episode_runs.empty else [],
                "best_average_runs": best_average_runs.to_dict("records") if not best_average_runs.empty else [],
                "experiment_id": experiment.experiment_id
            }
            
        except Exception as e:
            return {"error": f"Error getting best models info: {str(e)}"}
    
    def load_best_model_from_mlflow(self, agent, model_type: str = "episode") -> Dict[str, Any]:
        """
        Load the best model from MLflow and update the agent
        
        Args:
            agent: The DQNAgent instance to update
            model_type: Either "episode" (best single episode) or "average" (best 10-episode average)
            
        Returns:
            Dictionary with load status and details
        """
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                return {"success": False, "error": "Experiment not found"}
            
            # Determine which registered model to use
            if model_type == "episode":
                registered_model_name = "balatro_dqn_best_episode"
                metric_column = "metrics.best_episode_reward"
            elif model_type == "average":
                registered_model_name = "balatro_dqn_best_average"
                metric_column = "metrics.best_10_episode_average"
            else:
                return {"success": False, "error": f"Invalid model_type: {model_type}. Use 'episode' or 'average'"}
            
            # Find the best run
            best_runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=f"{metric_column} > 0",
                order_by=[f"{metric_column} DESC"],
                max_results=1
            )
            
            if best_runs.empty:
                return {"success": False, "error": f"No runs found with {model_type} models"}
            
            best_run = best_runs.iloc[0]
            run_id = best_run["run_id"]
            
            # Get the model artifacts for this run
            artifacts = mlflow.MlflowClient().list_artifacts(run_id, path="models")
            if not artifacts:
                return {"success": False, "error": f"No model artifacts found in run {run_id}"}
            
            # Find the best model artifact (latest timestamp)
            model_artifacts = [a for a in artifacts if a.path.startswith(f"models/best_{model_type}_model")]
            if not model_artifacts:
                return {"success": False, "error": f"No {model_type} model artifacts found"}
            
            # Sort by path (which includes timestamp) and get the latest
            latest_model = sorted(model_artifacts, key=lambda x: x.path)[-1]
            model_uri = f"runs:/{run_id}/{latest_model.path}"
            
            # Load the model
            loaded_model = mlflow.pytorch.load_model(model_uri)
            
            # Update the agent's networks
            agent.qnetwork_local.load_state_dict(loaded_model.state_dict())
            agent.qnetwork_target.load_state_dict(loaded_model.state_dict())
            
            # Update tracker's best performance tracking
            if model_type == "episode":
                self.best_episode_reward = best_run["metrics.best_episode_reward"]
            else:
                self.best_average_reward = best_run["metrics.best_10_episode_average"]
            
            return {
                "success": True,
                "run_id": run_id,
                "run_name": best_run.get("tags.mlflow.runName", "Unknown"),
                "model_path": latest_model.path,
                "performance": best_run[metric_column],
                "loaded_from": model_uri,
                "episode_count": int(best_run.get("metrics.episode_count", 0)) if "metrics.episode_count" in best_run else None
            }
            
        except Exception as e:
            return {"success": False, "error": f"Error loading model from MLflow: {str(e)}"}
    
    def list_available_models(self) -> Dict[str, Any]:
        """List all available models with their performance metrics"""
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                return {"error": "Experiment not found"}
            
            # Get all runs with models
            all_runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["start_time DESC"],
                max_results=50
            )
            
            models_info = []
            for _, run in all_runs.iterrows():
                try:
                    # Check if this run has model artifacts
                    artifacts = mlflow.MlflowClient().list_artifacts(run["run_id"], path="models")
                    if artifacts:
                        models = []
                        for artifact in artifacts:
                            if "best_episode_model" in artifact.path:
                                models.append({
                                    "type": "best_episode",
                                    "path": artifact.path,
                                    "performance": run.get("metrics.best_episode_reward", 0)
                                })
                            elif "best_average_model" in artifact.path:
                                models.append({
                                    "type": "best_average", 
                                    "path": artifact.path,
                                    "performance": run.get("metrics.best_10_episode_average", 0)
                                })
                        
                        if models:
                            models_info.append({
                                "run_id": run["run_id"],
                                "run_name": run.get("tags.mlflow.runName", "Unknown"),
                                "start_time": run["start_time"],
                                "episode_count": int(run.get("metrics.episode_count", 0)) if "metrics.episode_count" in run else None,
                                "models": models
                            })
                except Exception as e:
                    # Skip runs that can't be processed
                    continue
            
            return {
                "available_models": models_info,
                "total_runs": len(all_runs),
                "runs_with_models": len(models_info)
            }
            
        except Exception as e:
            return {"error": f"Error listing models: {str(e)}"}


def get_mlflow_ui_url() -> str:
    """Get the MLflow UI URL"""
    tracking_uri = mlflow.get_tracking_uri()
    if tracking_uri.startswith("file://"):
        return "http://localhost:5000"
    return tracking_uri 