#!/usr/bin/env python3
"""
Training plots generator for Balatro DQN training
Generates real-time plots and saves them as MLflow artifacts
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any
import os
from datetime import datetime

class TrainingPlotter:
    """Handles real-time plotting and MLflow artifact saving during training"""
    
    def __init__(self, mlflow_tracker=None, save_dir="training_plots"):
        self.mlflow_tracker = mlflow_tracker
        self.save_dir = save_dir
        self.plot_data = {
            'episode_rewards': [],
            'episode_scores': [],
            'win_rates': [],
            'epsilon_values': [],
            'buffer_sizes': [],
            'plays_used': [],
            'discards_used': []
        }
        
        # Create save directory with absolute path
        self.save_dir = os.path.abspath(save_dir)
        os.makedirs(self.save_dir, exist_ok=True)
        print(f"📁 Training plots will be saved to: {self.save_dir}")
        
        # Set up matplotlib style
        plt.style.use('seaborn-v0_8')
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
    
    def add_data_point(self, episode: int, data: Dict[str, Any]):
        """Add a new data point for plotting"""
        for key in self.plot_data:
            if key in data:
                self.plot_data[key].append(data[key])
    
    def calculate_running_averages(self, data: List[float], window: int = 100) -> List[float]:
        """Calculate running average with specified window size"""
        if len(data) < window:
            return data
        
        running_avg = []
        for i in range(len(data)):
            start_idx = max(0, i - window + 1)
            window_data = data[start_idx:i + 1]
            running_avg.append(np.mean(window_data))
        
        return running_avg
    
    def plot_training_progress(self, episode: int, save_plot: bool = True) -> str:
        """Generate comprehensive training progress plots"""
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Balatro DQN Training Progress - Episode {episode}', fontsize=16, fontweight='bold')
        
        # 1. Episode Rewards (with running average)
        if len(self.plot_data['episode_rewards']) > 0:
            rewards = self.plot_data['episode_rewards']
            episodes = range(1, len(rewards) + 1)
            
            ax1.plot(episodes, rewards, alpha=0.6, color='lightblue', label='Raw Rewards')
            
            # Add running average
            if len(rewards) >= 50:
                running_avg = self.calculate_running_averages(rewards, window=50)
                ax1.plot(episodes, running_avg, color='blue', linewidth=2, label='50-Episode Running Average')
            
            ax1.set_title('Episode Rewards', fontweight='bold')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Reward')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 2. Win Rate (100-episode moving average)
        if len(self.plot_data['win_rates']) > 0:
            win_rates = self.plot_data['win_rates']
            win_episodes = range(100, len(win_rates) * 100 + 1, 100)
            
            ax2.plot(win_episodes, win_rates, color='green', linewidth=2, marker='o', markersize=4)
            ax2.set_title('Win Rate (100-Episode Moving Average)', fontweight='bold')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Win Rate')
            ax2.set_ylim(0, 1)
            ax2.grid(True, alpha=0.3)
            
            # Add target line at 50%
            ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='50% Target')
            ax2.legend()
        
        # 3. Episode Scores (with running average)
        if len(self.plot_data['episode_scores']) > 0:
            scores = self.plot_data['episode_scores']
            episodes = range(1, len(scores) + 1)
            
            ax3.plot(episodes, scores, alpha=0.6, color='lightgreen', label='Raw Scores')
            
            # Add running average
            if len(scores) >= 50:
                running_avg = self.calculate_running_averages(scores, window=50)
                ax3.plot(episodes, running_avg, color='green', linewidth=2, label='50-Episode Running Average')
            
            # Add target line
            ax3.axhline(y=300, color='red', linestyle='--', alpha=0.7, label='Target Score (300)')
            
            ax3.set_title('Episode Scores', fontweight='bold')
            ax3.set_xlabel('Episode')
            ax3.set_ylabel('Score')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. Training Parameters
        if len(self.plot_data['epsilon_values']) > 0:
            epsilons = self.plot_data['epsilon_values']
            episodes = range(1, len(epsilons) + 1)
            
            ax4.plot(episodes, epsilons, color='orange', linewidth=2, label='Epsilon')
            ax4.set_title('Training Parameters', fontweight='bold')
            ax4.set_xlabel('Episode')
            ax4.set_ylabel('Epsilon')
            ax4.set_ylim(0, 1.1)
            ax4.grid(True, alpha=0.3)
            ax4.legend()
            
            # Add buffer size on secondary y-axis
            if len(self.plot_data['buffer_sizes']) > 0:
                ax4_twin = ax4.twinx()
                buffer_sizes = self.plot_data['buffer_sizes']
                ax4_twin.plot(episodes, buffer_sizes, color='purple', alpha=0.7, label='Buffer Size')
                ax4_twin.set_ylabel('Buffer Size', color='purple')
                ax4_twin.tick_params(axis='y', labelcolor='purple')
                ax4_twin.legend(loc='upper right')
        
        plt.tight_layout()
        
        # Save plot
        if save_plot:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plot_filename = f"training_progress_episode_{episode}_{timestamp}.png"
            plot_path = os.path.join(self.save_dir, plot_filename)
            
            # Ensure directory exists before saving
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)
            
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Log to MLflow if available
            if self.mlflow_tracker:
                try:
                    self.mlflow_tracker.log_artifact(plot_path, "training_plots")
                    print(f"📊 Training plot saved and logged to MLflow: {plot_filename}")
                except Exception as e:
                    print(f"⚠️ Failed to log plot to MLflow: {e}")
            
            return plot_path
        
        plt.show()
        return ""
    
    def plot_final_summary(self, episode: int, save_plot: bool = True) -> str:
        """Generate final training summary plot"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Balatro DQN Training Summary - Episode {episode}', fontsize=16, fontweight='bold')
        
        # 1. Reward distribution
        if len(self.plot_data['episode_rewards']) > 0:
            rewards = self.plot_data['episode_rewards']
            ax1.hist(rewards, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            ax1.set_title('Reward Distribution', fontweight='bold')
            ax1.set_xlabel('Reward')
            ax1.set_ylabel('Frequency')
            ax1.grid(True, alpha=0.3)
        
        # 2. Score distribution
        if len(self.plot_data['episode_scores']) > 0:
            scores = self.plot_data['episode_scores']
            ax2.hist(scores, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
            ax2.axvline(x=300, color='red', linestyle='--', linewidth=2, label='Target Score')
            ax2.set_title('Score Distribution', fontweight='bold')
            ax2.set_xlabel('Score')
            ax2.set_ylabel('Frequency')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. Learning curve (rewards over time)
        if len(self.plot_data['episode_rewards']) > 0:
            rewards = self.plot_data['episode_rewards']
            episodes = range(1, len(rewards) + 1)
            
            # Calculate moving averages
            window_50 = self.calculate_running_averages(rewards, 50)
            window_100 = self.calculate_running_averages(rewards, 100)
            
            ax3.plot(episodes, rewards, alpha=0.3, color='lightblue', label='Raw Rewards')
            if len(window_50) > 0:
                ax3.plot(episodes, window_50, color='blue', linewidth=2, label='50-Episode Average')
            if len(window_100) > 0:
                ax3.plot(episodes, window_100, color='darkblue', linewidth=2, label='100-Episode Average')
            
            ax3.set_title('Learning Curve', fontweight='bold')
            ax3.set_xlabel('Episode')
            ax3.set_ylabel('Reward')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. Training statistics
        if len(self.plot_data['epsilon_values']) > 0:
            epsilons = self.plot_data['epsilon_values']
            episodes = range(1, len(epsilons) + 1)
            
            ax4.plot(episodes, epsilons, color='orange', linewidth=2, label='Epsilon')
            ax4.set_title('Exploration Rate', fontweight='bold')
            ax4.set_xlabel('Episode')
            ax4.set_ylabel('Epsilon')
            ax4.set_ylim(0, 1.1)
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        if save_plot:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plot_filename = f"training_summary_episode_{episode}_{timestamp}.png"
            plot_path = os.path.join(self.save_dir, plot_filename)
            
            # Ensure directory exists before saving
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)
            
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Log to MLflow if available
            if self.mlflow_tracker:
                try:
                    self.mlflow_tracker.log_artifact(plot_path, "training_summary")
                    print(f"📊 Training summary plot saved and logged to MLflow: {plot_filename}")
                except Exception as e:
                    print(f"⚠️ Failed to log summary plot to MLflow: {e}")
            
            return plot_path
        
        plt.show()
        return ""
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get current training statistics"""
        stats = {}
        
        if len(self.plot_data['episode_rewards']) > 0:
            rewards = self.plot_data['episode_rewards']
            stats['total_episodes'] = len(rewards)
            stats['avg_reward'] = np.mean(rewards)
            stats['max_reward'] = np.max(rewards)
            stats['min_reward'] = np.min(rewards)
            
            # Recent performance (last 100 episodes)
            if len(rewards) >= 100:
                recent_rewards = rewards[-100:]
                stats['recent_avg_reward'] = np.mean(recent_rewards)
                stats['recent_max_reward'] = np.max(recent_rewards)
        
        if len(self.plot_data['episode_scores']) > 0:
            scores = self.plot_data['episode_scores']
            stats['avg_score'] = np.mean(scores)
            stats['max_score'] = np.max(scores)
            stats['min_score'] = np.min(scores)
            stats['wins'] = sum(1 for score in scores if score >= 300)
            stats['win_rate'] = stats['wins'] / len(scores) if len(scores) > 0 else 0
        
        if len(self.plot_data['win_rates']) > 0:
            stats['current_win_rate'] = self.plot_data['win_rates'][-1]
        
        if len(self.plot_data['epsilon_values']) > 0:
            stats['current_epsilon'] = self.plot_data['epsilon_values'][-1]
        
        return stats

def create_plotter(mlflow_tracker=None) -> TrainingPlotter:
    """Factory function to create a TrainingPlotter instance"""
    return TrainingPlotter(mlflow_tracker=mlflow_tracker)

# Example usage:
if __name__ == "__main__":
    # This would be used in your training script
    plotter = create_plotter()
    
    # Simulate some training data
    for episode in range(1, 101):
        data = {
            'episode_rewards': np.random.normal(10, 5),
            'episode_scores': np.random.normal(150, 50),
            'win_rates': np.random.uniform(0, 0.3),
            'epsilon_values': max(0.05, 1.0 - episode * 0.01),
            'buffer_sizes': min(100000, episode * 1000),
            'plays_used': np.random.randint(1, 4),
            'discards_used': np.random.randint(0, 4)
        }
        plotter.add_data_point(episode, data)
        
        # Generate plots every 20 episodes
        if episode % 20 == 0:
            plotter.plot_training_progress(episode)
    
    # Final summary
    plotter.plot_final_summary(100)
    
    # Print stats
    stats = plotter.get_training_stats()
    print("Training Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}") 