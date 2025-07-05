#!/usr/bin/env python3
"""
Training Diagnostics for Balatro DQN
Helps identify why training plateaus and where improvements can be made
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple
import os
from datetime import datetime
from collections import defaultdict

class TrainingDiagnostics:
    """Comprehensive training diagnostics to identify failure points"""
    
    def __init__(self, save_dir="diagnostics"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Detailed tracking
        self.episode_data = []
        self.failure_analysis = defaultdict(int)
        self.action_analysis = defaultdict(list)
        self.score_analysis = defaultdict(list)
        
    def add_episode_data(self, episode: int, data: Dict[str, Any]):
        """Add detailed episode data for analysis"""
        episode_info = {
            'episode': episode,
            'reward': data.get('episode_rewards', 0),
            'score': data.get('episode_scores', 0),
            'won': data.get('won', False),
            'plays_used': data.get('plays_used', 0),
            'discards_used': data.get('discards_used', 0),
            'epsilon': data.get('epsilon_values', 0),
            'steps_taken': data.get('steps_taken', 0),
            'final_hand_size': data.get('final_hand_size', 0),
            'max_hand_score': data.get('max_hand_score', 0),
            'avg_hand_score': data.get('avg_hand_score', 0),
            'invalid_actions': data.get('invalid_actions', 0),
            'game_end_reason': data.get('game_end_reason', 'unknown')
        }
        
        self.episode_data.append(episode_info)
        
        # Track failures
        if not episode_info['won']:
            self.failure_analysis[episode_info['game_end_reason']] += 1
        
        # Track action patterns
        self.action_analysis['plays_used'].append(episode_info['plays_used'])
        self.action_analysis['discards_used'].append(episode_info['discards_used'])
        
        # Track score patterns
        self.score_analysis['scores'].append(episode_info['score'])
        self.score_analysis['max_hand_scores'].append(episode_info['max_hand_score'])
        self.score_analysis['avg_hand_scores'].append(episode_info['avg_hand_score'])
    
    def analyze_training_progress(self, window_size: int = 100) -> Dict[str, Any]:
        """Analyze training progress and identify issues"""
        
        if len(self.episode_data) < window_size:
            return {"error": "Not enough data for analysis"}
        
        recent_data = self.episode_data[-window_size:]
        
        analysis = {
            'learning_progress': self._analyze_learning_progress(recent_data),
            'failure_patterns': self._analyze_failure_patterns(recent_data),
            'action_efficiency': self._analyze_action_efficiency(recent_data),
            'score_analysis': self._analyze_score_patterns(recent_data),
            'recommendations': []
        }
        
        # Generate recommendations
        analysis['recommendations'] = self._generate_recommendations(analysis)
        
        return analysis
    
    def _analyze_learning_progress(self, data: List[Dict]) -> Dict[str, Any]:
        """Analyze if the agent is actually learning"""
        
        rewards = [d['reward'] for d in data]
        scores = [d['score'] for d in data]
        epsilons = [d['epsilon'] for d in data]
        
        # Calculate trends
        reward_trend = np.polyfit(range(len(rewards)), rewards, 1)[0]
        score_trend = np.polyfit(range(len(scores)), scores, 1)[0]
        epsilon_trend = np.polyfit(range(len(epsilons)), epsilons, 1)[0]
        
        return {
            'reward_trend': reward_trend,
            'score_trend': score_trend,
            'epsilon_trend': epsilon_trend,
            'avg_reward': np.mean(rewards),
            'avg_score': np.mean(scores),
            'reward_std': np.std(rewards),
            'score_std': np.std(scores),
            'is_learning': reward_trend > 0 and score_trend > 0,
            'exploration_decreasing': epsilon_trend < 0
        }
    
    def _analyze_failure_patterns(self, data: List[Dict]) -> Dict[str, Any]:
        """Analyze why the agent is failing"""
        
        failures = [d for d in data if not d['won']]
        wins = [d for d in data if d['won']]
        
        if not failures:
            return {"status": "No failures in recent data"}
        
        failure_reasons = defaultdict(int)
        for failure in failures:
            failure_reasons[failure['game_end_reason']] += 1
        
        avg_failure_score = np.mean([f['score'] for f in failures])
        avg_win_score = np.mean([w['score'] for w in wins]) if wins else 0
        
        return {
            'failure_rate': len(failures) / len(data),
            'failure_reasons': dict(failure_reasons),
            'avg_failure_score': avg_failure_score,
            'avg_win_score': avg_win_score,
            'score_gap': avg_win_score - avg_failure_score,
            'most_common_failure': max(failure_reasons.items(), key=lambda x: x[1])[0] if failure_reasons else None
        }
    
    def _analyze_action_efficiency(self, data: List[Dict]) -> Dict[str, Any]:
        """Analyze how efficiently the agent uses actions"""
        
        plays_used = [d['plays_used'] for d in data]
        discards_used = [d['discards_used'] for d in data]
        
        avg_plays = np.mean(plays_used)
        avg_discards = np.mean(discards_used)
        
        # Analyze correlation with success
        wins = [d for d in data if d['won']]
        losses = [d for d in data if not d['won']]
        
        win_plays = [d['plays_used'] for d in wins] if wins else []
        loss_plays = [d['plays_used'] for d in losses] if losses else []
        
        return {
            'avg_plays_used': avg_plays,
            'avg_discards_used': avg_discards,
            'plays_efficiency': avg_plays / 3.0,  # How much of available plays are used
            'win_plays_avg': np.mean(win_plays) if win_plays else 0,
            'loss_plays_avg': np.mean(loss_plays) if loss_plays else 0,
            'plays_difference': (np.mean(win_plays) if win_plays else 0) - (np.mean(loss_plays) if loss_plays else 0)
        }
    
    def _analyze_score_patterns(self, data: List[Dict]) -> Dict[str, Any]:
        """Analyze score patterns and hand quality"""
        
        scores = [d['score'] for d in data]
        max_hand_scores = [d['max_hand_score'] for d in data]
        avg_hand_scores = [d['avg_hand_score'] for d in data]
        
        return {
            'score_distribution': {
                'min': np.min(scores),
                'max': np.max(scores),
                'mean': np.mean(scores),
                'median': np.median(scores),
                'std': np.std(scores)
            },
            'hand_quality': {
                'avg_max_hand': np.mean(max_hand_scores),
                'avg_avg_hand': np.mean(avg_hand_scores),
                'hand_consistency': np.std(avg_hand_scores)
            },
            'score_target_gap': 300 - np.mean(scores),  # How far from target
            'score_variance': np.var(scores)
        }
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate specific recommendations based on analysis"""
        
        recommendations = []
        
        # Learning progress recommendations
        learning = analysis['learning_progress']
        if not learning['is_learning']:
            if learning['reward_trend'] < 0:
                recommendations.append("⚠️ Rewards are decreasing - consider reducing learning rate or increasing exploration")
            if learning['score_trend'] < 0:
                recommendations.append("⚠️ Scores are decreasing - agent may be overfitting or reward function needs adjustment")
        
        if not learning['exploration_decreasing']:
            recommendations.append("⚠️ Exploration not decreasing - epsilon decay may be too slow")
        
        # Failure pattern recommendations
        failures = analysis['failure_patterns']
        if failures.get('failure_rate', 0) > 0.8:
            most_common = failures.get('most_common_failure')
            if most_common == 'no_plays_left':
                recommendations.append("🎯 Agent runs out of plays - consider improving card selection strategy")
            elif most_common == 'low_score':
                recommendations.append("🎯 Agent can't reach target score - consider improving hand evaluation")
        
        # Action efficiency recommendations
        actions = analysis['action_efficiency']
        if actions['plays_efficiency'] < 0.5:
            recommendations.append("🎯 Agent underutilizes plays - consider encouraging more aggressive play")
        
        if actions['plays_difference'] < 0:
            recommendations.append("🎯 Winning episodes use fewer plays - agent may be too conservative")
        
        # Score pattern recommendations
        scores = analysis['score_analysis']
        if scores['score_target_gap'] > 100:
            recommendations.append("🎯 Agent far from target score - consider reward shaping or curriculum learning")
        
        if scores['score_variance'] > 10000:
            recommendations.append("🎯 High score variance - agent may be unstable, consider reducing learning rate")
        
        # Architecture recommendations
        if len(recommendations) == 0:
            recommendations.append("✅ Training appears healthy - consider increasing model capacity or training time")
        elif len(recommendations) > 3:
            recommendations.append("🏗️ Multiple issues detected - consider simplifying state space or reducing complexity")
        
        return recommendations
    
    def plot_diagnostic_summary(self, episode: int, save_plot: bool = True) -> str:
        """Generate comprehensive diagnostic plots"""
        
        if len(self.episode_data) < 50:
            return ""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Training Diagnostics - Episode {episode}', fontsize=16, fontweight='bold')
        
        # 1. Learning Progress
        episodes = [d['episode'] for d in self.episode_data]
        rewards = [d['reward'] for d in self.episode_data]
        scores = [d['score'] for d in self.episode_data]
        
        ax1.plot(episodes, rewards, alpha=0.6, color='lightblue', label='Rewards')
        ax1.plot(episodes, scores, alpha=0.6, color='lightgreen', label='Scores')
        ax1.axhline(y=300, color='red', linestyle='--', alpha=0.7, label='Target Score')
        ax1.set_title('Learning Progress', fontweight='bold')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Failure Analysis
        if self.failure_analysis:
            failure_reasons = list(self.failure_analysis.keys())
            failure_counts = list(self.failure_analysis.values())
            
            ax2.bar(failure_reasons, failure_counts, color='salmon', alpha=0.7)
            ax2.set_title('Failure Reasons', fontweight='bold')
            ax2.set_xlabel('Failure Type')
            ax2.set_ylabel('Count')
            ax2.tick_params(axis='x', rotation=45)
        
        # 3. Action Usage
        plays_used = [d['plays_used'] for d in self.episode_data]
        discards_used = [d['discards_used'] for d in self.episode_data]
        
        ax3.plot(episodes, plays_used, color='blue', alpha=0.7, label='Plays Used')
        ax3.plot(episodes, discards_used, color='orange', alpha=0.7, label='Discards Used')
        ax3.axhline(y=3, color='red', linestyle='--', alpha=0.7, label='Max Available')
        ax3.set_title('Action Usage', fontweight='bold')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Actions Used')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Score Distribution
        recent_scores = scores[-100:] if len(scores) >= 100 else scores
        ax4.hist(recent_scores, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        ax4.axvline(x=300, color='red', linestyle='--', linewidth=2, label='Target Score')
        ax4.set_title('Recent Score Distribution', fontweight='bold')
        ax4.set_xlabel('Score')
        ax4.set_ylabel('Frequency')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plot_filename = f"diagnostics_episode_{episode}_{timestamp}.png"
            plot_path = os.path.join(self.save_dir, plot_filename)
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            return plot_path
        
        plt.show()
        return ""
    
    def print_diagnostic_report(self, episode: int):
        """Print a comprehensive diagnostic report"""
        
        if len(self.episode_data) < 50:
            print("⚠️ Not enough data for diagnosis (need at least 50 episodes)")
            return
        
        analysis = self.analyze_training_progress()
        
        print(f"\n🔍 Training Diagnostics Report - Episode {episode}")
        print("=" * 60)
        
        # Learning Progress
        learning = analysis['learning_progress']
        print(f"\n📈 Learning Progress:")
        print(f"   Reward trend: {learning['reward_trend']:.4f} {'✅' if learning['reward_trend'] > 0 else '❌'}")
        print(f"   Score trend: {learning['score_trend']:.4f} {'✅' if learning['score_trend'] > 0 else '❌'}")
        print(f"   Exploration decreasing: {'✅' if learning['exploration_decreasing'] else '❌'}")
        print(f"   Average reward: {learning['avg_reward']:.2f}")
        print(f"   Average score: {learning['avg_score']:.2f}")
        
        # Failure Analysis
        failures = analysis['failure_patterns']
        print(f"\n🎯 Failure Analysis:")
        print(f"   Failure rate: {failures.get('failure_rate', 0):.2%}")
        if failures.get('most_common_failure'):
            print(f"   Most common failure: {failures['most_common_failure']}")
        print(f"   Average failure score: {failures.get('avg_failure_score', 0):.2f}")
        print(f"   Score gap (win - loss): {failures.get('score_gap', 0):.2f}")
        
        # Action Efficiency
        actions = analysis['action_efficiency']
        print(f"\n⚡ Action Efficiency:")
        print(f"   Plays used: {actions['avg_plays_used']:.2f}/3 ({actions['plays_efficiency']:.1%})")
        print(f"   Discards used: {actions['avg_discards_used']:.2f}/3")
        print(f"   Plays difference (win - loss): {actions['plays_difference']:.2f}")
        
        # Score Analysis
        scores = analysis['score_analysis']
        score_dist = scores['score_distribution']
        print(f"\n📊 Score Analysis:")
        print(f"   Score range: {score_dist['min']:.0f} - {score_dist['max']:.0f}")
        print(f"   Average score: {score_dist['mean']:.2f}")
        print(f"   Score variance: {scores['score_variance']:.2f}")
        print(f"   Distance from target: {scores['score_target_gap']:.2f}")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        for i, rec in enumerate(analysis['recommendations'], 1):
            print(f"   {i}. {rec}")
        
        print("=" * 60)

def create_diagnostics() -> TrainingDiagnostics:
    """Factory function to create diagnostics instance"""
    return TrainingDiagnostics()

# Example usage
if __name__ == "__main__":
    diagnostics = create_diagnostics()
    
    # Simulate some training data
    for episode in range(1, 101):
        data = {
            'episode_rewards': np.random.normal(10, 5),
            'episode_scores': np.random.normal(150, 50),
            'won': np.random.random() < 0.2,
            'plays_used': np.random.randint(1, 4),
            'discards_used': np.random.randint(0, 4),
            'epsilon_values': max(0.05, 1.0 - episode * 0.01),
            'steps_taken': np.random.randint(5, 15),
            'final_hand_size': np.random.randint(3, 9),
            'max_hand_score': np.random.normal(200, 50),
            'avg_hand_score': np.random.normal(100, 30),
            'invalid_actions': np.random.randint(0, 3),
            'game_end_reason': np.random.choice(['no_plays_left', 'low_score', 'won'])
        }
        diagnostics.add_episode_data(episode, data)
    
    # Generate diagnostic report
    diagnostics.print_diagnostic_report(100)
    diagnostics.plot_diagnostic_summary(100) 