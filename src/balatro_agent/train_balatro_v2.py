import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from datetime import datetime

# Import our custom gym environment and DQN agent
from balatro_gym_v2 import BalatroGymEnv
from models.dqn_agent import DQNAgent
from mlflow_tracker import MLflowTracker
from training_plots import create_plotter
from training_diagnostics import create_diagnostics

# Training parameters
LEARNING_RATE = 0.001
N_EPISODES = 10000  # More episodes for this complex environment
START_EPSILON = 1.0
EPSILON_DECAY = START_EPSILON / (N_EPISODES / 2)
FINAL_EPSILON = 0.05

# Model parameters
STATE_SIZE = 229  # Updated for enhanced observation space
ACTION_SIZE = 436  # Valid actions with 1-5 card constraint
BUFFER_SIZE = int(1e5)
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 1e-3
UPDATE_EVERY = 4

def create_agent():
    """Create and return a DQN agent configured for Balatro"""
    return DQNAgent(
        state_size=STATE_SIZE,
        action_size=ACTION_SIZE,
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        lr=LEARNING_RATE,
        tau=TAU,
        update_every=UPDATE_EVERY,
        initial_epsilon=START_EPSILON,
        epsilon_decay=EPSILON_DECAY,
        final_epsilon=FINAL_EPSILON,
        seed=42
    )

def load_existing_model(agent, model_path):
    """Load existing model weights if available"""
    if os.path.exists(f"{model_path}_local.pth") and os.path.exists(f"{model_path}_target.pth"):
        try:
            agent.load(model_path)
            print(f"✅ Loaded existing model from {model_path}")
            return True
        except Exception as e:
            print(f"⚠️ Failed to load existing model: {e}")
            return False
    return False

def save_model(agent, model_path):
    """Save the trained model"""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save both local and target networks
        torch.save(agent.qnetwork_local.state_dict(), f"{model_path}_local.pth")
        torch.save(agent.qnetwork_target.state_dict(), f"{model_path}_target.pth")
        print(f"💾 Model saved to {model_path}")
        return True
    except Exception as e:
        print(f"❌ Failed to save model: {e}")
        return False

def evaluate_agent(agent, env, n_episodes=20):
    """Evaluate the trained agent"""
    print(f"\n🔍 Evaluating agent over {n_episodes} episodes...")
    
    win_count = 0
    total_scores = []
    avg_plays_used = []
    avg_discards_used = []
    
    # Add progress tracking
    from tqdm import tqdm
    
    for episode in tqdm(range(n_episodes), desc="Evaluation"):
        obs, _ = env.reset()
        done = False
        plays_used = 0
        discards_used = 0
        step_count = 0
        max_steps = 50  # Prevent infinite loops
        
        while not done and step_count < max_steps:
            # Use epsilon=0 for evaluation (no exploration)
            action = agent.get_action(obs, eps=0.0)
            obs, reward, done, truncated, info = env.step(action)
            
            # Track usage
            if info.get("action_type") == "play":
                plays_used += 1
            elif info.get("action_type") == "discard":
                discards_used += 1
            
            step_count += 1
            
            if truncated:
                break
        
        if info.get("won", False):
            win_count += 1
        
        total_scores.append(info.get("total_score", 0))
        avg_plays_used.append(plays_used)
        avg_discards_used.append(discards_used)
    
    win_rate = win_count / n_episodes
    avg_score = np.mean(total_scores)
    avg_plays = np.mean(avg_plays_used)
    avg_discards = np.mean(avg_discards_used)
    
    print(f"📈 Evaluation Results:")
    print(f"   Win Rate: {win_rate:.2%} ({win_count}/{n_episodes})")
    print(f"   Average Score: {avg_score:.2f}")
    print(f"   Average Plays Used: {avg_plays:.2f}/3")
    print(f"   Average Discards Used: {avg_discards:.2f}/3")
    print(f"   Best Score: {max(total_scores):.2f}")
    
    return win_rate, avg_score

def plot_training_results(episode_rewards, win_rates, scores, save_path="training_results_v2.png"):
    """Plot training results"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Episode rewards
    ax1.plot(episode_rewards)
    ax1.set_title('Episode Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.grid(True)
    
    # Win rates (moving average)
    if len(win_rates) > 0:
        ax2.plot(win_rates)
        ax2.set_title('Win Rate (100-episode moving average)')
        ax2.set_xlabel('Episode (hundreds)')
        ax2.set_ylabel('Win Rate')
        ax2.grid(True)
    
    # Scores
    ax3.plot(scores)
    ax3.set_title('Episode Scores')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Score')
    ax3.grid(True)
    
    # Score distribution
    if len(scores) > 0:
        ax4.hist(scores, bins=30, alpha=0.7)
        ax4.set_title('Score Distribution')
        ax4.set_xlabel('Score')
        ax4.set_ylabel('Frequency')
        ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"📊 Training plots saved to {save_path}")

def main():
    """Main training function"""
    print("🎰 Balatro DQN Training V2")
    print("=" * 50)
    
    # Create environment and agent
    env = BalatroGymEnv(blind_score=300)
    agent = create_agent()
    
    # Model path
    model_path = "weights/balatro_v2_agent"
    
    # Try to load existing model
    load_existing_model(agent, model_path)
    
    # Initialize MLflow tracking
    mlflow_tracker = MLflowTracker()
    run_name = f"balatro_v2_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    mlflow_tracker.start_run(run_name)
    
    # Initialize training plotter and diagnostics
    plotter = create_plotter(mlflow_tracker)
    diagnostics = create_diagnostics()
    
    # Training metrics
    episode_rewards = []
    episode_scores = []
    win_rates = []
    best_win_rate = 0.0
    
    print(f"\n🚀 Starting training for {N_EPISODES} episodes...")
    print(f"📊 State size: {STATE_SIZE}, Action size: {ACTION_SIZE}")
    print(f"🎯 Target epsilon: {FINAL_EPSILON}")
    print(f"🏆 Blind score target: {env.blind_score}")
    
    # Training loop
    for episode in tqdm(range(N_EPISODES), desc="Training Episodes"):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        
        # Play one episode
        step_count = 0
        max_hand_score = 0
        hand_scores = []
        invalid_actions = 0
        
        while not done:
            # Get action from agent
            action = agent.get_action(obs)
            
            # Take action in environment
            next_obs, reward, done, truncated, info = env.step(action)
            
            # Track hand scores and invalid actions
            if info.get("action_type") == "play":
                score_gained = info.get("score_gained", 0)
                hand_scores.append(score_gained)
                max_hand_score = max(max_hand_score, score_gained)
            
            if info.get("error"):
                invalid_actions += 1
            
            # Update agent
            agent.step(obs, action, float(reward), next_obs, done)
            
            # Update episode tracking
            episode_reward += reward
            obs = next_obs
            step_count += 1
            
            if truncated:
                break
        
        # Decay epsilon
        agent.decay_epsilon()
        
        # Record metrics
        episode_rewards.append(episode_reward)
        episode_scores.append(info.get("total_score", 0))
        
        # Determine game end reason
        game_end_reason = "won" if info.get("won", False) else "unknown"
        if not info.get("won", False):
            if info.get("plays_left", 0) <= 0:
                game_end_reason = "no_plays_left"
            elif info.get("total_score", 0) < env.blind_score:
                game_end_reason = "low_score"
        
        # Add data to plotter
        plot_data = {
            'episode_rewards': episode_reward,
            'episode_scores': info.get("total_score", 0),
            'epsilon_values': agent.epsilon,
            'buffer_sizes': len(agent.memory),
            'plays_used': 3 - info.get("plays_left", 0),
            'discards_used': 3 - info.get("discards_left", 0)
        }
        plotter.add_data_point(episode, plot_data)
        
        # Add detailed data to diagnostics
        diagnostic_data = {
            'episode_rewards': episode_reward,
            'episode_scores': info.get("total_score", 0),
            'won': info.get("won", False),
            'plays_used': 3 - info.get("plays_left", 0),
            'discards_used': 3 - info.get("discards_left", 0),
            'epsilon_values': agent.epsilon,
            'steps_taken': step_count,
            'final_hand_size': len(env.hand),
            'max_hand_score': max_hand_score,
            'avg_hand_score': np.mean(hand_scores) if hand_scores else 0,
            'invalid_actions': invalid_actions,
            'game_end_reason': game_end_reason
        }
        diagnostics.add_episode_data(episode, diagnostic_data)
        
        # Log to MLflow
        mlflow_tracker.log_custom_metric('episode_reward', episode_reward, step=episode)
        mlflow_tracker.log_custom_metric('episode_score', info.get("total_score", 0), step=episode)
        mlflow_tracker.log_custom_metric('won', int(info.get("won", False)), step=episode)
        mlflow_tracker.log_custom_metric('plays_used', 3 - info.get("plays_left", 0), step=episode)
        mlflow_tracker.log_custom_metric('discards_used', 3 - info.get("discards_left", 0), step=episode)
        mlflow_tracker.log_custom_metric('epsilon', agent.epsilon, step=episode)
        mlflow_tracker.log_custom_metric('buffer_size', len(agent.memory), step=episode)
        
        # Evaluate and print progress every 100 episodes
        if (episode + 1) % 100 == 0:
            recent_rewards = np.mean(episode_rewards[-100:])
            recent_scores = np.mean(episode_scores[-100:])
            recent_wins = sum(1 for i in range(-100, 0) if episode + i >= 0 and 
                            episode_scores[episode + i] >= env.blind_score) / 100
            
            win_rates.append(recent_wins)
            
            # Add win rate to plotter
            plotter.plot_data['win_rates'].append(recent_wins)
            
            print(f"\n📊 Episode {episode + 1}/{N_EPISODES}")
            print(f"   Recent 100-episode average reward: {recent_rewards:.2f}")
            print(f"   Recent 100-episode average score: {recent_scores:.2f}")
            print(f"   Recent 100-episode win rate: {recent_wins:.2%}")
            print(f"   Current epsilon: {agent.epsilon:.3f}")
            print(f"   Buffer size: {len(agent.memory)}")
            
            # Generate and save training plots every 500 episodes
            if (episode + 1) % 500 == 0:
                plotter.plot_training_progress(episode + 1)
                print(f"📊 Training plots generated and saved to MLflow")
            
            # Run diagnostics every 1000 episodes
            if (episode + 1) % 1000 == 0:
                print(f"\n🔍 Running training diagnostics...")
                diagnostics.print_diagnostic_report(episode + 1)
                diagnostic_plot_path = diagnostics.plot_diagnostic_summary(episode + 1)
                if diagnostic_plot_path and mlflow_tracker:
                    try:
                        mlflow_tracker.log_artifact(diagnostic_plot_path, "diagnostics")
                        print(f"📊 Diagnostic plot saved to MLflow")
                    except Exception as e:
                        print(f"⚠️ Failed to log diagnostic plot to MLflow: {e}")
            
            # Save model if we have a new best win rate
            if recent_wins > best_win_rate and episode >= 1000:  # Wait for some training
                best_win_rate = recent_wins
                save_model(agent, f"{model_path}_best")
                print(f"🏆 New best win rate! Saving model...")
        
        # Save model periodically
        if (episode + 1) % 2000 == 0:
            save_model(agent, f"{model_path}_checkpoint_{episode + 1}")
    
    # Save final model
    save_model(agent, model_path)
    
    # Generate final training plots
    print(f"\n📊 Generating final training plots...")
    plotter.plot_training_progress(N_EPISODES)
    plotter.plot_final_summary(N_EPISODES)
    
    # Also save the original plot for compatibility
    plot_training_results(
        episode_rewards, 
        win_rates, 
        episode_scores,
        "training_results_v2.png"
    )
    
    # Evaluate the trained agent (quick evaluation)
    print(f"\n⚡ Running quick evaluation...")
    win_rate, avg_score = evaluate_agent(agent, env, n_episodes=20)
    
    # Log final metrics
    mlflow_tracker.log_custom_metric('final_win_rate', win_rate)
    mlflow_tracker.log_custom_metric('final_avg_score', avg_score)
    mlflow_tracker.log_custom_metric('total_episodes', N_EPISODES)
    
    # End MLflow run
    mlflow_tracker.end_run()
    
    # Get final training statistics
    final_stats = plotter.get_training_stats()
    
    print(f"\n✅ Training completed!")
    print(f"🏆 Final win rate: {win_rate:.2%}")
    print(f"📈 Final average score: {avg_score:.2f}")
    print(f"💾 Model saved to: {model_path}")
    
    print(f"\n📊 Final Training Statistics:")
    print(f"   Total episodes: {final_stats.get('total_episodes', 0)}")
    print(f"   Average reward: {final_stats.get('avg_reward', 0):.2f}")
    print(f"   Best reward: {final_stats.get('max_reward', 0):.2f}")
    print(f"   Average score: {final_stats.get('avg_score', 0):.2f}")
    print(f"   Best score: {final_stats.get('max_score', 0):.2f}")
    print(f"   Total wins: {final_stats.get('wins', 0)}")
    print(f"   Overall win rate: {final_stats.get('win_rate', 0):.2%}")
    print(f"   Final epsilon: {final_stats.get('current_epsilon', 0):.3f}")
    
    # Test the agent with a few example games
    print(f"\n🎮 Testing agent with example games...")
    test_agent_behavior(agent, env)

def test_agent_behavior(agent, env, n_tests=3):
    """Test the agent's behavior with some example games"""
    for test in range(n_tests):
        print(f"\n🎯 Test Game {test + 1}")
        print("=" * 30)
        
        obs, _ = env.reset()
        step = 0
        max_steps = 15  # Reduced to prevent long runs
        
        while not env.game_over and step < max_steps:
            # Get agent's action
            action = agent.get_action(obs, eps=0.0)  # No exploration
            
            # Decode action
            action_type, card_indices = env._decode_action(action)
            selected_cards = [str(env.hand[i]) for i in card_indices if i < len(env.hand)]
            
            print(f"🤖 Step {step + 1}:")
            print(f"   Action: {action} -> {action_type}")
            print(f"   Selected cards: {selected_cards}")
            
            # Take the action
            obs, reward, done, truncated, info = env.step(action)
            
            print(f"   Reward: {reward:.2f}")
            if info.get("action_type") == "play":
                print(f"   Hand type: {info.get('hand_type', 'Unknown')}")
                print(f"   Score gained: {info.get('score_gained', 0)}")
            print(f"   Total score: {info.get('total_score', 0)}")
            print(f"   Plays left: {info.get('plays_left', 0)}")
            print(f"   Discards left: {info.get('discards_left', 0)}")
            print()
            
            step += 1
            
            if done or truncated:
                break
        
        env.render()
        result = "WON" if env.won else "LOST"
        print(f"🏁 Game {test + 1} Result: {result}")
        print(f"   Final Score: {env.current_score}")
        print(f"   Steps taken: {step}")
        print("-" * 50)

if __name__ == "__main__":
    main() 