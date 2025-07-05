#!/usr/bin/env python3
"""
Standalone script to load the best model from MLflow and continue training
This demonstrates how to use the new model loading functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.dqn_agent import DQNAgent
from mlflow_tracker import MLflowTracker
from config.settings import *

def main():
    print("🚀 Balatro DQN Model Loader")
    print("=" * 50)
    
    # Initialize tracker
    tracker = MLflowTracker()
    
    # Initialize agent
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
    
    print(f"🤖 Agent initialized with epsilon: {agent.epsilon:.4f}")
    
    # List available models
    print("\n📋 Checking available models...")
    models_info = tracker.list_available_models()
    
    if "error" in models_info:
        print(f"❌ Error: {models_info['error']}")
        return
    
    available_models = models_info.get("available_models", [])
    
    if not available_models:
        print("❌ No trained models found in MLflow.")
        print("💡 Train the agent first using the API to create models.")
        return
    
    print(f"✅ Found {len(available_models)} runs with trained models")
    
    # Show best models info
    print("\n🏆 Best Models Summary:")
    best_info = tracker.get_best_models_info()
    
    if "error" not in best_info:
        best_episode_runs = best_info.get("best_episode_runs", [])
        best_average_runs = best_info.get("best_average_runs", [])
        
        if best_episode_runs:
            best_episode = best_episode_runs[0]
            print(f"🥇 Best Episode Reward: {best_episode.get('metrics.best_episode_reward', 'N/A'):.2f}")
            print(f"   Run: {best_episode.get('tags.mlflow.runName', 'Unknown')}")
        
        if best_average_runs:
            best_average = best_average_runs[0]
            print(f"📈 Best 10-Episode Average: {best_average.get('metrics.best_10_episode_average', 'N/A'):.2f}")
            print(f"   Run: {best_average.get('tags.mlflow.runName', 'Unknown')}")
    
    # Interactive model loading
    print("\n🔄 Load Best Model Options:")
    print("1. Load best single episode model")
    print("2. Load best 10-episode average model")
    print("3. List all available models")
    print("4. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == "1":
                result = tracker.load_best_model_from_mlflow(agent, "episode")
                if result["success"]:
                    print(f"✅ Successfully loaded best episode model!")
                    print(f"📊 Performance: {result['performance']:.2f}")
                    print(f"🚀 Ready to continue training from this checkpoint")
                    
                    # Ask about epsilon reset
                    reset_eps = input("\nReset epsilon for more exploration? (y/n): ").strip().lower()
                    if reset_eps == 'y':
                        agent.epsilon = agent.initial_epsilon
                        print(f"🔄 Epsilon reset to: {agent.epsilon:.4f}")
                    
                    break
                else:
                    print(f"❌ Failed to load model: {result['error']}")
            
            elif choice == "2":
                result = tracker.load_best_model_from_mlflow(agent, "average")
                if result["success"]:
                    print(f"✅ Successfully loaded best average model!")
                    print(f"📊 Performance: {result['performance']:.2f}")
                    print(f"🚀 Ready to continue training from this checkpoint")
                    
                    # Ask about epsilon reset
                    reset_eps = input("\nReset epsilon for more exploration? (y/n): ").strip().lower()
                    if reset_eps == 'y':
                        agent.epsilon = agent.initial_epsilon
                        print(f"🔄 Epsilon reset to: {agent.epsilon:.4f}")
                    
                    break
                else:
                    print(f"❌ Failed to load model: {result['error']}")
            
            elif choice == "3":
                print("\n📋 All Available Models:")
                for i, model_run in enumerate(available_models):
                    print(f"\n{i+1}. Run: {model_run['run_name']}")
                    print(f"   Start Time: {model_run['start_time']}")
                    print(f"   Episodes: {model_run.get('episode_count', 'Unknown')}")
                    print(f"   Models:")
                    for model in model_run['models']:
                        print(f"     - {model['type']}: {model['performance']:.2f}")
            
            elif choice == "4":
                print("👋 Exiting...")
                return
            
            else:
                print("❌ Invalid choice. Please enter 1-4.")
        
        except KeyboardInterrupt:
            print("\n👋 Exiting...")
            return
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n🎯 Next Steps:")
    print("1. Start the API server: python main.py")
    print("2. The loaded model will be used for predictions and training")
    print("3. Training will continue from the loaded checkpoint")
    print("4. New best models will be saved automatically")

if __name__ == "__main__":
    main() 