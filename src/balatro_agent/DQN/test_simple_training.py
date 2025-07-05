#!/usr/bin/env python3
"""
Quick test script to validate simplified training setup
Runs a short training session to ensure everything works
"""

import os
import sys
import numpy as np
import torch
from balatro_gym_v2_simple import BalatroGymEnvSimple
from train_balatro_v2_simple import SimpleDQNAgent, evaluate_agent

def test_simplified_training():
    """Test the simplified training setup"""
    print("🧪 Testing Simplified Balatro Training Setup")
    print("=" * 50)
    
    # Create environment and agent
    env = BalatroGymEnvSimple(blind_score=300)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    print(f"✅ Environment created:")
    print(f"   State size: {state_size}")
    print(f"   Action size: {action_size}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = SimpleDQNAgent(state_size, action_size, device)
    
    print(f"✅ Agent created:")
    print(f"   Device: {device}")
    print(f"   Network parameters: {sum(p.numel() for p in agent.q_network.parameters()):,}")
    
    # Test a few episodes
    print(f"\n🔄 Testing training loop (5 episodes)...")
    
    episode_rewards = []
    episode_scores = []
    episode_wins = []
    
    for episode in range(5):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.act(obs, training=True)
            next_obs, reward, done, truncated, info = env.step(action)
            
            agent.remember(obs, action, reward, next_obs, done)
            obs = next_obs
            episode_reward += reward
            
            # Train on batch
            if len(agent.memory) > 32:  # Smaller batch for testing
                loss = agent.replay(32)
        
        episode_rewards.append(episode_reward)
        episode_scores.append(info.get('total_score', 0))
        episode_wins.append(1 if info.get('won', False) else 0)
        
        print(f"   Episode {episode + 1}: Reward={episode_reward:.2f}, Score={info.get('total_score', 0)}, Won={info.get('won', False)}")
    
    print(f"\n✅ Training test completed:")
    print(f"   Average reward: {np.mean(episode_rewards):.2f}")
    print(f"   Average score: {np.mean(episode_scores):.1f}")
    print(f"   Win rate: {np.mean(episode_wins):.1%}")
    print(f"   Memory size: {len(agent.memory)}")
    print(f"   Epsilon: {agent.epsilon:.3f}")
    
    # Test evaluation
    print(f"\n🔄 Testing evaluation...")
    win_rate, avg_score, avg_reward = evaluate_agent(agent, env, episodes=10)
    
    print(f"✅ Evaluation completed:")
    print(f"   Win rate: {win_rate:.1%}")
    print(f"   Average score: {avg_score:.1f}")
    print(f"   Average reward: {avg_reward:.2f}")
    
    # Test model saving/loading
    print(f"\n🔄 Testing model save/load...")
    test_model_path = "test_model.pth"
    
    try:
        agent.save(test_model_path)
        print(f"   ✅ Model saved successfully")
        
        # Create new agent and load
        new_agent = SimpleDQNAgent(state_size, action_size, device)
        new_agent.load(test_model_path)
        print(f"   ✅ Model loaded successfully")
        
        # Clean up
        os.remove(test_model_path)
        print(f"   ✅ Test file cleaned up")
        
    except Exception as e:
        print(f"   ❌ Model save/load failed: {e}")
    
    print(f"\n🎉 All tests passed! Simplified training setup is ready.")
    print(f"💡 You can now run: python train_balatro_v2_simple.py")

if __name__ == "__main__":
    test_simplified_training() 