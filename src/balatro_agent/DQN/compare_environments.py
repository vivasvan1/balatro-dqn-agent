#!/usr/bin/env python3
"""
Compare Original vs Simplified Balatro Environments
Shows the differences in state space, performance, and training characteristics
"""

import time
import numpy as np
from balatro_gym_v2 import BalatroGymEnv
from balatro_gym_v2_simple import BalatroGymEnvSimple

def compare_environments():
    """Compare the two environments"""
    print("🎰 Balatro Environment Comparison")
    print("=" * 50)
    
    # Create environments
    env_original = BalatroGymEnv(blind_score=300)
    env_simple = BalatroGymEnvSimple(blind_score=300)
    
    print(f"📊 State Space Comparison:")
    print(f"   Original: {env_original.observation_space.shape[0]} dimensions")
    print(f"   Simplified: {env_simple.observation_space.shape[0]} dimensions")
    print(f"   Reduction: {env_original.observation_space.shape[0] - env_simple.observation_space.shape[0]} dimensions ({((env_original.observation_space.shape[0] - env_simple.observation_space.shape[0]) / env_original.observation_space.shape[0] * 100):.1f}%)")
    
    print(f"\n🎯 Action Space:")
    print(f"   Original: {env_original.action_space.n} actions")
    print(f"   Simplified: {env_simple.action_space.n} actions")
    
    print(f"\n⚡ Performance Test (100 episodes each):")
    
    # Test original environment
    print(f"\n🔄 Testing Original Environment...")
    start_time = time.time()
    total_reward_orig = 0
    wins_orig = 0
    scores_orig = []
    
    for i in range(100):
        obs, _ = env_original.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = env_original.action_space.sample()
            obs, reward, done, truncated, info = env_original.step(action)
            episode_reward += reward
        
        total_reward_orig += episode_reward
        if info.get('won', False):
            wins_orig += 1
        scores_orig.append(info.get('total_score', 0))
    
    orig_time = time.time() - start_time
    orig_win_rate = wins_orig / 100
    orig_avg_score = np.mean(scores_orig)
    orig_avg_reward = total_reward_orig / 100
    
    # Test simplified environment
    print(f"🔄 Testing Simplified Environment...")
    start_time = time.time()
    total_reward_simple = 0
    wins_simple = 0
    scores_simple = []
    
    for i in range(100):
        obs, _ = env_simple.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = env_simple.action_space.sample()
            obs, reward, done, truncated, info = env_simple.step(action)
            episode_reward += reward
        
        total_reward_simple += episode_reward
        if info.get('won', False):
            wins_simple += 1
        scores_simple.append(info.get('total_score', 0))
    
    simple_time = time.time() - start_time
    simple_win_rate = wins_simple / 100
    simple_avg_score = np.mean(scores_simple)
    simple_avg_reward = total_reward_simple / 100
    
    print(f"\n📈 Performance Results:")
    print(f"   Original Environment:")
    print(f"     Time: {orig_time:.2f}s")
    print(f"     Win Rate: {orig_win_rate:.1%}")
    print(f"     Avg Score: {orig_avg_score:.1f}")
    print(f"     Avg Reward: {orig_avg_reward:.2f}")
    
    print(f"   Simplified Environment:")
    print(f"     Time: {simple_time:.2f}s")
    print(f"     Win Rate: {simple_win_rate:.1%}")
    print(f"     Avg Score: {simple_avg_score:.1f}")
    print(f"     Avg Reward: {simple_avg_reward:.2f}")
    
    speedup = orig_time / simple_time
    print(f"\n🚀 Speedup: {speedup:.1f}x faster")
    
    print(f"\n💡 Key Differences:")
    print(f"   ✅ Simplified environment removes:")
    print(f"      - Discarded cards tracking (208 dimensions)")
    print(f"      - Deck cards tracking (52 dimensions)")
    print(f"      - Complex card encoding")
    print(f"   ✅ Simplified environment adds:")
    print(f"      - Progress to target (1 dimension)")
    print(f"      - Hand quality score (1 dimension)")
    print(f"      - Better reward shaping")
    print(f"      - Improved penalties for invalid actions")
    
    print(f"\n🎯 Training Recommendations:")
    print(f"   1. Use simplified environment for faster training")
    print(f"   2. Reduced state space should improve learning stability")
    print(f"   3. Better reward shaping should lead to better performance")
    print(f"   4. Start with simplified, then graduate to original if needed")

if __name__ == "__main__":
    compare_environments() 