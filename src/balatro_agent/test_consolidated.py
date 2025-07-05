#!/usr/bin/env python3
"""
Test script for consolidated PPO training and evaluation
"""

import torch
import numpy as np
from train_ppo import ComprehensivePPOTrainer
from test_ppo import PPOEvaluator

def test_training():
    """Test the consolidated training system"""
    print("🧪 Testing Consolidated PPO Training System")
    print("=" * 50)
    
    # Create trainer with small parameters for quick test
    trainer = ComprehensivePPOTrainer(
        blind_score=300,
        learning_rate=3e-4,
        clip_ratio=0.2,
        hidden_dim=128,  # Smaller network for faster testing
        device="cpu"  # Use CPU for consistent testing
    )
    
    print("✅ Trainer created successfully")
    
    # Test a small training run
    print("\n🚀 Running small training test (10,000 timesteps)...")
    agent = trainer.train(
        total_timesteps=10000,
        batch_size=512,  # Smaller batch for faster testing
        eval_interval=2500,
        demo_interval=5000,
        debug_interval=1000,
        num_eval_episodes=5
    )
    
    print("✅ Training completed successfully")
    
    # Test evaluation
    print("\n📊 Testing evaluation...")
    eval_stats = trainer.evaluate(num_episodes=10)
    print(f"Evaluation results: {eval_stats}")
    print("✅ Evaluation completed successfully")
    
    return agent

def test_evaluation():
    """Test the evaluation system with a trained model"""
    print("\n🎯 Testing PPO Evaluation System")
    print("=" * 50)
    
    # Try to load the final model
    try:
        # Create environment to get the same hidden_dim as training
        from balatro_gym_v2_simple import BalatroGymEnvSimple
        env = BalatroGymEnvSimple(blind_score=300)
        
        evaluator = PPOEvaluator(
            model_path="ppo_balatro_final.pth",
            blind_score=300,
            device="cpu",
            hidden_dim=128  # Match the training hidden_dim
        )
        print("✅ Evaluator created successfully")
        
        # Test evaluation
        print("\n📊 Running evaluation...")
        eval_stats = evaluator.evaluate(num_episodes=20)
        print(f"Win rate: {eval_stats['win_rate']:.2%}")
        print(f"Average reward: {eval_stats['avg_reward']:.2f}")
        print(f"Action distribution: {eval_stats['action_distribution']}")
        print("✅ Evaluation completed successfully")
        
        # Test demo
        print("\n🎮 Running demo...")
        evaluator.demo_episodes(num_episodes=1, max_steps=5)
        print("✅ Demo completed successfully")
        
        # Test policy analysis
        print("\n🔍 Running policy analysis...")
        evaluator.analyze_policy(num_samples=100)
        print("✅ Policy analysis completed successfully")
        
    except FileNotFoundError:
        print("⚠️  No trained model found, skipping evaluation test")
        print("   Run training first to test evaluation")

def test_environment():
    """Test the environment works correctly"""
    print("\n🎰 Testing Balatro Environment")
    print("=" * 50)
    
    from balatro_gym_v2_simple import BalatroGymEnvSimple
    
    env = BalatroGymEnvSimple(blind_score=300)
    print("✅ Environment created successfully")
    
    # Test reset
    obs, info = env.reset()
    print(f"✅ Reset successful, observation shape: {obs.shape}")
    print(f"Initial hand: {[str(card) for card in env.hand]}")
    
    # Test step
    action = 0  # First action
    obs, reward, done, truncated, info = env.step(action)
    print(f"✅ Step successful, reward: {reward}, done: {done}")
    
    print("✅ Environment test completed")

def main():
    """Run all tests"""
    print("🧪 Running Comprehensive PPO System Tests")
    print("=" * 60)
    
    # Test environment
    test_environment()
    
    # Test training
    try:
        agent = test_training()
        print("\n🎉 All training tests passed!")
    except Exception as e:
        print(f"\n❌ Training test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test evaluation
    try:
        test_evaluation()
        print("\n🎉 All evaluation tests passed!")
    except Exception as e:
        print(f"\n❌ Evaluation test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🏁 All tests completed!")

if __name__ == "__main__":
    main() 