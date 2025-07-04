#!/usr/bin/env python3
"""
Detailed Model Evaluation Script
Shows deck, hand, actions, scores, and requirements for 5 game runs
"""

import os
import sys
import numpy as np
import torch
from typing import List, Dict, Any
import random

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from balatro_gym_v2_simple import BalatroGymEnvSimple
from train_balatro_v2_simple import SimpleDQNAgent

def format_card(card_str: str) -> str:
    """Format card for better display"""
    rank_map = {
        'A': 'A', 'K': 'K', 'Q': 'Q', 'J': 'J', '10': '10',
        '9': '9', '8': '8', '7': '7', '6': '6', '5': '5', '4': '4', '3': '3', '2': '2'
    }
    suit_map = {'H': '♥', 'D': '♦', 'C': '♣', 'S': '♠'}
    
    if len(card_str) >= 2:
        rank = card_str[:-1]
        suit = card_str[-1]
        return f"{rank_map.get(rank, rank)}{suit_map.get(suit, suit)}"
    return card_str

def format_hand(hand: List[str]) -> str:
    """Format hand for display"""
    return " ".join([format_card(card) for card in hand])

def get_hand_type_and_score(cards: List[str]) -> tuple:
    """Get hand type and score for a set of cards"""
    from balatro_gym_v2_simple import BalatroCard, BalatroHand
    
    balatro_cards = [BalatroCard(card[:-1], card[-1]) for card in cards]
    balatro_hand = BalatroHand(balatro_cards)
    hand_type, base_chips, multiplier, card_chips = balatro_hand.evaluate_hand()
    total_score = (base_chips + card_chips) * multiplier
    
    return hand_type, total_score

def evaluate_model_detailed(model_path: str = "weights/best_simple_model.pth", num_runs: int = 5):
    """Evaluate model with detailed game information"""
    
    print("🎰 Detailed Model Evaluation")
    print("=" * 60)
    
    # Setup environment and agent
    env = BalatroGymEnvSimple(blind_score=300)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = SimpleDQNAgent(state_size, action_size, device)
    
    # Load model
    if os.path.exists(model_path):
        try:
            agent.load(model_path)
            print(f"✅ Loaded model from: {model_path}")
            print(f"   Epsilon: {agent.epsilon:.3f}")
            print(f"   Steps: {agent.steps}")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return
    else:
        print(f"❌ Model not found at: {model_path}")
        return
    
    print(f"\n🎯 Target Score: {env.blind_score}")
    print(f"📊 Number of Runs: {num_runs}")
    print("=" * 60)
    
    # Run evaluations
    for run in range(num_runs):
        print(f"\n🎮 RUN {run + 1}")
        print("-" * 40)
        
        obs, _ = env.reset()
        total_reward = 0
        step_count = 0
        actions_taken = []
        
        # Show initial hand
        print(f"📋 Initial Hand: {format_hand([str(card) for card in env.hand])}")
        print(f"🎯 Target Score: {env.blind_score}")
        print(f"🃏 Cards in Deck: {len(env.deck)}")
        print()
        
        while not env.game_over and step_count < 20:  # Limit steps to prevent infinite loops
            step_count += 1
            
            # Show current state
            print(f"Step {step_count}:")
            print(f"   Current Score: {env.current_score}")
            print(f"   Plays Left: {env.plays_left}")
            print(f"   Discards Left: {env.discards_left}")
            print(f"   Hand: {format_hand([str(card) for card in env.hand])}")
            
            # Get agent action
            action = agent.act(obs, training=False)
            action_type, card_indices = env._decode_action(action)
            
            # Show selected cards
            selected_cards = [str(env.hand[i]) for i in card_indices if i < len(env.hand)]
            print(f"   Action: {action_type.upper()}")
            print(f"   Selected: {format_hand(selected_cards)}")
            
            # Take action
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            
            # Show result
            if action_type == "play":
                hand_type, score_gained = get_hand_type_and_score(selected_cards)
                print(f"   Hand Type: {hand_type}")
                print(f"   Score Gained: {score_gained}")
                print(f"   Total Score: {env.current_score}")
            else:
                print(f"   Cards Discarded: {len(selected_cards)}")
            
            print(f"   Reward: {reward:.2f}")
            print()
            
            # Store action info
            actions_taken.append({
                'step': step_count,
                'action_type': action_type,
                'cards': selected_cards,
                'score_gained': info.get('score_gained', 0) if action_type == "play" else 0,
                'hand_type': info.get('hand_type', 'N/A') if action_type == "play" else 'N/A',
                'reward': reward
            })
            
            if done:
                break
        
        # Show final results
        print(f"🏁 RUN {run + 1} RESULTS:")
        print(f"   Final Score: {env.current_score}")
        print(f"   Target Score: {env.blind_score}")
        print(f"   Result: {'✅ WON' if env.won else '❌ LOST'}")
        print(f"   Total Reward: {total_reward:.2f}")
        print(f"   Steps Taken: {step_count}")
        
        # Show action summary
        print(f"\n📋 Action Summary:")
        for action in actions_taken:
            if action['action_type'] == "play":
                print(f"   Step {action['step']}: PLAY {format_hand(action['cards'])} → {action['hand_type']} (+{action['score_gained']})")
            else:
                print(f"   Step {action['step']}: DISCARD {format_hand(action['cards'])}")
        
        print("-" * 40)
    
    # Overall statistics
    print(f"\n📊 OVERALL STATISTICS")
    print("=" * 40)
    
    # Run a quick evaluation for statistics
    wins = 0
    total_scores = []
    total_rewards = []
    
    for _ in range(100):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.act(obs, training=False)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
        
        if info.get('won', False):
            wins += 1
        total_scores.append(info.get('total_score', 0))
        total_rewards.append(episode_reward)
    
    win_rate = wins / 100
    avg_score = np.mean(total_scores)
    avg_reward = np.mean(total_rewards)
    
    print(f"Win Rate: {win_rate:.1%}")
    print(f"Average Score: {avg_score:.1f}")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Score Range: {min(total_scores)} - {max(total_scores)}")
    print(f"Best Score: {max(total_scores)}")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Balatro DQN model with detailed game information")
    parser.add_argument("--model", type=str, default="weights/best_simple_model.pth", 
                       help="Path to model file")
    parser.add_argument("--runs", type=int, default=5, 
                       help="Number of detailed runs to show")
    
    args = parser.parse_args()
    
    evaluate_model_detailed(args.model, args.runs)

if __name__ == "__main__":
    main() 