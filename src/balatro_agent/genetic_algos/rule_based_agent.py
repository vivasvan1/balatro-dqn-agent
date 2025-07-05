#!/usr/bin/env python3
"""
Rule-Based Agent for Balatro
Uses simple heuristics to make decisions - can serve as baseline or be evolved
"""

import os
import sys
import numpy as np
import random
from typing import List, Tuple, Dict, Any
import matplotlib.pyplot as plt
from datetime import datetime
import time

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from balatro_gym_v2_simple import BalatroGymEnvSimple

class RuleBasedAgent:
    """Rule-based agent that uses heuristics to play Balatro"""
    
    def __init__(self, state_size: int, action_size: int):
        self.state_size = state_size
        self.action_size = action_size
        
        # Rule parameters (can be evolved)
        self.high_card_threshold = 10  # Don't play cards below this value
        self.discard_threshold = 7     # Discard cards below this value
        self.score_threshold = 0.5     # Play aggressively if score > threshold * blind_score
        self.pass_threshold = 0.3      # Pass if score < threshold * blind_score
        
        # Hand type preferences (higher = more preferred)
        self.hand_preferences = {
            "Royal Flush": 100,
            "Straight Flush": 90,
            "Four of a Kind": 80,
            "Full House": 70,
            "Flush": 60,
            "Straight": 50,
            "Three of a Kind": 40,
            "Two Pair": 20,
            "Pair": 10,
            "High Card": 0
        }
    
    def decode_state(self, state: np.ndarray) -> Dict[str, Any]:
        """Decode the state into meaningful information"""
        # State format: [8 card ranks, plays_left, discards_left, current_score, blind_score, game_over, best_hand_value, worst_value, second_worst_value, can_play_high]
        
        card_ranks = state[:8].astype(int)
        plays_left = int(state[8])
        discards_left = int(state[9])
        current_score = state[10]
        blind_score = state[11]
        game_over = bool(state[12])
        best_hand_value = state[13] * 100  # Denormalize
        worst_value = int(state[14])
        second_worst_value = int(state[15])
        can_play_high = bool(state[16])
        
        # Calculate progress
        progress = current_score / blind_score if blind_score > 0 else 0
        
        # Find high cards (10+)
        high_cards = [i for i, rank in enumerate(card_ranks) if rank >= 10 and rank > 0]
        
        # Find low cards (2-6)
        low_cards = [i for i, rank in enumerate(card_ranks) if 2 <= rank <= 6 and rank > 0]
        
        return {
            'card_ranks': card_ranks,
            'plays_left': plays_left,
            'discards_left': discards_left,
            'current_score': current_score,
            'blind_score': blind_score,
            'progress': progress,
            'best_hand_value': best_hand_value,
            'worst_value': worst_value,
            'second_worst_value': second_worst_value,
            'can_play_high': can_play_high,
            'high_cards': high_cards,
            'low_cards': low_cards,
            'game_over': game_over
        }
    
    def act(self, state: np.ndarray) -> int:
        """Choose action based on rules"""
        info = self.decode_state(state)
        
        # If game is over, pass
        if info['game_over']:
            return 10  # Pass
        
        # If we can play a high-value hand, do it
        if info['can_play_high'] and info['plays_left'] > 0:
            # Find the best card to play
            best_card = self.find_best_card_to_play(info)
            if best_card is not None:
                return best_card
        
        # If we're doing well and have plays left, play a good card
        if info['progress'] > self.score_threshold and info['plays_left'] > 0:
            good_card = self.find_good_card_to_play(info)
            if good_card is not None:
                return good_card
        
        # If we're doing poorly and have discards left, discard bad cards
        if info['progress'] < self.pass_threshold and info['discards_left'] > 0:
            if info['worst_value'] <= self.discard_threshold:
                return 8  # Discard worst card
            elif info['second_worst_value'] <= self.discard_threshold:
                return 9  # Discard second worst card
        
        # If we have high cards and plays left, play them
        if info['high_cards'] and info['plays_left'] > 0:
            return info['high_cards'][0]  # Play first high card
        
        # If we have discards left and low cards, discard them
        if info['discards_left'] > 0 and info['low_cards']:
            return 8  # Discard worst card
        
        # Default: pass
        return 10
    
    def find_best_card_to_play(self, info: Dict[str, Any]) -> int:
        """Find the best card to play when we can make a high-value hand"""
        # Play the highest value card
        max_rank = 0
        best_card = None
        
        for i, rank in enumerate(info['card_ranks']):
            if rank > max_rank and rank > 0:
                max_rank = rank
                best_card = i
        
        return best_card
    
    def find_good_card_to_play(self, info: Dict[str, Any]) -> int:
        """Find a good card to play when we're doing well"""
        # Play cards above threshold
        for i, rank in enumerate(info['card_ranks']):
            if rank >= self.high_card_threshold and rank > 0:
                return i
        
        return None
    
    def set_parameters(self, **kwargs):
        """Update rule parameters"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get current rule parameters"""
        return {
            'high_card_threshold': self.high_card_threshold,
            'discard_threshold': self.discard_threshold,
            'score_threshold': self.score_threshold,
            'pass_threshold': self.pass_threshold
        }

class EvolvableRuleAgent(RuleBasedAgent):
    """Rule-based agent with evolvable parameters"""
    
    def __init__(self, state_size: int, action_size: int):
        super().__init__(state_size, action_size)
        self.fitness = 0.0
    
    def mutate(self, mutation_strength: float = 0.1):
        """Mutate the rule parameters"""
        self.high_card_threshold += random.uniform(-mutation_strength * 2, mutation_strength * 2)
        self.discard_threshold += random.uniform(-mutation_strength, mutation_strength)
        self.score_threshold += random.uniform(-mutation_strength, mutation_strength)
        self.pass_threshold += random.uniform(-mutation_strength, mutation_strength)
        
        # Keep parameters in reasonable ranges
        self.high_card_threshold = max(2, min(14, self.high_card_threshold))
        self.discard_threshold = max(2, min(10, self.discard_threshold))
        self.score_threshold = max(0.1, min(0.9, self.score_threshold))
        self.pass_threshold = max(0.1, min(0.5, self.pass_threshold))
    
    def crossover(self, other: 'EvolvableRuleAgent') -> 'EvolvableRuleAgent':
        """Create a child by crossing over parameters"""
        child = EvolvableRuleAgent(self.state_size, self.action_size)
        
        # Crossover parameters
        if random.random() < 0.5:
            child.high_card_threshold = self.high_card_threshold
        else:
            child.high_card_threshold = other.high_card_threshold
        
        if random.random() < 0.5:
            child.discard_threshold = self.discard_threshold
        else:
            child.discard_threshold = other.discard_threshold
        
        if random.random() < 0.5:
            child.score_threshold = self.score_threshold
        else:
            child.score_threshold = other.score_threshold
        
        if random.random() < 0.5:
            child.pass_threshold = self.pass_threshold
        else:
            child.pass_threshold = other.pass_threshold
        
        return child

def evaluate_rule_agent(agent: RuleBasedAgent, episodes: int = 10) -> Tuple[float, float]:
    """Evaluate a rule-based agent"""
    env = BalatroGymEnvSimple()
    total_fitness = 0.0
    wins = 0
    
    for episode in range(episodes):
        obs, _ = env.reset()
        episode_fitness = 0.0
        done = False
        step_count = 0
        
        while not done and step_count < 50:
            action = agent.act(obs)
            obs, reward, done, truncated, info = env.step(action)
            episode_fitness += reward
            step_count += 1
        
        if info.get('won', False):
            wins += 1
            episode_fitness += 100.0  # Bonus for winning
        
        total_fitness += episode_fitness
    
    avg_fitness = total_fitness / episodes
    win_rate = wins / episodes
    
    return avg_fitness, win_rate

def train_rule_agent():
    """Train an evolvable rule-based agent"""
    print("🧬 Training Evolvable Rule-Based Agent for Balatro")
    print("=" * 50)
    
    # Setup environment
    env = BalatroGymEnvSimple()
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    print(f"State size: {state_size}")
    print(f"Action size: {action_size}")
    
    # Create population of evolvable rule agents
    population_size = 50
    generations = 30
    population = []
    
    for _ in range(population_size):
        agent = EvolvableRuleAgent(state_size, action_size)
        population.append(agent)
    
    # Training history
    best_fitnesses = []
    best_win_rates = []
    
    start_time = time.time()
    
    for generation in range(generations):
        print(f"\n🧬 Generation {generation + 1}/{generations}")
        
        # Evaluate all agents
        for agent in population:
            fitness, win_rate = evaluate_rule_agent(agent)
            agent.fitness = fitness
        
        # Sort by fitness
        population.sort(key=lambda x: x.fitness, reverse=True)
        
        # Get best agent
        best_agent = population[0]
        best_fitness, best_win_rate = evaluate_rule_agent(best_agent, episodes=20)
        
        best_fitnesses.append(best_fitness)
        best_win_rates.append(best_win_rate)
        
        print(f"   Best Fitness: {best_fitness:.2f}")
        print(f"   Best Win Rate: {best_win_rate:.1%}")
        print(f"   Best Parameters: {best_agent.get_parameters()}")
        
        # Create new population
        new_population = []
        
        # Keep top 10 agents
        for i in range(10):
            new_population.append(population[i])
        
        # Create offspring
        while len(new_population) < population_size:
            # Select parents
            parent1 = random.choice(population[:20])  # Top 20
            parent2 = random.choice(population[:20])
            
            # Crossover
            child = parent1.crossover(parent2)
            
            # Mutate
            child.mutate()
            
            new_population.append(child)
        
        population = new_population
        
        # Save best agent periodically
        if generation % 10 == 0:
            os.makedirs("weights", exist_ok=True)
            with open(f"weights/rule_best_gen_{generation}.pkl", 'wb') as f:
                import pickle
                pickle.dump(best_agent, f)
        
        # Early stopping
        if best_win_rate > 0.8:
            print("🎯 Early stopping - excellent performance achieved!")
            break
    
    # Final evaluation
    total_time = time.time() - start_time
    print(f"\n🎯 Training completed in {total_time/60:.1f} minutes")
    
    # Test final best agent
    final_fitness, final_win_rate = evaluate_rule_agent(best_agent, episodes=50)
    print(f"Final Best Fitness: {final_fitness:.2f}")
    print(f"Final Best Win Rate: {final_win_rate:.1%}")
    print(f"Final Parameters: {best_agent.get_parameters()}")
    
    # Save final best agent
    with open("weights/rule_final_best.pkl", 'wb') as f:
        import pickle
        pickle.dump(best_agent, f)
    
    # Plot training progress
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(best_fitnesses, label='Best Fitness', color='blue')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Rule Agent Training Progress')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(best_win_rates, label='Best Win Rate', color='red')
    plt.xlabel('Generation')
    plt.ylabel('Win Rate')
    plt.title('Rule Agent Win Rate Progress')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_plots_simple/rule_training_progress.png')
    plt.close()
    
    return best_agent

if __name__ == "__main__":
    agent = train_rule_agent() 