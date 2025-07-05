#!/usr/bin/env python3
"""
Genetic Algorithm Agent for Balatro
Uses fixed neural network topologies but evolves weights and hyperparameters
"""

import os
import sys
import numpy as np
import random
from typing import List, Tuple, Dict, Any
import matplotlib.pyplot as plt
from datetime import datetime
import time
import pickle
import torch
import torch.nn as nn

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from balatro_gym_v2_simple import BalatroGymEnvSimple

# Genetic Algorithm Configuration
POPULATION_SIZE = 100
GENERATIONS = 50
ELITE_SIZE = 5
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.7
WEIGHT_MUTATION_STRENGTH = 0.2

class SimpleNN(nn.Module):
    """Simple neural network for genetic algorithm"""
    def __init__(self, input_size: int, output_size: int, hidden_size: int = 64):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Individual:
    """Represents an individual in the genetic algorithm"""
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 64):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.fitness = 0.0
        self.network = SimpleNN(state_size, action_size, hidden_size)
        self.initialize_weights()
    
    def initialize_weights(self):
        """Initialize network weights randomly"""
        for module in self.network.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def mutate(self):
        """Mutate the individual's weights"""
        for param in self.network.parameters():
            if random.random() < MUTATION_RATE:
                # Add random noise to weights
                noise = torch.randn_like(param) * WEIGHT_MUTATION_STRENGTH
                param.data += noise
    
    def crossover(self, other: 'Individual') -> 'Individual':
        """Create a child by crossing over with another individual"""
        child = Individual(self.state_size, self.action_size, self.hidden_size)
        
        # Crossover weights
        for child_param, self_param, other_param in zip(
            child.network.parameters(), 
            self.network.parameters(), 
            other.network.parameters()
        ):
            if random.random() < 0.5:
                child_param.data = self_param.data.clone()
            else:
                child_param.data = other_param.data.clone()
        
        return child
    
    def act(self, state: np.ndarray) -> int:
        """Get action from the neural network"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            outputs = self.network(state_tensor)
            return outputs.argmax().item()

class GeneticAgent:
    """Genetic Algorithm agent for Balatro"""
    
    def __init__(self, state_size: int, action_size: int):
        self.state_size = state_size
        self.action_size = action_size
        self.population = []
        self.generation = 0
        self.best_fitness = 0.0
        self.best_individual = None
        
        # Initialize population
        for _ in range(POPULATION_SIZE):
            individual = Individual(state_size, action_size)
            self.population.append(individual)
    
    def evaluate_fitness(self, individual: Individual, episodes: int = 3) -> float:
        """Evaluate the fitness of an individual by playing multiple episodes"""
        env = BalatroGymEnvSimple()
        total_fitness = 0.0
        
        for episode in range(episodes):
            obs, _ = env.reset()
            episode_fitness = 0.0
            done = False
            step_count = 0
            
            while not done and step_count < 50:  # Prevent infinite loops
                # Get action from individual
                action = individual.act(obs)
                
                # Take action
                obs, reward, done, truncated, info = env.step(action)
                episode_fitness += reward
                step_count += 1
            
            # Bonus for winning
            if info.get('won', False):
                episode_fitness += 100.0
            
            total_fitness += episode_fitness
        
        return total_fitness / episodes
    
    def evolve(self):
        """Evolve the population"""
        # Evaluate fitness
        for individual in self.population:
            individual.fitness = self.evaluate_fitness(individual)
            if individual.fitness > self.best_fitness:
                self.best_fitness = individual.fitness
                self.best_individual = individual
        
        # Sort population by fitness
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        
        # Create new population
        new_population = []
        
        # Elitism: keep best individuals
        for i in range(ELITE_SIZE):
            new_population.append(self.population[i])
        
        # Fill rest with offspring
        while len(new_population) < POPULATION_SIZE:
            # Select parents using tournament selection
            parent1 = self.tournament_select()
            parent2 = self.tournament_select()
            
            if random.random() < CROSSOVER_RATE:
                child = parent1.crossover(parent2)
            else:
                child = Individual(self.state_size, self.action_size)
                # Copy parent1's weights
                for child_param, parent_param in zip(child.network.parameters(), parent1.network.parameters()):
                    child_param.data = parent_param.data.clone()
            
            # Mutate child
            child.mutate()
            new_population.append(child)
        
        self.population = new_population[:POPULATION_SIZE]
        self.generation += 1
    
    def tournament_select(self, tournament_size: int = 3) -> Individual:
        """Select an individual using tournament selection"""
        tournament = random.sample(self.population, tournament_size)
        return max(tournament, key=lambda x: x.fitness)
    
    def act(self, state: np.ndarray) -> int:
        """Get action from the best individual"""
        if self.best_individual is None:
            return random.randint(0, self.action_size - 1)
        
        return self.best_individual.act(state)
    
    def save(self, filepath: str):
        """Save the best individual"""
        if self.best_individual:
            torch.save(self.best_individual.network.state_dict(), filepath)
    
    def load(self, filepath: str):
        """Load an individual"""
        individual = Individual(self.state_size, self.action_size)
        individual.network.load_state_dict(torch.load(filepath))
        self.best_individual = individual

def train_genetic_agent():
    """Train a Genetic Algorithm agent for Balatro"""
    print("🧬 Training Genetic Algorithm Agent for Balatro")
    print("=" * 50)
    
    # Setup environment
    env = BalatroGymEnvSimple()
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    print(f"State size: {state_size}")
    print(f"Action size: {action_size}")
    print(f"Population size: {POPULATION_SIZE}")
    print(f"Generations: {GENERATIONS}")
    
    # Create Genetic agent
    agent = GeneticAgent(state_size, action_size)
    
    # Training history
    best_fitnesses = []
    avg_fitnesses = []
    
    start_time = time.time()
    
    for generation in range(GENERATIONS):
        print(f"\n🧬 Generation {generation + 1}/{GENERATIONS}")
        
        # Evolve population
        agent.evolve()
        
        # Calculate statistics
        fitnesses = [ind.fitness for ind in agent.population]
        best_fitness = max(fitnesses)
        avg_fitness = np.mean(fitnesses)
        
        best_fitnesses.append(best_fitness)
        avg_fitnesses.append(avg_fitness)
        
        print(f"   Best Fitness: {best_fitness:.2f}")
        print(f"   Avg Fitness: {avg_fitness:.2f}")
        print(f"   Population Diversity: {np.std(fitnesses):.2f}")
        
        # Save best individual periodically
        if generation % 10 == 0:
            os.makedirs("weights", exist_ok=True)
            agent.save(f"weights/genetic_best_gen_{generation}.pth")
        
        # Early stopping if we're doing well
        if best_fitness > 300:
            print("🎯 Early stopping - excellent performance achieved!")
            break
    
    # Final evaluation
    total_time = time.time() - start_time
    print(f"\n🎯 Training completed in {total_time/60:.1f} minutes")
    print(f"Best fitness achieved: {agent.best_fitness:.2f}")
    
    # Save final best individual
    agent.save("weights/genetic_final_best.pth")
    
    # Plot training progress
    plt.figure(figsize=(12, 6))
    plt.plot(best_fitnesses, label='Best Fitness', color='blue')
    plt.plot(avg_fitnesses, label='Average Fitness', color='red')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Genetic Algorithm Training Progress')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_plots_simple/genetic_training_progress.png')
    plt.close()
    
    # Test the best agent
    print(f"\n🧪 Testing Best Agent:")
    test_episodes = 10
    wins = 0
    total_score = 0
    
    for episode in range(test_episodes):
        obs, _ = env.reset()
        episode_score = 0
        done = False
        step_count = 0
        
        while not done and step_count < 50:
            action = agent.act(obs)
            obs, reward, done, truncated, info = env.step(action)
            episode_score += reward
            step_count += 1
        
        if info.get('won', False):
            wins += 1
        total_score += episode_score
    
    win_rate = wins / test_episodes
    avg_score = total_score / test_episodes
    
    print(f"   Win Rate: {win_rate:.1%}")
    print(f"   Average Score: {avg_score:.1f}")
    
    return agent

if __name__ == "__main__":
    agent = train_genetic_agent() 