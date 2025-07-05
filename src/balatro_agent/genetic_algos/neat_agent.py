#!/usr/bin/env python3
"""
NEAT (NeuroEvolution of Augmenting Topologies) Agent for Balatro
NEAT is excellent for this type of problem because it can evolve both weights and topology
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

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from balatro_gym_v2_simple import BalatroGymEnvSimple

# NEAT Configuration
POPULATION_SIZE = 150
GENERATIONS = 100
ELITE_SIZE = 10
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.7
WEIGHT_MUTATION_RATE = 0.8
WEIGHT_MUTATION_STRENGTH = 0.1
ADD_NODE_RATE = 0.03
ADD_CONNECTION_RATE = 0.05
REMOVE_CONNECTION_RATE = 0.01

class Node:
    """Represents a node in the neural network"""
    def __init__(self, node_id: int, node_type: str = 'hidden'):
        self.node_id = node_id
        self.node_type = node_type  # 'input', 'hidden', 'output'
        self.value = 0.0
        self.connections = []  # List of Connection objects
        
    def activate(self, x: float) -> float:
        """Activation function (ReLU)"""
        return max(0, x)

class Connection:
    """Represents a connection between nodes"""
    def __init__(self, from_node: int, to_node: int, weight: float = None, enabled: bool = True):
        self.from_node = from_node
        self.to_node = to_node
        self.weight = weight if weight is not None else random.uniform(-1, 1)
        self.enabled = enabled
        self.innovation = None  # Innovation number for NEAT
        
    def mutate_weight(self):
        """Mutate the connection weight"""
        if random.random() < WEIGHT_MUTATION_RATE:
            self.weight += random.uniform(-WEIGHT_MUTATION_STRENGTH, WEIGHT_MUTATION_STRENGTH)
            # Keep weights in reasonable range
            self.weight = max(-5, min(5, self.weight))

class Genome:
    """Represents a complete neural network genome"""
    def __init__(self, input_size: int, output_size: int):
        self.input_size = input_size
        self.output_size = output_size
        self.nodes = {}
        self.connections = []
        self.fitness = 0.0
        self.adjusted_fitness = 0.0
        self.species_id = None
        
        # Initialize input nodes
        for i in range(input_size):
            self.nodes[i] = Node(i, 'input')
        
        # Initialize output nodes
        for i in range(output_size):
            self.nodes[input_size + i] = Node(input_size + i, 'output')
        
        # Add initial connections (fully connected)
        for i in range(input_size):
            for j in range(output_size):
                self.connections.append(Connection(i, input_size + j))
        
        # Innovation counter
        self.next_innovation = len(self.connections)
    
    def add_node(self):
        """Add a new hidden node"""
        if len(self.connections) == 0:
            return
        
        # Pick a random connection to split
        conn = random.choice(self.connections)
        if not conn.enabled:
            return
        
        # Disable the original connection
        conn.enabled = False
        
        # Create new hidden node
        new_node_id = len(self.nodes)
        new_node = Node(new_node_id, 'hidden')
        self.nodes[new_node_id] = new_node
        
        # Create two new connections
        conn1 = Connection(conn.from_node, new_node_id, 1.0)
        conn2 = Connection(new_node_id, conn.to_node, conn.weight)
        
        self.connections.append(conn1)
        self.connections.append(conn2)
    
    def add_connection(self):
        """Add a new connection between unconnected nodes"""
        # Find unconnected pairs
        unconnected = []
        for from_id in self.nodes:
            for to_id in self.nodes:
                if from_id >= to_id:  # Avoid self-connections and duplicates
                    continue
                
                # Check if connection already exists
                exists = any(c.from_node == from_id and c.to_node == to_id for c in self.connections)
                if not exists:
                    unconnected.append((from_id, to_id))
        
        if unconnected:
            from_id, to_id = random.choice(unconnected)
            self.connections.append(Connection(from_id, to_id))
    
    def remove_connection(self):
        """Remove a random connection"""
        if len(self.connections) > 1:
            conn = random.choice(self.connections)
            self.connections.remove(conn)
    
    def mutate(self):
        """Apply mutations to the genome"""
        # Weight mutations
        for conn in self.connections:
            conn.mutate_weight()
        
        # Structural mutations
        if random.random() < ADD_NODE_RATE:
            self.add_node()
        
        if random.random() < ADD_CONNECTION_RATE:
            self.add_connection()
        
        if random.random() < REMOVE_CONNECTION_RATE:
            self.remove_connection()
    
    def crossover(self, other: 'Genome') -> 'Genome':
        """Create a child genome from two parents"""
        child = Genome(self.input_size, self.output_size)
        child.nodes = self.nodes.copy()
        child.connections = []
        
        # Crossover connections
        for conn in self.connections:
            # Find matching connection in other parent
            matching = None
            for other_conn in other.connections:
                if (conn.from_node == other_conn.from_node and 
                    conn.to_node == other_conn.to_node):
                    matching = other_conn
                    break
            
            if matching:
                # Inherit from either parent
                if random.random() < 0.5:
                    child.connections.append(Connection(conn.from_node, conn.to_node, conn.weight, conn.enabled))
                else:
                    child.connections.append(Connection(matching.from_node, matching.to_node, matching.weight, matching.enabled))
            else:
                # Inherit from self (more fit parent)
                child.connections.append(Connection(conn.from_node, conn.to_node, conn.weight, conn.enabled))
        
        return child
    
    def forward(self, inputs: List[float]) -> List[float]:
        """Forward pass through the neural network"""
        # Reset node values
        for node in self.nodes.values():
            node.value = 0.0
        
        # Set input values
        for i, value in enumerate(inputs):
            if i in self.nodes:
                self.nodes[i].value = value
        
        # Forward pass
        outputs = []
        for i in range(self.output_size):
            output_id = self.input_size + i
            if output_id in self.nodes:
                output_node = self.nodes[output_id]
                
                # Calculate weighted sum of inputs
                total = 0.0
                for conn in self.connections:
                    if conn.to_node == output_id and conn.enabled:
                        from_node = self.nodes[conn.from_node]
                        total += from_node.value * conn.weight
                
                # Apply activation function
                output_node.value = output_node.activate(total)
                outputs.append(output_node.value)
            else:
                outputs.append(0.0)
        
        return outputs

class Species:
    """Represents a species of similar genomes"""
    def __init__(self, representative: Genome):
        self.representative = representative
        self.members = [representative]
        self.average_fitness = 0.0
        self.stagnation = 0
        self.best_fitness = 0.0
    
    def add_member(self, genome: Genome):
        """Add a member to the species"""
        self.members.append(genome)
        genome.species_id = id(self)
    
    def calculate_average_fitness(self):
        """Calculate the average fitness of the species"""
        if self.members:
            self.average_fitness = sum(m.fitness for m in self.members) / len(self.members)
            self.best_fitness = max(m.fitness for m in self.members)
    
    def cull(self, keep_percentage: float = 0.5):
        """Remove the worst performing members"""
        self.members.sort(key=lambda x: x.fitness, reverse=True)
        keep_count = max(1, int(len(self.members) * keep_percentage))
        self.members = self.members[:keep_count]

class NEATAgent:
    """NEAT-based agent for Balatro"""
    
    def __init__(self, state_size: int, action_size: int):
        self.state_size = state_size
        self.action_size = action_size
        self.population = []
        self.species = []
        self.generation = 0
        self.best_fitness = 0.0
        self.best_genome = None
        
        # Initialize population
        for _ in range(POPULATION_SIZE):
            genome = Genome(state_size, action_size)
            self.population.append(genome)
    
    def evaluate_fitness(self, genome: Genome, episodes: int = 5) -> float:
        """Evaluate the fitness of a genome by playing multiple episodes"""
        env = BalatroGymEnvSimple()
        total_fitness = 0.0
        
        for episode in range(episodes):
            obs, _ = env.reset()
            episode_fitness = 0.0
            done = False
            step_count = 0
            
            while not done and step_count < 50:  # Prevent infinite loops
                # Get action from genome
                outputs = genome.forward(obs.tolist())
                action = np.argmax(outputs)
                
                # Take action
                obs, reward, done, truncated, info = env.step(action)
                episode_fitness += reward
                step_count += 1
            
            # Bonus for winning
            if info.get('won', False):
                episode_fitness += 100.0
            
            total_fitness += episode_fitness
        
        return total_fitness / episodes
    
    def speciate(self):
        """Group genomes into species based on similarity"""
        self.species = []
        
        for genome in self.population:
            placed = False
            
            for species in self.species:
                # Simple distance metric (can be improved)
                distance = self.calculate_distance(genome, species.representative)
                if distance < 3.0:  # Threshold for species membership
                    species.add_member(genome)
                    placed = True
                    break
            
            if not placed:
                # Create new species
                new_species = Species(genome)
                self.species.append(new_species)
    
    def calculate_distance(self, genome1: Genome, genome2: Genome) -> float:
        """Calculate distance between two genomes"""
        # Simple distance metric based on connection differences
        # This can be improved with more sophisticated metrics
        return abs(len(genome1.connections) - len(genome2.connections))
    
    def evolve(self):
        """Evolve the population"""
        # Evaluate fitness
        for genome in self.population:
            genome.fitness = self.evaluate_fitness(genome)
            if genome.fitness > self.best_fitness:
                self.best_fitness = genome.fitness
                self.best_genome = genome
        
        # Speciate
        self.speciate()
        
        # Calculate adjusted fitness and species statistics
        for species in self.species:
            species.calculate_average_fitness()
        
        # Cull species
        for species in self.species:
            species.cull()
        
        # Create new population
        new_population = []
        
        # Elitism: keep best genomes
        all_genomes = [g for species in self.species for g in species.members]
        all_genomes.sort(key=lambda x: x.fitness, reverse=True)
        
        for i in range(min(ELITE_SIZE, len(all_genomes))):
            new_population.append(all_genomes[i])
        
        # Fill rest with offspring
        while len(new_population) < POPULATION_SIZE:
            # Select parents
            parent1 = self.select_parent()
            parent2 = self.select_parent()
            
            if random.random() < CROSSOVER_RATE:
                child = parent1.crossover(parent2)
            else:
                child = parent1
            
            # Mutate child
            child.mutate()
            new_population.append(child)
        
        self.population = new_population[:POPULATION_SIZE]
        self.generation += 1
    
    def select_parent(self) -> Genome:
        """Select a parent using tournament selection"""
        tournament_size = 3
        tournament = random.sample(self.population, tournament_size)
        return max(tournament, key=lambda x: x.fitness)
    
    def act(self, state: np.ndarray) -> int:
        """Get action from the best genome"""
        if self.best_genome is None:
            return random.randint(0, self.action_size - 1)
        
        outputs = self.best_genome.forward(state.tolist())
        return np.argmax(outputs)
    
    def save(self, filepath: str):
        """Save the best genome"""
        if self.best_genome:
            with open(filepath, 'wb') as f:
                pickle.dump(self.best_genome, f)
    
    def load(self, filepath: str):
        """Load a genome"""
        with open(filepath, 'rb') as f:
            self.best_genome = pickle.load(f)

def train_neat_agent():
    """Train a NEAT agent for Balatro"""
    print("🧬 Training NEAT Agent for Balatro")
    print("=" * 50)
    
    # Setup environment
    env = BalatroGymEnvSimple()
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    print(f"State size: {state_size}")
    print(f"Action size: {action_size}")
    print(f"Population size: {POPULATION_SIZE}")
    print(f"Generations: {GENERATIONS}")
    
    # Create NEAT agent
    agent = NEATAgent(state_size, action_size)
    
    # Training history
    best_fitnesses = []
    avg_fitnesses = []
    
    start_time = time.time()
    
    for generation in range(GENERATIONS):
        print(f"\n🧬 Generation {generation + 1}/{GENERATIONS}")
        
        # Evolve population
        agent.evolve()
        
        # Calculate statistics
        fitnesses = [g.fitness for g in agent.population]
        best_fitness = max(fitnesses)
        avg_fitness = np.mean(fitnesses)
        
        best_fitnesses.append(best_fitness)
        avg_fitnesses.append(avg_fitness)
        
        print(f"   Best Fitness: {best_fitness:.2f}")
        print(f"   Avg Fitness: {avg_fitness:.2f}")
        print(f"   Species: {len(agent.species)}")
        print(f"   Best Genome Connections: {len(agent.best_genome.connections) if agent.best_genome else 0}")
        
        # Save best genome periodically
        if generation % 10 == 0:
            os.makedirs("weights", exist_ok=True)
            agent.save(f"weights/neat_best_genome_gen_{generation}.pkl")
        
        # Early stopping if we're doing well
        if best_fitness > 500:
            print("🎯 Early stopping - excellent performance achieved!")
            break
    
    # Final evaluation
    total_time = time.time() - start_time
    print(f"\n🎯 Training completed in {total_time/60:.1f} minutes")
    print(f"Best fitness achieved: {agent.best_fitness:.2f}")
    
    # Save final best genome
    agent.save("weights/neat_final_best_genome.pkl")
    
    # Plot training progress
    plt.figure(figsize=(12, 6))
    plt.plot(best_fitnesses, label='Best Fitness', color='blue')
    plt.plot(avg_fitnesses, label='Average Fitness', color='red')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('NEAT Training Progress')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_plots_simple/neat_training_progress.png')
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
    agent = train_neat_agent() 