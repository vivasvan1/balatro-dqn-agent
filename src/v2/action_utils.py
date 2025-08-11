#!/usr/bin/env python3
"""
Action utilities for Balatro Gym Environment
Provides functions to generate random and smart actions
"""

import numpy as np
import random
from typing import List, Tuple, Dict, Any, Optional
from balatro_gym_v2_simple import BalatroGymEnvSimple


def generate_random_action(env: BalatroGymEnvSimple) -> np.ndarray:
    """
    Generate a completely random action for the Balatro environment.
    
    Args:
        env: The Balatro environment instance
        
    Returns:
        np.ndarray: Random action array with shape (7,) containing:
            - action_type: 0=play, 1=discard
            - card_1: index of first card (0-7)
            - card_2: index of second card (0-7)
            - card_3: index of third card (0-7)
            - card_4: index of fourth card (0-7)
            - card_5: index of fifth card (0-7)
            - priority: action priority (0-4)
    """
    # Generate random action type (play or discard)
    action_type = random.randint(0, 1)
    
    # Generate random card indices (0-7, where 0 means no card)
    card_indices = [random.randint(0, 7) for _ in range(5)]
    
    # Generate random priority (0-4)
    priority = random.randint(0, 4)
    
    # Combine into action array
    action = np.array([action_type] + card_indices + [priority], dtype=np.int64)
    
    return action


def generate_smart_random_action(env: BalatroGymEnvSimple) -> np.ndarray:
    """
    Generate a smart random action that respects game constraints.
    
    Args:
        env: The Balatro environment instance
        
    Returns:
        np.ndarray: Smart random action array
    """
    # Check if we can play or discard
    can_play = env.plays_left > 0
    can_discard = env.discards_left > 0
    
    if not can_play and not can_discard:
        # If we can't do anything, return a pass-like action
        return np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.int64)
    
    # Choose action type based on availability
    if can_play and can_discard:
        action_type = random.randint(0, 1)
    elif can_play:
        action_type = 0  # play
    else:
        action_type = 1  # discard
    
    # Determine how many cards to use (1-5, but not more than available)
    num_cards = random.randint(1, min(5, len(env.hand)))
    
    # Generate valid card indices
    available_indices = list(range(len(env.hand)))
    selected_indices = random.sample(available_indices, num_cards)
    
    # Pad with zeros to get 5 card slots
    card_indices = selected_indices + [0] * (5 - len(selected_indices))
    
    # Generate random priority
    priority = random.randint(0, 4)
    
    # Combine into action array
    action = np.array([action_type] + card_indices + [priority], dtype=np.int64)
    
    return action


def generate_heuristic_action(env: BalatroGymEnvSimple) -> np.ndarray:
    """
    Generate an action based on simple heuristics.
    
    Args:
        env: The Balatro environment instance
        
    Returns:
        np.ndarray: Heuristic-based action array
    """
    if not env.hand:
        return np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.int64)
    
    # Calculate hand strength
    from itertools import combinations
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from base import BalatroHand
    
    max_score = 0
    best_combo = []
    
    for r in range(1, min(6, len(env.hand) + 1)):
        for combo in combinations(env.hand, r):
            balatro_hand = BalatroHand(list(combo))
            hand_type, base_chips, multiplier, card_chips = balatro_hand.evaluate_hand()
            total_score = (base_chips + card_chips) * multiplier
            if total_score > max_score:
                max_score = total_score
                best_combo = list(combo)
    
    # Decide whether to play or discard
    if env.plays_left > 0 and max_score > 30:  # Good hand
        action_type = 0  # play
        cards_to_use = best_combo
    elif env.discards_left > 0 and max_score < 20:  # Bad hand
        action_type = 1  # discard
        # Discard worst cards (lowest rank)
        cards_to_use = sorted(env.hand, key=lambda c: c.get_rank_value())[:min(3, len(env.hand))]
    else:
        # Default to play if we have plays left, otherwise discard
        if env.plays_left > 0:
            action_type = 0
            cards_to_use = best_combo if best_combo else [env.hand[0]]
        else:
            action_type = 1
            cards_to_use = [env.hand[0]]
    
    # Convert cards to indices
    card_indices = []
    for card in cards_to_use:
        if card in env.hand:
            card_indices.append(env.hand.index(card))
    
    # Pad with zeros
    card_indices = card_indices + [0] * (5 - len(card_indices))
    
    # Set priority based on action type and hand strength
    if action_type == 0:  # play
        priority = min(4, int(max_score / 25))  # Higher priority for better hands
    else:  # discard
        priority = 2  # Medium priority for discards
    
    action = np.array([action_type] + card_indices + [priority], dtype=np.int64)
    
    return action


def generate_action_batch(env: BalatroGymEnvSimple, batch_size: int, action_type: str = "random") -> np.ndarray:
    """
    Generate a batch of actions.
    
    Args:
        env: The Balatro environment instance
        batch_size: Number of actions to generate
        action_type: Type of action generation ("random", "smart_random", "heuristic")
        
    Returns:
        np.ndarray: Batch of actions with shape (batch_size, 7)
    """
    action_generators = {
        "random": generate_random_action,
        "smart_random": generate_smart_random_action,
        "heuristic": generate_heuristic_action
    }
    
    if action_type not in action_generators:
        raise ValueError(f"Unknown action type: {action_type}. Use one of {list(action_generators.keys())}")
    
    generator = action_generators[action_type]
    actions = []
    
    for _ in range(batch_size):
        action = generator(env)
        actions.append(action)
    
    return np.array(actions)


def decode_action_for_display(action: np.ndarray, env: BalatroGymEnvSimple) -> Dict[str, Any]:
    """
    Decode an action for display purposes.
    
    Args:
        action: Action array
        env: The Balatro environment instance
        
    Returns:
        Dict containing decoded action information
    """
    action_type_idx = action[0]
    card_indices = action[1:6]
    priority = action[6]
    
    action_type = "play" if action_type_idx == 0 else "discard"
    
    # Get actual card indices (non-zero values)
    actual_indices = [idx for idx in card_indices if idx > 0 and idx < len(env.hand)]
    
    # Get card names
    card_names = []
    for idx in actual_indices:
        if idx < len(env.hand):
            card_names.append(str(env.hand[idx]))
    
    return {
        "action_type": action_type,
        "card_indices": actual_indices,
        "card_names": card_names,
        "priority": priority,
        "num_cards": len(actual_indices)
    }


def test_action_generation():
    """Test function to demonstrate action generation"""
    env = BalatroGymEnvSimple(blind_score=300)
    obs, _ = env.reset()
    
    print("Testing action generation:")
    print(f"Initial hand: {[str(card) for card in env.hand]}")
    print(f"Plays left: {env.plays_left}, Discards left: {env.discards_left}")
    print()
    
    # Test different action types
    action_types = ["random", "smart_random", "heuristic"]
    
    for action_type in action_types:
        print(f"--- {action_type.upper()} ACTION ---")
        action = generate_action_batch(env, 1, action_type)[0]
        decoded = decode_action_for_display(action, env)
        
        print(f"Action: {action}")
        print(f"Type: {decoded['action_type']}")
        print(f"Cards: {decoded['card_names']}")
        print(f"Priority: {decoded['priority']}")
        print()


if __name__ == "__main__":
    test_action_generation() 