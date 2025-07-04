#!/usr/bin/env python3
"""
Simplified Balatro Gym Environment with reduced state space
Focuses on essential information only to improve training stability
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import List, Tuple, Dict, Any, Optional
import random

# Import the card and hand classes from the original environment
from balatro_gym_v2 import BalatroCard, BalatroHand

class BalatroGymEnvSimple(gym.Env):
    """
    Simplified Balatro Gym Environment with reduced state space
    Focuses on essential game information only
    """
    
    def __init__(self, blind_score: int = 300):
        super().__init__()
        
        # Game state
        self.blind_score = blind_score
        self.deck = self._create_deck()
        self.hand = []
        self.discarded_cards = []
        self.plays_left = 3
        self.discards_left = 3
        self.current_score = 0
        self.game_over = False
        self.won = False
        
        # Action space: same as original
        self.action_space = spaces.Discrete(self._calculate_valid_actions())
        
        # Simplified observation space:
        # - 8 cards in hand (rank + suit) = 16 values
        # - plays_left = 1 value
        # - discards_left = 1 value  
        # - current_score = 1 value
        # - blind_score = 1 value
        # - game_over = 1 value
        # - progress_to_target = 1 value (new)
        # - hand_quality_score = 1 value (new)
        # Total: 23 values (much smaller than 229!)
        self.observation_space = spaces.Box(
            low=0, high=300, shape=(23,), dtype=np.float32
        )
    
    def _create_deck(self) -> List[BalatroCard]:
        """Create a standard 52-card deck"""
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        suits = ['H', 'D', 'C', 'S']
        deck = [BalatroCard(rank, suit) for rank in ranks for suit in suits]
        random.shuffle(deck)
        return deck
    
    def _deal_hand(self):
        """Deal 8 cards to hand"""
        if len(self.deck) < 8:
            self.deck = self._create_deck()
        
        self.hand = self.deck[:8]
        self.deck = self.deck[8:]
    
    def _encode_card(self, card: BalatroCard) -> Tuple[int, int]:
        """Encode card as (rank, suit) integers"""
        rank = card.get_rank_value()
        suit_map = {'H': 0, 'D': 1, 'C': 2, 'S': 3}
        suit = suit_map[card.suit]
        return rank, suit
    
    def _calculate_hand_quality(self) -> float:
        """Calculate a simple hand quality score"""
        if not self.hand:
            return 0.0
        
        # Calculate potential scores for all possible hand combinations
        max_score = 0
        from itertools import combinations
        
        for r in range(1, min(6, len(self.hand) + 1)):
            for combo in combinations(self.hand, r):
                balatro_hand = BalatroHand(list(combo))
                hand_type, base_chips, multiplier, card_chips = balatro_hand.evaluate_hand()
                total_score = (base_chips + card_chips) * multiplier
                max_score = max(max_score, total_score)
        
        return max_score / 100.0  # Normalize
    
    def _get_state(self) -> np.ndarray:
        """Get simplified state as numpy array"""
        # Encode hand (pad with zeros if less than 8 cards)
        hand_encoding = []
        for i in range(8):
            if i < len(self.hand):
                rank, suit = self._encode_card(self.hand[i])
                hand_encoding.extend([rank, suit])
            else:
                hand_encoding.extend([0, 0])
        
        # Calculate additional features
        progress_to_target = self.current_score / self.blind_score
        hand_quality = self._calculate_hand_quality()
        
        # Create simplified state vector
        state = np.array([
            *hand_encoding,      # 16 values (8 cards * 2 each)
            self.plays_left,     # 1 value
            self.discards_left,  # 1 value
            self.current_score,  # 1 value
            self.blind_score,    # 1 value
            int(self.game_over), # 1 value
            progress_to_target,  # 1 value (new)
            hand_quality         # 1 value (new)
        ], dtype=np.float32)
        
        return state
    
    def _calculate_valid_actions(self) -> int:
        """Calculate total number of valid actions"""
        from itertools import combinations
        
        valid_combinations = 0
        for r in range(1, 6):
            valid_combinations += len(list(combinations(range(8), r)))
        
        return valid_combinations * 2
    
    def _decode_action(self, action: int) -> Tuple[str, List[int]]:
        """Decode action into action_type and card_indices"""
        from itertools import combinations
        
        valid_combinations = 0
        for r in range(1, 6):
            valid_combinations += len(list(combinations(range(8), r)))
        
        if action < valid_combinations:
            action_type = "play"
            action_index = action
        else:
            action_type = "discard"
            action_index = action - valid_combinations
        
        card_indices = []
        current_index = 0
        
        for r in range(1, 6):
            combinations_r = list(combinations(range(8), r))
            if current_index <= action_index < current_index + len(combinations_r):
                combo_index = action_index - current_index
                card_indices = list(combinations_r[combo_index])
                break
            current_index += len(combinations_r)
        
        return action_type, card_indices
    
    def _play_hand(self, card_indices: List[int]) -> Tuple[int, str]:
        """Play selected cards and return score and hand type"""
        if not card_indices:
            return 0, "Invalid"
        
        if len(card_indices) > 5:
            card_indices = card_indices[:5]
        
        selected_cards = [self.hand[i] for i in card_indices if i < len(self.hand)]
        
        balatro_hand = BalatroHand(selected_cards)
        hand_type, base_chips, multiplier, card_chips = balatro_hand.evaluate_hand()
        
        total_score = (base_chips + card_chips) * multiplier
        
        cards_played = [self.hand[i] for i in card_indices if i < len(self.hand)]
        self.discarded_cards.extend(cards_played)
        
        self.hand = [card for i, card in enumerate(self.hand) if i not in card_indices]
        
        cards_to_draw = len(card_indices)
        if len(self.deck) < cards_to_draw:
            self.deck.extend(self._create_deck())
        
        new_cards = self.deck[:cards_to_draw]
        self.deck = self.deck[cards_to_draw:]
        self.hand.extend(new_cards)
        
        self.plays_left -= 1
        self.current_score += total_score
        
        return total_score, hand_type
    
    def _discard_cards(self, card_indices: List[int]) -> int:
        """Discard selected cards and return number discarded"""
        if not card_indices:
            return 0
        
        if len(card_indices) > 5:
            card_indices = card_indices[:5]
        
        cards_to_discard = [self.hand[i] for i in card_indices if i < len(self.hand)]
        self.discarded_cards.extend(cards_to_discard)
        
        self.hand = [card for i, card in enumerate(self.hand) if i not in card_indices]
        
        cards_to_draw = len(card_indices)
        if len(self.deck) < cards_to_draw:
            self.deck.extend(self._create_deck())
        
        new_cards = self.deck[:cards_to_draw]
        self.deck = self.deck[cards_to_draw:]
        self.hand.extend(new_cards)
        
        self.discards_left -= 1
        
        return len(card_indices)
    
    def _calculate_reward(self, action_type: str, result: Any) -> float:
        """Calculate reward with improved shaping"""
        if self.game_over:
            if self.won:
                return 200.0  # Increased win reward
            else:
                return -100.0  # Increased loss penalty
        
        if action_type == "play":
            score_gained, hand_type = result
            
            # Base reward (increased)
            reward = score_gained / 5.0  # More reward per point
            
            # Progress bonus - encourage getting closer to target
            progress_bonus = (self.current_score / self.blind_score) * 50  # Increased
            reward += progress_bonus
            
            # Hand quality bonus (increased)
            hand_bonuses = {
                "Royal Flush": 100, "Straight Flush": 60, "Four of a Kind": 40,
                "Full House": 30, "Flush": 20, "Straight": 15,
                "Three of a Kind": 10, "Two Pair": 6, "Pair": 3, "High Card": 0
            }
            reward += hand_bonuses.get(hand_type, 0)
            
            # Efficiency bonus - reward for using fewer cards effectively
            if score_gained > 50:  # Good score
                reward += 10.0
            
            return reward
        
        elif action_type == "discard":
            return -2.0  # Slightly increased discard penalty
        
        return 0.0
    
    def reset(self, seed=None, options=None):
        """Reset the environment"""
        super().reset(seed=seed)
        
        self.deck = self._create_deck()
        self._deal_hand()
        self.discarded_cards = []
        self.plays_left = 3
        self.discards_left = 3
        self.current_score = 0
        self.game_over = False
        self.won = False
        
        return self._get_state(), {}
    
    def step(self, action):
        """Take a step in the environment"""
        if self.game_over:
            return self._get_state(), 0.0, True, False, {}
        
        action_type, card_indices = self._decode_action(action)
        
        # Validate action
        if action_type == "play" and self.plays_left <= 0:
            return self._get_state(), -50.0, True, False, {"error": "No plays left"}
        
        if action_type == "discard" and self.discards_left <= 0:
            return self._get_state(), -50.0, True, False, {"error": "No discards left"}
        
        if not card_indices:
            return self._get_state(), -10.0, False, False, {"error": "No cards selected"}
        
        # Execute action
        if action_type == "play":
            cards_played = [str(self.hand[i]) for i in card_indices if i < len(self.hand)]
            result = self._play_hand(card_indices)
            score_gained, hand_type = result
            info = {
                "action_type": action_type,
                "cards_played": cards_played,
                "hand_type": hand_type,
                "score_gained": score_gained,
                "total_score": self.current_score,
                "cards_selected": card_indices
            }
        else:
            result = self._discard_cards(card_indices)
            info = {
                "action_type": action_type,
                "cards_discarded": result,
                "total_score": self.current_score,
                "cards_selected": card_indices
            }
        
        # Calculate reward
        reward = self._calculate_reward(action_type, result)
        
        # Check game end conditions
        if self.current_score >= self.blind_score:
            self.game_over = True
            self.won = True
            reward += 100.0
        elif self.plays_left <= 0:
            self.game_over = True
            self.won = False
            reward -= 100.0  # Severe penalty for running out of plays
        
        info.update({
            "plays_left": self.plays_left,
            "discards_left": self.discards_left,
            "game_over": self.game_over,
            "won": self.won
        })
        
        return self._get_state(), reward, self.game_over, False, info
    
    def render(self):
        """Render the current state"""
        print(f"Hand: {[str(card) for card in self.hand]}")
        print(f"Score: {self.current_score}/{self.blind_score}")
        print(f"Plays left: {self.plays_left}, Discards left: {self.discards_left}")
        if self.game_over:
            print(f"Game Over! {'WON' if self.won else 'LOST'}")
        print("-" * 50)

# Example usage
if __name__ == "__main__":
    env = BalatroGymEnvSimple(blind_score=300)
    
    print("🎰 Testing Simplified Balatro Gym Environment")
    print("=" * 50)
    
    obs, info = env.reset()
    print(f"State size: {len(obs)} (vs 229 in original)")
    print(f"Initial state: {obs}")
    
    total_reward = 0
    
    while not env.game_over:
        env.render()
        
        action = env.action_space.sample()
        action_type, card_indices = env._decode_action(action)
        
        print(f"Action: {action} -> {action_type} cards {card_indices}")
        
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        
        print(f"Reward: {reward:.2f}")
        print(f"Info: {info}")
        print()
        
        if done:
            break
    
    env.render()
    print(f"Total reward: {total_reward:.2f}")
    print("=" * 50) 