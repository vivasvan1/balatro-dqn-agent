#!/usr/bin/env python3
"""
Completely redesigned Simplified Balatro Gym Environment
Focus on core mechanics with much simpler action space and clearer rewards
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import List, Tuple, Dict, Any, Optional
import random

# Import the card and hand classes from the original environment
from balatro_gym_v2 import BalatroCard, BalatroHand

PLAYS_LEFT = 3
DISCARDS_LEFT = 3

class BalatroGymEnvSimple(gym.Env):
    """
    Completely redesigned Simplified Balatro Gym Environment
    Much simpler action space and clearer reward structure
    """
    
    def __init__(self, blind_score: int = 400):
        super().__init__()
        
        # Game state
        self.blind_score = blind_score
        self.deck = self._create_deck()
        self.hand = []
        self.discarded_cards = []
        self.plays_left = PLAYS_LEFT
        self.discards_left = DISCARDS_LEFT
        self.current_score = 0
        self.game_over = False
        self.won = False
        
        # Much simpler action space: 8 actions total
        # Actions 0-7: Play the card at that index (if valid)
        # Action 8: Discard the worst card
        # Action 9: Discard the second worst card
        # Action 10: Pass (do nothing)
        self.action_space = spaces.Discrete(11)
        
        # Simplified observation space: 17 values
        # - 8 cards in hand (rank only) = 8 values
        # - plays_left = 1 value
        # - discards_left = 1 value  
        # - current_score = 1 value
        # - blind_score = 1 value
        # - game_over = 1 value
        # - best_hand_value = 1 value (new)
        # - worst_card_value = 1 value (new)
        # - second_worst_card_value = 1 value (new)
        # - can_play_high_hand = 1 value (new)
        # Total: 17 values
        self.observation_space = spaces.Box(
            low=0, high=300, shape=(17,), dtype=np.float32
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
    
    def _get_card_value(self, card: BalatroCard) -> int:
        """Get numeric value of card (2=2, A=14)"""
        rank_values = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 
                      '10': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
        return rank_values.get(card.rank, 2)
    
    def _find_best_hand(self) -> Tuple[str, int]:
        """Find the best possible hand from current cards"""
        if len(self.hand) < 3:
            return "Invalid", 0
        
        best_hand_type = "High Card"
        best_score = 0
        
        # Try all combinations of 3-5 cards
        from itertools import combinations
        for r in range(3, min(6, len(self.hand) + 1)):
            for combo in combinations(self.hand, r):
                balatro_hand = BalatroHand(list(combo))
                hand_type, base_chips, multiplier, card_chips = balatro_hand.evaluate_hand()
                total_score = (base_chips + card_chips) * multiplier
                
                if total_score > best_score:
                    best_score = total_score
                    best_hand_type = hand_type
        
        return best_hand_type, best_score
    
    def _find_worst_cards(self) -> Tuple[int, int]:
        """Find indices of the two worst cards to discard"""
        if len(self.hand) < 2:
            return 0, 0
        
        # Sort cards by value (lowest first)
        card_values = [(i, self._get_card_value(card)) for i, card in enumerate(self.hand)]
        card_values.sort(key=lambda x: x[1])
        
        return card_values[0][0], card_values[1][0]
    
    def _can_play_high_hand(self) -> bool:
        """Check if we can play a high-value hand (Three of a Kind or better)"""
        best_hand_type, _ = self._find_best_hand()
        high_hands = ["Three of a Kind", "Straight", "Flush", "Full House", "Four of a Kind", "Straight Flush", "Royal Flush"]
        return best_hand_type in high_hands
    
    def _get_state(self) -> np.ndarray:
        """Get simplified state as numpy array"""
        # Encode hand (rank values only)
        hand_encoding = []
        for i in range(8):
            if i < len(self.hand):
                hand_encoding.append(self._get_card_value(self.hand[i]))
            else:
                hand_encoding.append(0)
        
        # Find best hand and worst cards
        best_hand_type, best_hand_value = self._find_best_hand()
        worst_idx, second_worst_idx = self._find_worst_cards()
        worst_value = self._get_card_value(self.hand[worst_idx]) if worst_idx < len(self.hand) else 0
        second_worst_value = self._get_card_value(self.hand[second_worst_idx]) if second_worst_idx < len(self.hand) else 0
        can_play_high = 1.0 if self._can_play_high_hand() else 0.0
        
        # Create simplified state vector
        state = np.array([
            *hand_encoding,           # 8 values (card ranks)
            self.plays_left,          # 1 value
            self.discards_left,       # 1 value
            self.current_score,       # 1 value
            self.blind_score,         # 1 value
            int(self.game_over),      # 1 value
            best_hand_value / 100.0,  # 1 value (normalized)
            worst_value,              # 1 value
            second_worst_value,       # 1 value
            can_play_high             # 1 value
        ], dtype=np.float32)
        
        return state
    
    def _play_single_card(self, card_index: int) -> Tuple[int, str]:
        """Play a single card and return score and hand type"""
        if card_index >= len(self.hand):
            return 0, "Invalid"
        
        card = self.hand[card_index]
        balatro_hand = BalatroHand([card])
        hand_type, base_chips, multiplier, card_chips = balatro_hand.evaluate_hand()
        total_score = (base_chips + card_chips) * multiplier
        
        # Remove card from hand and add to discarded
        self.discarded_cards.append(card)
        self.hand.pop(card_index)
        
        # Draw new card
        if len(self.deck) < 1:
            self.deck.extend(self._create_deck())
        new_card = self.deck.pop(0)
        self.hand.append(new_card)
        
        self.plays_left -= 1
        self.current_score += total_score
        
        return total_score, hand_type
    
    def _discard_worst_card(self, which_worst: int = 0) -> int:
        """Discard the worst card (0=worst, 1=second worst)"""
        if len(self.hand) < 1:
            return 0
        
        worst_idx, second_worst_idx = self._find_worst_cards()
        discard_idx = worst_idx if which_worst == 0 else second_worst_idx
        
        if discard_idx >= len(self.hand):
            return 0
        
        # Remove card and add to discarded
        card = self.hand.pop(discard_idx)
        self.discarded_cards.append(card)
        
        # Draw new card
        if len(self.deck) < 1:
            self.deck.extend(self._create_deck())
        new_card = self.deck.pop(0)
        self.hand.append(new_card)
        
        self.discards_left -= 1
        
        return 1
    
    def _calculate_reward(self, action: int, result: Any) -> float:
        """Simple, clear reward function"""
        # Win/Loss rewards
        if self.game_over:
            if self.won:
                return 100.0  # Big win reward
            else:
                return -20.0  # Moderate loss penalty
        
        # Action-specific rewards
        if action <= 7:  # Play card action
            if isinstance(result, tuple):
                score_gained, hand_type = result
                
                # Reward based on hand type
                if hand_type in ["Royal Flush", "Straight Flush", "Four of a Kind"]:
                    return 50.0
                elif hand_type in ["Full House", "Flush", "Straight"]:
                    return 30.0
                elif hand_type == "Three of a Kind":
                    return 20.0
                elif hand_type == "Two Pair":
                    return 5.0
                elif hand_type == "Pair":
                    return -1.0
                elif hand_type == "High Card":
                    return -5.0  # Small penalty for High Card
                else:
                    return 0.0
            else:
                return -10.0  # Invalid play
        
        elif action == 8 or action == 9:  # Discard actions
            return 2.0  # Small positive reward for discarding
        
        elif action == 10:  # Pass action
            return 0.0  # Neutral for passing
        
        return 0.0
    
    def reset(self, seed=None, options=None):
        """Reset the environment"""
        super().reset(seed=seed)
        
        self.deck = self._create_deck()
        self._deal_hand()
        self.discarded_cards = []
        self.plays_left = PLAYS_LEFT
        self.discards_left = DISCARDS_LEFT
        self.current_score = 0
        self.game_over = False
        self.won = False
        
        return self._get_state(), {}
    
    def step(self, action):
        """Take a step in the environment"""
        if self.game_over:
            return self._get_state(), 0.0, True, False, {}
        
        # Validate action
        if action <= 7 and self.plays_left <= 0:
            return self._get_state(), -50.0, True, False, {"error": "No plays left"}
        
        if (action == 8 or action == 9) and self.discards_left <= 0:
            return self._get_state(), -50.0, True, False, {"error": "No discards left"}
        
        # Execute action
        if action <= 7:  # Play card action
            # Store the card info before playing it
            card_played = str(self.hand[action]) if action < len(self.hand) else "Invalid"
            result = self._play_single_card(action)
            info = {
                "action_type": "play",
                "card_played": card_played,
                "result": result,
                "total_score": self.current_score
            }
            
            # Add hand type info if result is a tuple
            if isinstance(result, tuple) and len(result) >= 2:
                info["hand_type"] = result[1]
        elif action == 8:  # Discard worst card
            result = self._discard_worst_card(0)
            info = {
                "action_type": "discard_worst",
                "result": result,
                "total_score": self.current_score
            }
        elif action == 9:  # Discard second worst card
            result = self._discard_worst_card(1)
            info = {
                "action_type": "discard_second_worst",
                "result": result,
                "total_score": self.current_score
            }
        elif action == 10:  # Pass
            result = None
            info = {
                "action_type": "pass",
                "result": None,
                "total_score": self.current_score
            }
        else:
            return self._get_state(), -10.0, False, False, {"error": "Invalid action"}
        
        # Calculate reward
        reward = self._calculate_reward(action, result)
        
        # Check game end conditions
        if self.current_score >= self.blind_score:
            self.game_over = True
            self.won = True
            reward += 50.0  # Bonus for winning
        elif self.plays_left <= 0:
            self.game_over = True
            self.won = False
            reward -= 10.0  # Penalty for running out of plays
        
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
    
    print("🎰 Testing Redesigned Simplified Balatro Gym Environment")
    print("=" * 50)
    
    obs, info = env.reset()
    print(f"State size: {len(obs)}")
    print(f"Action space: {env.action_space.n}")
    print(f"Initial state: {obs}")
    
    total_reward = 0
    
    while not env.game_over:
        env.render()
        
        action = env.action_space.sample()
        print(f"Action: {action}")
        
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