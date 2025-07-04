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
    
    def __init__(self, blind_score: int = 1500):  # Increased score requirement
        super().__init__()
        
        # Game state
        self.blind_score = blind_score
        self.deck = self._create_deck()
        self.hand = []
        self.discarded_cards = []
        self.plays_left = 10  # Increased from 3 to 10
        self.discards_left = 10  # Increased from 3 to 10
        self.current_score = 0
        self.game_over = False
        self.won = False
        
        # Track hand quality progression
        self.previous_hand_types = []
        self.hand_quality_improvement_bonus = 0
        
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
        """Calculate a simple hand quality score (optimized)"""
        if not self.hand:
            return 0.0
        
        # Simple heuristic: count high cards and potential pairs
        high_cards = sum(1 for card in self.hand if card.rank in ['A', 'K', 'Q', 'J', '10'])
        rank_counts = {}
        for card in self.hand:
            rank_counts[card.rank] = rank_counts.get(card.rank, 0) + 1
        
        # Count pairs and better
        pairs = sum(1 for count in rank_counts.values() if count >= 2)
        three_of_kind = sum(1 for count in rank_counts.values() if count >= 3)
        
        # Simple quality score
        quality = high_cards * 2 + pairs * 10 + three_of_kind * 30
        return quality / 100.0  # Normalize
    
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
        """Calculate total number of valid actions (optimized)"""
        # Use cached combinations if available
        if not hasattr(self, '_cached_combinations'):
            from itertools import combinations
            self._cached_combinations = {}
            for r in range(1, 6):
                self._cached_combinations[r] = list(combinations(range(8), r))
        
        valid_combinations = 0
        for r in range(1, 6):
            valid_combinations += len(self._cached_combinations[r])
        
        return valid_combinations * 2
    
    def _decode_action(self, action: int) -> Tuple[str, List[int]]:
        """Decode action into action_type and card_indices (optimized)"""
        # Use cached combinations if available
        if not hasattr(self, '_cached_combinations'):
            from itertools import combinations
            self._cached_combinations = {}
            for r in range(1, 6):
                self._cached_combinations[r] = list(combinations(range(8), r))
        
        valid_combinations = 0
        for r in range(1, 6):
            valid_combinations += len(self._cached_combinations[r])
        
        if action < valid_combinations:
            action_type = "play"
            action_index = action
        else:
            action_type = "discard"
            action_index = action - valid_combinations
        
        card_indices = []
        current_index = 0
        
        for r in range(1, 6):
            combinations_r = self._cached_combinations[r]
            if current_index <= action_index < current_index + len(combinations_r):
                combo_index = action_index - current_index
                card_indices = list(combinations_r[combo_index])
                break
            current_index += len(combinations_r)
        
        # Prevent High Card plays by checking the hand type
        if action_type == "play" and len(card_indices) >= 3:  # Check all play actions, not just 5-card
            # Check if this would result in a High Card hand
            selected_cards = [self.hand[i] for i in card_indices if i < len(self.hand)]
            if len(selected_cards) >= 3:  # Check 3+ card hands
                balatro_hand = BalatroHand(selected_cards)
                hand_type, _, _, _ = balatro_hand.evaluate_hand()
                if hand_type == "High Card":
                    # Return pass instead of playing High Card
                    # Also add a penalty for attempting High Card
                    self.high_card_attempt_penalty = -30.0  # Increased penalty
                    return "pass", []
        
        # Reset penalty if not attempting High Card
        if hasattr(self, 'high_card_attempt_penalty'):
            delattr(self, 'high_card_attempt_penalty')
        
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
    
    def _evaluate_potential_hands(self) -> Dict[str, int]:
        """Evaluate what hands are possible with the current cards"""
        if not self.hand:
            return {}
        
        ranks = [card.rank for card in self.hand]
        suits = [card.suit for card in self.hand]
        
        # Count ranks and suits
        rank_counts = {}
        suit_counts = {}
        for rank in ranks:
            rank_counts[rank] = rank_counts.get(rank, 0) + 1
        for suit in suits:
            suit_counts[suit] = suit_counts.get(suit, 0) + 1
        
        potential_hands = {}
        
        # Check for potential hands
        max_rank_count = max(rank_counts.values()) if rank_counts else 0
        max_suit_count = max(suit_counts.values()) if suit_counts else 0
        
        # Potential for Three of a Kind
        if max_rank_count >= 2:  # Need 2 of same rank to potentially get 3
            potential_hands["Three of a Kind"] = 3
        
        # Potential for Four of a Kind
        if max_rank_count >= 3:  # Need 3 of same rank to potentially get 4
            potential_hands["Four of a Kind"] = 4
        
        # Potential for Full House
        if len(rank_counts) >= 2 and max_rank_count >= 2:
            potential_hands["Full House"] = 6
        
        # Potential for Flush
        if max_suit_count >= 3:  # Need 3 of same suit to potentially get flush
            potential_hands["Flush"] = 5
        
        # Potential for Straight
        rank_values = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
        numeric_ranks = [rank_values.get(rank, 0) for rank in ranks]
        numeric_ranks.sort()
        
        # Check for potential straight (need 3 consecutive or close)
        for i in range(len(numeric_ranks) - 2):
            if numeric_ranks[i+2] - numeric_ranks[i] <= 2:
                potential_hands["Straight"] = 4
                break
        
        return potential_hands
    
    def _assess_discard_quality(self, card_indices: List[int]) -> str:
        """Assess how good the discard decision was"""
        if not card_indices:
            return "no_cards"
        
        # Get the cards that were discarded
        discarded_cards = []
        for i in card_indices:
            if i < len(self.hand):
                discarded_cards.append(self.hand[i])
        
        if not discarded_cards:
            return "invalid"
        
        # Analyze discard quality
        ranks = [card.rank for card in discarded_cards]
        suits = [card.suit for card in discarded_cards]
        
        # Check if discarding high-value cards (generally bad)
        high_ranks = ['A', 'K', 'Q', 'J', '10']
        high_cards_discarded = sum(1 for rank in ranks if rank in high_ranks)
        
        # Check if discarding potential flush cards
        suit_counts = {}
        for suit in suits:
            suit_counts[suit] = suit_counts.get(suit, 0) + 1
        
        # Check if discarding potential straight cards
        rank_values = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
        numeric_ranks = [rank_values.get(rank, 0) for rank in ranks]
        numeric_ranks.sort()
        
        # Assess quality
        if high_cards_discarded >= 2:
            return "poor"  # Discarding too many high cards
        elif max(suit_counts.values()) >= 3:
            return "poor"  # Discarding potential flush
        elif len(numeric_ranks) >= 3:
            # Check for potential straight
            for i in range(len(numeric_ranks) - 2):
                if numeric_ranks[i+2] - numeric_ranks[i] <= 2:
                    return "poor"  # Discarding potential straight
        
        return "good"  # Smart discard
    
    def _calculate_reward(self, action_type: str, result: Any) -> float:
        """Calculate reward with simplified strategy: 1 High Card + 2 High Hands"""
        if self.game_over:
            if self.won:
                return 800.0  # Increased win reward for longer game
            else:
                return -300.0  # Increased loss penalty for longer game
        
        if action_type == "play":
            score_gained, hand_type = result
            
            # Track hand types played in this episode
            if not hasattr(self, 'episode_hand_types'):
                self.episode_hand_types = []
            self.episode_hand_types.append(hand_type)
            
            # Define high-level hands (Three of a Kind or better)
            high_hands = ["Three of a Kind", "Straight", "Flush", "Full House", "Four of a Kind", "Straight Flush", "Royal Flush"]
            
            # Check if this follows the winning pattern: 1 High Card + 2 High Hands
            high_card_count = self.episode_hand_types.count("High Card")
            high_hand_count = sum(1 for hand in self.episode_hand_types if hand in high_hands)
            total_hands = len(self.episode_hand_types)
            
            # Base reward - encourage taking actions
            reward = score_gained / 5.0  # Increased base reward for points
            reward += 5.0  # Positive reward for taking any action (prevents freezing)
            
            # REWARD: For building better hands through strategic play
            if hand_type in ["Three of a Kind", "Straight", "Flush", "Full House", "Four of a Kind", "Straight Flush", "Royal Flush"]:
                reward += 150.0  # Increased reward for high hands (longer game)
                
                # Extra reward if we're close to winning
                if self.current_score >= self.blind_score * 0.8:
                    reward += 75.0
            elif hand_type == "Two Pair":
                reward += 30.0  # Increased reward for two pair (longer game)
            elif hand_type == "Pair":
                reward += 8.0   # Small reward for pair (better than high card)
            
            # BONUS: For discovering and playing high hands
            if hand_type in high_hands:
                # General bonus for playing high hands
                reward += 40.0  # Base bonus for any high hand
                
                # Check if this is the first time playing this hand type in recent episodes
                if not hasattr(self, 'recent_high_hands'):
                    self.recent_high_hands = []
                
                if hand_type not in self.recent_high_hands:
                    reward += 80.0  # Big bonus for discovering new high hand types
                    self.recent_high_hands.append(hand_type)
                    
                    # Keep only last 10 unique high hands to encourage continued exploration
                    if len(self.recent_high_hands) > 10:
                        self.recent_high_hands = self.recent_high_hands[-10:]
                
                # BONUS: For consistency - reward playing the same good hand types repeatedly
                if len(self.episode_hand_types) >= 2:
                    if hand_type == self.episode_hand_types[-2]:  # Same hand type as last play
                        reward += 30.0  # Consistency bonus
            
            # PENALTY: If we're not following the winning pattern
            else:
                if hand_type == "High Card":
                    reward -= 200.0  # Stronger penalty for High Card to make it more consistent
                elif hand_type in ["Pair"]:
                    # Less penalty for pairs since they're common, but encourage building better
                    reward -= 10.0  # Reduced penalty for pairs
                elif hand_type in ["Two Pair"]:
                    reward -= 5.0  # Small penalty for two pair (it's actually decent)
                
                # PENALTY: For regression - playing worse hands after good ones
                if len(self.episode_hand_types) >= 2:
                    hand_rank = {
                        "High Card": 0, "Pair": 1, "Two Pair": 2, "Three of a Kind": 3,
                        "Straight": 4, "Flush": 5, "Full House": 6, "Four of a Kind": 7,
                        "Straight Flush": 8, "Royal Flush": 9
                    }
                    
                    current_rank = hand_rank.get(hand_type, 0)
                    previous_rank = hand_rank.get(self.episode_hand_types[-2], 0)
                    
                    if current_rank < previous_rank:
                        regression_penalty = (previous_rank - current_rank) * 20
                        reward -= regression_penalty  # Penalty for playing worse hands
            
            # PENALTY: If we're not making progress toward the target
            if self.current_score < self.blind_score * 0.3:
                reward -= 10.0
            
            # PENALTY: If we have discards available but play low hands
            if self.discards_left > 0 and hand_type in ["High Card", "Pair", "Two Pair"]:
                if hand_type == "High Card":
                    reward -= 50.0  # Much stronger penalty for High Card with discards available
                else:
                    reward -= 20.0
            
            # PENALTY: If we have high hands available but play low hands
            if hasattr(self, 'current_hand') and self.current_hand:
                potential_hands = self._evaluate_potential_hands()
                if potential_hands:
                    best_potential = max(potential_hands.values())
                    if best_potential >= 5 and hand_type in ["High Card", "Pair"]:  # Flush or better available
                        reward -= 30.0  # Reduced penalty for not playing available high hands
                    elif best_potential >= 3 and hand_type == "High Card":  # Three of a Kind or better available
                        reward -= 15.0  # Reduced penalty for not playing available medium hands
            
            # PENALTY: For attempting to play High Card (even if prevented)
            if hasattr(self, 'high_card_attempt_penalty'):
                reward += self.high_card_attempt_penalty
            
            # BONUS: For improving hand quality over consecutive plays
            if len(self.episode_hand_types) >= 2:
                hand_rank = {
                    "High Card": 0, "Pair": 1, "Two Pair": 2, "Three of a Kind": 3,
                    "Straight": 4, "Flush": 5, "Full House": 6, "Four of a Kind": 7,
                    "Straight Flush": 8, "Royal Flush": 9
                }
                
                current_rank = hand_rank.get(hand_type, 0)
                previous_rank = hand_rank.get(self.episode_hand_types[-2], 0)
                
                if current_rank > previous_rank:
                    improvement_bonus = (current_rank - previous_rank) * 15
                    reward += improvement_bonus  # Bonus for improving hand quality
            
            # Penalty for passing when actions are available
            if action_type == "pass":
                if self.plays_left > 0 or self.discards_left > 0:
                    return -10.0  # Penalty for passing when actions available
            
            return reward
        
        elif action_type == "discard":
            # Simple discard reward aligned with the 1 High Card + 2 High Hands strategy
            reward = 10.0  # Increased base reward for discarding (encourage action)
            
            # Bonus for discarding when we have low hands (encourage building better hands)
            if hasattr(self, 'episode_hand_types') and self.episode_hand_types:
                last_hand = self.episode_hand_types[-1] if self.episode_hand_types else "High Card"
                if last_hand in ["High Card", "Pair"]:
                    reward += 15.0  # Increased bonus for discarding low hands
                elif last_hand == "Two Pair":
                    reward += 8.0   # Smaller bonus for discarding two pair
            
            # Bonus for discarding when we have potential for high hands
            if hasattr(self, 'current_hand') and self.current_hand:
                potential_hands = self._evaluate_potential_hands()
                if potential_hands:
                    best_potential = max(potential_hands.values())
                    if best_potential >= 5:  # Flush or better
                        reward += 25.0  # Big bonus for discarding to pursue flush or better
                    elif best_potential >= 3:  # Three of a Kind or better
                        reward += 15.0  # Bonus for discarding to pursue three of a kind
            
            return reward
        
        
        return 0.0
    
    def reset(self, seed=None, options=None):
        """Reset the environment"""
        super().reset(seed=seed)
        
        self.deck = self._create_deck()
        self._deal_hand()
        self.discarded_cards = []
        self.plays_left = 10  # Increased from 3 to 10
        self.episode_hand_types = []  # Reset episode hand tracking
        self.discards_left = 10  # Increased from 3 to 10
        self.current_score = 0
        self.game_over = False
        self.won = False
        
        # Reset hand quality tracking
        self.previous_hand_types = []
        self.hand_quality_improvement_bonus = 0
        
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
            
            # Assess discard quality
            discard_quality = self._assess_discard_quality(card_indices)
            
            info = {
                "action_type": action_type,
                "cards_discarded": result,
                "total_score": self.current_score,
                "cards_selected": card_indices,
                "discard_quality": discard_quality
            }
        
        # Calculate reward
        reward = self._calculate_reward(action_type, result)
        
        # Add quality-based adjustments for discards
        if action_type == "discard" and "discard_quality" in info:
            discard_quality = info["discard_quality"]
            if discard_quality == "good":
                reward += 5.0  # Bonus for smart discarding
            elif discard_quality == "poor":
                reward -= 3.0  # Penalty for poor discarding
        
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