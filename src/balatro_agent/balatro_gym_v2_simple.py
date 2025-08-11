#!/usr/bin/env python3
"""
Simplified Balatro Gym Environment with OpenAI Five-style multi-head actions
Uses separate action heads for action type, card selection, etc.
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
    Simplified Balatro Gym Environment with OpenAI Five-style multi-head actions
    Uses separate action heads for better training stability
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

        # OpenAI Five-style multi-head action space (no 'pass' action):
        # - action_type: 0=play, 1=discard (2 values)
        # - card_count: 1-5 cards to select (5 values)
        # - card_1: index of first card (8 values, 0-7)
        # - card_2: index of second card (8 values, 0-7, or 0 if not used)
        # - card_3: index of third card (8 values, 0-7, or 0 if not used)
        # - card_4: index of fourth card (8 values, 0-7, or 0 if not used)
        # - card_5: index of fifth card (8 values, 0-7, or 0 if not used)
        # - priority: action priority/confidence (5 values, 0-4)
        # Total: 7 action heads with varying dimensions
        self.action_space = spaces.MultiDiscrete([2, 5, 8, 8, 8, 8, 8, 5])

        # Enhanced observation space:
        # - 8 cards in hand (rank + suit) = 16 values
        # - plays_left = 1 value
        # - discards_left = 1 value
        # - current_score = 1 value
        # - blind_score = 1 value
        # - game_over = 1 value
        # - progress_to_target = 1 value
        # - hand_quality = 1 value
        # - strategic_features = 8 values
        # Total: 31 values
        self.observation_space = spaces.Box(
            low=0, high=300, shape=(31,), dtype=np.float32
        )

    def _create_deck(self) -> List[BalatroCard]:
        """Create a standard 52-card deck"""
        ranks = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
        suits = ["H", "D", "C", "S"]
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
        suit_map = {"H": 0, "D": 1, "C": 2, "S": 3}
        suit = suit_map[card.suit]
        return rank, suit

    def _calculate_hand_quality(self) -> float:
        """Calculate a comprehensive hand quality score with strategic considerations"""
        if not self.hand:
            return 0.0

        # Calculate potential scores for all possible hand combinations
        max_score = 0
        from itertools import combinations

        for r in range(1, min(6, len(self.hand) + 1)):
            for combo in combinations(self.hand, r):
                balatro_hand = BalatroHand(list(combo))
                hand_type, base_chips, multiplier, card_chips = (
                    balatro_hand.evaluate_hand()
                )
                total_score = (base_chips + card_chips) * multiplier
                max_score = max(max_score, total_score)

        # Add strategic bonuses for hand potential
        strategic_bonus = self._calculate_hand_potential_bonus()

        return (max_score + strategic_bonus) / 100.0  # Normalize

    def _calculate_hand_potential_bonus(self) -> float:
        """Calculate bonus for hand potential (flush, straight opportunities)"""
        if not self.hand:
            return 0.0

        bonus = 0.0

        # Count cards by suit
        suit_counts = {"H": 0, "D": 0, "C": 0, "S": 0}
        for card in self.hand:
            suit_counts[card.suit] += 1

        # Flush potential
        max_suit_count = max(suit_counts.values())
        if max_suit_count >= 4:
            bonus += 50.0  # Very strong flush potential
        elif max_suit_count == 3:
            bonus += 25.0  # Good flush potential
        elif max_suit_count == 2:
            bonus += 10.0  # Some flush potential

        # Count cards by rank
        rank_counts = {}
        for card in self.hand:
            rank = card.get_rank_value()
            rank_counts[rank] = rank_counts.get(rank, 0) + 1

        # Pair/three-of-a-kind potential
        for rank, count in rank_counts.items():
            if count >= 3:
                bonus += 30.0  # Three of a kind potential
            elif count == 2:
                bonus += 15.0  # Pair potential

        # High card potential
        high_cards = sum(1 for card in self.hand if card.get_rank_value() >= 10)
        if high_cards >= 3:
            bonus += 10.0  # Good high card potential

        return bonus

    def _get_state(self) -> np.ndarray:
        """Get enhanced state as numpy array with strategic information"""
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
        print(f"Progress to target: {progress_to_target}")
        
        print(f"Calculating hand quality...")
        hand_quality = self._calculate_hand_quality()
        print(f"Hand quality: {hand_quality}")

        # Strategic features for better decision making
        strategic_features = self._calculate_strategic_features()

        # Create enhanced state vector
        state = np.array(
            [
                *hand_encoding,  # 16 values (8 cards * 2 each)
                self.plays_left,  # 1 value
                self.discards_left,  # 1 value
                self.current_score,  # 1 value
                self.blind_score,  # 1 value
                int(self.game_over),  # 1 value
                progress_to_target,  # 1 value
                hand_quality,  # 1 value
                *strategic_features,  # strategic features
            ],
            dtype=np.float32,
        )

        return state

    def _calculate_strategic_features(self) -> List[float]:
        """Calculate strategic features to help the agent make better decisions"""
        if not self.hand:
            return [0.0] * 8  # Return zeros if no hand

        features = []

        # 1. Flush potential (0-1 scale)
        suit_counts = {"H": 0, "D": 0, "C": 0, "S": 0}
        for card in self.hand:
            suit_counts[card.suit] += 1
        max_suit_count = max(suit_counts.values())
        flush_potential = min(max_suit_count / 5.0, 1.0)  # Normalize to 0-1
        features.append(flush_potential)

        # 2. Three-of-a-kind potential (0-1 scale)
        rank_counts = {}
        for card in self.hand:
            rank = card.get_rank_value()
            rank_counts[rank] = rank_counts.get(rank, 0) + 1
        max_rank_count = max(rank_counts.values()) if rank_counts else 0
        three_kind_potential = min(max_rank_count / 3.0, 1.0)
        features.append(three_kind_potential)

        # 3. High card ratio (0-1 scale)
        high_cards = sum(1 for card in self.hand if card.get_rank_value() >= 10)
        high_card_ratio = high_cards / len(self.hand)
        features.append(high_card_ratio)

        # 4. Current best hand score (normalized)
        from itertools import combinations

        max_score = 0
        for r in range(1, min(6, len(self.hand) + 1)):
            for combo in combinations(self.hand, r):
                balatro_hand = BalatroHand(list(combo))
                hand_type, base_chips, multiplier, card_chips = (
                    balatro_hand.evaluate_hand()
                )
                total_score = (base_chips + card_chips) * multiplier
                max_score = max(max_score, total_score)
        normalized_best_score = min(max_score / 100.0, 1.0)  # Normalize to 0-1
        features.append(normalized_best_score)

        # 5. Discard urgency (0-1 scale) - higher when we have bad hands and few discards left
        if self.discards_left > 0:
            if max_score < 30:  # Bad hand
                discard_urgency = 1.0 - (
                    self.discards_left / 3.0
                )  # Higher urgency with fewer discards
            else:
                discard_urgency = 0.0
        else:
            discard_urgency = 0.0
        features.append(discard_urgency)

        # 6. Play urgency (0-1 scale) - higher when we have good hands and few plays left
        if self.plays_left > 0:
            if max_score > 50:  # Good hand
                play_urgency = 1.0 - (
                    self.plays_left / 3.0
                )  # Higher urgency with fewer plays
            else:
                play_urgency = 0.0
        else:
            play_urgency = 0.0
        features.append(play_urgency)

        # 7. Hand diversity (0-1 scale) - measures how diverse the hand is
        unique_ranks = len(set(card.get_rank_value() for card in self.hand))
        unique_suits = len(set(card.suit for card in self.hand))
        diversity = (unique_ranks + unique_suits) / 12.0  # Normalize to 0-1
        features.append(diversity)

        # 8. Strategic value (0-1 scale) - overall strategic value of the hand
        strategic_value = (
            flush_potential
            + three_kind_potential
            + high_card_ratio
            + normalized_best_score
        ) / 4.0
        features.append(strategic_value)

        return features

    def _decode_multi_head_action(self, action: np.ndarray) -> Tuple[str, List[int]]:
        """Decode multi-head action into action_type and card_indices (no pass)"""
        # action is a numpy array with 8 values: [action_type, card_count, card_1, card_2, card_3, card_4, card_5, priority]
        action_type_idx = action[0]
        card_count = action[1]
        card_indices = []
        # Map action type
        if action_type_idx == 0:
            action_type = "play"
        else:
            action_type = "discard"
        # Extract card indices (only use the number specified by card_count)
        for i in range(2, 2 + card_count):
            card_idx = action[i]
            if card_idx < len(self.hand) and card_idx not in card_indices:
                card_indices.append(card_idx)
        # Ensure we have valid cards
        if not card_indices:
            if len(self.hand) > 0:
                card_indices = [random.randint(0, len(self.hand) - 1)]
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
        reward = 0.0

        if action_type == "play":
            score_gained, hand_type = result
            # Reward for score progress
            reward += (score_gained / 25.0)
            # Bonus for winning hand types
            hand_bonuses = {
                "Royal Flush": 200, "Straight Flush": 150, "Four of a Kind": 100,
                "Full House": 80, "Flush": 60, "Straight": 40,
                "Three of a Kind": 30, "Two Pair": 15, "Pair": 5, "High Card": -10
            }
            reward += hand_bonuses.get(hand_type, 0)
            # Penalty for playing weak hands if discards are available
            if hand_type in ["High Card", "Pair"] and score_gained < 30 and self.discards_left > 0:
                reward -= 50.0
            # Small reward for getting closer to the blind score
            reward += 2.0 * (self.current_score / self.blind_score)
        elif action_type == "discard":
            # Penalize discarding unless the hand is truly bad
            if self._calculate_hand_quality() < 0.2:
                reward -= 5.0  # Mild penalty for discarding bad hands
            else:
                reward -= 20.0  # Strong penalty for discarding decent hands
            # Penalty for discarding too many times in a row
            if hasattr(self, 'last_action') and self.last_action == "discard":
                reward -= 10.0
            # Small bonus for discarding when plays_left is low
            if self.plays_left == 1:
                reward += 5.0
        
        # Save last action for next step
        self.last_action = action_type
        return reward

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
        """Take a step in the environment (no pass action)"""
        if self.game_over:
            return self._get_state(), 0.0, True, False, {}
        # Decode multi-head action
        action_type, card_indices = self._decode_multi_head_action(action)
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
            reward += 50.0  # Bonus for winning
        elif self.plays_left <= 0:
            self.game_over = True
            self.won = False
            reward -= 20.0  # Penalty for running out of plays
        elif self.discards_left <= 0:
            # Penalty for running out of discards (encourages strategic discarding)
            reward -= 15.0
            # Additional penalty if we still have weak hands and no discards left
            if hasattr(self, 'hand') and self.hand:
                from itertools import combinations
                max_score = 0
                for r in range(1, min(6, len(self.hand) + 1)):
                    for combo in combinations(self.hand, r):
                        balatro_hand = BalatroHand(list(combo))
                        hand_type, base_chips, multiplier, card_chips = balatro_hand.evaluate_hand()
                        total_score = (base_chips + card_chips) * multiplier
                        max_score = max(max_score, total_score)
                
                if max_score < 30:
                    reward -= 10.0  # Extra penalty for running out of discards with bad hands
        
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

    def get_action_mask(self):
        """
        Returns a binary mask for each action head indicating valid actions.
        Returns a list of masks, one for each action head.
        """
        masks = []
        # Action type mask: [play, discard]
        action_type_mask = [1, 1]
        if self.plays_left <= 0:
            action_type_mask[0] = 0  # Can't play
        if self.discards_left <= 0:
            action_type_mask[1] = 0  # Can't discard
        masks.append(action_type_mask)
        # Card count mask: [1, 2, 3, 4, 5]
        card_count_mask = [1, 1, 1, 1, 1]
        masks.append(card_count_mask)
        # Card index masks: each card can be selected if it exists
        for i in range(5):
            card_mask = [1] * 8
            for j in range(8):
                if j >= len(self.hand):
                    card_mask[j] = 0
            masks.append(card_mask)
        # Priority mask: [0, 1, 2, 3, 4]
        priority_mask = [1, 1, 1, 1, 1]
        masks.append(priority_mask)
        return masks


# Example usage
if __name__ == "__main__":
    env = BalatroGymEnvSimple(blind_score=300)
    
    print("🎰 Testing OpenAI Five-style Balatro Gym Environment")
    print("=" * 60)
    
    obs, info = env.reset()
    print(f"State size: {len(obs)}")
    print(f"Action space: {env.action_space}")
    print(f"Initial state: {obs}")
    
    total_reward = 0
    
    while not env.game_over:
        env.render()
        
        # Sample a multi-head action
        action = env.action_space.sample()
        action_type, card_indices = env._decode_multi_head_action(action)
        
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
    print("=" * 60)
