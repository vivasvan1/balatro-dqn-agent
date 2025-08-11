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
import logging

# Import the card and hand classes from the original environment
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from base import BalatroCard, BalatroHand

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")


class ObservationState(np.ndarray):
    """
    Custom numpy ndarray subclass for Balatro observation state.
    This allows us to print the observation state in a more readable format.
    """

    def __new__(cls, input_array):
        # Convert input array to float32 and create the ndarray instance
        obj = np.asarray(input_array, dtype=np.float32).view(cls)
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return

    def decoded_observation_space(self):
        # Ensure the array is at least 22 elements (no strategic features)
        arr = np.asarray(self)
        # Defensive: pad with zeros if needed
        if arr.shape[0] < 22:
            arr = np.pad(arr, (0, 22 - arr.shape[0]), mode="constant")
        hand = []
        for i in range(0, 16, 2):
            rank = int(arr[i])
            suit = int(arr[i + 1])
            if rank == 0 and suit == 0:
                hand.append("--")
            else:
                hand.append(self._get_card_formatted(rank, suit, without_color=True))
        return {
            "hand": hand,  # 8 cards (formatted)
            "plays_left": float(arr[16]),
            "discards_left": float(arr[17]),
            "current_score": float(arr[18]),
            "blind_score": float(arr[19]),
            "game_over": bool(arr[20]),
            "progress_to_target": float(arr[21]),
        }

    def _get_card_formatted(self, rank, suit, without_color: bool = False):
        """
        Formats and returns a card with suit icons and colors.
        """
        # ANSI escape codes for colors
        RESET_COLOR = "\033[0m"
        BOLD = "\033[1m"
        BLACK = "\033[30m"
        WHITE = "\033[37m"
        RED = "\033[31m"
        PURPLE = "\033[35m"
        ORANGE = "\033[38;5;208m"
        # Mapping suit letters to Unicode icons
        suit_to_icon = {
            "S": f"{WHITE}♠{RESET_COLOR}",
            "D": f"{ORANGE}♦{RESET_COLOR}",
            "C": f"{PURPLE}♣{RESET_COLOR}",
            "H": f"{RED}♥{RESET_COLOR}",
        }
        # Map suit int to letter
        suit_map = {0: "H", 1: "D", 2: "C", 3: "S"}
        rank_map = {11: "J", 12: "Q", 13: "K", 14: "A"}
        rank_str = rank_map.get(rank, str(rank))
        suit_letter = suit_map.get(suit, "?")
        icon = suit_to_icon.get(suit_letter, suit_letter)
        if without_color:
            # Strip ANSI if requested
            icon = {
                "S": "♠",
                "D": "♦",
                "C": "♣",
                "H": "♥",
            }.get(suit_letter, suit_letter)
            return f"{rank}{icon}"
        formatted_card = f"{rank_str}{icon}"
        return f"{BOLD}{formatted_card}{RESET_COLOR}"

    def __str__(self):
        decoded = self.decoded_observation_space()

        return f"<Decoded Observation Space: {decoded}>"

    def __repr__(self):
        return self.__str__()


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
        # - card_count: number of cards to use (encoded 0-4 => 1-5 actual)
        # - card_1: index of first card (8 values, 0-7)
        # - card_2: index of second card (8 values, 0-7)
        # - card_3: index of third card (8 values, 0-7)
        # - card_4: index of fourth card (8 values, 0-7)
        # - card_5: index of fifth card (8 values, 0-7)
        # - priority: action priority/confidence (5 values, 0-4)
        # Total: 8 action heads with varying dimensions
        self.action_space = spaces.MultiDiscrete([2, 5, 8, 8, 8, 8, 8, 5])

        # Enhanced observation space (no strategic features):
        # - 8 cards in hand (rank + suit) = 16 values
        # - plays_left = 1 value
        # - discards_left = 1 value
        # - current_score = 1 value
        # - blind_score = 1 value
        # - game_over = 1 value
        # - progress_to_target = 1 value
        # Total: 22 values
        self.observation_space = spaces.Box(
            low=0, high=300, shape=(22,), dtype=np.float32
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

    def _get_state(self) -> ObservationState:
        """Get enhanced state as numpy array with strategic information"""
        # Encode hand (pad with zeros if less than 8 cards)
        hand_encoding = []
        for i in range(8):
            if i < len(self.hand):
                rank, suit = self._encode_card(self.hand[i])
                hand_encoding.extend([rank, suit])
            else:
                hand_encoding.extend([0, 0])

        # Additional features
        progress_to_target = self.current_score / self.blind_score

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
            ],
            dtype=np.float32,
        )

        return ObservationState(state)

    # Strategic features removed

    def _decode_multi_head_action(self, action: np.ndarray) -> Tuple[str, List[int]]:
        """Decode multi-head action into action_type and card_indices (no pass)"""
        # action is a numpy array with 8 values: [action_type, card_count(0-4), card_1, card_2, card_3, card_4, card_5, priority]
        action_type_idx = action[0]
        card_count = int(action[1]) + 1  # map 0-4 => 1-5
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
            reward += score_gained / 25.0
            # Bonus for winning hand types
            hand_bonuses = {
                "Royal Flush": 200,
                "Straight Flush": 150,
                "Four of a Kind": 100,
                "Full House": 80,
                "Flush": 60,
                "Straight": 40,
                "Three of a Kind": 30,
                "Two Pair": 15,
                "Pair": 5,
                "High Card": -10,
            }
            reward += hand_bonuses.get(hand_type, 0)
            # Penalty for playing weak hands if discards are available
            if hand_type in ["High Card", "Pair"] and self.discards_left > 0:
                reward -= 50.0
            # Small reward for getting closer to the blind score
            reward += 2.0 * (self.current_score / self.blind_score)
        elif action_type == "discard":
            if self.discards_left == 0:
                reward -= 100.0

        # Save last action for next step
        self.last_action = action_type

        # Check game end conditions
        if self.current_score >= self.blind_score:
            self.game_over = True
            self.won = True
            reward += 100.0  # Bonus for winning (increased from 50.0)
        elif self.plays_left <= 0:
            self.game_over = True
            self.won = False
            reward -= 50.0  # Penalty for running out of plays (increased from 20.0)

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
            return (
                self._get_state(),
                -10.0,
                False,
                False,
                {"error": "No cards selected"},
            )
        # Execute action
        if action_type == "play":
            result = self._play_hand(card_indices)
            score_gained, hand_type = result
            info = {
                "action_type": action_type,
                "card_indices": card_indices,
                "hand_type": hand_type,
                "score_gained": score_gained,
                "total_score": self.current_score,
                "cards_selected": card_indices,
            }
        else:
            result = self._discard_cards(card_indices)
            info = {
                "action_type": action_type,
                "cards_discarded": result,
                "total_score": self.current_score,
                "cards_selected": card_indices,
            }
        # Calculate reward
        reward = self._calculate_reward(action_type, result)

        info.update(
            {
                "plays_left": self.plays_left,
                "discards_left": self.discards_left,
                "game_over": self.game_over,
                "won": self.won,
            }
        )
        return self._get_state(), reward, self.game_over, False, info

    def render(self):
        """Render the current state"""
        print(f"Hand: {' '.join([str(card) for card in self.hand])}")
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

    logging.info("🎰 Testing OpenAI Five-style Balatro Gym Environment")
    logging.info("=" * 60)

    obs, info = env.reset()
    logging.info(f"State size: {len(obs)}")
    logging.info(f"Action space: {env.action_space}")
    logging.info(f"Initial state: {obs}")

    total_reward = 0

    while not env.game_over:
        env.render()

        # Sample a multi-head action
        action = env.action_space.sample()
        action_type, card_indices = env._decode_multi_head_action(action)

        # logging.info(f"Action: {action} -> {action_type} cards {card_indices}")

        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward

        logging.info(f"Reward: {reward:.2f}")
        # logging.info(f"Info: {info}")
        logging.info("-" * 50)

        if done:
            break

    env.render()
    logging.info(f"Total reward: {total_reward:.2f}")
    logging.info("=" * 60)
