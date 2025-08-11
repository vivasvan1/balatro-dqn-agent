import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import List, Tuple, Dict, Any, Optional
import random
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from base import BalatroCard, BalatroHand

class BalatroGymEnv(gym.Env):
    """
    Balatro Gym Environment that properly simulates the game mechanics
    """

    def __init__(self, blind_score: int = 300):
        super().__init__()

        # Game state
        self.blind_score = blind_score  # Score needed to beat the blind
        self.deck = self._create_deck()
        self.hand = []  # 8 cards in hand
        self.discarded_cards = []  # Track discarded cards
        self.plays_left = 3
        self.discards_left = 3
        self.current_score = 0
        self.game_over = False
        self.won = False

        # Action space:
        # - Action type (0=play, 1=discard)
        # - Card selection (only combinations with 1-5 cards)
        # We'll calculate the total valid actions
        self.action_space = spaces.Discrete(self._calculate_valid_actions())

        # Observation space:
        # - 8 cards in hand (rank + suit) = 16 values
        # - 52 discarded cards (rank + suit) = 104 values (max possible discards)
        # - 52 remaining deck cards (rank + suit) = 104 values
        # - plays_left = 1 value
        # - discards_left = 1 value
        # - current_score = 1 value
        # - blind_score = 1 value
        # - game_over = 1 value
        # Total: 228 values
        self.observation_space = spaces.Box(
            low=0, high=300, shape=(228,), dtype=np.float32
        )

    def _create_deck(self) -> List[BalatroCard]:
        """Create a standard 52-card deck"""
        ranks = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
        suits = ["H", "D", "C", "S"]  # Hearts, Diamonds, Clubs, Spades
        deck = [BalatroCard(rank, suit) for rank in ranks for suit in suits]
        random.shuffle(deck)
        return deck

    def _deal_hand(self):
        """Deal 8 cards to hand"""
        if len(self.deck) < 8:
            self.deck = self._create_deck()  # Reshuffle if needed

        self.hand = self.deck[:8]
        self.deck = self.deck[8:]

    def _encode_card(self, card: BalatroCard) -> Tuple[int, int]:
        """Encode card as (rank, suit) integers"""
        rank = card.get_rank_value()
        suit_map = {"H": 0, "D": 1, "C": 2, "S": 3}
        suit = suit_map[card.suit]
        return rank, suit

    def _get_state(self) -> np.ndarray:
        """Get current state as numpy array"""
        # Encode hand (pad with zeros if less than 8 cards)
        hand_encoding = []
        for i in range(8):
            if i < len(self.hand):
                rank, suit = self._encode_card(self.hand[i])
                hand_encoding.extend([rank, suit])
            else:
                hand_encoding.extend([0, 0])  # Padding

        # Encode discarded cards (pad with zeros if less than 52 cards)
        discarded_encoding = []
        for i in range(52):
            if i < len(self.discarded_cards):
                rank, suit = self._encode_card(self.discarded_cards[i])
                discarded_encoding.extend([rank, suit])
            else:
                discarded_encoding.extend([0, 0])  # Padding

        # Encode remaining deck cards (pad with zeros if less than 52 cards)
        deck_encoding = []
        for i in range(52):
            if i < len(self.deck):
                rank, suit = self._encode_card(self.deck[i])
                deck_encoding.extend([rank, suit])
            else:
                deck_encoding.extend([0, 0])  # Padding

        # Create state vector
        state = np.array(
            [
                *hand_encoding,  # 16 values (8 cards * 2 each)
                *discarded_encoding,  # 104 values (52 cards * 2 each)
                *deck_encoding,  # 104 values (52 cards * 2 each)
                self.plays_left,  # 1 value
                self.discards_left,  # 1 value
                self.current_score,  # 1 value
                self.blind_score,  # 1 value
                int(self.game_over),  # 1 value
            ],
            dtype=np.float32,
        )

        # Ensure correct shape
        assert len(state) == 229, f"Expected 229 values, got {len(state)}"

        return state

    def _calculate_valid_actions(self) -> int:
        """Calculate total number of valid actions (1-5 cards per action)"""
        from itertools import combinations

        # Count valid card combinations (1-5 cards from 8 cards)
        valid_combinations = 0
        for r in range(1, 6):  # 1 to 5 cards
            valid_combinations += len(list(combinations(range(8), r)))

        # Multiply by 2 for play/discard actions
        return valid_combinations * 2

    def _decode_action(self, action: int) -> Tuple[str, List[int]]:
        """Decode action into action_type and card_indices"""
        from itertools import combinations

        # Calculate how many valid combinations there are
        valid_combinations = 0
        for r in range(1, 6):  # 1 to 5 cards
            valid_combinations += len(list(combinations(range(8), r)))

        # Determine action type
        if action < valid_combinations:
            action_type = "play"
            action_index = action
        else:
            action_type = "discard"
            action_index = action - valid_combinations

        # Find the card combination for this action index
        card_indices = []
        current_index = 0

        for r in range(1, 6):  # 1 to 5 cards
            combinations_r = list(combinations(range(8), r))
            if current_index <= action_index < current_index + len(combinations_r):
                # Found the right combination size
                combo_index = action_index - current_index
                card_indices = list(combinations_r[combo_index])
                break
            current_index += len(combinations_r)

        return action_type, card_indices

    def _play_hand(self, card_indices: List[int]) -> Tuple[int, str]:
        """Play selected cards and return score and hand type"""
        if not card_indices:
            return 0, "Invalid"

        # Limit to 5 cards maximum
        if len(card_indices) > 5:
            card_indices = card_indices[:5]

        # Get selected cards
        selected_cards = [self.hand[i] for i in card_indices]

        # Evaluate hand
        balatro_hand = BalatroHand(selected_cards)
        hand_type, base_chips, multiplier, card_chips = balatro_hand.evaluate_hand()

        # Calculate total score: (base_chips + card_chips) * multiplier
        total_score = (base_chips + card_chips) * multiplier

        # Get the cards to be played before removing them
        cards_played = [self.hand[i] for i in card_indices if i < len(self.hand)]

        # Add played cards to the discarded pile (they're removed from the game)
        self.discarded_cards.extend(cards_played)

        # Remove played cards from hand
        self.hand = [card for i, card in enumerate(self.hand) if i not in card_indices]

        # Draw replacement cards
        cards_to_draw = len(card_indices)
        if len(self.deck) < cards_to_draw:
            self.deck.extend(self._create_deck())  # Add new deck if needed

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

        # Limit to 5 cards maximum
        if len(card_indices) > 5:
            card_indices = card_indices[:5]

        # Get the cards to be discarded before removing them
        cards_to_discard = [self.hand[i] for i in card_indices if i < len(self.hand)]

        # Add discarded cards to the discarded pile
        self.discarded_cards.extend(cards_to_discard)

        # Remove discarded cards from hand
        self.hand = [card for i, card in enumerate(self.hand) if i not in card_indices]

        # Draw replacement cards
        cards_to_draw = len(card_indices)
        if len(self.deck) < cards_to_draw:
            self.deck.extend(self._create_deck())  # Add new deck if needed

        new_cards = self.deck[:cards_to_draw]
        self.deck = self.deck[cards_to_draw:]
        self.hand.extend(new_cards)

        self.discards_left -= 1

        return len(card_indices)

    def _calculate_reward(self, action_type: str, result: Any) -> float:
        """Calculate reward based on action and result"""
        if self.game_over:
            if self.won:
                return 100.0  # Big reward for winning
            else:
                return -50.0  # Penalty for losing

        if action_type == "play":
            score_gained, hand_type = result

            # Base reward is the score gained
            reward = score_gained / 10.0  # Scale down

            # Bonus for better hands
            hand_bonuses = {
                "Royal Flush": 50,
                "Straight Flush": 30,
                "Four of a Kind": 20,
                "Full House": 15,
                "Flush": 10,
                "Straight": 8,
                "Three of a Kind": 5,
                "Two Pair": 3,
                "Pair": 1,
                "High Card": 0,
            }
            reward += hand_bonuses.get(hand_type, 0)

            # Progress toward goal
            progress_reward = (self.current_score / self.blind_score) * 10
            reward += progress_reward

            return reward

        elif action_type == "discard":
            # Small penalty for discarding, but necessary for strategy
            reward = -1.0

            # Additional penalty if no discards are left (this should be caught earlier, but just in case)
            if self.discards_left <= 0:
                reward -= 20.0  # Extra penalty for attempting invalid discard

            return reward

        return 0.0

    def reset(self, seed=None, options=None):
        """Reset the environment"""
        super().reset(seed=seed)

        # Reset game state
        self.deck = self._create_deck()
        self._deal_hand()
        self.discarded_cards = []  # Reset discarded cards
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

        # Decode action
        action_type, card_indices = self._decode_action(action)

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
            # Get the cards before they're removed
            cards_played = [
                str(self.hand[i]) for i in card_indices if i < len(self.hand)
            ]
            result = self._play_hand(card_indices)
            score_gained, hand_type = result
            info = {
                "action_type": action_type,
                "cards_played": cards_played,
                "hand_type": hand_type,
                "score_gained": score_gained,
                "total_score": self.current_score,
                "cards_selected": card_indices,
            }
        else:  # discard
            result = self._discard_cards(card_indices)
            info = {
                "action_type": action_type,
                "cards_discarded": result,
                "total_score": self.current_score,
                "cards_selected": card_indices,
            }

        # Calculate reward
        reward = self._calculate_reward(action_type, result)

        # Check game end conditions
        if self.current_score >= self.blind_score:
            self.game_over = True
            self.won = True
            reward += 100.0  # Win bonus
        elif self.plays_left <= 0:
            self.game_over = True
            self.won = False
            reward -= 50.0  # Loss penalty

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
        print(f"Hand: {[str(card) for card in self.hand]}")
        print(f"Discarded: {[str(card) for card in self.discarded_cards]}")
        print(f"Deck remaining: {len(self.deck)} cards")
        print(f"Score: {self.current_score}/{self.blind_score}")
        print(f"Plays left: {self.plays_left}, Discards left: {self.discards_left}")
        if self.game_over:
            print(f"Game Over! {'WON' if self.won else 'LOST'}")
        print("-" * 50)


# Example usage and testing
if __name__ == "__main__":
    env = BalatroGymEnv(blind_score=300)

    print("🎰 Testing Balatro Gym Environment V2")
    print("=" * 50)

    obs, info = env.reset()
    total_reward = 0

    while not env.game_over:
        env.render()

        # Take random action
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
