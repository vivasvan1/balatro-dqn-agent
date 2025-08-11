from typing import List, Tuple

class BalatroCard:
    """Represents a single Balatro card with chip value"""

    def __init__(self, rank: str, suit: str):
        self.rank = rank
        self.suit = suit
        self.card_str = f"{rank}{suit}"

    def get_chip_value(self) -> int:
        """Get the chip value of the card"""
        if self.rank == "A":
            return 11
        elif self.rank in ["K", "Q", "J", "10"]:
            return 10
        else:
            return int(self.rank)

    def get_rank_value(self) -> int:
        """Get numeric rank for poker hand evaluation (2-14)"""
        rank_map = {
            "2": 2,
            "3": 3,
            "4": 4,
            "5": 5,
            "6": 6,
            "7": 7,
            "8": 8,
            "9": 9,
            "10": 10,
            "J": 11,
            "Q": 12,
            "K": 13,
            "A": 14,
        }
        return rank_map[self.rank]

    def _get_card_formatted(self):
        """
        Formats and prints a card with suit icons and colors.
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

        # Format each card with its icon and specified black color
        rank = self.rank if hasattr(self, "rank") else self[0:-1]
        suit = self.suit if hasattr(self, "suit") else self[-1]
        icon = suit_to_icon.get(suit, suit)  # Fallback to letter if suit not found
        # Apply black color to the icon
        colored_icon = f"{BLACK}{icon}{RESET_COLOR}"
        formatted_card = f"{rank}{colored_icon}"

        # Print the hand with the specified prefix in bold blue for emphasis
        return f"{BOLD}{formatted_card}{RESET_COLOR}"

    def __str__(self):
        return self._get_card_formatted()

    def __repr__(self):
        return self.card_str


class BalatroHand:
    """Evaluates poker hands with Balatro scoring"""

    # Base hand values (chips, multiplier)
    HAND_VALUES = {
        "High Card": (5, 1),
        "Pair": (10, 2),
        "Two Pair": (20, 2),
        "Three of a Kind": (30, 3),
        "Straight": (30, 4),
        "Flush": (35, 4),
        "Full House": (40, 4),
        "Four of a Kind": (60, 7),
        "Straight Flush": (100, 8),
        "Royal Flush": (100, 8),
    }

    def __init__(self, cards: List[BalatroCard]):
        self.cards = cards
        self.ranks = [card.get_rank_value() for card in cards]
        self.suits = [card.suit for card in cards]

    def evaluate_hand(self) -> Tuple[str, int, int, int]:
        """
        Evaluate poker hand and return (hand_type, base_chips, multiplier, card_chips)
        """
        if not (1 <= len(self.cards) <= 5):
            return "Invalid", 0, 1, 0

        # Calculate card chips
        card_chips = sum(card.get_chip_value() for card in self.cards)

        if len(self.cards) == 1:
            return "High Card", 5, 1, card_chips

        # For hands with 2-5 cards, evaluate poker hand
        hand_type = self._get_poker_hand_type()
        base_chips, multiplier = self.HAND_VALUES[hand_type]

        return hand_type, base_chips, multiplier, card_chips

    def _get_poker_hand_type(self) -> str:
        """Determine the poker hand type"""
        if len(self.cards) < 2:
            return "High Card"

        # Sort ranks for easier evaluation
        sorted_ranks = sorted(self.ranks)
        rank_counts = {}
        for rank in sorted_ranks:
            rank_counts[rank] = rank_counts.get(rank, 0) + 1

        # Check for flush (only if 5 cards)
        is_flush = len(self.cards) == 5 and len(set(self.suits)) == 1

        # Check for straight (only if 5 cards)
        is_straight = False
        if len(self.cards) == 5:
            is_straight = (
                max(sorted_ranks) - min(sorted_ranks) == 4
                and len(set(sorted_ranks)) == 5
            )
            # Special case for Ace-low straight (A-2-3-4-5)
            if sorted_ranks == [2, 3, 4, 5, 14]:
                is_straight = True

        # Determine hand type
        if len(self.cards) == 5:
            if is_straight and is_flush:
                if sorted_ranks == [10, 11, 12, 13, 14]:  # Royal flush
                    return "Royal Flush"
                else:  # Straight flush
                    return "Straight Flush"

            if is_flush:
                return "Flush"

            if is_straight:
                return "Straight"

        # Four of a kind
        if 4 in rank_counts.values():
            return "Four of a Kind"

        # Full house (only possible with 5 cards)
        if (
            len(self.cards) == 5
            and 3 in rank_counts.values()
            and 2 in rank_counts.values()
        ):
            return "Full House"

        # Three of a kind
        if 3 in rank_counts.values():
            return "Three of a Kind"

        # Two pair
        pairs = [r for r, count in rank_counts.items() if count == 2]
        if len(pairs) == 2:
            return "Two Pair"

        # One pair
        if 2 in rank_counts.values():
            return "Pair"

        # High card
        return "High Card"

