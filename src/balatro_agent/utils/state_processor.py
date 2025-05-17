import numpy as np

class StateProcessor:
    def __init__(self):
        # Define the size of different state components
        self.hand_size = 8  # Maximum number of cards in hand
        self.joker_slots = 5  # Maximum number of joker slots
        self.card_features = 4  # Suit, rank, enhancement, edition
        self.joker_features = 3  # Type, level, edition
        
        # Calculate total state size
        self.state_size = (
            self.hand_size * self.card_features +  # Hand cards
            self.joker_slots * self.joker_features +  # Jokers
            3  # Additional game state info (chips, mult, blind)
        )
    
    def process_state(self, game_state):
        """
        Convert the raw game state from Balatro into a fixed-size numpy array.
        
        Args:
            game_state (dict): Raw game state from Balatro containing:
                - hand: List of cards in hand
                - jokers: List of active jokers
                - chips: Current chips
                - mult: Current multiplier
                - blind: Current blind info
        
        Returns:
            np.ndarray: Processed state vector
        """
        # Initialize state vector
        state = np.zeros(self.state_size)
        idx = 0
        
        # Process hand cards
        for i in range(self.hand_size):
            if i < len(game_state['hand']):
                card = game_state['hand'][i]
                state[idx:idx+self.card_features] = [
                    card['suit'],
                    card['rank'],
                    card['enhancement'],
                    card['edition']
                ]
            idx += self.card_features
        
        # Process jokers
        for i in range(self.joker_slots):
            if i < len(game_state['jokers']):
                joker = game_state['jokers'][i]
                state[idx:idx+self.joker_features] = [
                    joker['type'],
                    joker['level'],
                    joker['edition']
                ]
            idx += self.joker_features
        
        # Add game state info
        state[idx] = game_state['chips']
        state[idx+1] = game_state['mult']
        state[idx+2] = game_state['blind']['level']
        
        return state
    
    def get_state_size(self):
        """Return the size of the processed state vector."""
        return self.state_size 