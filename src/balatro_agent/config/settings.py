import os

# Server settings
HOST = '0.0.0.0'
PORT = 5000

# Model settings
STATE_SIZE = 19  # Size of the state representation (19 elements: current_chips, hand_left, discards_left, 8 pairs of index-card)
ACTION_SIZE = 256  # Binary representation: 8 bits for cards (1 bit per card) + 1 bit for play/discard
HIDDEN_SIZE = 256  # Size of hidden layers in the neural network

# Training settings
BUFFER_SIZE = int(1e5)  # Replay buffer size
BATCH_SIZE = 64  # Minibatch size
GAMMA = 0.99  # Discount factor
LEARNING_RATE = 5e-4  # Learning rate
TAU = 1e-3  # For soft update of target parameters
UPDATE_EVERY = 4  # How often to update the network

# Exploration settings
INITIAL_EPSILON = 1.0  # Starting value of epsilon
EPSILON_DECAY = 0.995  # Epsilon decay rate
FINAL_EPSILON = 0.01  # Minimum value of epsilon

# File paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_WEIGHTS_PATH = os.path.join(BASE_DIR, 'weights', 'dqn_agent')

# Create weights directory if it doesn't exist
os.makedirs(os.path.dirname(MODEL_WEIGHTS_PATH), exist_ok=True) 