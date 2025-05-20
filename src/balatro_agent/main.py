from typing import Dict, Any, Optional, List, Union, Tuple
import traceback
import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from models.dqn_agent import DQNAgent
from config.settings import (
    STATE_SIZE,
    ACTION_SIZE,
    BUFFER_SIZE,
    BATCH_SIZE,
    GAMMA,
    LEARNING_RATE,
    TAU,
    UPDATE_EVERY,
    INITIAL_EPSILON,
    EPSILON_DECAY,
    FINAL_EPSILON,
    MODEL_WEIGHTS_PATH,
)

app = FastAPI(
    title="Balatro DQN Agent API",
    description="API for interacting with the Balatro DQN agent",
    version="1.0.0",
)

# Initialize the DQN agent
agent = DQNAgent(
    state_size=STATE_SIZE,
    action_size=ACTION_SIZE,
    buffer_size=BUFFER_SIZE,
    batch_size=BATCH_SIZE,
    gamma=GAMMA,
    lr=LEARNING_RATE,
    tau=TAU,
    update_every=UPDATE_EVERY,
    initial_epsilon=INITIAL_EPSILON,
    epsilon_decay=EPSILON_DECAY,
    final_epsilon=FINAL_EPSILON,
)

# Load pre-trained weights if they exist
try:
    agent.load(MODEL_WEIGHTS_PATH)
    print(f"Loaded model weights from {MODEL_WEIGHTS_PATH}")
except:
    print("No pre-trained weights found. Starting with fresh model.")

def decode_action(action_value: int) -> Tuple[List[int], str]:
    """
    Decodes an integer action_value into selected card indices and action type.
    """
    # Last bit (0 or 1) determines play/discard
    action_type = "discard" if action_value & 1 else "play"

    # The other 8 bits determine card selection (up to 8 cards, we will cap at 5)
    card_selection_bits = action_value >> 1
    selected_indices = []
    for i in range(8):  # Iterate through bits for cards 1 to 8
        if (card_selection_bits >> i) & 1:  # Check if the i-th bit is set
            selected_indices.append(i + 1)  # Add 1-based card index
            if len(selected_indices) == 5:  # Max 5 cards
                break
    return selected_indices, action_type

def encode_action(selected_indices: List[int], action_type: str) -> int:
    """
    Encodes selected card indices and action type into an integer action_value.
    """
    action_value = 0
    if action_type == "discard":
        action_value = 1
    
    card_selection_bits = 0
    for idx in selected_indices:
        if 1 <= idx <= 8: # Ensure index is valid
            card_selection_bits |= (1 << (idx - 1)) # Set the (idx-1)-th bit
            
    action_value = (card_selection_bits << 1) | action_value
    return action_value

@app.post("/predict")
async def predict(request: dict):
    """
    Get action predictions from the DQN agent.

    Args:
        request: the current game state.
        Example: {'state': [0(current chips), 2(hands left), 3(discards left), 1, 'QC', 2, '10C', 3, '10D', 4, '7C', 5, '6C', 6, '6D', 7, '4H', 8, '2S']}

    Returns:
        ActionResponse containing:
        - indices: List of up to 5 card indices (values between 1-8), guaranteed non-empty.
        - action: "play" or "discard"
    """
    print("Received state:", request)
    try:
        # Convert the state to a numeric representation
        numeric_state = []
        for item in request["state"]:
            if isinstance(item, int):
                numeric_state.append(item)
            else:
                # Convert card strings to numeric values
                numeric_state.append(
                    hash(item) % 100
                )  # Simple hash-based encoding for now

        state_array = np.array(numeric_state, dtype=np.float32)
        action_value = agent.get_action(state_array)  # This is the raw action from the agent

        selected_indices, action_type = decode_action(action_value)
        
        # Enforce game rule: selected_indices cannot be empty
        if not selected_indices:
            selected_indices.append(1) # Default to selecting the first card
            print(f"Override: No cards selected by agent. Forcing selection of card 1.")

        # Enforce game rule: if discards_left is 0, action must be "play"
        discards_left = request["state"][2] # Index 2 is discards_left as per your example
        if discards_left == 0 and action_type == "discard":
            action_type = "play"
            print(f"Override: Discards left is 0. Forcing action to 'play'.")

        print(f"Decoded - Selected indices: {selected_indices}, Action: {action_type}")

        return {"indices": selected_indices, "action": action_type}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/train")
async def train(request: dict):
    """
    Train the agent with new experiences.

    Args:
        request: the experience tuple
        Example: {
            'state': [0(current chips), 2(hands left), 3(discards left), 1, 'QC', 2, '10C', 3, '10D', 4, '7C', 5, '6C', 6, '6D', 7, '4H', 8, '2S'],
            'action': {'indices': [1, 2, 3, 4, 5], 'action': 'play'},
            'reward': 1.0,
            'next_state': [0(current chips), 2(hands left), 3(discards left), 1, 'QC', 2, '10C', 3, '10D', 4, '7C', 5, '6C', 6, '6D', 7, '4H', 8, '2S'],
            'done': false
        }

    Returns:
        TrainResponse confirming the experience was added
    """
    try:
        # Convert states to numeric representations
        def convert_state(state_list):
            numeric_state = []
            for item in state_list:
                if isinstance(item, (int, float)):
                    numeric_state.append(float(item))
                else:
                    numeric_state.append(float(hash(item) % 100))
            return np.array(numeric_state, dtype=np.float32)

        state = convert_state(request["state"])
        next_state = convert_state(request["next_state"])
        action = encode_action(request["action"]["indices"], request["action"]["action"])
        reward = request["reward"]
        done = request["done"]

        agent.step(state, action, reward, next_state, done)
        return {"message": "Experience added to replay buffer"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/save")
async def save_model():
    """
    Save the current model weights.

    Returns:
        SaveResponse confirming the save operation
    """
    try:
        agent.save(MODEL_WEIGHTS_PATH)
        return f"Model saved to {MODEL_WEIGHTS_PATH}"
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000)
