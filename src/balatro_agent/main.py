from typing import Dict, Any, Optional
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
    MODEL_WEIGHTS_PATH
)

app = FastAPI(
    title="Balatro DQN Agent API",
    description="API for interacting with the Balatro DQN agent",
    version="1.0.0"
)

# Pydantic models for request/response validation
class StateRequest(BaseModel):
    state: list[float] = Field(..., description="Current game state vector")

class ActionResponse(BaseModel):
    action: int = Field(..., description="Predicted action index")
    status: str = Field("success", description="Response status")

class TrainRequest(BaseModel):
    state: list[float] = Field(..., description="Current state vector")
    action: int = Field(..., description="Action taken")
    reward: float = Field(..., description="Reward received")
    next_state: list[float] = Field(..., description="Next state vector")
    done: bool = Field(..., description="Whether episode is done")

class TrainResponse(BaseModel):
    status: str = Field("success", description="Response status")
    message: str = Field(..., description="Response message")

class SaveResponse(BaseModel):
    status: str = Field("success", description="Response status")
    message: str = Field(..., description="Response message")

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
    final_epsilon=FINAL_EPSILON
)

# Load pre-trained weights if they exist
try:
    agent.load(MODEL_WEIGHTS_PATH)
    print(f"Loaded model weights from {MODEL_WEIGHTS_PATH}")
except:
    print("No pre-trained weights found. Starting with fresh model.")

@app.post("/predict", response_model=ActionResponse)
async def predict(request: StateRequest) -> ActionResponse:
    """
    Get action predictions from the DQN agent.
    
    Args:
        request: StateRequest containing the current game state
        
    Returns:
        ActionResponse containing the predicted action
    """
    try:
        state = np.array(request.state)
        action = agent.get_action(state)
        return ActionResponse(action=action)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/train", response_model=TrainResponse)
async def train(request: TrainRequest) -> TrainResponse:
    """
    Train the agent with new experiences.
    
    Args:
        request: TrainRequest containing the experience tuple
        
    Returns:
        TrainResponse confirming the experience was added
    """
    try:
        state = np.array(request.state)
        next_state = np.array(request.next_state)
        agent.step(state, request.action, request.reward, next_state, request.done)
        return TrainResponse(message="Experience added to replay buffer")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/save", response_model=SaveResponse)
async def save_model() -> SaveResponse:
    """
    Save the current model weights.
    
    Returns:
        SaveResponse confirming the save operation
    """
    try:
        agent.save(MODEL_WEIGHTS_PATH)
        return SaveResponse(message=f"Model saved to {MODEL_WEIGHTS_PATH}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000) 