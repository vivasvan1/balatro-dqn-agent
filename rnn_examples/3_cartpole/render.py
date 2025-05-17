import os
import gymnasium as gym
import numpy as np
import torch
# Assuming dqn_agent.py exists in the same directory or is accessible
from dqn_agent import DQNAgent, QNetwork # Assuming QNetwork is also defined there

# --- Parameters ---
ENV_NAME = "CartPole-v1"
WEIGHTS_DIR = "3_cartpole/weights"
MODEL_NAME = "dqn_agent" # Base name used during saving
N_EPISODES_TO_RENDER = 5
SEED = 42 # Optional: for reproducibility of environment reset if needed

# --- Environment Setup ---
env = gym.make(ENV_NAME, render_mode="human")

# Get state and action sizes from the environment
# CartPole state is a Box(4,), action is Discrete(2)
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# --- Agent Setup ---
# Instantiate the agent - Ensure DQNAgent is compatible with CartPole state/action space
# Set epsilon to 0 for deterministic (greedy) actions during evaluation
agent = DQNAgent(
    env=env, # Pass env if needed by agent internals, else None
    state_size=state_size,
    action_size=action_size,
    initial_epsilon=0.0, # Greedy policy for evaluation
    final_epsilon=0.0,
    epsilon_decay=0.0, # Not needed for evaluation
    seed=SEED
    # Add other necessary parameters if your DQNAgent requires them (e.g., buffer_size, batch_size - usually not needed for eval)
)

# --- Load Weights ---
weights_path_local = os.path.join(WEIGHTS_DIR, f"{MODEL_NAME}_local.pth")
weights_path_target = os.path.join(WEIGHTS_DIR, f"{MODEL_NAME}_target.pth")

if os.path.exists(weights_path_local) and os.path.exists(weights_path_target):
    print(f"Loading weights from {WEIGHTS_DIR}/{MODEL_NAME}...")
    try:
        agent.load(os.path.join(WEIGHTS_DIR, MODEL_NAME))
        agent.qnetwork_local.eval() # Set model to evaluation mode
        if hasattr(agent, 'qnetwork_target'): # Check if target network exists
             agent.qnetwork_target.eval()
    except Exception as e:
        print(f"Error loading weights: {e}")
        print("Ensure the DQNAgent structure matches the saved weights.")
        env.close()
        exit()
else:
    print(f"Error: Weights not found at {weights_path_local} or {weights_path_target}")
    print("Please ensure the agent has been trained and weights are saved correctly.")
    env.close()
    exit()

# --- Rendering Loop ---
print(f"Rendering {N_EPISODES_TO_RENDER} episodes...")

for episode in range(N_EPISODES_TO_RENDER):
    obs, info = env.reset(seed=SEED + episode if SEED is not None else None) # Seed reset for consistency
    done = False
    truncated = False
    total_reward = 0
    step_count = 0

    while not done and not truncated:
        # Get action from the agent (using greedy policy, epsilon=0)
        action = agent.get_action(obs) # Assumes get_action takes state and uses internal epsilon=0

        # Step the environment
        next_obs, reward, done, truncated, info = env.step(action)

        # Accumulate reward
        total_reward += reward
        step_count += 1

        # Update observation
        obs = next_obs

        # Rendering is handled by `render_mode="human"` in gym.make

    print(f"Episode {episode + 1}: Total Reward = {total_reward}, Steps = {step_count}, Done = {done}, Truncated = {truncated}")

# --- Cleanup ---
env.close()
print("Rendering finished.")
