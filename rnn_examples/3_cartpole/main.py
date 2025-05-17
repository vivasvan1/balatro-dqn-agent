import os
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
import numpy as np
from dqn_agent import DQNAgent

learning_rate = 0.001
n_episodes = 5000
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)  # Recalculated decay
final_epsilon = 0.1


env = gym.make("CartPole-v1")
# Wrap the environment to flatten the observation space
env = FlattenObservation(env)
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

# Add prints to inspect the spaces - REMOVED
print("Observation Space:", env.observation_space)
print("Action Space:", env.action_space)
# import sys; sys.exit() # Optional: uncomment to exit after printing - REMOVED

# # Calculate state_size from the flattened space shape
state_size = np.prod(env.observation_space.shape)

agent = DQNAgent(
    env=env,
    state_size=state_size, # Use calculated state_size
    action_size=env.action_space.n,
    buffer_size=int(1e5),
    batch_size=64,
    gamma=0.99,
    lr=5e-4,
    tau=1e-3,
    update_every=4,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
    seed=0,
)

if os.path.exists("3_cartpole/weights/dqn_agent_local.pth") and os.path.exists("3_cartpole/weights/dqn_agent_target.pth"):
    agent.load("3_cartpole/weights/dqn_agent")

from tqdm import tqdm

for episode in tqdm(range(n_episodes)):
    obs, info = env.reset()
    done = False

    # play one episode
    while not done:
        action = agent.get_action(obs)
        # print("action", action)
        # print("obs", obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        # print("next_obs", next_obs)
        # print("reward", reward)
        # print("terminated", terminated)
        # print("truncated", truncated)
        # print("info", info)
        # print("--------------------------------")
        # if (input("continue?") == "n"):
        #     break
        # update the agent
        agent.step(obs, action, float(reward), next_obs, terminated)

        # update if the environment is done and the current obs
        done = terminated or truncated
        obs = next_obs

    agent.decay_epsilon()

agent.save("3_cartpole/weights/dqn_agent")

from matplotlib import pyplot as plt

def get_moving_avgs(arr, window, convolution_mode):
    return np.convolve(
        np.array(arr).flatten(),
        np.ones(window),
        mode=convolution_mode
    ) / window

# Smooth over a 500 episode window
rolling_length = 500
fig, axs = plt.subplots(ncols=3, figsize=(12, 5))

axs[0].set_title("Episode rewards")
reward_moving_average = get_moving_avgs(
    env.return_queue,
    rolling_length,
    "valid"
)
axs[0].plot(range(len(reward_moving_average)), reward_moving_average)

axs[1].set_title("Episode lengths")
length_moving_average = get_moving_avgs(
    env.length_queue,
    rolling_length,
    "valid"
)
axs[1].plot(range(len(length_moving_average)), length_moving_average)

axs[2].set_title("Training Loss")
training_loss_moving_average = get_moving_avgs(
    agent.training_loss,
    rolling_length,
    "same"
)
axs[2].plot(range(len(training_loss_moving_average)), training_loss_moving_average)
plt.tight_layout()
plt.show()


