#!/usr/bin/env python3
"""
Render a video of the trained PPO agent on CartPole-v1
"""
import torch
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
import os
from ppo_agent import ActorCritic

# Video output directory
VIDEO_DIR = "cartpole_videos"
MODEL_PATH = "checkpoints/ppo_CartPole-v1_final.pth"
ENV_NAME = "CartPole-v1"


def load_actor_critic(model_path, state_dim, action_dim, hidden_dim=64, device="cpu"):
    model = ActorCritic(state_dim, action_dim, hidden_dim)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['actor_critic_state_dict'])
    model.eval()
    return model

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = gym.make(ENV_NAME, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(env, VIDEO_DIR, episode_trigger=lambda ep: True, name_prefix="ppo_cartpole")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Load model
    model = load_actor_critic(MODEL_PATH, state_dim, action_dim, hidden_dim=64, device=device)

    obs, info = env.reset()
    obs = torch.FloatTensor(obs).to(device)
    done = False
    truncated = False
    total_reward = 0
    step = 0

    while not (done or truncated):
        with torch.no_grad():
            action_logits, _ = model(obs.unsqueeze(0))
            action_probs = F.softmax(action_logits, dim=-1)
            action = torch.argmax(action_probs).item()
        next_obs, reward, done, truncated, info = env.step(action)
        obs = torch.FloatTensor(next_obs).to(device)
        total_reward += reward
        step += 1

    env.close()
    print(f"Episode finished in {step} steps, total reward: {total_reward}")
    print(f"Video saved to: {VIDEO_DIR}")
    print("If you want a single mp4, you can use ffmpeg or moviepy to combine frames.")

if __name__ == "__main__":
    main() 