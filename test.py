import os
import sys
import time
import glob
import numpy as np
import torch

from environment import CarEnvironment, ActorCritic


def load_policy(checkpoint_path, env: CarEnvironment):
    num_actions = int(env.action_space.nvec[0])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy = ActorCritic(num_actions=num_actions).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("policy_state_dict", checkpoint)
    policy.load_state_dict(state_dict)
    policy.eval()
    return policy


def drive_loop(checkpoint_path):
    env = CarEnvironment()
    obs, _ = env.reset()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = load_policy(checkpoint_path, env)

    episode = 0
    episode_reward = 0.0

    try:
        step = 0
        while True:
            with torch.no_grad():
                action, _, _ = policy.act(obs, device)

            next_obs, reward, terminated, truncated, _ = env.step(np.array([action]))

            episode_reward += reward
            step += 1
            obs = next_obs

            if terminated or truncated:
                episode += 1
                print(
                    f"Episode {episode} finished | "
                    f"steps: {step} | total reward: {episode_reward:.1f}"
                )
                episode_reward = 0.0
                step = 0
                obs, _ = env.reset()

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("Driving interrupted by user.")
    finally:
        try:
            env.cleanup()
        except Exception:
            pass


if __name__ == "__main__":
    drive_loop("./models/1769514634/ppo_final_stap_8168.pt")