import os
import time
import random
import cv2
from environment import ActorCritic, CarEnvironment
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import carla


def compute_gae(
    rewards, dones, values, gamma=0.99, lam=0.95
):
    advantages = np.zeros_like(rewards, dtype=np.float32)
    gae = 0.0
    for step in reversed(range(len(rewards))):
        next_non_terminal = 1.0 - dones[step]
        next_value = values[step + 1] if step + 1 < len(values) else 0.0
        delta = rewards[step] + gamma * next_value * next_non_terminal - values[step]
        gae = delta + gamma * lam * next_non_terminal * gae
        advantages[step] = gae
    returns = advantages + values[:-1]
    return advantages, returns


def main():
    print("Setting folders for logs and models")
    models_dir = f"models/{int(time.time())}/"
    logdir = f"logs/{int(time.time())}/"

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logdir, exist_ok=True)

    print("Connecting to env..")
    env = CarEnvironment()
    # 
    seed = 45
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    obs, _ = env.reset(seed=seed)
    print("Env has been reset as part of launch")
    print("Observation space:", env.observation_space)

    num_actions = int(env.action_space.nvec[0])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = ActorCritic(num_actions=num_actions).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=3e-4)

    # PPO hyperparameters
    max_env_steps = 2_000_000
    update_timesteps = 16
    epochs = 10
    batch_size = 16
    gamma = 0.99
    gae_lambda = 0.95
    clip_eps = 0.2
    vf_coef = 0.5
    ent_coef = 0.01

    timestep = 0

    obs_buffer = []
    angle_buffer = []
    action_buffer = []
    logprob_buffer = []
    reward_buffer = []
    done_buffer = []
    value_buffer = []

    episode_rewards = []
    ep_reward = 0.0

    try:
        while timestep < max_env_steps:
            # collect trajectories
            while len(reward_buffer) < update_timesteps:
                action, logprob, value = policy.act(obs, device)

                # env expects MultiDiscrete([9]) -> wrap action in array
                next_obs, reward, terminated, truncated, _ = env.step(np.array([action]))
                done = terminated or truncated

                obs_buffer.append(obs["image"])
                angle_buffer.append(obs["angle"])
                action_buffer.append(action)
                logprob_buffer.append(logprob)
                reward_buffer.append(reward)
                done_buffer.append(float(done))
                value_buffer.append(value)

                ep_reward += reward
                timestep += 1

                obs = next_obs

                if done:
                    episode_rewards.append(ep_reward)
                    print(
                        f"Step {timestep} | Episode reward: {ep_reward:.1f} | Episodes: {len(episode_rewards)}"
                    )
                    ep_reward = 0.0
                    obs, _ = env.reset()

                if timestep >= max_env_steps:
                    break

            # Add last value for GAE
            with torch.no_grad():
                _, last_value = policy.forward(
                    torch.from_numpy(obs["image"])
                    .float()
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .to(device),
                    torch.from_numpy(obs["angle"].astype(np.float32))
                    .view(1, -1)
                    .to(device),
                )
                last_value = last_value.cpu().numpy()[0, 0]

            values_np = np.array(value_buffer + [last_value], dtype=np.float32)
            rewards_np = np.array(reward_buffer, dtype=np.float32)
            dones_np = np.array(done_buffer, dtype=np.float32)

            advantages, returns = compute_gae(
                rewards_np, dones_np, values_np, gamma=gamma, lam=gae_lambda
            )

            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Prepare tensors
            images_t = (
                torch.from_numpy(np.stack(obs_buffer))
                .float()
                .permute(0, 3, 1, 2)
                .to(device)
            )
            angles_t = (
                torch.from_numpy(np.stack(angle_buffer).astype(np.float32))
                .view(-1, 1)
                .to(device)
            )
            actions_t = torch.from_numpy(np.array(action_buffer)).long().to(device)
            old_logprobs_t = torch.from_numpy(np.array(logprob_buffer)).float().to(device)
            returns_t = torch.from_numpy(returns).float().to(device)
            advantages_t = torch.from_numpy(advantages).float().to(device)

            dataset_size = len(reward_buffer)
            indices = np.arange(dataset_size)

            for _ in range(epochs):
                np.random.shuffle(indices)
                for start in range(0, dataset_size, batch_size):
                    end = start + batch_size
                    mb_idx = indices[start:end]

                    mb_images = images_t[mb_idx]
                    mb_angles = angles_t[mb_idx]
                    mb_actions = actions_t[mb_idx]
                    mb_old_logprobs = old_logprobs_t[mb_idx]
                    mb_returns = returns_t[mb_idx]
                    mb_advantages = advantages_t[mb_idx]

                    new_logprobs, values, entropy = policy.evaluate_actions(
                        mb_images, mb_angles, mb_actions
                    )

                    ratio = torch.exp(new_logprobs - mb_old_logprobs)
                    surr1 = ratio * mb_advantages
                    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()

                    value_loss = nn.functional.mse_loss(values, mb_returns)
                    entropy_loss = -entropy.mean()

                    loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                    optimizer.step()

            print(
                f"PPO update at step {timestep} | mean ep reward (last 10): "
                f"{np.mean(episode_rewards[-10:]) if episode_rewards else 0.0:.1f}"
            )

            # Clear buffers
            obs_buffer.clear()
            angle_buffer.clear()
            action_buffer.clear()
            logprob_buffer.clear()
            reward_buffer.clear()
            done_buffer.clear()
            value_buffer.clear()

            # Save model periodically
            torch.save(
                {
                    "policy_state_dict": policy.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "timestep": timestep,
                },
                os.path.join(models_dir, f"ppo_step_{timestep}.pt"),
            )

    except KeyboardInterrupt:
        print("Training interrupted by user.")
    finally:
        print("Saving final model...")
        torch.save(
            {
                "policy_state_dict": policy.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "timestep": timestep,
            },
            os.path.join(models_dir, f"ppo_final_step_{timestep}.pt"),
        )
        try:
            env.cleanup()
        except Exception:
            pass


if __name__ == "__main__":
    main()