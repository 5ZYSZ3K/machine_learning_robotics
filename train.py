import datetime
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from environment import ActorCritic, CarEnvironment

# Match environment constants
HEIGHT = 180
WIDTH = 240
N_CHANNELS = 3


def dict_obs_to_tensor(obs, device):
    """Convert dict observation to batched tensors (1, ...)."""
    image = torch.from_numpy(obs["image"]).float().to(device)
    # (H, W, C) -> (1, C, H, W)
    image = image.permute(2, 0, 1).unsqueeze(0)
    angle = torch.from_numpy(np.asarray(obs["angle"], dtype=np.float32)).float().to(device).reshape(1, 1)
    return image, angle

def compute_gae_and_returns(rewards, dones, values, next_value, next_done, gamma=0.99, gae_lambda=0.95):
    """Compute advantages and returns for a rollout."""
    advantages = torch.zeros_like(values)
    lastgaelam = 0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            nextnonterminal = 1.0 - next_done
            nextvalues = next_value
        else:
            nextnonterminal = 1.0 - dones[t + 1]
            nextvalues = values[t + 1]
        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        advantages[t] = lastgaelam = delta + gamma * gae_lambda * lastgaelam * nextnonterminal
    returns = advantages + values
    return advantages, returns


def ppo_update(
    model,
    images,
    angles,
    actions,
    log_probs_old,
    advantages,
    returns,
    clip_eps=0.2,
    value_coef=0.5,
    entropy_coef=0.01,
):
    """One PPO update over the collected batch."""
    _, log_prob, entropy, value = model.get_action_and_value(images, angles, action=actions)
    ratio = (log_prob - log_probs_old).exp()
    surr1 = ratio * advantages
    surr2 = ratio.clamp(1 - clip_eps, 1 + clip_eps) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    value_loss = F.mse_loss(value, returns)
    entropy_loss = -entropy.mean()
    loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss
    model.optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    model.optimizer.step()
    return policy_loss.item(), value_loss.item(), entropy_loss.item()


def collect_rollout(env, model, device, n_steps):
    """Collect n_steps of experience. Returns lists of tensors (on device)."""
    images_list, angles_list = [], []
    actions_list, rewards_list, dones_list, log_probs_list, values_list = [], [], [], [], []

    obs, _ = env.reset()
    image, angle = dict_obs_to_tensor(obs, device)
    next_done = torch.zeros(1, device=device)

    for _ in range(n_steps):
        with torch.no_grad():
            action, log_prob, _, value = model.get_action_and_value(image, angle)
        act_np = action.cpu().numpy().squeeze(0)
        next_obs, reward, terminated, truncated, _ = env.step(act_np)
        done = terminated or truncated

        images_list.append(image)
        angles_list.append(angle)
        actions_list.append(action)
        rewards_list.append(reward)
        dones_list.append(torch.tensor([float(done)], device=device))
        log_probs_list.append(log_prob)
        values_list.append(value)

        if done:
            obs, _ = env.reset()
        else:
            obs = next_obs
        image, angle = dict_obs_to_tensor(obs, device)
        next_done = torch.tensor([float(done)], device=device)

    with torch.no_grad():
        next_value = model.get_value(image, angle)

    images = torch.cat(images_list, dim=0)
    angles = torch.cat(angles_list, dim=0)
    actions = torch.cat(actions_list, dim=0)
    rewards = torch.tensor(rewards_list, device=device, dtype=torch.float32)
    dones = torch.cat(dones_list, dim=0)
    log_probs_old = torch.cat(log_probs_list, dim=0)
    values = torch.cat(values_list, dim=0)

    advantages, returns = compute_gae_and_returns(
        rewards, dones, values, next_value, next_done
    )
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return images, angles, actions, log_probs_old, advantages, returns, rewards


def main():
    models_directory = f"models/{datetime.datetime.now().strftime('%I-%M%p-%B-%d-%Y')}/"
    logs_directory = f"logs/{datetime.datetime.now().strftime('%I-%M%p-%B-%d-%Y')}/"

    if not os.path.exists(models_directory):
        os.makedirs(models_directory)
    if not os.path.exists(logs_directory):
        os.makedirs(logs_directory)

    print("connecting to env..")
    environment = CarEnvironment()
    seed = 2025
    environment.reset(seed=seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ActorCritic(lr=0.001).to(device)
    writer = SummaryWriter(logs_directory, comment="PPO")

    TIMESTEPS_PER_ITERATION = 100
    training_iteration = 0

    while training_iteration < 4:
        training_iteration += 1
        print("Iteration ", training_iteration, " is to commence...")

        images, angles, actions, log_probs_old, advantages, returns, rewards = collect_rollout(
            environment, model, device, TIMESTEPS_PER_ITERATION
        )

        # PPO update (several epochs over the same batch)
        for _ in range(4):
            pl, vl, el = ppo_update(
                model, images, angles, actions, log_probs_old, advantages, returns
            )
        mean_reward = rewards.mean().item()
        writer.add_scalar("train/mean_reward", mean_reward, training_iteration * TIMESTEPS_PER_ITERATION)
        writer.add_scalar("train/policy_loss", pl, training_iteration * TIMESTEPS_PER_ITERATION)
        writer.add_scalar("train/value_loss", vl, training_iteration * TIMESTEPS_PER_ITERATION)

        print("Iteration ", training_iteration, " has been trained")
        save_path = f"{models_directory}/{TIMESTEPS_PER_ITERATION * training_iteration}"
        torch.save(model.state_dict(), save_path)

    writer.close()


if __name__ == "__main__":
    main()
