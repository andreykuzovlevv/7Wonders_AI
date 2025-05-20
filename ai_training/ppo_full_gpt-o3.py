# ppo_7wonders_ai.py
"""A self-contained PPO implementation for the 7 Wonders match-3 simulator.

This single file bundles:
• SevenWondersEnv – a Gymnasium wrapper around your `SevenWondersSimulator`
• ActorCritic     – a CNN-based policy/value network that handles action masks
• PPOAgent        – on-policy trainer with GAE, clipped surrogate objective
• train()         – minimal training script & CLI flags

The code assumes that `game_simulator.py` (with the classes you pasted) is in
PYTHONPATH.  All heavy lifting lives here so you only need to keep three files
around: this one, `game_simulator.py`, and (optionally) `visual.py` for human
play/inspection.

Tested with: Python 3.11, PyTorch 2.2, Gymnasium 0.29.
"""
from __future__ import annotations

import argparse
import math
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Dict, Any

from .game_simulator import SevenWondersSimulator

import config
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from torch.utils.tensorboard import SummaryWriter
import imageio.v3 as iio
import cProfile
import pstats
from pstats import SortKey

# -----------------------------------------------------------------------------
# -- environment wrapper --
# -----------------------------------------------------------------------------

Swap = Tuple[Tuple[int, int], Tuple[int, int]]  # ((r1,c1),(r2,c2))





# -----------------------------------------------------------------------------
# -- neural network --
# -----------------------------------------------------------------------------

class ActorCritic(nn.Module):
    def __init__(self, rows: int, cols: int, n_actions: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(17, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128,128,3,padding=1),   nn.ReLU(),
            nn.Conv2d(128,128,3,padding=1),   nn.ReLU(),
            nn.Flatten(),
        )
        # Calculate correct output dimension: 128 channels * rows * cols
        conv_out_dim = 128 * rows * cols
        self.fc = nn.Linear(conv_out_dim + 3, 256)
        self.policy_head = nn.Linear(256, n_actions)
        self.value_head = nn.Linear(256, 1)

    def forward(self, boards: torch.Tensor, globals_: torch.Tensor):
        x = self.conv(boards)
        x = torch.cat([x, globals_], dim=1)
        x = F.relu(self.fc(x))
        logits = self.policy_head(x)
        value = self.value_head(x).squeeze(-1)
        return logits, value


# -----------------------------------------------------------------------------
# -- PPO agent --
# -----------------------------------------------------------------------------

def masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1):
    """Apply softmax to logits while masking out invalid actions.
    
    Args:
        logits: Action logits tensor
        mask: Boolean mask where True indicates valid actions
        dim: Dimension to apply softmax over
    """
    # Handle case where all actions are masked out
    if not mask.any():
        return torch.zeros_like(logits)
        
    # Normalize logits for numerical stability
    logits = logits - logits.max(dim=dim, keepdim=True)[0]
    
    # Set invalid actions to a large negative number
    logits = logits.masked_fill(~mask, float('-inf'))
    
    # Apply softmax
    probs = torch.softmax(logits, dim=dim)
    
    # Ensure we have a valid probability distribution
    probs = probs.masked_fill(~mask, 0.0)
    probs = probs / (probs.sum(dim=dim, keepdim=True) + 1e-8)
    
    return probs


@dataclass
class Trajectory:
    boards: torch.Tensor
    globals: torch.Tensor
    actions: torch.Tensor
    masks: torch.Tensor
    logprobs: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    values: torch.Tensor
    next_value: torch.Tensor


class PPOAgent:
    def __init__(self, env: gym.Env, gamma: float = 0.99, lam: float = 0.95, clip_eps: float = 0.3,
                 lr: float = 3e-4, batch_size: int = 512, minibatch: int = 256, epochs: int = 4):
        self.env = env
        self.gamma, self.lam, self.clip_eps = gamma, lam, clip_eps
        self.batch_size, self.minibatch, self.epochs = batch_size, minibatch, epochs
        self.device = config.DEVICE

        self.model = ActorCritic(env.rows, env.cols, env.n_actions).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.tb = SummaryWriter(log_dir="ai_training/runs/7wonders_ppo")
        self.step = 0


    # --- rollout collection -------------------------------------------------------------
    def collect(self) -> Tuple[Trajectory, int]:
        boards_l, globals_l, actions_l, masks_l = [], [], [], []
        logp_l, rewards_l, dones_l, values_l = [], [], [], []

        obs, info = self.env.reset()
        total_steps = 0
        done = False

        while total_steps < self.batch_size or not done:
            board = torch.from_numpy(obs["board"]).unsqueeze(0).to(self.device)
            globs = torch.from_numpy(obs["globals"]).unsqueeze(0).to(self.device)
            mask = torch.from_numpy(info["action_mask"]).unsqueeze(0).to(self.device)

            mask_true = info["action_mask"].sum()
            self.tb.add_scalar("train/mask_true", mask_true, self.step)


            with torch.no_grad():
                logits, value = self.model(board, globs)
                probs = masked_softmax(logits, mask)
                dist = Categorical(probs)
                action = dist.sample()

            next_obs, reward, term, next_info = self.env.step(action.item())

            boards_l.append(board.squeeze(0).cpu())
            globals_l.append(globs.squeeze(0).cpu())
            actions_l.append(action.cpu())
            masks_l.append(mask.squeeze(0).cpu())
            logp_l.append(dist.log_prob(action).cpu())
            rewards_l.append(torch.tensor(reward, dtype=torch.float32))
            dones_l.append(torch.tensor(term, dtype=torch.float32))
            values_l.append(value.cpu())

            total_steps += 1
            obs, info = next_obs, next_info
            done = term
            if done:
                obs, info = self.env.reset()

        # get next_value after last state
        board = torch.from_numpy(obs["board"]).unsqueeze(0).to(self.device)
        globs = torch.from_numpy(obs["globals"]).unsqueeze(0).to(self.device)
        with torch.no_grad():
            next_value = self.model(board, globs)[1].cpu()

        return Trajectory(
            boards=torch.stack(boards_l),
            globals=torch.stack(globals_l),
            actions=torch.stack(actions_l),
            masks=torch.stack(masks_l),
            logprobs=torch.stack(logp_l),
            rewards=torch.tensor(rewards_l),
            dones=torch.tensor(dones_l),
            values=torch.stack(values_l),
            next_value=next_value.squeeze(0),
        ), total_steps


    # --- advantage / return computation -----------------------------------------------
    @staticmethod
    def _gae(rewards, dones, values, next_value, gamma, lam):
        adv = torch.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * (1 - dones[t]) * (next_value if t == len(rewards) - 1 else values[t + 1]) - values[t]
            gae = delta + gamma * lam * (1 - dones[t]) * gae
            adv[t] = gae
        returns = adv + values.squeeze(-1)
        return adv, returns

    # --- training step -----------------------------------------------------------------
    def update(self, traj: Trajectory):
        adv, returns = self._gae(traj.rewards, traj.dones, traj.values.squeeze(-1), traj.next_value, self.gamma, self.lam)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        total_policy_loss, total_value_loss, total_entropy = 0.0, 0.0, 0.0
        n_minibatches = 0

        # flatten
        boards = traj.boards.to(self.device)
        globals_ = traj.globals.to(self.device)
        actions = traj.actions.to(self.device)
        old_logp = traj.logprobs.to(self.device)
        masks = traj.masks.to(self.device)
        returns = returns.to(self.device)
        adv = adv.to(self.device)

        idxs = np.arange(len(actions))
        for _ in range(self.epochs):
            np.random.shuffle(idxs)
            for start in range(0, len(actions), self.minibatch):
                mb_idx = idxs[start:start + self.minibatch]

                logits, value = self.model(boards[mb_idx], globals_[mb_idx])
                probs = masked_softmax(logits, masks[mb_idx])
                dist = Categorical(probs)
                logp = dist.log_prob(actions[mb_idx])

                ratio = torch.exp(logp - old_logp[mb_idx])
                surr1 = ratio * adv[mb_idx]
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv[mb_idx]
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(value, returns[mb_idx])
                entropy = dist.entropy().mean()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                n_minibatches += 1

                loss = policy_loss + 0.25 * value_loss - 0.05 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

        return {
            "policy_loss": total_policy_loss / n_minibatches,
            "value_loss": total_value_loss / n_minibatches,
            "entropy": total_entropy / n_minibatches,
            "adv": adv,
        }

    def play_episode(self, render=False):
        obs, info = self.env.reset()
        moves, won = 0, 0
        frames = []
        while True:
            board = torch.from_numpy(obs["board"]).unsqueeze(0).to(self.device)
            globs = torch.from_numpy(obs["globals"]).unsqueeze(0).to(self.device)
            mask  = torch.from_numpy(info["action_mask"]).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits, _ = self.model(board, globs)
                probs = masked_softmax(logits, mask)
                action = probs.argmax(-1).item()          # greedy

            obs, reward, term, trunc, info = self.env.step(action)
            moves += 1
            if render and moves % 3 == 0:
                frames.append(self.env.sim.render_rgb()) # you add this helper

            if term or trunc:
                won = int(reward > 0)
                break
        return moves, won, frames

    # --- top-level training loop --------------------------------------------------------
    def train(self, total_steps: int = 1_000_000, log_every: int = 10_000, save_path: str | None = None):
        print("Device: ", self.device)
        
        t0 = time.time()
        while self.step < total_steps:
            traj, traj_steps = self.collect()
            stats = self.update(traj)
            self.step += traj_steps
            print(f"Step: {self.step}, Traj steps: {traj_steps}")
          
            fps = self.step / (time.time() - t0)
            avg_r = traj.rewards.sum().item() / traj_steps


            # Reward stats
            self.tb.add_scalar("train/avg_reward_per_step", avg_r, self.step)
            self.tb.add_scalar("train/reward_mean", traj.rewards.mean().item(), self.step)
            self.tb.add_scalar("train/reward_std", traj.rewards.std().item(), self.step)

            # Episode stats
            self.tb.add_scalar("train/episode_avg_len", traj_steps/traj.dones.sum().item(), self.step)
            self.tb.add_scalar("train/fps", fps, self.step)
            self.tb.add_scalar("train/episodes_in_batch", traj.dones.sum().item(), self.step)

            # Loss stats
            self.tb.add_scalar("train/policy_loss", stats["policy_loss"], self.step)
            self.tb.add_scalar("train/value_loss", stats["value_loss"], self.step)
            self.tb.add_scalar("train/entropy", stats["entropy"], self.step)

            # Adv stats
            self.tb.add_scalar("train/adv_mean", stats["adv"].mean().item(), self.step)
            self.tb.add_scalar("train/adv_std", stats["adv"].std().item(), self.step)

            
            # # GIF every 100 k steps
            # if self.step // 100_000 > (self.step-self.batch_size)//100_000 and frames:
            #     iio.imwrite(f"ai_training/runs/vid_{self.step//1000:06d}.gif", frames, duration=0.08)

            if self.step // 100_000 > (self.step - len(traj.actions)) // 100_000:
                torch.save(self.model.state_dict(), save_path)
        print("Training complete")


# -----------------------------------------------------------------------------
# -- entry point --
# -----------------------------------------------------------------------------

def train():
    parser = argparse.ArgumentParser(description="Train PPO on 7 Wonders match-3")
    parser.add_argument("--total_steps", type=int, default=2_000_000)
    parser.add_argument("--save", type=Path, default=Path("ai_training/runs/ppo_7wonders.pt"))
    parser.add_argument("--profile", action="store_true", help="Enable profiling")
    args = parser.parse_args()

    env = SevenWondersEnv()
    agent = PPOAgent(env)
    
    if args.profile:
        profiler = cProfile.Profile()
        profiler.enable()
        agent.train(total_steps=args.total_steps, save_path=str(args.save))
        profiler.disable()
        
        # Save profiling results to a file
        stats = pstats.Stats(profiler)
        stats.sort_stats(SortKey.CUMULATIVE)
        stats.dump_stats("ai_training/runs/ppo_profile.prof")
        
        # Print top 20 time-consuming functions
        print("\nProfiling Results (Top 20):")
        stats.print_stats(20)
    else:
        agent.train(total_steps=args.total_steps, save_path=str(args.save))


if __name__ == "__main__":
    train()