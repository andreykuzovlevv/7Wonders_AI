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

# -----------------------------------------------------------------------------
# -- environment wrapper --
# -----------------------------------------------------------------------------

Swap = Tuple[Tuple[int, int], Tuple[int, int]]  # ((r1,c1),(r2,c2))


class SevenWondersEnv(gym.Env):
    """A lightweight Gym wrapper exposing the simulator to RL algorithms.

    Action encoding: we give the agent a *fixed* discrete action space of
    size `rows*cols*4` (swap UP, DOWN, LEFT, RIGHT for every cell).  Before
    sampling an action we mask‐out illegal swaps so the policy never selects
    them.  Invalid-action penalties inside the simulator therefore become an
    additional safety net rather than the primary guard rail.
    """

    metadata = {"render_modes": [None]}

    def __init__(self, rows=config.GRID_ROWS, cols=config.GRID_COLS, level=config.LEVEL_1, seed=config.SEED):

        super().__init__()
        self.sim = SevenWondersSimulator(level=level)
        self.rows, self.cols = rows, cols
        # --- generate unique swaps (right + down only) ---
        self.swap_list = []
        for r in range(rows):
            for c in range(cols):
                if c + 1 < cols:
                    self.swap_list.append(((r, c), (r, c + 1)))
                if r + 1 < rows:
                    self.swap_list.append(((r, c), (r + 1, c)))

        self.n_actions = len(self.swap_list)
        self.action_space = gym.spaces.Discrete(self.n_actions)

        # (17, rows, cols) board tensor  +  (3,) global features  →  Dict space
        board_low = np.zeros((17, rows, cols), dtype=np.float32)
        board_high = np.ones_like(board_low)
        globals_low = np.zeros(3, dtype=np.float32)
        globals_high = np.ones(3, dtype=np.float32)
        self.observation_space = gym.spaces.Dict(
            {
                "board": gym.spaces.Box(board_low, board_high, dtype=np.float32),
                "globals": gym.spaces.Box(globals_low, globals_high, dtype=np.float32),
            }
        )

        self._rng = np.random.default_rng(seed)
        self.reset(seed=seed)

    # --- utility: action index  ↔  Swap tuple ---------------------------------------------------
    def _decode_action(self, action: int) -> Swap:
        """
        Given an integer action index, return the corresponding Swap tuple from swap_list.
        """
        return self.swap_list[action]

    def _encode_action(self, swap: Swap) -> int:
        """
        Given a Swap tuple, return its index in swap_list.
        Raises ValueError if the swap is not in the list.
        """
        return self.swap_list.index(swap)

    # --- gym core ---------------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        super().reset(seed=seed)
        state_tuple = self.sim.reset()  # simulator returns state_tuple
        obs_planes, obs_globals = state_tuple  # unpack the state tuple
        obs = {"board": obs_planes, "globals": obs_globals}
        info = {"action_mask": self._get_action_mask()}
        return obs, info

    def step(self, action: int):
        swap = self._decode_action(action)
        state_tuple, reward, done = self.sim.step(swap)  # simulator returns (state_tuple, reward, done)
        obs_planes, obs_globals = state_tuple  # unpack the state tuple
        terminated = done
        truncated = self.sim.step_count >= 400
        info = {"action_mask": self._get_action_mask()}
        obs = {"board": obs_planes, "globals": obs_globals}
        return obs, reward, terminated, truncated, info

    # --- helpers ---------------------------------------------------------------------------
    def _get_obs(self):
        planes, globs = self.sim.get_state_tuple()
        return {"board": planes, "globals": globs}

    def _get_action_mask(self):
        mask = np.zeros(self.n_actions, dtype=bool)
        for swap in self.sim.get_valid_swaps():
            idx = self._encode_action(swap)
            mask[idx] = True
        return mask

    # optional but handy for manual play -------------------------------------------------------
    def render(self):
        self.sim.display()


# -----------------------------------------------------------------------------
# -- neural network --
# -----------------------------------------------------------------------------

class ActorCritic(nn.Module):
    def __init__(self, rows: int, cols: int, n_actions: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(17, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        conv_out_dim = 64 * rows * cols
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
    logits = logits.masked_fill(~mask, -1e9)
    return torch.softmax(logits, dim=dim)


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
    def __init__(self, env: gym.Env, gamma: float = 0.99, lam: float = 0.95, clip_eps: float = 0.2,
                 lr: float = 2.5e-4, batch_size: int = 512, minibatch: int = 64, epochs: int = 4, device: str = "cuda"):
        self.env = env
        self.gamma, self.lam, self.clip_eps = gamma, lam, clip_eps
        self.batch_size, self.minibatch, self.epochs = batch_size, minibatch, epochs
        self.device = config.DEVICE

        self.model = ActorCritic(env.rows, env.cols, env.n_actions).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.tb = SummaryWriter(log_dir="ai_training/runs/7wonders_ppo")
        self.global_step = 0


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

            with torch.no_grad():
                logits, value = self.model(board, globs)
                probs = masked_softmax(logits, mask)
                dist = Categorical(probs)
                action = dist.sample()

            next_obs, reward, term, trunc, next_info = self.env.step(action.item())

            boards_l.append(board.squeeze(0).cpu())
            globals_l.append(globs.squeeze(0).cpu())
            actions_l.append(action.cpu())
            masks_l.append(mask.squeeze(0).cpu())
            logp_l.append(dist.log_prob(action).cpu())
            rewards_l.append(torch.tensor(reward, dtype=torch.float32))
            dones_l.append(torch.tensor(term or trunc, dtype=torch.float32))
            values_l.append(value.cpu())

            total_steps += 1
            obs, info = next_obs, next_info
            done = term or trunc
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

                value_loss = F.mse_loss(value.squeeze(-1), returns[mb_idx])
                entropy = dist.entropy().mean()

                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

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
        step = 0
        t0 = time.time()
        while step < total_steps:
            traj, traj_steps = self.collect()
            self.update(traj)
            step += traj_steps
          
            fps = step / (time.time() - t0)
            avg_r = traj.rewards.sum().item() / traj_steps

            # eval_moves, eval_win, frames = self.play_episode()
            # TensorBoard scalars
            self.tb.add_scalar("train/avg_return", avg_r, step)
            self.tb.add_scalar("train/episode_len", traj_steps/traj.dones.sum().item(), step)
            self.tb.add_scalar("train/fps", fps, step)
            self.tb.add_scalar("train/episodes_in_batch", traj.dones.sum().item(), step)
            # self.tb.add_scalar("eval/moves", eval_moves, step)
            # self.tb.add_scalar("eval/win",   eval_win,   step)
            
            # # GIF every 100 k steps
            # if step // 100_000 > (step-self.batch_size)//100_000 and frames:
            #     iio.imwrite(f"ai_training/runs/vid_{step//1000:06d}.gif", frames, duration=0.08)

            if step // 100_000 > (step - len(traj.actions)) // 100_000:
                torch.save(self.model.state_dict(), save_path)
        print("Training complete")


# -----------------------------------------------------------------------------
# -- entry point --
# -----------------------------------------------------------------------------

def train():
    parser = argparse.ArgumentParser(description="Train PPO on 7 Wonders match-3")
    parser.add_argument("--total_steps", type=int, default=2_000_000)
    parser.add_argument("--save", type=Path, default=Path("ai_training/runs/ppo_7wonders.pt"))
    args = parser.parse_args()

    env = SevenWondersEnv()
    agent = PPOAgent(env)
    agent.train(total_steps=args.total_steps, save_path=str(args.save))


if __name__ == "__main__":
    train()
