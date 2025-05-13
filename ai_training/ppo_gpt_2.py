# ppo_seven_wonders.py
"""
Proximal Policy Optimization (PPO) trainer for the **Seven Wonders** match‑3 environment.


```bash
python ppo_seven_wonders.py  # trains until --total‑timesteps reached

# inspect tensorboard
tensorboard --logdir runs

# evaluate a saved checkpoint
python ppo_seven_wonders.py --eval weights/latest.pt --episodes 20
```

The script’s default hyper‑parameters match those that work well on
10×10 boards.  Feel free to tweak with the CLI flags.
"""

###############################################################################
# IMPORT_PATHS – tweak if your filenames differ
###############################################################################
import os, sys, math, random, time, argparse, collections, datetime
from pathlib import Path

# ➜  adjust this if `seven_wonders_simulator.py` / `config.py` live elsewhere
ROOT_DIR = Path(__file__).absolute().parent

from .game_simulator import SevenWondersSimulator  # ⬅️  your env class
import config                                               # ⬅️  your config

###############################################################################
# THIRD‑PARTY
###############################################################################
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

import gymnasium as gym

from tqdm import trange  # progress bar


def set_seed(seed: int = config.SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


################################################################################
# ENVIRONMENT WRAPPER
################################################################################

class SevenWondersGym(gym.Env):
    """Gym wrapper around `SevenWondersSimulator`.  Discrete action space.

    **Action encoding**  – We enumerate every *right* or *down* swap on the
    10×10 grid.  That is, for each cell `(r, c)` we add:
        * `(r, c) ↔ (r, c+1)`  if `c+1 < COLS`
        * `(r, c) ↔ (r+1, c)`  if `r+1 < ROWS`
    This yields `rows*cols*2 – rows – cols` distinct swap actions.
    """

    metadata = {"render.modes": []}

    def __init__(self, seed: int | None = None):
        super().__init__()
        self.sim = SevenWondersSimulator()
        self.rows, self.cols = self.sim.rows, self.sim.cols

        # --- build static action lookup table --------------------------------
        self._actions: list[tuple[tuple[int,int], tuple[int,int]]] = []
        for r in range(self.rows):
            for c in range(self.cols):
                if c + 1 < self.cols:
                    self._actions.append(((r, c), (r, c + 1)))  # swap right
                if r + 1 < self.rows:
                    self._actions.append(((r, c), (r + 1, c)))  # swap down
        self.action_space = gym.spaces.Discrete(len(self._actions))

        # Observation: (17, rows, cols) planes + 3 globals  ➔  float32 tensor
        board_planes = (config.N_PLANES, self.rows, self.cols)
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(config.N_PLANES, self.rows, self.cols + 1),  # +1 col for globals
            dtype=np.float32,
        )
        if seed is not None:
            self.reset(seed=seed)

    # ---------------------------------------------------------------------
    def _combine_obs(self, state_tuple):
        board, global_feats = state_tuple  # board: (17,10,10)  globals: (3,)
        # tack globals as an extra column (broadcast along planes)
        board_combined = np.concatenate(
            [
                board,
                np.tile(global_feats.reshape(3, 1, 1), (1, 1, self.cols)),
            ],
            axis=0,
        )
        return board_combined.astype(np.float32)

    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            super().reset(seed=seed)
        obs_tuple = self.sim.reset()
        return self._combine_obs(obs_tuple), {}

    def step(self, action_idx: int):
        swap_action = self._actions[action_idx]
        next_state, reward, done = self.sim.step(swap_action)
        obs = self._combine_obs(next_state)
        return obs, reward, done, False, {}

    # no rendering for now -------------------------------------------------
    def render(self):
        pass


################################################################################
# NETWORK
################################################################################

class ActorCritic(nn.Module):
    def __init__(self, n_actions: int):
        super().__init__()
        planes = config.N_PLANES
        # CNN trunk for 18×10×10 input (17 board + 1 global aggregate plane)
        self.cnn = nn.Sequential(
            nn.Conv2d(planes, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        dummy = torch.zeros(1, planes, 10, 10)
        with torch.no_grad():
            cnn_out = self.cnn(dummy).shape[1]
        hidden = 256
        self.policy_net = nn.Sequential(
            nn.Linear(cnn_out, hidden), nn.ReLU(), nn.Linear(hidden, n_actions)
        )
        self.value_net = nn.Sequential(
            nn.Linear(cnn_out, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )

    # ---------------------------------------------------------------------
    def forward(self, x):
        z = self.cnn(x)
        return self.policy_net(z), self.value_net(z).squeeze(-1)


################################################################################
# ROLLOUT BUFFER – GAE
################################################################################

class RolloutBuffer:
    def __init__(self, size: int, obs_shape, device):
        self.size = size
        self.device = device
        self.ptr = 0
        self.full = False
        self.obs = torch.empty((size,) + obs_shape, device=device)
        self.actions = torch.empty(size, dtype=torch.int64, device=device)
        self.rewards = torch.empty(size, device=device)
        self.dones = torch.empty(size, dtype=torch.bool, device=device)
        self.logprobs = torch.empty(size, device=device)
        self.values = torch.empty(size, device=device)
        self.advantages = torch.empty(size, device=device)
        self.returns = torch.empty(size, device=device)

    # ------------------------------------------------------------------
    def add(self, obs, action, reward, done, logprob, value):
        self.obs[self.ptr].copy_(obs)
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.logprobs[self.ptr] = logprob
        self.values[self.ptr] = value
        self.ptr = (self.ptr + 1)
        if self.ptr == self.size:
            self.full = True
            self.ptr = 0

    # ------------------------------------------------------------------
    def compute_returns_and_advantages(self, last_value, gamma, lam):
        last_adv = 0
        for t in reversed(range(self.size)):
            non_terminal = 1.0 - self.dones[t].float()
            delta = self.rewards[t] + gamma * last_value * non_terminal - self.values[t]
            last_adv = delta + gamma * lam * non_terminal * last_adv
            self.advantages[t] = last_adv
            self.returns[t] = self.advantages[t] + self.values[t]
            last_value = self.values[t]

    # ------------------------------------------------------------------
    def get(self, batch_size):
        idxs = torch.randperm(self.size, device=self.device)
        for start in range(0, self.size, batch_size):
            end = start + batch_size
            batch_idx = idxs[start:end]
            yield (
                self.obs[batch_idx],
                self.actions[batch_idx],
                self.logprobs[batch_idx],
                self.returns[batch_idx],
                self.advantages[batch_idx],
            )


################################################################################
# PPO TRAINER
################################################################################

def ppo_update(model, optimizer, buffer: RolloutBuffer, clip_eps, vf_coef, ent_coef, epochs, batch_size):
    for _ in range(epochs):
        for obs, actions, old_logp, returns, adv in buffer.get(batch_size):
            logits, values = model(obs)
            dist = Categorical(logits=logits)
            logp = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            ratio = torch.exp(logp - old_logp)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = 0.5 * (returns - values).pow(2).mean()

            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()


################################################################################
# MAIN
################################################################################

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--total-timesteps", type=int, default=1_000_000)
    p.add_argument("--update-steps", type=int, default=2048)
    p.add_argument("--mini-batch", type=int, default=64)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--eval", type=str, default="")
    p.add_argument("--episodes", type=int, default=20, help="eval episodes if --eval is used")
    return p.parse_args()


def evaluate(model, env, episodes=20, device="cpu"):
    model.eval()
    returns, steps = [], []
    with torch.no_grad():
        for _ in range(episodes):
            obs, _ = env.reset()
            obs = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            done = False
            ep_return = 0
            ep_steps = 0
            while not done:
                logits, _ = model(obs)
                action = torch.argmax(logits, dim=-1).item()
                next_obs, reward, done, _, _ = env.step(action)
                obs = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)
                ep_return += reward
                ep_steps += 1
            returns.append(ep_return)
            steps.append(ep_steps)
    return np.mean(returns), np.std(returns), np.mean(steps)


def main():
    args = parse_args()
    set_seed(args.seed)

    env = SevenWondersGym(seed=args.seed)
    n_actions = env.action_space.n

    device = torch.device(args.device)
    model = ActorCritic(n_actions).to(device)

    if args.eval:
        model.load_state_dict(torch.load(args.eval, map_location=device))
        mean_r, std_r, mean_steps = evaluate(model, env, args.episodes, device)
        print(f"Evaluation over {args.episodes} eps: R={mean_r:.1f}±{std_r:.1f}, steps={mean_steps:.1f}")
        return

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)
    writer = SummaryWriter()

    obs, _ = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32, device=device)
    global_step = 0
    rollout = RolloutBuffer(args.update_steps, obs.shape, device)

    start_time = time.time()
    pbar = trange(args.total_timesteps // args.update_steps, desc="updates")
    for update in pbar:
        model.eval()
        for step in range(args.update_steps):
            logits, value = model(obs.unsqueeze(0))
            dist = Categorical(logits=logits)
            action = dist.sample()
            logprob = dist.log_prob(action)

            next_obs_np, reward, done, _, _ = env.step(action.item())
            reward = torch.tensor(reward, dtype=torch.float32, device=device)
            done_bool = torch.tensor(done, dtype=torch.bool, device=device)

            rollout.add(obs, action, reward, done_bool, logprob, value.squeeze(0))

            obs = torch.tensor(next_obs_np, dtype=torch.float32, device=device)
            global_step += 1

            if done:
                obs_np, _ = env.reset()
                obs = torch.tensor(obs_np, dtype=torch.float32, device=device)

        # bootstrap value
        with torch.no_grad():
            _, last_value = model(obs.unsqueeze(0))
        rollout.compute_returns_and_advantages(last_value.squeeze(0), args.gamma, args.gae_lambda)

        # PPO update ------------------------------------------------------
        model.train()
        ppo_update(
            model, optimizer, rollout, args.clip_eps, args.vf_coef, args.ent_coef,
            args.epochs, args.mini_batch,
        )

        # logging ---------------------------------------------------------
        writer.add_scalar("train/episode", update, global_step)
        writer.add_scalar("time/fps", int(global_step / (time.time()-start_time)), global_step)

        # evaluate occasionally
        if (update + 1) % 20 == 0:
            mean_r, std_r, mean_steps = evaluate(model, env, episodes=10, device=device)
            writer.add_scalar("eval/return_mean", mean_r, global_step)
            writer.add_scalar("eval/steps_mean", mean_steps, global_step)
            pbar.set_postfix(ret=f"{mean_r:.0f}±{std_r:.0f}", steps=f"{mean_steps:.1f}")

        # save checkpoint
        if (update + 1) % 50 == 0:
            ckpt_dir = ROOT_DIR / "weights"
            ckpt_dir.mkdir(exist_ok=True)
            ckpt_path = ckpt_dir / f"ckpt_{update+1:05d}.pt"
            torch.save(model.state_dict(), ckpt_path)
            torch.save(model.state_dict(), ckpt_dir / "latest.pt")

    writer.close()


if __name__ == "__main__":
    main()
