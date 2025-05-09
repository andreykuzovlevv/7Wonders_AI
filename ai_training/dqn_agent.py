# --- dqn_agent.py ----------------------------------------------------------
"""Dueling‑DDQN agent with n‑step returns, action masking and soft target update.

Features
--------
*   ε‑greedy or greedy evaluation with legal‑action masking
*   n‑step replay (length `config.N_STEPS`, default 20)
*   Correct γⁿ boot‑strapping even when episode ends before n steps
*   Compatible with the QNetwork defined in `dqn_model.py`
"""

import random
from collections import deque
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .dqn_model import QNetwork
import config

# ---------------------------------------------------------------------------
# Helper: deterministic mapping ⟺ flat action index
# ---------------------------------------------------------------------------
DIR2OFF = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}  # U, D, L, R

def swap_to_idx(r1: int, c1: int, r2: int, c2: int) -> int:
    """Map a swap of two adjacent cells to a unique integer in
    `[0, MAX_ACTIONS)`.
    """
    if   (r2 == r1 - 1 and c2 == c1):  d = 0  # up
    elif (r2 == r1 + 1 and c2 == c1):  d = 1  # down
    elif (r2 == r1 and c2 == c1 - 1):  d = 2  # left
    elif (r2 == r1 and c2 == c1 + 1):  d = 3  # right
    else:
        raise ValueError("swap_to_idx: coordinates are not adjacent")
    return ((r1 * config.GRID_COLS) + c1) * 4 + d

def idx_to_swap(idx: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    cell, d = divmod(idx, 4)
    r1, c1  = divmod(cell, config.GRID_COLS)
    dr, dc  = DIR2OFF[d]
    return (r1, c1), (r1 + dr, c1 + dc)

# ---------------------------------------------------------------------------
#  Agent
# ---------------------------------------------------------------------------
class DQNAgent:
    """Dueling‑DDQN agent with n‑step replay and legal‑action masking."""

    def __init__(self):
        self.batch_size   = config.BATCH_SIZE
        self.gamma        = config.GAMMA
        self.tau          = config.TAU
        self.update_every = config.UPDATE_EVERY
        self.memory       = deque(maxlen=config.BUFFER_SIZE)
        self.t_step       = 0

        # n‑step settings
        self.n_step   = config.N_STEPS
        self.n_buffer = deque(maxlen=self.n_step)  # (state, action, reward)

        random.seed(config.SEED)
        torch.manual_seed(config.SEED)

        self.q_local  = QNetwork().to(config.DEVICE)
        self.q_target = QNetwork().to(config.DEVICE)
        self.opt      = torch.optim.Adam(self.q_local.parameters(), lr=config.LR)

    # ---------------------------------------------------------------------
    # Act
    # ---------------------------------------------------------------------
    def act(self, board: np.ndarray, gfeat: np.ndarray,
            valid_swaps: List[Tuple[Tuple[int, int], Tuple[int, int]]],
            eps: float = 0.0):
        """Return a legal swap chosen ε‑greedily from Q‑values."""
        if not valid_swaps:
            return None  # no legal action – terminate the episode upstream

        # exploration
        if random.random() < eps:
            return random.choice(valid_swaps)

        # evaluation
        b = torch.from_numpy(board).unsqueeze(0).to(config.DEVICE)
        g = torch.from_numpy(gfeat).unsqueeze(0).to(config.DEVICE)
        q = self.q_local(b, g).squeeze(0)  # (MAX_ACTIONS,)

        # mask illegal actions
        mask = torch.full_like(q, -1e9)
        for s in valid_swaps:
            idx = swap_to_idx(*s[0], *s[1])
            mask[idx] = 0.0
        q = q + mask

        best_idx = int(torch.argmax(q).item())
        return idx_to_swap(best_idx)

    # ---------------------------------------------------------------------
    # Step – store experience & learn periodically
    # ---------------------------------------------------------------------
    def step(self, state, action, reward, next_state, done, valid_next):
        """Save n‑step transition then perform a learning step if due."""
        # 1️⃣  push latest (s, a, r) into the short FIFO
        self.n_buffer.append((state, action, reward))

        # 2️⃣  once we have n entries *or* the episode ended, compute return R
        if len(self.n_buffer) == self.n_step or done:
            R, γ = 0.0, 1.0
            for (_, _, r) in reversed(self.n_buffer):
                R += γ * r
                γ *= self.gamma

            effective_n = len(self.n_buffer)  # may be < n_step at episode end
            first_state, first_action, _ = self.n_buffer[0]

            self.memory.append(
                (
                    first_state[0].astype(np.float16),  # board
                    first_state[1].astype(np.float16),  # global features
                    swap_to_idx(*first_action[0], *first_action[1]),
                    R,
                    next_state[0].astype(np.float16),   # board n‑steps ahead
                    next_state[1].astype(np.float16),   # global features n‑steps ahead
                    done,
                    [swap_to_idx(*s[0], *s[1]) for s in valid_next],
                    effective_n,
                )
            )

        # 3️⃣  Learn every `update_every` env steps
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and len(self.memory) >= self.batch_size:
            batch = random.sample(self.memory, self.batch_size)
            self.learn(batch)

        # 4️⃣  reset the FIFO at episode boundary
        if done:
            self.n_buffer.clear()

    # ---------------------------------------------------------------------
    # Learn
    # ---------------------------------------------------------------------
    def learn(self, batch):
        (boards, gfeats, a_indices, returns,
         nboards, ngfeats, dones, valid_next, n_effective) = zip(*batch)

        boards  = torch.from_numpy(np.stack(boards)).float().to(config.DEVICE)
        gfeats  = torch.from_numpy(np.stack(gfeats)).float().to(config.DEVICE)
        actions = torch.tensor(a_indices, dtype=torch.int64,
                               device=config.DEVICE).unsqueeze(1)
        returns = torch.tensor(returns, dtype=torch.float32,
                               device=config.DEVICE).unsqueeze(1)
        nboards = torch.from_numpy(np.stack(nboards)).float().to(config.DEVICE)
        ngfeats = torch.from_numpy(np.stack(ngfeats)).float().to(config.DEVICE)
        dones   = torch.tensor(dones, dtype=torch.float32,
                               device=config.DEVICE).unsqueeze(1)
        n_eff   = torch.tensor(n_effective, dtype=torch.float32,
                               device=config.DEVICE).unsqueeze(1)

        # 1️⃣  Q(s,a) from local network
        q_all = self.q_local(boards, gfeats)               # (B, A)
        q_exp = q_all.gather(1, actions)                   # (B, 1)

        # 2️⃣  Boot‑strap Q‑target using DDQN selection
        with torch.no_grad():
            q_next_local  = self.q_local(nboards, ngfeats)   # (B, A)
            q_next_target = self.q_target(nboards, ngfeats)

            q_tar_next = []
            for i, valids in enumerate(valid_next):
                if dones[i] or not valids:
                    q_tar_next.append(0.0)
                else:
                    mask = torch.full_like(q_next_local[i], -1e9)
                    mask[valids] = 0.0
                    best_a = torch.argmax(q_next_local[i] + mask).item()
                    q_tar_next.append(q_next_target[i, best_a].item())

            q_tar_next = torch.tensor(q_tar_next, dtype=torch.float32,
                                       device=config.DEVICE).unsqueeze(1)
            # γ raised to effective n for each sample
            q_tar = returns + (self.gamma ** n_eff) * q_tar_next * (1 - dones)

        # 3️⃣  optimise
        loss = F.mse_loss(q_exp, q_tar)
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_local.parameters(), 10.0)
        self.opt.step()

        # 4️⃣  soft update
        self._soft_update()

    # ---------------------------------------------------------------------
    # Soft update
    # ---------------------------------------------------------------------
    def _soft_update(self):
        with torch.no_grad():
            for t, l in zip(self.q_target.parameters(), self.q_local.parameters()):
                t.copy_(self.tau * l + (1.0 - self.tau) * t)
