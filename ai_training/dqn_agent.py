# --- dqn_agent.py ----------------------------------------------------------
import random, numpy as np, torch, torch.nn.functional as F
from collections import deque
from .dqn_model import QNetwork
import config

# helper --------------------------------------------------------------------
DIR2OFF = {0:(-1,0), 1:(1,0), 2:(0,-1), 3:(0,1)}          # U,D,L,R

def swap_to_idx(r1, c1, r2, c2):
    """Deterministic mapping ⟹  idx in [0, MAX_ACTIONS)."""
    if   (r2 == r1-1 and c2 == c1):  d = 0                 # up
    elif (r2 == r1+1 and c2 == c1):  d = 1                 # down
    elif (r2 == r1   and c2 == c1-1):d = 2                 # left
    elif (r2 == r1   and c2 == c1+1):d = 3                 # right
    else: raise ValueError("swap_to_idx: not adjacent")
    return ((r1 * config.GRID_COLS) + c1) * 4 + d

def idx_to_swap(idx):
    cell, d = divmod(idx, 4)
    r1, c1  = divmod(cell, config.GRID_COLS)
    dr, dc  = DIR2OFF[d]
    return (r1, c1), (r1+dr, c1+dc)
# ---------------------------------------------------------------------------


class DQNAgent:
    """Dueling DDQN with action‑indexing & masking."""
    def __init__(self):
        self.batch_size   = config.BATCH_SIZE
        self.gamma        = config.GAMMA
        self.tau          = config.TAU
        self.update_every = config.UPDATE_EVERY
        self.memory       = deque(maxlen=config.BUFFER_SIZE)
        self.t_step       = 0
        random.seed(config.SEED); torch.manual_seed(config.SEED)

        self.q_local  = QNetwork().to(config.DEVICE)
        self.q_target = QNetwork().to(config.DEVICE)
        self.opt      = torch.optim.Adam(self.q_local.parameters(), lr=config.LR)

    # -------- act ----------------------------------------------------------
    def act(self, board, gfeat, valid_swaps, eps=0.0):
        if not valid_swaps: return None
        if random.random() < eps:
            return random.choice(valid_swaps)

        b = torch.from_numpy(board).unsqueeze(0).to(config.DEVICE)
        g = torch.from_numpy(gfeat).unsqueeze(0).to(config.DEVICE)
        q = self.q_local(b, g).squeeze(0)                    # (MAX_ACTIONS,)

        # build mask
        mask = torch.full_like(q, -1e9)
        for s in valid_swaps:
            idx = swap_to_idx(*s[0], *s[1])
            mask[idx] = 0.0
        q = q + mask
        best_idx = int(torch.argmax(q).item())
        return idx_to_swap(best_idx)

    # -------- step (store) -------------------------------------------------
    def step(self, state, action, reward, next_state, done, valid_next):
        board, gfeat   = state
        nb, ng         = next_state
        a_idx          = swap_to_idx(*action[0], *action[1])

        self.memory.append((
            board.astype(np.float16), gfeat.astype(np.float16),
            a_idx, reward,
            nb.astype(np.float16), ng.astype(np.float16),
            done, [swap_to_idx(*s[0], *s[1]) for s in valid_next]
        ))

        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and len(self.memory) >= self.batch_size:
            self.learn(random.sample(self.memory, self.batch_size))

    # -------- learn --------------------------------------------------------
    def learn(self, batch):
        (boards, gfeats, a_indices, rewards,
         nboards, ngfeats, dones, valid_next) = zip(*batch)

        boards  = torch.from_numpy(np.stack(boards)).float().to(config.DEVICE)
        gfeats  = torch.from_numpy(np.stack(gfeats)).float().to(config.DEVICE)
        actions = torch.tensor(a_indices, dtype=torch.int64,
                               device=config.DEVICE).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32,
                               device=config.DEVICE).unsqueeze(1)
        nboards = torch.from_numpy(np.stack(nboards)).float().to(config.DEVICE)
        ngfeats = torch.from_numpy(np.stack(ngfeats)).float().to(config.DEVICE)
        dones   = torch.tensor(dones, dtype=torch.float32,
                               device=config.DEVICE).unsqueeze(1)

        # current Q
        q_all   = self.q_local(boards, gfeats)               # (B,A)
        q_exp   = q_all.gather(1, actions)                   # (B,1)

        # target Q using DDQN selection
        with torch.no_grad():
            q_next_local  = self.q_local(nboards, ngfeats)   # (B,A)
            q_next_target = self.q_target(nboards, ngfeats)
            q_tar = []
            for i, valids in enumerate(valid_next):
                if dones[i] or not valids:
                    q_tar.append(0.0)
                    continue
                # mask invalids
                mask = torch.full_like(q_next_local[i], -1e9)
                mask[valids] = 0.0
                best_a = torch.argmax(q_next_local[i] + mask).item()
                q_tar.append(q_next_target[i, best_a].item())
            q_tar = torch.tensor(q_tar, dtype=torch.float32,
                                  device=config.DEVICE).unsqueeze(1)
            q_tar = rewards + self.gamma * q_tar * (1 - dones)

        loss = F.mse_loss(q_exp, q_tar)
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        self._soft_update()

    # -------- soft update --------------------------------------------------
    def _soft_update(self):
        for t, l in zip(self.q_target.parameters(), self.q_local.parameters()):
            t.data.copy_(self.tau * l.data + (1 - self.tau) * t.data)
