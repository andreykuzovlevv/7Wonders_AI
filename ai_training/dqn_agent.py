# --- dqn_agent.py ---
import random, numpy as np, torch, torch.nn.functional as F
from collections import deque
from dqn_model import QNetwork, device
import config

class DQNAgent:
    """Stores (board, gfeat) separately so we don’t squeeze channels together."""
    def __init__(self, state_shape, action_representation_size=4):
        self.batch_size  = config.BATCH_SIZE
        self.gamma       = config.GAMMA
        self.tau         = config.TAU
        self.update_every= config.UPDATE_EVERY
        self.memory      = deque(maxlen=config.BUFFER_SIZE)
        self.t_step      = 0
        random.seed(config.SEED)
        torch.manual_seed(config.SEED)

        self.q_local  = QNetwork(state_shape[0]).to(device)
        self.q_target = QNetwork(state_shape[0]).to(device)
        self.opt      = torch.optim.Adam(self.q_local.parameters(), lr=config.LR)

    # ---------- utilities ----------
    def _norm_swap(self, swap, rows, cols):
        (r1,c1),(r2,c2) = swap
        return torch.tensor([[r1/rows, c1/cols, r2/rows, c2/cols]],
                            dtype=torch.float32, device=device)

    def act(self, board, gfeat, valid_swaps, eps=0.0):
        if not valid_swaps:                       # no moves – let caller handle
            return None
        if random.random() < eps:                 # explore
            return random.choice(valid_swaps)

        self.q_local.eval()
        with torch.no_grad():
            b = torch.from_numpy(board).unsqueeze(0).to(device)   # (1,C,H,W)
            g = torch.from_numpy(gfeat).unsqueeze(0).to(device)   # (1,3)
            q_best, a_best = -1e9, None
            for s in valid_swaps:
                q = self.q_local(b, g, self._norm_swap(s,*board.shape[-2:])).item()
                if q > q_best:
                    q_best, a_best = q, s
        self.q_local.train()
        return a_best

    def step(self, state, action, reward, next_state, done, valid_next):
        board, gfeat = state
        nb, ng       = next_state
        a_vec        = list(action[0]) + list(action[1])          # int coords

        self.memory.append((
            board.astype(np.float16), gfeat.astype(np.float16),
            a_vec, reward,
            nb.astype(np.float16), ng.astype(np.float16),
            done, valid_next                                         # store valid_next too!
        ))

        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and len(self.memory) >= self.batch_size:
            self.learn(random.sample(self.memory, self.batch_size))

    # ---------- learning ----------
    def learn(self, experiences):
        (boards, gfeats, action_vecs, rewards,
         nboards, ngfeats, dones, valids) = zip(*experiences)

        B = len(boards)
        boards  = torch.from_numpy(np.stack(boards)).float().to(device)          # (B,C,H,W)
        gfeats  = torch.from_numpy(np.stack(gfeats)).float().to(device)          # (B,3)
        actions = torch.tensor(action_vecs, dtype=torch.float32,
                               device=device) / torch.tensor(
                               [config.GRID_ROWS, config.GRID_COLS,
                                config.GRID_ROWS, config.GRID_COLS],
                               dtype=torch.float32, device=device)

        rewards = torch.tensor(rewards, dtype=torch.float32,
                               device=device).unsqueeze(1)
        nboards = torch.from_numpy(np.stack(nboards)).float().to(device)
        ngfeats = torch.from_numpy(np.stack(ngfeats)).float().to(device)
        dones   = torch.tensor(dones, dtype=torch.float32,
                               device=device).unsqueeze(1)

        # Q_expected
        Q_exp = self.q_local(boards, gfeats, actions)

        # Q_targets: for each next state, take max_a' Q_target(s',a')
        Q_next = torch.zeros(B,1, device=device)
        with torch.no_grad():
            for i, vlist in enumerate(valids):
                if (dones[i] > 0) or (not vlist):
                    Q_next[i] = 0.
                else:
                    b  = nboards[i].unsqueeze(0)
                    g  = ngfeats[i].unsqueeze(0)
                    qmax = -1e9
                    for s in vlist:
                        q = self.q_target(b, g,
                            self._norm_swap(s,config.GRID_ROWS,config.GRID_COLS)).item()
                        qmax = max(qmax, q)
                    Q_next[i] = qmax
        Q_tar = rewards + (self.gamma * Q_next * (1-dones))

        loss = F.mse_loss(Q_exp, Q_tar)
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        self._soft_update()

    def _soft_update(self):
        for t,l in zip(self.q_target.parameters(), self.q_local.parameters()):
            t.data.copy_(self.tau*l.data + (1-self.tau)*t.data)
