# --- File: dqn_model.py ----------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import config                           # same file you already use

print(f"Using device: {config.DEVICE}")

# --------------------------------------------------------------------------
# 1️⃣  Residual Block
# --------------------------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x):
        skip = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = F.relu(x + skip)
        return x

# --------------------------------------------------------------------------
# 2️⃣  Dueling‑DQN Network
#    • Takes the whole state once and returns Q‑values for every swap id
#    • You mask illegal actions *outside* the network
# --------------------------------------------------------------------------
class QNetwork(nn.Module):
    def __init__(self, num_actions=config.MAX_ACTIONS, num_global_features: int = 3):
        super().__init__()

        # ---------- visual backbone ---------------------------------------
        self.stem = nn.Sequential(
            nn.Conv2d(config.N_PLANES, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.res_layers = nn.Sequential(*[ResBlock(64) for _ in range(6)])

        flat_size = 64 * config.GRID_ROWS * config.GRID_COLS

        # ---------- non‑visual branch -------------------------------------
        self.fc_global = nn.Sequential(
            nn.Linear(num_global_features, 64),
            nn.ReLU(inplace=True),
        )

        # ---------- head ---------------------------------------------------
        self.fc_shared = nn.Sequential(
            nn.Linear(flat_size + 64, 256),
            nn.ReLU(inplace=True),
        )

        # Dueling split: value & advantage
        self.fc_value = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))
        self.fc_adv   = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, num_actions))

    def forward(self, state, global_feat):
        """
        state        : (B, 17, H, W) float32
        global_feat  : (B, 3)         float32
        returns      : (B, num_actions) Q‑values  (no masking here)
        """
        x = self.stem(state)
        x = self.res_layers(x)
        x = x.flatten(start_dim=1)                     # (B, flat_size)

        g = self.fc_global(global_feat)

        x = torch.cat([x, g], dim=1)
        x = self.fc_shared(x)

        v = self.fc_value(x)                           # (B, 1)
        a = self.fc_adv(x)                             # (B, num_actions)
        q = v + (a - a.mean(dim=1, keepdim=True))      # dueling combine
        return q
