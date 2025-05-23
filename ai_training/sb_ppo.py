# train_7wonders_ppo_v4.py
from __future__ import annotations
import gymnasium as gym
import torch
import torch.nn as nn
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
import config                   
from .sw_gym_env import SevenWondersEnv
import os

# --------------------------- global hyper-parameters ---------------------------
N_ENVS        = 8
HORIZON       = 2048    
TOTAL_STEPS   = 10_000_000

LR_INITIAL    = 1e-3
LR_FINAL      = 1e-4

ENT           = 1e-2             
               

CLIP_RANGE    = 0.25
TARGET_KL     = 0.03                   

BATCH_SIZE    = 1024
EPOCHS        = 6

GAMMA         = 0.7
GAE_LAMBDA    = 0.65

MAX_MOVES     = 400

# Current setup - only Level 1
LEVELS = [config.LEVEL_1] * N_ENVS

# Future setup - progressive difficulty levels
# LEVELS = [
#     config.LEVEL_1, config.LEVEL_1, config.LEVEL_1,  # 3× easy
#     config.LEVEL_2,
#     config.LEVEL_3,
#     config.LEVEL_4,
#     config.LEVEL_5,
#     config.LEVEL_6,
# ]

# ------------------------- custom feature extractor ----------------------------

class DWConv(nn.Module):                   # depth-wise separable 3×3 conv
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.depth = nn.Conv2d(ch_in, ch_in, 3, padding=1, groups=ch_in, bias=False)
        self.point = nn.Conv2d(ch_in, ch_out, 1, bias=False)

    def forward(self, x):
        return self.point(self.depth(x))


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = DWConv(ch, ch)
        self.conv2 = DWConv(ch, ch)
        self.norm1 = nn.GroupNorm(8, ch)
        self.norm2 = nn.GroupNorm(8, ch)
        self.act    = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.act(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        return self.act(x + y)


# ─── main extractor ──────────────────────────────────────────
class Match3Extractor(BaseFeaturesExtractor):
    """
    Observation space must contain:
        board   : (17, H, W)   channels = 13 content + 3 bg + 1 mask
        globals : (4,)         any scalar features you track
    features_dim is fixed to 512.
    """
    def __init__(self, observation_space, features_dim: int = 512):
        super().__init__(observation_space, features_dim)

        H, W = observation_space["board"].shape[1:]
        assert (H, W) == (10, 10), "model tuned for 10×10; adapt if different"
        self.H, self.W = H, W

        # ── low-level encoders ───────────────────────────────
        self.content_conv = nn.Sequential(          # gems + specials (13)
            nn.Conv2d(13, 32, 3, padding=1, bias=False),
            nn.GroupNorm(8, 32), nn.ReLU(inplace=True)
        )

        self.bg_conv = nn.Sequential(               # background (3)
            nn.Conv2d(3, 16, 3, padding=1, bias=False),
            nn.GroupNorm(8, 16), nn.ReLU(inplace=True)
        )

        # content-aware attention on background (32+16 → 1 weight map)
        self.bg_attention = nn.Sequential(
            nn.Conv2d(32 + 16, 32, 1, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1), nn.Sigmoid()
        )

        # ── deep stack (pattern detection) ───────────────────
        # inputs: 32 (content) + 16 (bg*) + 1 (mask) + 2 (coords) = 51 ch
        self.deep = nn.Sequential(
            nn.Conv2d(51, 64, 3, padding=1, bias=False),
            nn.GroupNorm(8, 64), nn.ReLU(inplace=True),
            ResBlock(64),
            ResBlock(64),
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.GroupNorm(8, 128), nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),  # global pool
            nn.Flatten()                   # → (B, 128)
        )

        self.board_head = nn.Sequential(
            nn.Linear(128, 256), nn.ReLU(inplace=True), nn.Dropout(0.2)
        )

        self.global_head = nn.Sequential(
            nn.Linear(4, 64), nn.ReLU(inplace=True)
        )

        self.final = nn.Linear(256 + 64, features_dim)

        # ── pre-compute coordinate planes ────────────────────
        xs = torch.linspace(-1, 1, W).view(1, 1, 1, W).expand(1, 1, H, W)
        ys = torch.linspace(-1, 1, H).view(1, 1, H, 1).expand(1, 1, H, W)
        self.register_buffer("coords", torch.cat([xs, ys], dim=1))

    # ─────────────────────────────────────────────────────────
    def forward(self, obs):
        board   = obs["board"]          # (B,17,H,W)
        globals_ = obs["globals"]       # (B,4)

        content   = board[:,  :13]      # (B,13,H,W)
        bg_raw    = board[:, 13:16]     # (B,3,H,W)
        mask      = board[:, 16:17]     # (B,1,H,W)

        f_content = self.content_conv(content)      # (B,32)
        f_bg      = self.bg_conv(bg_raw)            # (B,16)

        w = self.bg_attention(torch.cat([f_content, f_bg], 1))  # (B,1)
        f_bg = f_bg * w                                         # weighted bg

        B = board.size(0)
        coords = self.coords.expand(B, -1, -1, -1)              # (B,2,H,W)

        deep_in = torch.cat([f_content, f_bg, mask, coords], 1) # (B,51,H,W)
        board_lat = self.board_head(self.deep(deep_in))         # (B,256)

        glob_lat  = self.global_head(globals_)                  # (B,64)

        return self.final(torch.cat([board_lat, glob_lat], 1))  # (B,512)


# ----------------------------- schedules ---------------------------------------
def linear_schedule(start: float, end: float):
    """SB3 linear schedule helper."""
    def _schedule(progress_remaining: float):
        return end + (start - end) * progress_remaining
    return _schedule

lr_schedule   = linear_schedule(LR_INITIAL,  LR_FINAL)


# --------------------------- vector-env factory --------------------------------
def make_env(idx: int):
    level = LEVELS[idx % len(LEVELS)]
    def _init() -> gym.Env:
        env = SevenWondersEnv(level=level)
        env = gym.wrappers.TimeLimit(env, MAX_MOVES)
        return env
    return _init

def main():
    vec_env = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])
    vec_env = VecMonitor(vec_env)

    # ------------------------------- Masked PPO -------------------------------------------
    policy_kwargs = dict(
        features_extractor_class   = Match3Extractor,
        features_extractor_kwargs  = dict(features_dim=512),
        net_arch                   = dict(pi=[256, 256], vf=[256, 128]),
    )

    # Create save directory
    save_dir = "ai_training/models/7wonders_ppo_v4"
    os.makedirs(save_dir, exist_ok=True)

    model = MaskablePPO(
        policy               = "MultiInputPolicy",
        env                  = vec_env,
        learning_rate        = LR_INITIAL,
        ent_coef             = ENT,
        n_steps              = HORIZON,
        batch_size           = BATCH_SIZE,
        n_epochs             = EPOCHS,
        gamma                = GAMMA,
        gae_lambda           = GAE_LAMBDA,
        clip_range           = CLIP_RANGE,
        target_kl            = TARGET_KL,
        max_grad_norm        = 0.5,
        verbose              = 0,
        tensorboard_log      = "ai_training/runs/7wonders_ppo_v4",
        policy_kwargs        = policy_kwargs,
        vf_coef=0.25
    )

    try:
        model.learn(
            total_timesteps=TOTAL_STEPS,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("Training interrupted by user")
    finally:
        print("Saving model...")
        model.save(os.path.join(save_dir, "7wonders_ppo_v4_final"))

if __name__ == "__main__":
    main()
