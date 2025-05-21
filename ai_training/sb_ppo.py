# train_7wonders_ppo_v4.py
# detached, minimal-boilerplate version – ready to run
from __future__ import annotations
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
import config                           # your constants + LEVEL_1 … LEVEL_6
from .sw_gym_env import SevenWondersEnv

# --------------------------- global hyper-parameters ---------------------------
N_ENVS        = 8
HORIZON       = 4_096                   # n_steps
TOTAL_STEPS   = 10_000_000

LR_INITIAL    = 1e-3
LR_FINAL      = 1e-4

ENT_INITIAL   = 1e-2                    # start with exploration
ENT_FINAL     = 0.0                     # end fully greedy

CLIP_RANGE    = 0.25
TARGET_KL     = 0.03                    # stricter than before

BATCH_SIZE    = 2_048
EPOCHS        = 10

GAMMA         = 0.99
GAE_LAMBDA    = 0.95

MAX_MOVES     = 250                     # TimeLimit

LEVELS        = [
    config.LEVEL_1, config.LEVEL_1, config.LEVEL_1,  # 3× easy
    config.LEVEL_2,
    config.LEVEL_3,
    config.LEVEL_4,
    config.LEVEL_5,
    config.LEVEL_6,
]

# ------------------------- custom feature extractor ----------------------------
class ResidualBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.act   = nn.ReLU()

    def forward(self, x):
        y = self.act(self.conv1(x))
        y = self.conv2(y)
        return self.act(x + y)

class ResCNNWithGlobals(BaseFeaturesExtractor):
    """
    6-layer ResNet on 17×H×W board planes + FC on 4 global scalars.
    Output dim = 512 (384 from CNN, 128 from globals)
    """
    def __init__(self, obs_space, features_dim: int = 512):
        super().__init__(obs_space, features_dim)
        c, h, w = obs_space["board"].shape

        self.stem = nn.Sequential(
            nn.Conv2d(c, 64, 3, padding=1), nn.ReLU(),
            ResidualBlock(64),
            ResidualBlock(64),
            nn.Conv2d(64, 96, 3, padding=1), nn.ReLU(),
            ResidualBlock(96),
            ResidualBlock(96),
            nn.Flatten(),                         # → 96 × H × W
        )
        self.head = nn.Sequential(
            nn.Linear(96 * h * w, 384), nn.ReLU()
        )
        self.globals = nn.Sequential(
            nn.Linear(4, 128), nn.ReLU()          # globals dim unchanged
        )

    def forward(self, obs):
        board_lat = self.head(self.stem(obs["board"]))
        glob_lat  = self.globals(obs["globals"])
        return torch.cat([board_lat, glob_lat], dim=1)

# ----------------------------- schedules ---------------------------------------
def linear_schedule(start: float, end: float):
    """SB3 linear schedule helper."""
    def _schedule(progress_remaining: float):
        return end + (start - end) * progress_remaining
    return _schedule

lr_schedule   = linear_schedule(LR_INITIAL,  LR_FINAL)

class EntropyDecayCallback(BaseCallback):
    """Linearly decays model.ent_coef from ENT_INITIAL to ENT_FINAL."""
    def _on_rollout_end(self) -> None:
        progress = self.model._current_progress_remaining
        new_coef = ENT_FINAL + (ENT_INITIAL - ENT_FINAL) * progress
        # needs to be a torch tensor on the same device as the model
        self.model.ent_coef = torch.as_tensor(new_coef, device=self.model.device)

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
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,                       # observation normalisation
        norm_reward=True,                    # reward normalisation
        clip_obs=5.0,
    )

    # ------------------------------- PPO -------------------------------------------
    policy_kwargs = dict(
        features_extractor_class   = ResCNNWithGlobals,
        features_extractor_kwargs  = dict(features_dim=512),
        net_arch                   = dict(pi=[256, 128], vf=[256, 128]),
    )

    model = MaskablePPO(
        policy               = "MultiInputPolicy",
        env                  = vec_env,
        learning_rate        = lr_schedule,
        ent_coef             = ENT_INITIAL,
        n_steps              = HORIZON,
        batch_size           = BATCH_SIZE,
        n_epochs             = EPOCHS,
        gamma                = GAMMA,
        gae_lambda           = GAE_LAMBDA,
        clip_range           = CLIP_RANGE,
        target_kl            = TARGET_KL,
        max_grad_norm        = 0.5,
        verbose              = 1,
        tensorboard_log      = "ai_training/runs/7wonders_ppo_v4",
        policy_kwargs        = policy_kwargs,
    )
    model.learn(total_timesteps=TOTAL_STEPS, callback=EntropyDecayCallback())
    model.save("ai_training/models/7wonders_ppo_v4")

if __name__ == "__main__":
    main()
