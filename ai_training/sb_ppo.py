# train_7wonders_ppo_v4.py
from __future__ import annotations
import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3 import PPO
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

ENT           = 7e-3             
               

CLIP_RANGE    = 0.25
TARGET_KL     = 0.03                   

BATCH_SIZE    = 1024
EPOCHS        = 6

GAMMA         = 0.7
GAE_LAMBDA    = 0.65

MAX_MOVES     = 250

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

class ImprovedFeatureExtractor(BaseFeaturesExtractor):
    """
    Add attention mechanism to focus on stone-breaking opportunities
    """
    
    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)
        
        # Separate processing for different plane types
        self.content_conv = nn.Sequential(
            nn.Conv2d(13, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
        )
        
        self.background_conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
        )
        
        # Attention mechanism for stone locations
        self.stone_attention = nn.Sequential(
            nn.Conv2d(16, 8, 1), nn.Sigmoid()  # Attention weights for background
        )
        
        # Combined processing
        self.combined_conv = nn.Sequential(
            nn.Conv2d(48 + 1, 64, 3, padding=1), nn.ReLU(),  # 32+16+1 input channels
            ResidualBlock(64),
            ResidualBlock(64),
            nn.Conv2d(64, 96, 3, padding=1), nn.ReLU(),
            nn.Flatten()
        )
        
        # Assume 10x10 board
        self.head = nn.Sequential(
            nn.Linear(96 * 100, 384), nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.globals_net = nn.Sequential(
            nn.Linear(4, 128), nn.ReLU(),  # Updated for new global features
            nn.Dropout(0.2)
        )
        
    def forward(self, observations):
        board = observations["board"]
        globals_feat = observations["globals"]
        
        # Split board into content, background, and mask
        content = board[:, :13, :, :]      # First 13 channels
        background = board[:, 13:16, :, :] # Next 3 channels  
        mask = board[:, 16:17, :, :]       # Last channel
        
        # Process content and background separately
        content_feat = self.content_conv(content)
        background_feat = self.background_conv(background)
        
        # Apply attention to background features (focus on stones)
        attention_weights = self.stone_attention(background_feat)
        background_feat = background_feat * attention_weights
        
        # Combine all features
        combined = torch.cat([content_feat, background_feat, mask], dim=1)
        combined_feat = self.head(self.combined_conv(combined))
        
        # Process global features
        global_feat = self.globals_net(globals_feat)
        
        return torch.cat([combined_feat, global_feat], dim=1)

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


    # ------------------------------- PPO -------------------------------------------
    policy_kwargs = dict(
        features_extractor_class   = ImprovedFeatureExtractor,
        features_extractor_kwargs  = dict(features_dim=512),
        net_arch                   = dict(pi=[256, 128], vf=[256, 128]),
    )

    # Create save directory
    save_dir = "ai_training/models/7wonders_ppo_v4"
    os.makedirs(save_dir, exist_ok=True)

    model = PPO(
        policy               = "MultiInputPolicy",
        env                  = vec_env,
        learning_rate        = lr_schedule,
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



    model.learn(
        total_timesteps=TOTAL_STEPS,
        progress_bar=True,
    )
    model.save(os.path.join(save_dir, "7wonders_ppo_v4_final"))

if __name__ == "__main__":
    main()
