from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from .sw_gym_env import SevenWondersEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import torch
import gymnasium as gym     

N_ENVS       = 8                 # more decorrelated data
HORIZON      = 4096              # n_steps
TOTAL_STEPS  = 10_000_000        # let it actually learn
LR_INITIAL   = 1e-3
LR_FINAL     = 1e-4              # linear decay
ENT_COEF     = 1e-2              # keep entropy up
CLIP_RANGE   = 0.25              # allow larger moves
BATCH_SIZE   = 1024
EPOCHS       = 10
GAMMA        = 0.995             # long episodes
GAE_LAMBDA   = 0.95
MAX_MOVES = 250

class ResidualBlock(nn.Module):
    def __init__(self, ch):
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
    6-layer ResNet on the 17×HxW board planes  +  FC on the 3 global scalars.
    Output dim = 512  (384 from CNN, 128 from globals)
    """
    def __init__(self, obs_space, features_dim=512):
        super().__init__(obs_space, features_dim)
        c, h, w = obs_space["board"].shape

        self.stem = nn.Sequential(
            nn.Conv2d(c, 64, 3, padding=1), nn.ReLU(),
            ResidualBlock(64),
            ResidualBlock(64),
            nn.Conv2d(64, 96, 3, padding=1), nn.ReLU(),
            ResidualBlock(96),
            ResidualBlock(96),
            nn.Flatten(),                                    # → 96 × H × W
        )
        self.head = nn.Sequential(
            nn.Linear(96*h*w, 384), nn.ReLU()
        )
        self.globals = nn.Sequential(
            nn.Linear(3, 128), nn.ReLU()
        )

    def forward(self, obs):
        b = self.head(self.stem(obs["board"]))
        g = self.globals(obs["globals"])
        return torch.cat([b, g], dim=1)


def main():

    def make_env():
        env = gym.wrappers.TimeLimit(SevenWondersEnv(), MAX_MOVES)
        return env

    vec_env = VecMonitor(SubprocVecEnv([make_env]*N_ENVS))

    def lr_schedule_function(progress_remaining):
        return LR_FINAL + (LR_INITIAL - LR_FINAL) * progress_remaining

    lr_schedule = lr_schedule_function

    policy_kwargs = dict(
        features_extractor_class = ResCNNWithGlobals,
        features_extractor_kwargs= dict(features_dim=512),
        net_arch                = [dict(pi=[256,128], vf=[256,128])]
    )

    model = MaskablePPO(
        "MultiInputPolicy",
        vec_env,
        learning_rate        = lr_schedule,
        n_steps              = HORIZON,
        batch_size           = BATCH_SIZE,
        n_epochs             = EPOCHS,
        gamma                = GAMMA,
        gae_lambda           = GAE_LAMBDA,
        clip_range           = CLIP_RANGE,
        ent_coef             = ENT_COEF,
        max_grad_norm        = 0.5,
        target_kl            = 0.03,        # let it move, but stop divergence
        verbose              = 1,
        tensorboard_log      = "ai_training/runs/7wonders_ppo_v3",
        policy_kwargs        = policy_kwargs,
    )
    model.learn(total_timesteps=TOTAL_STEPS)
    model.save("ai_training/models/7wonders_ppo_v2")

if __name__ == "__main__":
    main()