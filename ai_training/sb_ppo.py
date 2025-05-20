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

class GridWithGlobals(BaseFeaturesExtractor):
    """
    • CNN on the board planes
    • Linear layer on the 3-element global vector
    • Concatenate → final feature vector
    """
    def __init__(self, obs_space, features_dim=256):
        super().__init__(obs_space, features_dim)

        n_chan, H, W = obs_space["board"].shape
        self.cnn = nn.Sequential(
            nn.Conv2d(n_chan, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * H * W, 512 * 3 // 4), nn.ReLU(),     
        )
        self.glb = nn.Sequential(
            nn.Linear(3, 512 * 1 // 4), nn.ReLU(),                
        )
        # final layer just concatenates, so no extra parameters

    def forward(self, obs):
        board_f = self.cnn(obs["board"])
        glob_f  = self.glb(obs["globals"])
        return torch.cat([board_f, glob_f], dim=1)


def main():

    def make_env():
        env = gym.wrappers.TimeLimit(SevenWondersEnv(), MAX_MOVES)
        return env

    vec_env = VecMonitor(SubprocVecEnv([make_env]*N_ENVS))

    lr_schedule = lambda frac: LR_INITIAL - frac*(LR_INITIAL-LR_FINAL)

    policy_kwargs = dict(
        features_extractor_class = GridWithGlobals,
        features_extractor_kwargs= dict(features_dim=512),   # ↑ capacity
        net_arch = [dict(pi=[256,128], vf=[256,128])]
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
        tensorboard_log      = "ai_training/runs/7wonders_ppo_v2",
        policy_kwargs        = policy_kwargs,
    )
    model.learn(total_timesteps=TOTAL_STEPS)

if __name__ == "__main__":
    main()