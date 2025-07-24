# train_7wonders_ppo_v4.py
from __future__ import annotations
import gymnasium as gym

from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
from typing import Callable
import config                   
from .sw_gym_env import SevenWondersEnv
from .feature_extractor import RotInvMatch3Extractor
import os

# --------------------------- global hyper-parameters ---------------------------
N_ENVS        = 8
HORIZON       = 2048    
TOTAL_STEPS   = 10_000_000

LR_INITIAL    = 1e-3
LR_FINAL      = 1e-4

ENT           = 1e-3            
               

CLIP_RANGE    = 0.15
TARGET_KL     = None                  

BATCH_SIZE    = 1024
EPOCHS        = 6

GAMMA         = 0.7
GAE_LAMBDA    = 0.65

MAX_MOVES     = 400

LEVELS = [config.LEVEL_1, config.LEVEL_2, config.LEVEL_3,
          config.LEVEL_4, config.LEVEL_5, config.LEVEL_6,
          config.LEVEL_7]

# Level distribution: 3 x Level 1, 1 each of others
LEVEL_DISTRIBUTION = [
    config.LEVEL_1, config.LEVEL_1,
    config.LEVEL_2, config.LEVEL_3, config.LEVEL_4,
    config.LEVEL_5, config.LEVEL_6, config.LEVEL_7
]

# --------------------------- learning rate schedule ---------------------------
def linear_lr_schedule(initial_lr: float, final_lr: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule that decays from initial_lr to final_lr.
    
    Args:
        initial_lr: Starting learning rate
        final_lr: Ending learning rate
        
    Returns:
        Schedule function that takes remaining progress (1.0 -> 0.0) and returns lr
    """
    def lr_func(progress_remaining: float) -> float:
        # progress_remaining goes from 1.0 (start) to 0.0 (end)
        # We want lr to go from initial_lr to final_lr
        return final_lr + (initial_lr - final_lr) * progress_remaining
    
    return lr_func

# --------------------------- vector-env factory --------------------------------
def make_env(level_config):
    """Create environment factory for specific level"""
    def _init() -> gym.Env:
        env = SevenWondersEnv(level=level_config)
        env = gym.wrappers.TimeLimit(env, MAX_MOVES)
        return env
    return _init

def main():
    # Initialize environments with distributed levels
    vec_env = SubprocVecEnv([make_env(LEVEL_DISTRIBUTION[i]) for i in range(N_ENVS)])
    vec_env = VecMonitor(vec_env)

    # ------------------------------- Recurrent PPO -------------------------------------------
    policy_kwargs = dict(
        features_extractor_class   = RotInvMatch3Extractor,
        net_arch                   = dict(pi=[384, 256], vf=[128, 32]),
        # lstm_hidden_size = 512,
    )

    # Create save directory
    save_dir = "ai_training/models/7wonders_recurrent_ppo_v1_all_levels"
    os.makedirs(save_dir, exist_ok=True)

    # Create learning rate schedule
    lr_schedule = linear_lr_schedule(LR_INITIAL, LR_FINAL)

    model = MaskablePPO(
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
        verbose              = 1,
        tensorboard_log      = "ai_training/runs/7wonders_recurrent_ppo_v1_all_levels",
        policy_kwargs        = policy_kwargs,
        vf_coef=1,
    )

    try:
        model.learn(total_timesteps=TOTAL_STEPS)
    except KeyboardInterrupt:
        print("Training interrupted by user")
    finally:
        print("Saving model...")
        model.save(os.path.join(save_dir, "7wonders_recurrent_ppo_v1_all_levels_final"))

if __name__ == "__main__":
    main()
