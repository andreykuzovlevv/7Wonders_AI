# train_7wonders_dqn_v1.py
from __future__ import annotations
import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed
import config                   
from .sw_gym_env import SevenWondersEnv
import os

# --------------------------- global hyper-parameters ---------------------------
N_ENVS        = 4               # DQN typically uses fewer parallel envs
TOTAL_STEPS   = 2_000_000       # DQN often needs fewer total steps

LR_INITIAL    = 1e-4            # DQN typically uses lower learning rates
LR_FINAL      = 1e-5

BUFFER_SIZE   = 100_000         # Replay buffer size
BATCH_SIZE    = 32              # Smaller batches for DQN
LEARNING_STARTS = 10_000        # Steps before learning starts

# Exploration schedule
EXPLORATION_INITIAL = 1.0       # Start with full exploration
EXPLORATION_FINAL   = 0.05      # End with 5% exploration
EXPLORATION_FRACTION = 0.3      # Fraction of total steps for exploration decay

TARGET_UPDATE_INTERVAL = 1000   # Update target network every N steps
TRAIN_FREQ = 4                  # Train every N steps
GRADIENT_STEPS = 1              # Number of gradient steps per training

GAMMA = 0.99                    # Discount factor (higher for DQN)
TAU = 1.0                       # Hard update (1.0) vs soft update (< 1.0)

MAX_MOVES = 400

# Levels configuration
LEVELS = [
    config.LEVEL_1, config.LEVEL_1, config.LEVEL_1,  # 3× easy
    config.LEVEL_2,
]

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
class Match3ExtractorDQN(BaseFeaturesExtractor):
    """
    Feature extractor optimized for DQN.
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
            nn.Linear(128, 256), nn.ReLU(inplace=True), nn.Dropout(0.1)  # Lower dropout for DQN
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


# ----------------------------- action masking wrapper -------------------------
class ActionMaskingWrapper(gym.Wrapper):
    """
    Wrapper that applies action masking to DQN by modifying Q-values.
    Sets Q-values of invalid actions to a very negative value.
    """
    def __init__(self, env):
        super().__init__(env)
        self.mask_value = -1e8  # Very negative value for invalid actions
        
    def get_action_mask(self):
        """Get current action mask from environment"""
        if hasattr(self.env, 'action_masks'):
            return self.env.action_masks()
        elif hasattr(self.env.unwrapped, 'action_masks'):
            return self.env.unwrapped.action_masks()
        else:
            # If no masking available, all actions are valid
            return np.ones(self.action_space.n, dtype=bool)

    def predict_with_mask(self, model, obs, deterministic=True):
        """
        Custom predict function that applies action masking.
        This should be used instead of model.predict() during evaluation.
        """
        # Get Q-values from model
        q_values, _ = model.predict(obs, deterministic=False)  # Get raw Q-values
        
        # Get action mask
        action_mask = self.get_action_mask()
        
        # Apply mask: set invalid actions to very negative values
        if hasattr(q_values, 'cpu'):  # If tensor
            q_values = q_values.cpu().numpy()
        
        masked_q_values = q_values.copy()
        masked_q_values[~action_mask] = self.mask_value
        
        # Select best valid action
        if deterministic:
            action = np.argmax(masked_q_values)
        else:
            # Epsilon-greedy with masking
            if np.random.random() < model.exploration_rate:
                valid_actions = np.where(action_mask)[0]
                action = np.random.choice(valid_actions)
            else:
                action = np.argmax(masked_q_values)
                
        return action


# ----------------------------- schedules ---------------------------------------
def linear_schedule(start: float, end: float):
    """SB3 linear schedule helper."""
    def _schedule(progress_remaining: float):
        return end + (start - end) * progress_remaining
    return _schedule

lr_schedule = linear_schedule(LR_INITIAL, LR_FINAL)
exploration_schedule = linear_schedule(EXPLORATION_INITIAL, EXPLORATION_FINAL)


# --------------------------- vector-env factory --------------------------------
def make_env(idx: int):
    level = LEVELS[idx % len(LEVELS)]
    def _init() -> gym.Env:
        env = SevenWondersEnv(level=level)
        env = gym.wrappers.TimeLimit(env, MAX_MOVES)
        # Note: We'll apply masking during evaluation, not training
        # This is because SB3's DQN doesn't natively support action masking
        return env
    return _init

# ----------------------------- training callback --------------------------------
class DQNProgressCallback(BaseCallback):
    """Callback to log DQN-specific metrics"""
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # Log exploration rate
        if hasattr(self.model, 'exploration_rate'):
            self.logger.record('train/exploration_rate', self.model.exploration_rate)
        return True

def main():
    # Set random seed for reproducibility
    set_random_seed(42)
    
    vec_env = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])
    vec_env = VecMonitor(vec_env)

    # ------------------------------- Standard DQN -------------------------------------------
    policy_kwargs = dict(
        features_extractor_class   = Match3ExtractorDQN,
        features_extractor_kwargs  = dict(features_dim=512),
        net_arch                   = [512, 256, 128],  # DQN uses single network architecture
    )

    # Create save directory
    save_dir = "ai_training/models/7wonders_dqn_v1"
    os.makedirs(save_dir, exist_ok=True)

    model = DQN(
        policy               = "MultiInputPolicy",
        env                  = vec_env,
        learning_rate        = lr_schedule,
        buffer_size          = BUFFER_SIZE,
        learning_starts      = LEARNING_STARTS,
        batch_size           = BATCH_SIZE,
        tau                  = TAU,
        gamma                = GAMMA,
        train_freq           = TRAIN_FREQ,
        gradient_steps       = GRADIENT_STEPS,
        target_update_interval = TARGET_UPDATE_INTERVAL,
        exploration_fraction = EXPLORATION_FRACTION,
        exploration_initial_eps = EXPLORATION_INITIAL,
        exploration_final_eps = EXPLORATION_FINAL,
        max_grad_norm        = 10.0,  # Higher for DQN
        verbose              = 1,
        tensorboard_log      = "ai_training/runs/7wonders_dqn_v1",
        policy_kwargs        = policy_kwargs,
    )

    # Create callback
    callback = DQNProgressCallback()

    try:
        model.learn(
            total_timesteps=TOTAL_STEPS,
            callback=callback,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("Training interrupted by user")
    finally:
        print("Saving model...")
        model.save(os.path.join(save_dir, "7wonders_dqn_v1_final"))
        print(f"Model saved to {save_dir}")

def evaluate_with_masking():
    """
    Example of how to evaluate the trained model with action masking.
    """
    print("\n=== Evaluating DQN with Action Masking ===")
    
    # Load the model
    save_dir = "ai_training/models/7wonders_dqn_v1"
    model_path = os.path.join(save_dir, "7wonders_dqn_v1_final")
    
    if not os.path.exists(model_path + ".zip"):
        print(f"Model not found at {model_path}. Please train first.")
        return
    
    model = DQN.load(model_path)
    
    # Create a single environment with action masking
    env = SevenWondersEnv(level=config.LEVEL_1)
    env = gym.wrappers.TimeLimit(env, MAX_MOVES)
    env = ActionMaskingWrapper(env)
    
    # Evaluate for a few episodes
    for episode in range(3):
        obs, _ = env.reset()
        total_reward = 0
        steps = 0
        
        while True:
            # Use our custom predict function with masking
            action = env.predict_with_mask(model, obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
        
        print(f"Episode {episode+1}: Reward = {total_reward:.2f}, Steps = {steps}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "evaluate":
        evaluate_with_masking()
    else:
        main() 