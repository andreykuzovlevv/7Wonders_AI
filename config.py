# config.py
# ───── CONFIGURATION ─────

import torch
from typing import Tuple, List, Dict

Coord = Tuple[int, int]
Swap = Tuple[Coord, Coord]

GRID_ROWS = 10
GRID_COLS = 10

TILE_SIZE = 50

GRID_PIXEL_LEFT = 220
GRID_PIXEL_TOP = 20
GRID_PIXEL_RIGHT = GRID_PIXEL_LEFT + TILE_SIZE * 10
GRID_PIXEL_BOTTOM = GRID_PIXEL_TOP + 493

# ── label sets ──────────────────────────────────────────────────────
CONTENT_CLASSES = (
    ["empty"]  # 0
    + [f"gem_{i}" for i in range(8)]  # 1-8
    + ["fragment"]  # 9
    + ["bonus_0", "bonus_1", "bonus_2"]  # 10-12
)  # total 13
BACKGROUND_CLASSES = ["none", "stone", "shield"]  # 0-2


N_CONTENT = len(CONTENT_CLASSES)           # 13
N_BG      = len(BACKGROUND_CLASSES)        # 3
N_PLANES  = N_CONTENT + N_BG + 1           # +1 for mask ⟹ 17

# --- Improved State Representation Constants ---
N_GEM_TYPES = 8                            # Number of gem types (gem_0 to gem_7)
N_IMPROVED_PLANES = N_GEM_TYPES + 2        # 8 gem planes + 1 background + 1 mask = 10 total
MAX_FRAGMENTS_PER_BOARD = 2                # Maximum expected fragments on board at once
MAX_BONUSES_PER_BOARD = 10                  # Maximum expected bonuses on board at once

MAP_FG = {c: i for i, c in enumerate(CONTENT_CLASSES)}
MAP_BG = {c: i for i, c in enumerate(BACKGROUND_CLASSES)}

# --- Action Space Definition ---
# Canonical swap: ((r1, c1), (r2, c2)) where (r1, c1) is lexicographically smaller or equal to (r2, c2)
def get_canonical_swap(swap: Tuple[Tuple[int, int], Tuple[int, int]]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    p1, p2 = swap
    # Sort by r, then by c to ensure a unique representation for dictionary keys
    if p1[0] > p2[0] or (p1[0] == p2[0] and p1[1] > p2[1]):
        return (p2, p1)
    return (p1, p2)

def _generate_all_possible_swaps(rows: int, cols: int) -> Tuple[List[Tuple[Tuple[int, int], Tuple[int, int]]], Dict[Tuple[Tuple[int, int], Tuple[int, int]], int]]:
    possible_swaps = []
    # Horizontal swaps: (r, c) with (r, c+1)
    for r_idx in range(rows):
        for c_idx in range(cols - 1):
            swap = ((r_idx, c_idx), (r_idx, c_idx + 1))
            possible_swaps.append(get_canonical_swap(swap)) # Store canonical form
    # Vertical swaps: (r, c) with (r+1, c)
    for r_idx in range(rows - 1):
        for c_idx in range(cols):
            swap = ((r_idx, c_idx), (r_idx + 1, c_idx))
            possible_swaps.append(get_canonical_swap(swap)) # Store canonical form
    
    # Remove duplicates if any (shouldn't be with this generation method for canonical swaps)
    # and create mapping
    unique_swaps = sorted(list(set(possible_swaps))) # Sort for consistent indexing
    swap_to_idx = {swap: i for i, swap in enumerate(unique_swaps)}
    return unique_swaps, swap_to_idx

ALL_POSSIBLE_SWAPS_LIST, SWAP_TO_INDEX_MAP = _generate_all_possible_swaps(GRID_ROWS, GRID_COLS)
MAX_ACTIONS = len(ALL_POSSIBLE_SWAPS_LIST)

# DQN Hyperparameters
SEED = 42             # Random seed for reproducibility
LR            = 1e-4            # Learning rate
BUFFER_SIZE   = int(1e6)  # Replay buffer size
BATCH_SIZE    = 512       # Minibatch size

TAU           = 5e-3           # For soft update of target parameters
UPDATE_EVERY  = 4         # ↑ 2× the old replay frequency
N_STEPS       = 4        # How often to update the network
NUM_GLOBAL_FEATURES = 3 # e.g., stone_norm, shield_norm, fragment_flag
ACTION_DIM = 4         # r1, c1, r2, c2
MAX_ACTIONS = GRID_ROWS * GRID_COLS * 4   

# --- PPO Hyperparameters ---
LEARNING_RATE = 3e-5       # Adam optimizer learning rate
GAMMA = 0.99               # Discount factor for future rewards
GAE_LAMBDA = 0.95          # Lambda for Generalized Advantage Estimation
CLIP_EPSILON = 0.2         # PPO clipping parameter for surrogate loss
N_EPOCHS_PPO = 10          # Number of epochs to train on a batch of data in PPO
MINIBATCH_SIZE_PPO = 64    # Mini-batch size for PPO updates
ENTROPY_COEFF = 0.02      # Coefficient for entropy bonus (encourages exploration)
VALUE_LOSS_COEFF = 0.5     # Coefficient for value function loss

# --- Training Loop Configuration ---
N_ROLLOUT_STEPS = 2048     # Number of steps to collect from the environment per PPO update cycle
                           # This is often (num_parallel_envs * steps_per_env)
TOTAL_TRAINING_TIMESTEPS = 2_000_000 # Total timesteps for the entire training duration
SAVE_MODEL_FREQ = 50000    # Save model checkpoint every X timesteps
LOG_FREQ = 1000            # Log training statistics every X timesteps

# --- Game Specific ---
MAX_MOVES_PER_EPISODE = 400 # Max moves before game ends (used for step_count normalization in global_features)
                            # Also useful as a timeout for episodes if needed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 

LEVEL_1 = {
    "mask": [
        "##########",
        "#........#",
        "#........#",
        "#........#",
        "#........#",
        "#........#",
        "#........#",
        "#........#",
        "#........#",
        "##########",
    ]
}

LEVEL_2 = {
    "mask": [
        "##########",
        "#...##...#",
        "#........#",
        "#........#",
        "#........#",
        "#........#",
        "#........#",
        "#........#",
        "#........#",
        "##########",
    ]
}

LEVEL_3 = {
    "mask": [
        "##########",
        "#........#",
        "#........#",
        "#..ssss..#",
        "#..ssss..#",
        "#..ssss..#",
        "#..ssss..#",
        "#........#",
        "#........#",
        "##########",
    ]
}

LEVEL_4 = {
    "mask": [
        "########..",
        "#######...",
        "######....",
        "#####.....",
        "####......",
        "###.......",
        "##........",
        "#.........",
        "..........",
        "..........",
    ]
}

LEVEL_5 = {
    "mask": [
        "##########",
        "##########",
        "#######...",
        "###.......",
        "###.......",
        "###.......",
        "###.......",
        "###.......",
        "..........",
        "..........",
    ]
}

LEVEL_6 = {
    "mask": [
        "###....###",
        "..........",
        "...ssss...",
        "..........",
        "###....###",
        "..........",
        "..........",
        "...ssss...",
        "..........",
        "...####...",
    ]
}

LEVEL_7 = {
    "mask": [
        "###.##.###",
        "....##....",
        "..........",
        "#........#",
        "###....###",
        "#........#",
        "...ssss...",
        "...ssss...",
        "..........",
        "...####...",
    ]
}

