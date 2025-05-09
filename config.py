# config.py
# ───── CONFIGURATION ─────

import torch

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

MAP_FG = {c: i for i, c in enumerate(CONTENT_CLASSES)}
MAP_BG = {c: i for i, c in enumerate(BACKGROUND_CLASSES)}

# DQN Hyperparameters
SEED = 42             # Random seed for reproducibility
LR            = 1e-4            # Learning rate
BUFFER_SIZE   = int(5e5)  # Replay buffer size
BATCH_SIZE    = 512       # Minibatch size
GAMMA         = 0.995          # Discount factor
TAU           = 5e-3           # For soft update of target parameters
UPDATE_EVERY  = 4         # ↑ 2× the old replay frequency
N_STEPS       = 20        # How often to update the network
NUM_GLOBAL_FEATURES = 3 # e.g., stone_norm, shield_norm, fragment_flag
ACTION_DIM = 4         # r1, c1, r2, c2
MAX_ACTIONS = GRID_ROWS * GRID_COLS * 4   

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

LEVEL_6 = {
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

