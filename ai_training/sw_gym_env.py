import gymnasium as gym
import numpy as np
from typing import Any, Dict, List

from .game_simulator import SevenWondersSimulator
import config


class SevenWondersEnv(gym.Env):
    """A lightweight Gym wrapper exposing the simulator to RL algorithms.

    Action encoding: we give the agent a *fixed* discrete action space of
    size `rows*cols*4` (swap UP, DOWN, LEFT, RIGHT for every cell).  Before
    sampling an action we mask‐out illegal swaps so the policy never selects
    them.  Invalid-action penalties inside the simulator therefore become an
    additional safety net rather than the primary guard rail.
    
    Note: This includes symmetric actions (e.g., both (1,1)→(1,2) and (1,2)→(1,1))
    which may help with exploration and representation learning.
    """

    metadata = {"render_modes": [None]}

    def __init__(self, rows=config.GRID_ROWS, cols=config.GRID_COLS, level=config.LEVEL_1, seed=config.SEED):

        super().__init__()
        self.sim = SevenWondersSimulator(level=level)
        self.rows, self.cols = rows, cols
        self.current_level = level  # Track current level for debugging
        
        # curriculum support  ────────────────────────────────
        self._unlocked_levels: List[int] = [level]    # start with provided level
        
        # --- generate all directional swaps (UP, DOWN, LEFT, RIGHT for each cell) ---
        self.swap_list = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
        
        for r in range(rows):
            for c in range(cols):
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        self.swap_list.append(((r, c), (nr, nc)))

        self.n_actions = len(self.swap_list)
        self.action_space = gym.spaces.Discrete(self.n_actions)

        # (17, rows, cols) board tensor  +  (4,) global features  →  Dict space
        board_low = np.zeros((17, rows, cols), dtype=np.float32)
        board_high = np.ones_like(board_low)
        globals_low = np.zeros(4, dtype=np.float32)
        globals_high = np.ones(4, dtype=np.float32)
        self.observation_space = gym.spaces.Dict(
            {
                "board": gym.spaces.Box(board_low, board_high, dtype=np.float32),
                "globals": gym.spaces.Box(globals_low, globals_high, dtype=np.float32),
            }
        )
        self._current_mask = np.zeros(self.n_actions, dtype=bool)
        self._rng = np.random.default_rng(seed)
        self.reset(seed=seed)

    # -------------------------------------------------------------------------
    # curriculum helpers (called from trainer-side callback)
    def unlock_level(self, lvl: int):
        """Expose a new difficulty to the sampler."""
        if lvl not in self._unlocked_levels:
            self._unlocked_levels.append(lvl)
            print(f"Environment unlocked level {lvl}. Available levels: {self._unlocked_levels}")

    # backward-compat shim (still used by make_env)
    def set_level(self, new_level):
        """Change the level configuration. Takes effect on next reset."""
        self._unlocked_levels = [new_level]
        self.current_level = new_level
        self.sim.level = new_level

    def action_masks(self) -> np.ndarray:
        return self._current_mask

    # --- utility: action index  ↔  Swap tuple ---------------------------------------------------
    def _decode_action(self, action: int) -> config.Swap:
        """
        Given an integer action index, return the corresponding Swap tuple from swap_list.
        """
        return self.swap_list[action]

    def _encode_action(self, swap: config.Swap) -> int:
        """
        Given a Swap tuple, return its index in swap_list.
        Raises ValueError if the swap is not in the list.
        """
        return self.swap_list.index(swap)

    # --- gym core ---------------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        super().reset(seed=seed)
        
        # pick one of the unlocked presets at random
        chosen = self._rng.choice(self._unlocked_levels)
        if chosen != self.current_level:
            self.current_level = chosen
            self.sim.level = chosen
        
        valid_swaps = self.sim.reset()  # simulator returns state_tuple
        self._current_mask = self._get_action_mask(valid_swaps)
        state_tuple = self.sim.get_state_tuple()
        obs_planes, obs_globals = state_tuple  # unpack the state tuple
        obs = {"board": obs_planes, "globals": obs_globals}
        info = {"action_mask": self._current_mask}
        return obs, info
    
    def step(self, action: int):
        swap = self._decode_action(action)
        reward, done, valid_swaps = self.sim.step(swap)  # simulator returns (state_tuple, reward, done)
        self._current_mask = self._get_action_mask(valid_swaps)
        state_tuple = self.sim.get_state_tuple()
        obs_planes, obs_globals = state_tuple  # unpack the state tuple
        info = {"action_mask": self._current_mask}
        obs = {"board": obs_planes, "globals": obs_globals}
        truncated = False
        return obs, reward, done, truncated, info

    # --- helpers ---------------------------------------------------------------------------
    def _get_obs(self):
        planes, globs = self.sim.get_state_tuple()
        return {"board": planes, "globals": globs}

    def _get_action_mask(self, valid_swaps: List[config.Swap]):
        mask = np.zeros(self.n_actions, dtype=bool)
        for swap in valid_swaps:
            try:
                idx = self._encode_action(swap)
                mask[idx] = True
            except ValueError:
                # Skip swaps not in our action space (shouldn't happen)
                pass
        return mask

    def render(self):
        self.sim.display()