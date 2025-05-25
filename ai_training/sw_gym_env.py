import gymnasium as gym
import numpy as np
from typing import Any, Dict

from .game_simulator import SevenWondersSimulator
import config


class SevenWondersEnv(gym.Env):
    """A lightweight Gym wrapper exposing the simulator to RL algorithms.

    Action encoding: we give the agent a *fixed* discrete action space of
    size `rows*cols*4` (swap UP, DOWN, LEFT, RIGHT for every cell).  Before
    sampling an action we mask‐out illegal swaps so the policy never selects
    them.  Invalid-action penalties inside the simulator therefore become an
    additional safety net rather than the primary guard rail.
    """

    metadata = {"render_modes": [None]}

    def __init__(self, rows=config.GRID_ROWS, cols=config.GRID_COLS, level=config.LEVEL_1, seed=config.SEED):

        super().__init__()
        self.sim = SevenWondersSimulator(level=level)
        self.rows, self.cols = rows, cols
        # --- generate unique swaps (right + down only) ---
        self.swap_list = []
        for r in range(rows):
            for c in range(cols):
                if c + 1 < cols:
                    self.swap_list.append(((r, c), (r, c + 1)))
                if r + 1 < rows:
                    self.swap_list.append(((r, c), (r + 1, c)))

        self.n_actions = len(self.swap_list)
        self.action_space = gym.spaces.Discrete(self.n_actions)

        # (17, rows, cols) board tensor  +  (3,) global features  →  Dict space
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

    def _get_action_mask(self, valid_swaps):
        mask = np.zeros(self.n_actions, dtype=bool)
        for swap in valid_swaps:
            idx = self._encode_action(swap)
            mask[idx] = True
        return mask

    def render(self):
        self.sim.display()