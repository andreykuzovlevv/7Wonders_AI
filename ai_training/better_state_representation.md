# Better state representation

We train agent to play match-3 game.
Currently we represent state like this:
```
 def get_global_features(self) -> np.ndarray:
        """4 floats in [0,1] – tweak as you like."""
        stones_ratio = self.stones_cleared / max(1, self.initial_stones)
        fragments = self.fragments_on_board
        step_count = self.step_count / 250
        bonus_count = self.bonus2_trigger_count % 4 / 4
        return np.array([stones_ratio, fragments, step_count, bonus_count], dtype=np.float32)
    
    def get_state_representation(self) -> np.ndarray:
        """
        Returns a (17, rows, cols) float32 tensor:
        • 13 one‑hot planes for `content`
        • 3  one‑hot planes for `background`
        • 1  binary plane for `mask` (holes)
        """
        # --- 13 content planes --------------------------------------------------
        content_oh = np.eye(config.N_CONTENT, dtype=np.float32)[self.content]          # (rows, cols, 13)
        content_oh = np.transpose(content_oh, (2, 0, 1))                        # (13, rows, cols)

        # --- 3 background planes ----------------------------------------------
        bg_oh = np.eye(config.N_BG, dtype=np.float32)[self.background]                 # (rows, cols, 3)
        bg_oh = np.transpose(bg_oh, (2, 0, 1))                                  # (3, rows, cols)

        # --- 1 mask plane ------------------------------------------------------
        mask_plane = self.mask.astype(np.float32)[None, ...]                    # (1, rows, cols)

        state = np.concatenate([content_oh, bg_oh, mask_plane], axis=0)         # (17, rows, cols)
        return state

    # convenience – one call returns everything the agent stores
    def get_state_tuple(self):
        return (self.get_state_representation(), self.get_global_features()) 
```

game works like this:

    Game have different tables, 10x10, they can be different shapes: 8x8, 10x6+6x4(bottom part for example), 10x8 but in bottom row 3 and 8 is empty. Like that.

    If tile is not empty it has some content and background.
    For content It could have:
    - 8 types of gem
    - 3 types of bonuses
    - Fragment peace (stone block)
    For background it could have:
    - No background (already broken)
    - stone
    - stone + shield

    Main game action is to swap contents of tiles.

```
# config.py
CONTENT_CLASSES = (
    ["empty"]  # 0
    + [f"gem_{i}" for i in range(8)]  # 1-8
    + ["fragment"]  # 9
    + ["bonus_0", "bonus_1", "bonus_2"]  # 10-12
)  # total 13
BACKGROUND_CLASSES = ["none", "stone", "shield"]  # 0-2
```

Would be more logical approah if we represent like this:

    8x10x10 one-hot gems plane
    1x10x10 background plane with 0-1-2 values for 'none'-'stone'-'shield'
    1x10x10 mask plane

    I think it would be more logical to represent every fragment location as coord, like (1,1).
    Can we have different multiple of those in representation? Like list? what we can do here?

    And same for bonus, but with coord we need also pair it with bonus type like ((1,5), 1),
    and we also can have one or multiple of those.


# New feature extractor
how we can then reimplement feature extractor for new state representation?
current:
```
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
class Match3Extractor(BaseFeaturesExtractor):
    """
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
            nn.Linear(128, 256), nn.ReLU(inplace=True), nn.Dropout(0.2)
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
```
## What we need
I mean we need somehow extract patterns from each gem plane SEPARATLY.

We need to extract features like this (most common+variations, so you can understand what we dealing with)
gg0g

gg0
00g

gg0g
00g0

etc,
any pre-match pattern could fit into 5x5
with the biggest i think something like this
0g00
0g00
g0gg
0g00
0g00

so any patterns shown can be rotated, mirored and still create pre-match state feature.
do i need to know exact number of every possible pattern for filter of something?

And standalone gems are important too, coz they can form a cascade match,
or lead to valuable states  in the future.

We dont need to extract any particular feature from mask or background,
Ai just needs to be aware what each tile is

And Fragment cant be swapped, it can only be lifted down by making mathes below it.
And Bonuses can be swapped with anything except Fragment and masked tile.


# Action space

Here is enviroment wrapper that defines action space:
```
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
```
Right now it's 180 actions for board 10x10.

What type of actions space there are could be with ppo?

I think it would be cool if model somethow predict exact coords of the swap like (1,1)(1,2).
Is it practical here?