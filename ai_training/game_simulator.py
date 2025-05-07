# --- game_simulator.py ---
import numpy as np
import random
from typing import List, Tuple, Set
from collections import deque
import config
import math
# Type hint for a swap action
Coord = Tuple[int, int]
Swap = Tuple[Coord, Coord]

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
        "###.##.###",
        "....##....",
        "..........",
        "#........#",
        "###....###",
        "#........#",
        "...ssss...",
        "...ssss...",
        "..........",
        "..######..",
    ]
}

class SevenWondersSimulator:
    def __init__(self, rows=config.GRID_ROWS, cols=config.GRID_COLS, level=LEVEL_1, debug_mode=False):
        # Allow configurable size, but default to config
        self.rows = rows
        self.cols = cols
        self.content = np.zeros((self.rows, self.cols), dtype=int)
        self.background = np.zeros((self.rows, self.cols), dtype=int)
        self.score = 0
        self.bonus2_trigger_count = 0  # Counter for bonus_0/bonus_1 activations
        self.initial_stones = 0
        self.stones_cleared = 0
        self.fragments_on_board = 0
        self.max_fragments = 5  # Example limit
        self.fragment_spawned = False 
        self.debug_mode = debug_mode  # Flag to control debug output

        # Map string names to integers using config
        self.map_fg = config.MAP_FG  # {'empty':0, 'gem_0':1, …, 'bonus_2':12}
        self.map_bg = config.MAP_BG  # {'none':0, 'stone':1, 'shield':2}
        # Reverse map for debugging/display if needed
        self.rev_map_fg = {v: k for k, v in self.map_fg.items()}
        self.rev_map_bg = {v: k for k, v in self.map_bg.items()}

        # Define integer constants for easier access
        self.EMPTY = self.map_fg["empty"]
        self.FRAGMENT = self.map_fg["fragment"]
        self.GEM_START_IDX = self.map_fg["gem_0"]
        self.GEM_END_IDX = self.map_fg["gem_7"]
        self.GEMS = [self.map_fg[f"gem_{i}"] for i in range(8)]
        self.BONUS_0 = self.map_fg["bonus_0"]
        self.BONUS_1 = self.map_fg["bonus_1"]
        self.BONUS_2 = self.map_fg["bonus_2"]
        self.BG_NONE = self.map_bg["none"]
        self.BG_STONE = self.map_bg["stone"]
        self.BG_SHIELD = self.map_bg["shield"]

        self.level = level  # store level
        self.mask = None  # will be set in reset

        self.step_count = 0

        self.reset()

    def reset(self):
        """Resets the board to a starting state."""
        self.score = 0
        self.bonus2_trigger_count = 0
        self.stones_cleared = 0
        self.fragments_on_board = 0
        self.initial_stones = 0
        self.fragment_spawned = False
        self.step_count = 0

        # --- Board Initialization ---
        self._init_from_level(self.level)

        return self.get_state_representation()

    def _init_from_level(self, level_dict):
        self.mask = np.ones((self.rows, self.cols), dtype=bool)
        self.content = np.full((self.rows, self.cols), self.EMPTY, dtype=int)
        self.background = np.full((self.rows, self.cols), self.BG_NONE, dtype=int)

        for r, row in enumerate(level_dict["mask"]):
            for c, ch in enumerate(row):
                if ch == "#":
                    self.mask[r, c] = False  # mark hole
                    self.content[r, c] = self.EMPTY
                    self.background[r, c] = self.BG_NONE
                elif ch == "s":
                    self.mask[r, c] = True
                    self.content[r, c] = random.randint(
                        self.GEM_START_IDX, self.GEM_END_IDX
                    )
                    self.background[r, c] = self.BG_SHIELD
                elif ch == ".":
                    self.mask[r, c] = True
                    self.content[r, c] = random.randint(
                        self.GEM_START_IDX, self.GEM_END_IDX
                    )
                    self.background[r, c] = self.BG_STONE
                else:
                    raise ValueError(f"Invalid level character: {ch}")

        # Ensure no initial matches
        while self._find_matches():
            for r in range(self.rows):
                for c in range(self.cols):
                    if self.mask[r, c]:
                        self.content[r, c] = random.randint(
                            self.GEM_START_IDX, self.GEM_END_IDX
                        )

        self.initial_stones = np.sum(
            (self.background == self.BG_STONE) | (self.background == self.BG_SHIELD)
        )

    def _is_valid_coord(self, r, c) -> bool:
        """Checks if coordinates are within the board bounds."""
        return 0 <= r < self.rows and 0 <= c < self.cols and self.mask[r, c]

    def _run_length(self, line, pos, val, direction):
        """Helper to count consecutive matching values in a line."""
        count = 0
        pos += direction
        while 0 <= pos < len(line) and line[pos] == val:
            count += 1
            pos += direction
        return count

    def _swap_cells(self, r1, c1, r2, c2):
        """Helper to swap two cells on the board."""
        self.content[r1, c1], self.content[r2, c2] = self.content[r2, c2], self.content[r1, c1]

    def swap_creates_match(self, r1, c1, r2, c2):
        """Check if swapping two cells creates a match, only looking at affected rows/columns."""
        for (r, c) in [(r1, c1), (r2, c2)]:
            val = self.content[r, c]
            
            # Check horizontal matches
            count = 1 + self._run_length(self.content[r, :], c, val, -1) \
                      + self._run_length(self.content[r, :], c, val, +1)
            if count >= 3:
                return True

            # Check vertical matches
            count = 1 + self._run_length(self.content[:, c], r, val, -1) \
                      + self._run_length(self.content[:, c], r, val, +1)
            if count >= 3:
                return True

        return False

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

    def _find_matches(self) -> Set[Tuple[int, int]]:
        """
        Finds all coordinates of gems involved in matches (3 or more).
        Does NOT match bonuses or fragments.
        Returns a set of (row, col) tuples.
        """
        matches = set()
        gem_mask = (self.content >= self.GEM_START_IDX) & (
            self.content <= self.GEM_END_IDX
        )

        # First pass: find all horizontal and vertical matches
        h_matches = set()
        v_matches = set()
        
        for r in range(self.rows):
            for c in range(self.cols):
                if not self._is_valid_coord(r, c) or self.content[r, c] not in self.GEMS:
                    continue 

                gem_type = self.content[r, c]

                # Check horizontal
                h_len = 0
                while (
                    self._is_valid_coord(r, c + h_len)
                    and self.content[r, c + h_len] == gem_type
                ):
                    h_len += 1

                # Add horizontal matches
                if h_len >= 3:
                    for i in range(h_len):
                        h_matches.add((r, c + i))

                # Check vertical
                v_len = 0
                while (
                    self._is_valid_coord(r + v_len, c)
                    and self.content[r + v_len, c] == gem_type
                ):
                    v_len += 1

                # Add vertical matches
                if v_len >= 3:
                    for i in range(v_len):
                        v_matches.add((r + i, c))

        # Combine horizontal and vertical matches
        matches.update(h_matches)
        matches.update(v_matches)

        return matches


    def _get_match_details(self, matches: Set[Tuple[int, int]], swap_action=None):
        """
        Analyse the set of match‑coordinates and split them into true clusters
        (connected, **same‑colour** groups).  Returns bonus placements and reward.
        """
        clusters = []
        remaining = set(matches)           # still unprocessed squares

        DIRS = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        # ────────────────────────────────────────────────────────────────
        # 1.  Build connected components, colour‑aware
        # ────────────────────────────────────────────────────────────────
        while remaining:
            start          = remaining.pop()                 # seed
            colour         = self.content[start]             # gem ID at the seed
            cluster        = {start}
            queue          = deque([start])

            while queue:
                r, c = queue.popleft()
                for dr, dc in DIRS:
                    nr, nc = r + dr, c + dc
                    if (nr, nc) in remaining and self.content[nr, nc] == colour:
                        remaining.remove((nr, nc))
                        cluster.add((nr, nc))
                        queue.append((nr, nc))

            clusters.append(cluster)

        # ────────────────────────────────────────────────────────────────
        # 2.  Reward + bonus logic (unchanged apart from the loop header)
        # ────────────────────────────────────────────────────────────────
        bonus_placements = []
        total_reward     = 0

        for cluster in clusters:
            cluster_size = len(cluster)
            colour       = self.content[next(iter(cluster))]

            cluster_reward = 2 * cluster_size
            if cluster_size >= 6:
                cluster_reward += 30
            elif cluster_size == 5:
                cluster_reward += 20
            elif cluster_size == 4:
                cluster_reward += 10
            total_reward += cluster_reward

            # 4‑match → BONUS_0, 5+ → BONUS_1
            if cluster_size >= 4:
                bonus_type = self.BONUS_0 if cluster_size == 4 else self.BONUS_1

                # choose bonus position
                if swap_action and swap_action in cluster:
                    bonus_pos = swap_action
                else:                                  # centre‑of‑mass fallback
                    ar = sum(r for r, _ in cluster) / cluster_size
                    ac = sum(c for _, c in cluster) / cluster_size
                    bonus_pos = min(cluster, key=lambda p: (p[0]-ar)**2 + (p[1]-ac)**2)

                bonus_placements.append((*bonus_pos, bonus_type))

        return {
            "clusters":          clusters,
            "bonus_placements":  bonus_placements,
            "total_reward":      total_reward,
        }


    def get_valid_swaps(self) -> List[Swap]:
        """
        Return every legal swap the player can make.

        Rules
        -----
        • Adjacent only (no diagonals).
        • Tiles may NOT be EMPTY or FRAGMENT.
        • If either tile is a BONUS (bonus_0, bonus_1, bonus_2) → the swap is
        immediately valid (bonuses always fire when moved).
        • Otherwise the swap must create ≥ 1 gem match.
        """
        valid_swaps: List[Swap] = []

        def is_swappable(r: int, c: int) -> bool:
            """Can the tile at (r, c) take part in a swap?"""
            if not self._is_valid_coord(r, c):
                return False
            # If you're using a mask for holes, skip them
            if not self.mask[r, c]:
                return False
            t = self.content[r, c]
            return t != self.EMPTY and t != self.FRAGMENT

        def is_bonus(tile_val: int) -> bool:
            return self.BONUS_0 <= tile_val <= self.BONUS_2

        # Iterate through every cell; consider right‑ and bottom‑ neighbour only → no duplicates
        for r in range(self.rows):
            for c in range(self.cols):
                if not is_swappable(r, c):
                    continue

                # --- swap with right neighbour ---------------------------------
                if c + 1 < self.cols and is_swappable(r, c + 1):
                    t1, t2 = self.content[r, c], self.content[r, c + 1]

                    if is_bonus(t1) or is_bonus(t2):
                        valid_swaps.append(((r, c), (r, c + 1)))
                    else:
                        # need to test for a resulting match
                        self._swap_cells(r, c, r, c + 1)
                        if self.swap_creates_match(r, c, r, c + 1):
                            valid_swaps.append(((r, c), (r, c + 1)))
                        # restore board
                        self._swap_cells(r, c, r, c + 1)

                # --- swap with bottom neighbour -------------------------------
                if r + 1 < self.rows and is_swappable(r + 1, c):
                    t1, t2 = self.content[r, c], self.content[r + 1, c]

                    if is_bonus(t1) or is_bonus(t2):
                        valid_swaps.append(((r, c), (r + 1, c)))
                    else:
                        self._swap_cells(r, c, r + 1, c)
                        if self.swap_creates_match(r, c, r + 1, c):
                            valid_swaps.append(((r, c), (r + 1, c)))
                        self._swap_cells(r, c, r + 1, c)

        return valid_swaps

    def _apply_gravity(self) -> bool:
        """
        Drop everything as far as it can fall, then collect any fragment that reaches
        the lowest playable cell of its column.

        Returns
        -------
        moved : bool
            True if any tile moved **or** a fragment was collected.
        """
        moved = False

        # ──────────────────────────────────────────
        # A.   drop tiles column by column
        # ──────────────────────────────────────────
        for c in range(self.cols):
            # 'write' walks upward, always pointing at the next empty spot
            write = None
            for r in range(self.rows - 1, -1, -1):          # bottom → top
                if not self._is_valid_coord(r, c):
                    continue                                # skip holes

                if self.content[r, c] == self.EMPTY:
                    if write is None:
                        write = r                           # first empty found
                else:
                    if write is not None:                   # found a piece above an empty
                        self.content[write, c] = self.content[r, c]
                        self.content[r, c] = self.EMPTY
                        moved = True
                        write -= 1                          # next empty is just above
            # end for r
        # end for c

        # ──────────────────────────────────────────
        # B.   collect fragments on the bottom
        # ──────────────────────────────────────────
        fragment_removed = False
        for c in range(self.cols):
            # locate the lowest playable square in this column
            for r in range(self.rows - 1, -1, -1):
                if self._is_valid_coord(r, c):
                    if self.content[r, c] == self.FRAGMENT:
                        self.content[r, c] = self.EMPTY
                        self.fragments_on_board -= 1
                        fragment_removed = moved = True
                    break                                   # only the lowest cell matters

        # if removing a fragment created gaps, do one extra gravity pass (iteration, not recursion)
        if fragment_removed:
            for c in range(self.cols):
                write = None
                for r in range(self.rows - 1, -1, -1):
                    if not self._is_valid_coord(r, c):
                        continue
                    if self.content[r, c] == self.EMPTY:
                        if write is None:
                            write = r
                    else:
                        if write is not None:
                            self.content[write, c] = self.content[r, c]
                            self.content[r, c] = self.EMPTY
                            moved = True
                            write -= 1

        return moved


    def _refill_board(self) -> bool:
        """
        Fill every EMPTY cell with a random gem.  The very first playable square
        in each column is treated as the "top‑row" candidate for spawning specials.

        Returns
        -------
        refilled : bool
            True if any tile was created (gems, bonus_2 or fragment).
        """
        refilled = False
        # track which top‑row squares were filled during this pass
        freshly_filled_top = []

        # ──────────────────────────────────────────
        # A.   fill all empties + remember topmost ones
        # ──────────────────────────────────────────
        for c in range(self.cols):
            top_playable_row = None
            for r in range(self.rows):
                if self._is_valid_coord(r, c):
                    top_playable_row = r
                    break
            if top_playable_row is None:
                continue                                   # column full of holes

            for r in range(self.rows - 1, -1, -1):
                if not self._is_valid_coord(r, c):
                    continue
                if self.content[r, c] == self.EMPTY:
                    self.content[r, c] = random.randint(self.GEM_START_IDX,
                                                    self.GEM_END_IDX)
                    refilled = True
                    if r == top_playable_row:
                        freshly_filled_top.append((r, c))

        # nothing special to place
        if not freshly_filled_top:
            return refilled

        # ──────────────────────────────────────────
        # B.   maybe place BONUS_2
        # ──────────────────────────────────────────
        if self.bonus2_trigger_count > 0 and self.bonus2_trigger_count % 4 == 0:
            r, c = random.choice(freshly_filled_top)
            self.content[r, c] = self.BONUS_2
            freshly_filled_top.remove((r, c))
            refilled = True
            self.bonus2_trigger_count = 0

        # ──────────────────────────────────────────
        # C.   maybe place one‑time FRAGMENT
        # ──────────────────────────────────────────
        if (not self.fragment_spawned                               # only once, ever
            and self.stones_cleared > self.initial_stones * 0.5
            and self.fragments_on_board < self.max_fragments
            and freshly_filled_top):
            r, c = random.choice(freshly_filled_top)
            self.content[r, c] = self.FRAGMENT
            self.fragments_on_board += 1
            self.fragment_spawned = True                            # lock out future spawns
            refilled = True

        return refilled

    # ---------------------------------------------------------------------
    # helper: reshuffle when stuck (gems+bonuses only)
    # ---------------------------------------------------------------------
    def _shuffle_board(self):
        """Shuffle movable tiles until at least one valid swap exists and no matches are present."""
        movable = []
        for r in range(self.rows):
            for c in range(self.cols):
                if not self._is_valid_coord(r, c):
                    continue
                t = self.content[r, c]
                if t != self.EMPTY and t != self.FRAGMENT:
                    movable.append(t)

        if not movable:  # pathological case
            return False

        while True:
            random.shuffle(movable)
            idx = 0
            for r in range(self.rows):
                for c in range(self.cols):
                    if not self._is_valid_coord(r, c):
                        continue
                    t = self.content[r, c]
                    if t != self.EMPTY and t != self.FRAGMENT:
                        self.content[r, c] = movable[idx]
                        idx += 1

            # Check if there are matches on the board
            matches = self._find_matches()
            if not matches and self.get_valid_swaps():  # success: no matches, but valid swaps exist
                return True
          

    # ---------------------------------------------------------------------
    # patched activate_bonus
    # ---------------------------------------------------------------------
    def _activate_bonus(self, r, c):
        """
        Activate the bonus at (r,c).
        Returns: (affected_coords, chained_bonus_coords)
        Increments self.bonus2_trigger_count internally for B0/B1.
        """
        bonus_type = self.content[r, c]
        affected, chained = set(), set()
        
        # Store the current bonus position to prevent re-adding itself
        current_bonus_pos = (r, c)
        
        # Add the bonus location itself to affected (it will get cleared)
        affected.add(current_bonus_pos)

        def add_row(rr):
            # Add tiles to the right of the bonus
            for cc in range(c + 1, self.cols):
                if not self._is_valid_coord(rr, cc):
                    break
                if self.content[rr, cc] == self.FRAGMENT:
                    break  # stop at fragment
                affected.add((rr, cc))
            
            # Add tiles to the left of the bonus
            for cc in range(c - 1, -1, -1):
                if not self._is_valid_coord(rr, cc):
                    continue
                if self.content[rr, cc] == self.FRAGMENT:
                    break  # stop at fragment
                affected.add((rr, cc))

        def add_col(cc):
            # Add tiles above the bonus
            for rr in range(r - 1, -1, -1):
                if not self._is_valid_coord(rr, cc):
                    continue
                if self.content[rr, cc] == self.FRAGMENT:
                    break  # stop at fragment
                affected.add((rr, cc))
            
            # Add tiles below the bonus
            for rr in range(r + 1, self.rows):
                if not self._is_valid_coord(rr, cc):
                    continue
                if self.content[rr, cc] == self.FRAGMENT:
                    break  # stop at fragment
                affected.add((rr, cc))

        if bonus_type == self.BONUS_0:
            self.bonus2_trigger_count += 1
            add_row(r)

        elif bonus_type == self.BONUS_1:
            self.bonus2_trigger_count += 1
            add_row(r)
            add_col(c)

        elif bonus_type == self.BONUS_2:
            # pick 15‑20 random clearable tiles
            pool = [
                (rr, cc)
                for rr in range(self.rows)
                for cc in range(self.cols)
                if self._is_valid_coord(rr, cc)
                and self.content[rr, cc] not in (self.EMPTY, self.FRAGMENT)
                and not (self.BONUS_0 <= self.content[rr, cc] <= self.BONUS_2)
            ]
            random.shuffle(pool)
            affected.update(pool[: random.randint(15, 20)])

        # chain‑react: pull out bonuses encountered
        for rr, cc in list(affected):
            if self.BONUS_0 <= self.content[rr, cc] <= self.BONUS_2 and (rr, cc) != current_bonus_pos:
                chained.add((rr, cc))
                affected.remove((rr, cc))   # remove the chained bonus so it won't be cleared twice
        # keep current_bonus_pos in `affected` so it gets cleared this round


        return affected, chained

    # ==========================================================================
    # ===                        CORE STEP FUNCTION                          ===
    # ==========================================================================
    def step(self, swap_action: Swap):
        """
        Execute one player swap, resolve all bonus chains + cascades, then return:
            (next_state, reward, done)
        """
        self.step_count += 1

        if self.debug_mode:
            print(f"Stepping with swap: {swap_action}")
        (r1, c1), (r2, c2) = swap_action

        # ---- 0. basic legality checks ------------------------------------
        if not (self._is_valid_coord(r1, c1) and self._is_valid_coord(r2, c2)):
            if self.debug_mode:
                self.display()
            return self.get_state_representation(), -100, True

        if any(
            self.content[r, c] in (self.FRAGMENT, self.EMPTY)
            for r, c in ((r1, c1), (r2, c2))
        ):
            if self.debug_mode:
                self.display()
            return self.get_state_representation(), -100, True

        # ---- 1. perform swap (background never moves) --------------------
        self.content[r1, c1], self.content[r2, c2] = (
            self.content[r2, c2],
            self.content[r1, c1],
        )

        step_reward = -1 * self.step_count  # move cost

        bonuses_queue = set()  # bonuses that will explode immediately

        # if the swap moved a bonus, queue it
        for pos in ((r1, c1), (r2, c2)):
            if self.BONUS_0 <= self.content[pos] <= self.BONUS_2:
                bonuses_queue.add(pos)
                if self.debug_mode:
                    print(f"Added bonus at {pos} to queue")

        # ---- 2. CASCADE LOOP --------------------------------------------
        while True:
            # Cleared contents (gems, bonuses); To break background tiles
            cleared, to_break, bonus_breaks = set(), set(), set()
            processed_bonuses = set()          # NEW: bonuses that already exploded
            # A. process bonuses until no new ones appear ------------------
            while bonuses_queue:
                br, bc = bonuses_queue.pop()
                if (br, bc) in processed_bonuses:
                    continue
                processed_bonuses.add((br, bc))

                affected, chained = self._activate_bonus(br, bc)
                if self.debug_mode:
                    print(f"Activated bonus at {br, bc}")

                for ar, ac in affected:
                    if self.content[ar, ac] != self.FRAGMENT:
                        cleared.add((ar, ac))
                        bonus_breaks.add((ar, ac))

                # only queue bonuses that haven't fired yet
                bonuses_queue.update(chained - processed_bonuses)   

                step_reward += 30  # reward per bonus trigger

          

            # B. match regular gems on the current static board ------------
            matches = self._find_matches()
            if self.debug_mode:
                print(f"Found {len(matches)} matches")
            if matches:
                cleared.update(matches)
                to_break.update(matches)
                
                # Get match details to determine bonus placement and additional rewards
                md = self._get_match_details(matches, swap_action)
                step_reward += md['total_reward']
                
                # Place bonuses at the appropriate positions
                for bonus_r, bonus_c, bonus_type in md['bonus_placements']:
                    if self.debug_mode:
                        print(f"Placing bonus at {bonus_r, bonus_c}")
                    # The position will be cleared, then we'll place the bonus there
                    self.content[bonus_r, bonus_c] = bonus_type
                    # Remove from the cleared set so the bonus doesn't get removed
                    if (bonus_r, bonus_c) in cleared:
                        cleared.remove((bonus_r, bonus_c))

            # stop cascade when nothing happened this round
            if not cleared and not matches:
                break

            # C. clear contents -------------------------------------------
            for cr, cc in cleared:
                self.content[cr, cc] = self.EMPTY

            step_reward += len(cleared)

            floor = 1.0
            A, mid, k = 350.0, 70.0, 0.07
            m = (A - floor) / (1 + math.exp(k * (self.step_count - mid))) + floor

            # D. break background tiles -----------------------------------
            for br, bc in bonus_breaks:
                if self.background[br, bc] != self.BG_NONE:
                    self.background[br, bc] = self.BG_NONE
                    self.stones_cleared += 1
                    step_reward += m

            for br, bc in to_break:
                if self.background[br, bc] == self.BG_SHIELD:
                    self.background[br, bc] = self.BG_STONE
                    step_reward += m * 0.5
                elif self.background[br, bc] == self.BG_STONE:
                    self.background[br, bc] = self.BG_NONE
                    self.stones_cleared += 1
                    step_reward += m

            # E. gravity + refill (handles bonus‑2 & fragment drops) -------
            
            if self.fragments_on_board > 0:
                fragment_coords_before = []
                # Find each coordinate of the fragments before gravity
                for r in range(self.rows):
                    for c in range(self.cols):
                        if self.content[r, c] == self.FRAGMENT:
                            fragment_coords_before.append((r, c))
                            
            self._apply_gravity()
            if self.debug_mode:
                print(f"Applied gravity")

            # Calculate reward for fragment movement after gravity
            if self.fragments_on_board > 0:
                fragment_coords_after = []
                # Find each coordinate of the fragments after gravity
                for r in range(self.rows):
                    for c in range(self.cols):
                        if self.content[r, c] == self.FRAGMENT:
                            fragment_coords_after.append((r, c))
                
                # Calculate movement and reward
                fragment_movement_reward = 0
                for before in fragment_coords_before:
                    for after in fragment_coords_after:
                        # Check if fragments are in the same column (they should be)
                        if before[1] == after[1]:
                            # Calculate how many rows the fragment moved down
                            rows_moved = after[0] - before[0]
                            if rows_moved > 0:  # Only reward downward movement
                                fragment_movement_reward += rows_moved
                
                # Add the fragment movement reward to the total score
                self.score += fragment_movement_reward * 20
                
                if self.debug_mode and fragment_movement_reward > 0:
                    print(f"Fragment movement reward: {fragment_movement_reward}")

            self._refill_board()
            if self.debug_mode:
                print(f"Refilled board")

        # ---- 3. SHUFFLE IF STUCK ----------------------------------------
        if not self.get_valid_swaps():
            if self.debug_mode:
                print(f"No valid swaps, shuffling board")
            if not self._shuffle_board():  # shuffle failed to produce a move
                return self.get_state_representation(), step_reward, True

        # ---- 4. WIN CHECK -----------------------------------------------
        if (
            self.fragments_on_board == 0
            and np.all(self.background != self.BG_SHIELD)
            and np.all(self.background != self.BG_STONE)
        ):
            if self.debug_mode:
                print(f"Win check passed")
                self.display()

            def win_reward(step_count):
                A = 5000
                B = np.log(3) / 50
                floor = 100
                return A * np.exp(-B * step_count) + floor


            return self.get_state_representation(), step_reward + win_reward(self.step_count), True

        # ---- 5. continue playing ----------------------------------------
        self.score += step_reward  # Add the step reward to the total score
        if self.debug_mode:
            self.display()
        return self.get_state_representation(), step_reward, False
    
    def get_global_features(self) -> np.ndarray:
        """3 floats in [0,1] – tweak as you like."""
        stones_ratio = self.stones_cleared / max(1, self.initial_stones)
        fragments = self.fragments_on_board 
        step_count = self.step_count / 400
        return np.array([stones_ratio, fragments, step_count], dtype=np.float32)

    # convenience – one call returns everything the agent stores
    def get_state_tuple(self):
        return (self.get_state_representation(), self.get_global_features())    

    # --- Optional: Helper for Display ---
    def display(self):
        """Prints a textual representation of the board."""
        hline = "+" + ("-" * 10 + "+") * self.cols
        print(hline)
        for r in range(self.rows):
            row_str_fg = "| "
            row_str_bg = "| "
            for c in range(self.cols):
                if not self.mask[r, c]:
                    fg_tile = "#HOLE#"  # Indicate hole
                    bg_tile = "      "
                else:
                    fg_tile = self.rev_map_fg.get(self.content[r, c], "UNK").ljust(7)[
                        :7
                    ]  # Use get for safety, limit length
                    bg_tile = self.rev_map_bg.get(self.background[r, c], "UNK").ljust(
                        7
                    )[:7]

                row_str_fg += f"{fg_tile} | "
                row_str_bg += f"{bg_tile} | "

            print(f"FG: {row_str_fg}")
            print(f"BG: {row_str_bg}")
            print(hline)
        print(
            f"Score: {self.score}, Stones Cleared: {self.stones_cleared}/{self.initial_stones}, Fragments: {self.fragments_on_board}, B2 Count: {self.bonus2_trigger_count}"
        )
        print("=" * (len(hline)))



if __name__ == "__main__":
    pass