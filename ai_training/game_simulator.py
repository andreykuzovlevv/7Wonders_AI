# --- game_simulator.py ---
import numpy as np
import random
from typing import List, Tuple, Set, Optional  # For type hinting
from collections import defaultdict, deque

# Assuming config.py is in the same directory or accessible
import config

# Type hint for a swap action
Swap = Tuple[Tuple[int, int], Tuple[int, int]]

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


class SevenWondersSimulator:
    def __init__(self, rows=config.GRID_ROWS, cols=config.GRID_COLS, level=LEVEL_1):
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
        self.BONUS_0 = self.map_fg["bonus_0"]
        self.BONUS_1 = self.map_fg["bonus_1"]
        self.BONUS_2 = self.map_fg["bonus_2"]
        self.BG_NONE = self.map_bg["none"]
        self.BG_STONE = self.map_bg["stone"]
        self.BG_SHIELD = self.map_bg["shield"]

        self.level = level  # store level
        self.mask = None  # will be set in reset

        self.reset()

    def reset(self):
        """Resets the board to a starting state."""
        self.score = 0
        self.bonus2_trigger_count = 0
        self.stones_cleared = 0
        self.fragments_on_board = 0
        self.initial_stones = 0

        # --- Board Initialization ---
        self._init_from_level(self.level)

        return self._get_state_representation()

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
        while self._find_matches(self.content):
            for r in range(self.rows):
                for c in range(self.cols):
                    if self.mask[r, c]:
                        self.content[r, c] = random.randint(
                            self.GEM_START_IDX, self.GEM_END_IDX
                        )

        self.initial_stones = np.sum(
            (self.background == self.BG_STONE) | (self.background == self.BG_SHIELD)
        )

    def _is_valid_coord(self, r, c):
        """Checks if coordinates are within the board bounds."""
        return 0 <= r < self.rows and 0 <= c < self.cols and self.mask[r, c]

    def _get_state_representation(self):
        """
        Converts the current board state (content, background) into a numerical
        representation suitable for the DQN agent (e.g., a multi-channel NumPy array).
        Channels could represent:
        - Content type (one-hot encoded or integer mapped)
        - Background type
        - Special properties (e.g., is_bonus, is_fragment)
        """
        # Simple example: 2 channels (content index, background index)
        # Normalize? Maybe later. DQN conv layers can learn scaling.
        state = np.stack([self.content, self.background], axis=0).astype(np.float32)
        # Shape: (2, self.rows, self.cols) - Matches QNetwork input_channels=2 expectation
        return state

    def _find_matches(self, board_content) -> Set[Tuple[int, int]]:
        """
        Finds all coordinates of gems involved in matches (3 or more).
        Does NOT match bonuses or fragments.
        Returns a set of (row, col) tuples.
        """
        matches = set()
        gem_mask = (board_content >= self.GEM_START_IDX) & (
            board_content <= self.GEM_END_IDX
        )

        for r in range(self.rows):
            for c in range(self.cols):
                if not gem_mask[r, c]:
                    continue  # Only match gems

                gem_type = board_content[r, c]

                # Check horizontal
                h_len = 0
                while (
                    c + h_len < self.cols
                    and gem_mask[r, c + h_len]
                    and board_content[r, c + h_len] == gem_type
                ):
                    h_len += 1
                if h_len >= 3:
                    for i in range(h_len):
                        matches.add((r, c + i))

                # Check vertical
                v_len = 0
                while (
                    r + v_len < self.rows
                    and gem_mask[r + v_len, c]
                    and board_content[r + v_len, c] == gem_type
                ):
                    v_len += 1
                if v_len >= 3:
                    for i in range(v_len):
                        matches.add((r + i, c))
        return matches

    def _get_match_details(
        self,
        matches: Set[Tuple[int, int]],
        swapped_loc: Optional[Tuple[int, int]] = None,
    ) -> dict:
        """
        Classifies every connected match cluster into:
            • 5_matches  → size ≥5 (straight or L/T)
            • 4_matches  → size ==4 (straight)
            • other_matches → size ==3
        Also returns a sensible bonus placement coordinate.
        """
        details = {"4_matches": set(), "5_matches": set(), "other_matches": set()}
        bonus_loc = None

        for cluster in self._clusters(matches):
            size = len(cluster)

            # Measure max row‑run and col‑run inside this cluster
            rows = defaultdict(int)
            cols = defaultdict(int)
            for r, c in cluster:
                rows[r] += 1
                cols[c] += 1
            longest_row = max(rows.values())
            longest_col = max(cols.values())

            if size >= 5:  # could be straight‑5 OR L/T
                details["5_matches"].update(cluster)
            elif size == 4:
                details["4_matches"].update(cluster)
            else:
                details["other_matches"].update(cluster)

            # Choose a good bonus location for this cluster
            if bonus_loc is None:  # take the first eligible cluster
                if swapped_loc and swapped_loc in cluster:
                    bonus_loc = swapped_loc
                else:
                    # Prefer the intersection of an L/T; otherwise the mid‑point
                    inter = next(
                        ((r, c) for (r, c) in cluster if rows[r] >= 3 and cols[c] >= 3),
                        None,
                    )
                    if inter:
                        bonus_loc = inter
                    else:
                        bonus_loc = next(iter(cluster))

        details["bonus_loc"] = bonus_loc
        return details

    def _clusters(self, coords):
        """Split a set of coordinates into 4‑connected clusters."""
        coords = set(coords)
        clusters = []
        while coords:
            q = deque([coords.pop()])
            cluster = set()
            while q:
                r, c = q.popleft()
                cluster.add((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    neigh = (r + dr, c + dc)
                    if neigh in coords:
                        coords.remove(neigh)
                        q.append(neigh)
            clusters.append(cluster)
        return clusters

    def get_valid_swaps(self) -> List[Swap]:
        """
        Return every legal swap the player can make.

        Rules
        -----
        • Adjacent only (no diagonals).
        • Tiles may NOT be EMPTY or FRAGMENT.
        • If either tile is a BONUS (bonus_0, bonus_1, bonus_2) → the swap is
        immediately valid (bonuses always fire when moved).
        • Otherwise the swap must create ≥ 1 gem match.
        """
        valid_swaps: List[Swap] = []

        def is_swappable(r: int, c: int) -> bool:
            """Can the tile at (r, c) take part in a swap?"""
            if not self._is_valid_coord(r, c):
                return False
            # If you’re using a mask for holes, skip them
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
                        self.content[r, c], self.content[r, c + 1] = t2, t1
                        if self._find_matches(self.content):
                            valid_swaps.append(((r, c), (r, c + 1)))
                        # restore board
                        self.content[r, c], self.content[r, c + 1] = t1, t2

                # --- swap with bottom neighbour -------------------------------
                if r + 1 < self.rows and is_swappable(r + 1, c):
                    t1, t2 = self.content[r, c], self.content[r + 1, c]

                    if is_bonus(t1) or is_bonus(t2):
                        valid_swaps.append(((r, c), (r + 1, c)))
                    else:
                        self.content[r, c], self.content[r + 1, c] = t2, t1
                        if self._find_matches(self.content):
                            valid_swaps.append(((r, c), (r + 1, c)))
                        self.content[r, c], self.content[r + 1, c] = t1, t2

        return valid_swaps

    def _apply_gravity(self) -> bool:
        """
        Move every tile straight downward until it lands on:
            • the bottom row, or
            • the first non‑hole cell whose mask == False below it,
            • or another occupied tile.

        Holes (`mask == False`) are treated as solid walls:
        content never occupies them; gravity skips over them.
        """
        moved = False

        # Copy so we can edit in‑place without breaking look‑ahead
        new_content = np.copy(self.content)
        new_background = np.copy(self.background)

        for c in range(self.cols):
            # Find the lowest *playable* cell in this column
            write_idx = self.rows - 1
            while write_idx >= 0 and not self.mask[write_idx, c]:
                write_idx -= 1  # skip bottom holes entirely

            # Scan upward
            r = write_idx
            while r >= 0:
                if not self.mask[r, c]:
                    # Hole: keep write_idx where it is but step past the hole
                    r -= 1
                    continue

                if new_content[r, c] != self.EMPTY:
                    if r != write_idx:
                        # Move the tile + its background
                        new_content[write_idx, c] = new_content[r, c]
                        new_background[write_idx, c] = new_background[r, c]

                        new_content[r, c] = self.EMPTY
                        new_background[r, c] = self.BG_NONE
                        moved = True
                    write_idx -= 1  # next landing spot
                    # Skip any intervening holes
                    while write_idx >= 0 and not self.mask[write_idx, c]:
                        write_idx -= 1
                r -= 1

            # Clear remaining playable cells above write_idx
            while write_idx >= 0:
                if self.mask[write_idx, c] and new_content[write_idx, c] != self.EMPTY:
                    new_content[write_idx, c] = self.EMPTY
                    new_background[write_idx, c] = self.BG_NONE
                    moved = True
                write_idx -= 1
                while write_idx >= 0 and not self.mask[write_idx, c]:
                    write_idx -= 1

        self.content = new_content
        self.background = new_background
        return moved

    def _refill_board(self) -> bool:
        """Refill top of each column, then maybe drop Fragment and/or Bonus‑2."""
        refilled = False

        # ---- 1. Normal gem refill (respect mask) ---------------------------
        for c in range(self.cols):
            # first playable row in this column
            top_playable = next(
                (r for r in range(self.rows) if self._is_valid_coord(r, c)), None
            )
            if top_playable is None:
                continue  # column is entirely holes

            r = top_playable
            while (
                r < self.rows
                and self._is_valid_coord(r, c)
                and self.content[r, c] == self.EMPTY
            ):
                self.content[r, c] = random.randint(
                    self.GEM_START_IDX, self.GEM_END_IDX
                )
                self.background[r, c] = self.BG_NONE
                refilled = True
                r += 1

        # ---- 2. Bonus‑2 drop (every 4 bonus0/1 activations) ---------------
        if self.bonus2_trigger_count >= 4:
            self.bonus2_trigger_count %= 4  # reset counter but keep overflow
            candidate_cols = []
            for c in range(self.cols):
                top_row = next(
                    (r for r in range(self.rows) if self._is_valid_coord(r, c)), None
                )
                if top_row is None:
                    continue
                if self.content[top_row, c] == self.EMPTY:
                    candidate_cols.append((top_row, c))
            if candidate_cols:
                r_b2, c_b2 = random.choice(candidate_cols)
                self.content[r_b2, c_b2] = self.BONUS_2
                #  Bonus‑2 sits on whatever background is there (stone or none)
                refilled = True

        # ---- 3. Fragment drop (>50 % stones cleared) ----------------------
        stones_ratio = (
            (self.stones_cleared / self.initial_stones) if self.initial_stones else 0
        )
        if stones_ratio > 0.5 and self.fragments_on_board < self.max_fragments:
            candidate_cols = []
            for c in range(self.cols):
                top_row = next(
                    (r for r in range(self.rows) if self._is_valid_coord(r, c)), None
                )
                if top_row is None:
                    continue
                if (
                    self.content[top_row, c] == self.EMPTY
                    and self.background[top_row, c] == self.BG_NONE
                ):
                    candidate_cols.append((top_row, c))
            if candidate_cols:
                r_f, c_f = random.choice(candidate_cols)
                self.content[r_f, c_f] = self.FRAGMENT
                # fragment always sits on BG_NONE by rule
                self.fragments_on_board += 1
                refilled = True

        return refilled

    # ---------------------------------------------------------------------
    # helper: bonus clears shield in one hit
    # ---------------------------------------------------------------------
    def _bonus_breaks_shield(self, r: int, c: int):
        """Break background at (r,c) according to bonus rules."""
        if self.background[r, c] == self.BG_SHIELD:
            self.background[r, c] = self.BG_NONE  # bonus skips the stone phase
            self.stones_cleared += 1  # counts as breaking a stone
            return 5  # reward as if stone was broken
        elif self.background[r, c] == self.BG_STONE:
            self.background[r, c] = self.BG_NONE
            self.stones_cleared += 1
            return 5
        return 0

    # ---------------------------------------------------------------------
    # helper: reshuffle when stuck (gems+bonuses only)
    # ---------------------------------------------------------------------
    def _shuffle_board(self):
        """Shuffle movable tiles until at least one valid swap exists."""
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

        tries = 20
        while tries:
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

            if self.get_valid_swaps():  # success
                return True
            tries -= 1

        return False  # give up; unlikely, but step() will end the game

    # ---------------------------------------------------------------------
    # patched _activate_bonus
    # ---------------------------------------------------------------------
    def _activate_bonus(self, r, c):
        """
        Activate the bonus at (r,c).
        Returns: (affected_coords, chained_bonus_coords)
        Increments self.bonus2_trigger_count internally for B0/B1.
        """
        bonus_type = self.content[r, c]
        affected, chained = set(), set()

        def add_row(rr):
            for cc in range(self.cols):
                if cc == c or not self._is_valid_coord(rr, cc):
                    continue
                if self.content[rr, cc] == self.FRAGMENT:
                    break  # stop at fragment
                affected.add((rr, cc))

        def add_col(cc):
            for rr in range(self.rows):
                if rr == r or not self._is_valid_coord(rr, cc):
                    continue
                if self.content[rr, cc] == self.FRAGMENT:
                    break
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
            if self.BONUS_0 <= self.content[rr, cc] <= self.BONUS_2:
                chained.add((rr, cc))
                affected.remove((rr, cc))

        return affected, chained

    # ==========================================================================
    # ===                        CORE STEP FUNCTION                          ===
    # ==========================================================================
    def step(self, swap_action: Swap):
        """
        Execute one player swap, resolve all bonus chains + cascades, then return:
            (next_state, reward, done)
        """
        (r1, c1), (r2, c2) = swap_action

        # ---- 0. basic legality checks ------------------------------------
        if not (self._is_valid_coord(r1, c1) and self._is_valid_coord(r2, c2)):
            self.display()
            return self._get_state_representation(), -100, True

        if any(
            self.content[r, c] in (self.FRAGMENT, self.EMPTY)
            for r, c in ((r1, c1), (r2, c2))
        ):
            self.display()
            return self._get_state_representation(), -100, True

        # ---- 1. perform swap (background never moves) --------------------
        self.content[r1, c1], self.content[r2, c2] = (
            self.content[r2, c2],
            self.content[r1, c1],
        )

        step_reward = -1  # tiny move cost
        bonuses_queue = set()  # bonuses that will explode immediately

        # if the swap moved a bonus, queue it
        for pos in ((r1, c1), (r2, c2)):
            if self.BONUS_0 <= self.content[pos] <= self.BONUS_2:
                bonuses_queue.add(pos)

        # ---- 2. CASCADE LOOP --------------------------------------------
        while True:
            cleared, to_break = set(), set()

            # A. process bonuses until no new ones appear ------------------
            while bonuses_queue:
                br, bc = bonuses_queue.pop()
                affected, chained = self._activate_bonus(br, bc)

                cleared.add((br, bc))
                to_break.add((br, bc))

                for ar, ac in affected:
                    if self.content[ar, ac] != self.FRAGMENT:
                        cleared.add((ar, ac))
                        to_break.add((ar, ac))

                # chain all B0/B1, but never B2
                bonuses_queue.update(chained)

                step_reward += 5  # reward per bonus trigger

            # B. match regular gems on the current static board ------------
            matches = self._find_matches(self.content)
            if matches:
                swapped_loc = (
                    (r1, c1)
                    if (r1, c1) in matches
                    else (r2, c2) if (r2, c2) in matches else None
                )
                md = self._get_match_details(matches, swapped_loc)

                bonus_loc = md["bonus_loc"] or (r1, c1)
                cleared.update(matches)
                to_break.update(matches)
                step_reward += 2 * len(matches)

                # spawn new bonus if 4‑ or 5‑match
                if bonus_loc in md["5_matches"]:
                    self.content[bonus_loc] = self.BONUS_1
                    cleared.discard(bonus_loc)
                    to_break.discard(bonus_loc)
                elif bonus_loc in md["4_matches"]:
                    self.content[bonus_loc] = self.BONUS_0
                    cleared.discard(bonus_loc)
                    to_break.discard(bonus_loc)

            # stop cascade when nothing happened this round
            if not cleared and not matches:
                break

            # C. clear contents -------------------------------------------
            bonus_clears = cleared.copy()
            for cr, cc in cleared:
                self.content[cr, cc] = self.EMPTY

            # D. break background tiles -----------------------------------
            for br, bc in to_break:
                if self.content[br, bc] != self.EMPTY:
                    continue  # something replaced it (e.g., new bonus)

                if (br, bc) in bonus_clears:
                    step_reward += self._bonus_breaks_shield(br, bc)  # bonus smash
                else:  # ordinary gem match
                    if self.background[br, bc] == self.BG_SHIELD:
                        self.background[br, bc] = self.BG_STONE
                        step_reward += 3
                    elif self.background[br, bc] == self.BG_STONE:
                        self.background[br, bc] = self.BG_NONE
                        self.stones_cleared += 1
                        step_reward += 5

            # E. gravity + refill (handles bonus‑2 & fragment drops) -------
            while self._apply_gravity():
                pass
            self._refill_board()

            # any bonus_2 spawned by refill should explode next loop
            for r in range(self.rows):
                for c in range(self.cols):
                    if self.content[r, c] == self.BONUS_2 and self._is_valid_coord(
                        r, c
                    ):
                        bonuses_queue.add((r, c))

        # ---- 3. SHUFFLE IF STUCK ----------------------------------------
        if not self.get_valid_swaps():
            if not self._shuffle_board():  # shuffle failed to produce a move
                self.display()
                return self._get_state_representation(), step_reward - 500, True

        # ---- 4. WIN CHECK -----------------------------------------------
        if (
            self.fragments_on_board == 0
            and np.all(self.background != self.BG_SHIELD)
            and np.all(self.background != self.BG_STONE)
        ):
            self.display()
            return self._get_state_representation(), step_reward + 1000, True

        # ---- 5. continue playing ----------------------------------------
        self.display()
        return self._get_state_representation(), step_reward, False

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


# --- Example Usage (for testing) ---
if __name__ == "__main__":
    # Example Level with Stones and Shields
    TEST_LEVEL = {
        "mask": [
            "##########",
            "#..ss..ss#",  # . = stone, s = shield
            "#........#",
            "#..####..#",  # Add some holes
            "#..#ss#..#",
            "#..#..#..#",
            "#........#",
            "#ss....ss#",
            "#........#",
            "##########",
        ]
    }
    # Use a fixed seed for reproducible tests
    random.seed(42)
    np.random.seed(42)

    # sim = SevenWondersSimulator(level=TEST_LEVEL) # Pass the specific level
    sim = (
        SevenWondersSimulator()
    )  # Use default LEVEL_1 for now if TEST_LEVEL causes issues

    print("Initial Board:")
    sim.display()

    done = False
    step_count = 0
    max_steps = 50  # Limit steps for testing

    while not done and step_count < max_steps:
        print(f"\n M A I N   L O O P --- Step {step_count + 1} ---")
        valid_swaps = sim.get_valid_swaps()
        print(f"Found {len(valid_swaps)} valid swaps.")
        # print(valid_swaps) # Uncomment to see the swaps

        if not valid_swaps:
            Exception("No valid swaps available!")
            break

        # --- Simple Test Strategy: prioritize bonus swaps/matches ---
        best_action = None
        # 1. Swap involving a bonus?
        bonus_swaps = [
            s
            for s in valid_swaps
            if sim.BONUS_0 <= sim.content[s[0]] <= sim.BONUS_2
            or sim.BONUS_0 <= sim.content[s[1]] <= sim.BONUS_2
        ]
        if bonus_swaps:
            best_action = random.choice(bonus_swaps)
            print("Choosing a bonus swap action.")
        else:
            # 2. Swap creating a 4+ match? (Simulate swap and check)
            potential_big_match_swaps = []
            for swap in valid_swaps:
                (r1, c1), (r2, c2) = swap
                t1, t2 = sim.content[r1, c1], sim.content[r2, c2]
                sim.content[r1, c1], sim.content[r2, c2] = t2, t1  # Test swap
                matches = sim._find_matches(sim.content)
                if matches:
                    details = sim._get_match_details(
                        matches, (r1, c1)
                    )  # Pass one loc for potential bonus check
                    if details["5_matches"] or details["4_matches"]:
                        potential_big_match_swaps.append(swap)
                sim.content[r1, c1], sim.content[r2, c2] = t1, t2  # Swap back
            if potential_big_match_swaps:
                best_action = random.choice(potential_big_match_swaps)
                print("Choosing a swap creating potential 4/5 match.")
            else:
                # 3. Just pick a random valid swap
                best_action = random.choice(valid_swaps)
                print("Choosing a random valid swap.")

        action = best_action
        print(f"Performing action: {action}")

        # Execute the step
        new_state, reward, done = sim.step(action)

        print(f"Step {step_count+1} completed. Reward: {reward}, Done: {done}")
        # display is called inside step now
        # sim.display()
        step_count += 1

        # Optional: Add a small delay for visual inspection
        # import time
        # time.sleep(0.5)

    print("\nSimulation finished.")
    if done:
        print(f"Game ended naturally after {step_count} steps.")
    else:
        print(f"Max steps ({max_steps}) reached.")
    print(f"Final Score: {sim.score}")
