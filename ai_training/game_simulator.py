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
        return 0 <= r < self.rows and 0 <= c < self.cols

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

    def _apply_gravity(self):
        """Moves tiles down to fill empty spaces. Fragments also fall."""
        new_content = np.copy(self.content)
        new_background = np.copy(self.background)
        moved = False

        for c in range(self.cols):
            write_idx = self.rows - 1
            for r in range(self.rows - 1, -1, -1):
                if new_content[r, c] != self.EMPTY:
                    if r != write_idx:
                        # Move content and background together
                        new_content[write_idx, c] = new_content[r, c]
                        new_background[write_idx, c] = new_background[r, c]
                        new_content[r, c] = self.EMPTY
                        new_background[r, c] = (
                            self.BG_NONE
                        )  # Ensure background is cleared too
                        moved = True
                    write_idx -= 1
            # Fill remaining top spots with empty
            while write_idx >= 0:
                if new_content[write_idx, c] != self.EMPTY:
                    new_content[write_idx, c] = self.EMPTY
                    new_background[write_idx, c] = self.BG_NONE
                    moved = True
                write_idx -= 1

        self.content = new_content
        self.background = new_background
        return moved

    def _refill_board(self):
        """Fills empty cells in the top row with new random gems."""
        refilled = False
        for c in range(self.cols):
            # Check if the *topmost* cell in the column is empty after gravity
            if self.content[0, c] == self.EMPTY:
                # Need to find the actual top empty slot if gravity didn't fill column
                top_empty_row = 0
                while (
                    top_empty_row < self.rows
                    and self.content[top_empty_row, c] == self.EMPTY
                ):
                    # Spawn new tile (gem)
                    new_gem = random.randint(self.GEM_START_IDX, self.GEM_END_IDX + 1)
                    self.content[top_empty_row, c] = new_gem
                    self.background[top_empty_row, c] = (
                        self.BG_NONE
                    )  # New tiles have no background
                    refilled = True
                    top_empty_row += (
                        1  # Continue if multiple empty cells stacked at top
                    )

        # --- Fragment Spawning Logic ---
        stones_cleared_ratio = (
            (self.stones_cleared / self.initial_stones)
            if self.initial_stones > 0
            else 0
        )
        # Example condition: Spawn a fragment if >50% stones cleared and below max fragments
        if stones_cleared_ratio > 0.5 and self.fragments_on_board < self.max_fragments:
            # Try to find an empty top spot to spawn
            empty_top_cols = [
                c for c in range(self.cols) if self.content[0, c] == self.EMPTY
            ]
            if empty_top_cols:
                spawn_col = random.choice(empty_top_cols)
                # Spawn fragment in the highest empty cell of that column
                spawn_row = 0
                while (
                    spawn_row < self.rows
                    and self.content[spawn_row, spawn_col] == self.EMPTY
                ):
                    spawn_row += 1
                spawn_row -= 1  # Place in the last empty cell found from top

                if spawn_row >= 0:  # Make sure we found an empty spot
                    self.content[spawn_row, spawn_col] = self.FRAGMENT
                    self.background[spawn_row, spawn_col] = (
                        self.BG_NONE
                    )  # Fragments have no background? Assume so.
                    self.fragments_on_board += 1
                    refilled = True
                    # Optional: Reset stone clear count or use a different trigger mechanism
                    # self.stones_cleared = 0 # Reset count after spawning? Depends on game rules.

        return refilled

    def _activate_bonus(self, r, c) -> Tuple[Set[Tuple[int, int]], Optional[int], int]:
        """
        Determines the effect of activating a bonus at (r, c).
        Returns:
            - Set of affected coordinates (excluding the bonus itself).
            - Type of next bonus activated (if any).
            - Bonus trigger count increment (1 for B0/B1, 0 otherwise).
        """
        bonus_type = self.content[r, c]
        affected_coords = set()
        next_activated_bonus_type = None
        bonus_trigger_inc = 0

        if bonus_type == self.BONUS_0:  # Row clear
            bonus_trigger_inc = 1
            for col in range(self.cols):
                if col != c:  # Don't affect self initially
                    affected_coords.add((r, col))
        elif bonus_type == self.BONUS_1:  # Plus clear
            bonus_trigger_inc = 1
            for col in range(self.cols):  # Row
                if col != c:
                    affected_coords.add((r, col))
            for row in range(self.rows):  # Column
                if row != r:
                    affected_coords.add((row, c))
        elif bonus_type == self.BONUS_2:  # Random clear
            # Find all clearable tiles (non-empty, non-fragment, non-bonus)
            clearable = []
            for rr in range(self.rows):
                for cc in range(self.cols):
                    tile = self.content[rr, cc]
                    if (
                        tile != self.EMPTY
                        and tile != self.FRAGMENT
                        and not (self.BONUS_0 <= tile <= self.BONUS_2)
                    ):
                        clearable.append((rr, cc))

            num_to_clear = random.randint(15, 20)
            random.shuffle(clearable)
            affected_coords.update(clearable[:num_to_clear])

        # Check affected coords for other bonuses to chain activate
        activated_bonuses_coords = set()
        for ar, ac in list(affected_coords):  # Iterate copy as we might modify
            if not self._is_valid_coord(ar, ac):
                continue
            affected_tile = self.content[ar, ac]
            if self.BONUS_0 <= affected_tile <= self.BONUS_2:
                # Add to chain reaction list, remove from direct clear list
                activated_bonuses_coords.add((ar, ac))
                affected_coords.remove((ar, ac))

        return affected_coords, activated_bonuses_coords, bonus_trigger_inc

    def step(self, swap_action: Swap):
        """
        Performs the swap action, resolves all cascades, and returns the result.
        """
        (r1, c1), (r2, c2) = swap_action
        step_reward = 0
        done = False

        # --- 1. Validate and Perform Swap ---
        # Basic check (more thorough check in get_valid_swaps)
        if not (self._is_valid_coord(r1, c1) and self._is_valid_coord(r2, c2)):
            print(f"Warning: Invalid swap coordinates {swap_action}")
            return (
                self._get_state_representation(),
                -100,
                True,
            )  # Penalize invalid action heavily

        t1 = self.content[r1, c1]
        t2 = self.content[r2, c2]
        if (
            t1 == self.FRAGMENT
            or t2 == self.FRAGMENT
            or t1 == self.EMPTY
            or t2 == self.EMPTY
        ):
            print(f"Warning: Invalid swap involving Fragment/Empty {swap_action}")
            return self._get_state_representation(), -100, True  # Penalize invalid swap

        self.content[r1, c1], self.content[r2, c2] = t2, t1
        self.background[r1, c1], self.background[r2, c2] = (
            self.background[r2, c2],
            self.background[r1, c1],
        )  # Swap background too? Assume yes.

        # Keep track of the location where a potential generated bonus should appear
        # One of the two swapped locations is a good candidate
        potential_bonus_spawn_loc = (r1, c1)  # Arbitrary choice

        # Small penalty for taking a step? Or reward based on outcome?
        step_reward -= 1  # Small penalty encourages efficiency

        # --- 2. Cascade Loop ---
        cascade_bonus_trigger_count = 0

        # Store bonuses activated *during* the cascade to process in the *next* iteration
        bonuses_to_activate_next = set()

        while True:
            coords_cleared_this_iter = set()
            coords_to_break_bg = set()

            # --- A. Process Activated Bonuses (from previous iteration) ---
            activated_bonuses_coords_this_iter = set()  # To check for B2 trigger
            if bonuses_to_activate_next:
                current_bonuses_to_process = bonuses_to_activate_next.copy()
                bonuses_to_activate_next.clear()

                for br, bc in current_bonuses_to_process:
                    if not self._is_valid_coord(br, bc) or not (
                        self.BONUS_0 <= self.content[br, bc] <= self.BONUS_2
                    ):
                        continue  # Bonus might have been cleared already

                    # Mark bonus location itself for clearing and background break
                    coords_cleared_this_iter.add((br, bc))
                    coords_to_break_bg.add((br, bc))

                    affected_coords, chained_bonuses, trigger_inc = (
                        self._activate_bonus(br, bc)
                    )
                    activated_bonuses_coords_this_iter.add(
                        (br, bc)
                    )  # Record which bonus was activated

                    # Add affected non-fragment tiles to clear/break lists
                    for ar, ac in affected_coords:
                        if (
                            self._is_valid_coord(ar, ac)
                            and self.content[ar, ac] != self.FRAGMENT
                        ):
                            coords_cleared_this_iter.add((ar, ac))
                            coords_to_break_bg.add((ar, ac))

                    bonuses_to_activate_next.update(
                        chained_bonuses
                    )  # Add chained bonuses for the *next* iteration
                    cascade_bonus_trigger_count += trigger_inc
                    step_reward += 5  # Reward for activating a bonus

            # --- B. Find Gem Matches ---
            matches = self._find_matches(self.content)

            # Stop condition: No matches found AND no bonuses were activated in this iteration
            if not matches and not activated_bonuses_coords_this_iter:
                # Check if there were bonuses queued for next iter; if so, continue
                if not bonuses_to_activate_next:
                    break  # End cascade

            # --- C. Process Matches ---
            if matches:
                # Determine match details (4-match, 5-match) for bonus generation
                # Pass one of the swapped locs if it was part of the match, else None
                swapped_loc_in_match = None
                if (r1, c1) in matches:
                    swapped_loc_in_match = (r1, c1)
                elif (r2, c2) in matches:
                    swapped_loc_in_match = (r2, c2)

                match_details = self._get_match_details(matches, swapped_loc_in_match)

                bonus_spawn_loc = potential_bonus_spawn_loc  # Use the initial swap location as default
                if match_details[
                    "bonus_loc"
                ]:  # Prefer location derived from match analysis
                    bonus_spawn_loc = match_details["bonus_loc"]

                # Add matched coordinates to clear/break lists
                coords_cleared_this_iter.update(matches)
                coords_to_break_bg.update(matches)

                step_reward += len(matches) * 2  # Reward per matched gem

                # --- D. Generate Bonuses (Overwrite one cleared tile) ---
                # Prioritize 5-match over 4-match if location overlaps
                generated_bonus_type = None
                if (
                    match_details["5_matches"]
                    and bonus_spawn_loc in match_details["5_matches"]
                ):
                    generated_bonus_type = self.BONUS_1
                    step_reward += 20  # Larger reward for creating bonus 1
                elif (
                    match_details["4_matches"]
                    and bonus_spawn_loc in match_details["4_matches"]
                ):
                    generated_bonus_type = self.BONUS_0
                    step_reward += 10  # Reward for creating bonus 0

                # If a bonus was generated, place it (it replaces the cleared tile)
                if generated_bonus_type is not None:
                    self.content[bonus_spawn_loc] = generated_bonus_type
                    # Ensure this location isn't immediately cleared again in this iteration
                    coords_cleared_this_iter.discard(bonus_spawn_loc)
                    coords_to_break_bg.discard(bonus_spawn_loc)

            # --- E. Clear Tiles and Break Backgrounds ---
            temp_stones_cleared_count = 0
            for r_cl, c_cl in coords_cleared_this_iter:
                if not self._is_valid_coord(r_cl, c_cl):
                    continue

                # Clear content (unless it's a newly generated bonus - handled above)
                # Don't clear bonuses that are scheduled for activation next iter
                if (r_cl, c_cl) not in bonuses_to_activate_next:
                    self.content[r_cl, c_cl] = self.EMPTY

            for r_bg, c_bg in coords_to_break_bg:
                if not self._is_valid_coord(r_bg, c_bg):
                    continue
                bg_type = self.background[r_bg, c_bg]
                if bg_type == self.BG_SHIELD:
                    self.background[r_bg, c_bg] = self.BG_STONE
                    step_reward += 3  # Reward for breaking shield
                elif bg_type == self.BG_STONE:
                    self.background[r_bg, c_bg] = self.BG_NONE
                    step_reward += 5  # Reward for breaking stone
                    temp_stones_cleared_count += 1

            self.stones_cleared += temp_stones_cleared_count

            # --- F+G. Apply Gravity and Refill ---
            # Apply gravity repeatedly until no more tiles move
            while self._apply_gravity():
                pass
            refilled = self._refill_board()  # Refill includes fragment spawning check

            # Add any newly spawned bonuses (e.g. B2) to activate next cycle
            # Check for B2 Trigger
            self.bonus2_trigger_count += (
                cascade_bonus_trigger_count  # Add count from B0/B1 activated this step
            )
            cascade_bonus_trigger_count = (
                0  # Reset for next potential cascade iteration
            )
            if self.bonus2_trigger_count >= 4:
                self.bonus2_trigger_count %= 4  # Reset counter
                # Spawn Bonus 2 - Where? Random empty? Replace a gem? Rule needed.
                # Simple approach: Find a random non-empty, non-fragment, non-bonus spot
                possible_b2_locs = []
                for r_b2 in range(self.rows):
                    for c_b2 in range(self.cols):
                        tile = self.content[r_b2, c_b2]
                        if (
                            tile != self.EMPTY
                            and tile != self.FRAGMENT
                            and not (self.BONUS_0 <= tile <= self.BONUS_2)
                        ):
                            possible_b2_locs.append((r_b2, c_b2))
                if possible_b2_locs:
                    b2_r, b2_c = random.choice(possible_b2_locs)
                    self.content[b2_r, b2_c] = self.BONUS_2
                    # Add it to be activated in the next iteration
                    bonuses_to_activate_next.add((b2_r, b2_c))
                    step_reward += 15  # Reward for triggering B2

            # If board was refilled, need to check for new matches in the next loop iteration
            if (
                not refilled
                and not matches
                and not activated_bonuses_coords_this_iter
                and not bonuses_to_activate_next
            ):
                break  # Absolutely nothing happened and nothing pending

        # --- 3. After Cascade: Check Game End Conditions ---
        self.score += step_reward  # Add accumulated step reward to total score

        # Win Condition: Example - all background tiles cleared
        if np.all(self.background == self.BG_NONE):
            done = True
            step_reward += 1000  # Large reward for winning
            self.score += 1000
            print("\n--- Level Cleared! ---")
        else:
            # Lose Condition: No valid swaps left
            valid_swaps = self.get_valid_swaps()
            if not valid_swaps:
                done = True
                step_reward -= 500  # Large penalty for getting stuck
                self.score -= 500
                print("\n--- No Valid Moves! Game Over ---")
                # Implement shuffle or just end episode? Ending is simpler for RL.

        # --- 4. Return New State, Reward, Done ---
        next_state = self._get_state_representation()
        return next_state, step_reward, done

    # --- Optional: Helper for Display ---
    def display(self):
        """Prints a textual representation of the board."""
        for r in range(self.rows):
            row_str_fg = " ".join(
                f"{self.rev_map_fg[self.content[r, c]]:<9}" for c in range(self.cols)
            )
            row_str_bg = " ".join(
                f"{self.rev_map_bg[self.background[r, c]]:<9}" for c in range(self.cols)
            )
            print(f"FG: {row_str_fg}  | BG: {row_str_bg}")
        print(
            f"Score: {self.score}, Stones Cleared: {self.stones_cleared}/{self.initial_stones}, Fragments: {self.fragments_on_board}"
        )
        print("-" * 20)


# --- Example Usage (for testing) ---
if __name__ == "__main__":
    sim = SevenWondersSimulator()
    print("Initial Board:")
    sim.display()

    done = False
    step_count = 0
    max_steps = 50

    while not done and step_count < max_steps:
        print(f"\n--- Step {step_count + 1} ---")
        valid_swaps = sim.get_valid_swaps()
        print(f"Found {len(valid_swaps)} valid swaps.")

        if not valid_swaps:
            print("No valid moves found by simulation.")
            break

        # Choose a random valid swap for testing
        action = random.choice(valid_swaps)
        print(f"Performing random valid swap: {action}")

        new_state, reward, done = sim.step(action)

        print(f"Reward received: {reward}")
        print("Board after step:")
        sim.display()
        step_count += 1

    print("\nSimulation finished.")
    if done:
        print("Game ended.")
    else:
        print("Max steps reached.")
