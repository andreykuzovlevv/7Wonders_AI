# --- File: game_simulator.py ---

import random
import copy  # Use deepcopy for state isolation if needed, but be mindful of performance
from typing import List, Tuple, Set, Dict, Optional
import numpy as np
import logging  # Optional, but helpful for debugging

# --- Configuration ---
# (Ideally load from a config file or pass during init)
DEFAULT_GRID_ROWS = 10
DEFAULT_GRID_COLS = 10

# Tile Types (Using strings for clarity in logic)
EMPTY = "empty"
FRAGMENT = "fragment"
GEM_TYPES = [f"gem_{i}" for i in range(8)]
BONUS_0 = "bonus_0"  # Row clear
BONUS_1 = "bonus_1"  # Plus clear
BONUS_2 = "bonus_2"  # Random clear
BONUS_TYPES = [BONUS_0, BONUS_1, BONUS_2]
ALL_GEMS_AND_BONUSES = GEM_TYPES + BONUS_TYPES

# Background Types
BG_NONE = "none"
BG_STONE = "stone"
BG_SHIELD = "shield"
BG_BLOCKED = "blocked"  # For irregular grids if needed

# Mapping for state representation
CONTENT_CLASSES = [EMPTY] + GEM_TYPES + [FRAGMENT] + BONUS_TYPES
BACKGROUND_CLASSES = [BG_NONE, BG_STONE, BG_SHIELD, BG_BLOCKED]  # Add blocked if used

MAP_FG = {c: i for i, c in enumerate(CONTENT_CLASSES)}
MAP_BG = {c: i for i, c in enumerate(BACKGROUND_CLASSES)}

# --- Type Hint Definitions ---
GridCoord = Tuple[int, int]  # (row, col)
Swap = Tuple[GridCoord, GridCoord]
GridState = List[List[str]]

# --- Logging Setup (Optional) ---
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger("GameSim")

# --- Helper Functions ---


def is_valid_coord(r, c, rows, cols):
    """Checks if coordinates are within grid bounds."""
    return 0 <= r < rows and 0 <= c < cols


def get_neighbors(r, c, rows, cols) -> List[GridCoord]:
    """Gets valid orthogonal neighbors."""
    neighbors = []
    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        nr, nc = r + dr, c + dc
        if is_valid_coord(nr, nc, rows, cols):
            neighbors.append((nr, nc))
    return neighbors


# --- Simulator Class ---


class SevenWondersSimulator:
    def __init__(self, level_config: Optional[Dict] = None):
        """
        Initializes the simulator.
        level_config (dict, optional): Defines the level layout, goals, etc.
                                        If None, uses default random setup.
        """
        self.rows = DEFAULT_GRID_ROWS
        self.cols = DEFAULT_GRID_COLS
        self.content: GridState = []
        self.background: GridState = []
        self.fragments_on_board: int = 0
        self.fragments_collected: int = 0
        self.target_fragments: int = 1  # Example default
        self.total_initial_background_tiles: int = 0
        self.background_destroyed_count: int = 0
        self.bonus_activation_counter: int = 0  # Counts bonus_0/bonus_1 activations

        # RL specific state
        self.current_score: int = 0
        self.moves_made: int = 0
        self.max_moves: Optional[int] = None  # Set by level_config if applicable
        # Add time tracking if needed

        # Load level or use default
        self._load_level(level_config)
        log.info(f"Simulator initialized: {self.rows}x{self.cols} grid.")

    def _load_level(self, config: Optional[Dict]):
        """Sets up the board based on level config or defaults."""
        # TODO: Implement proper level loading from config dict
        # Config could specify: rows, cols, initial content, background,
        # fragment positions, target_fragments, max_moves/time etc.

        # --- Default Random Initialization (Placeholder) ---
        if config is None:
            self.rows = DEFAULT_GRID_ROWS
            self.cols = DEFAULT_GRID_COLS
            self.content = [
                [random.choice(GEM_TYPES) for _ in range(self.cols)]
                for _ in range(self.rows)
            ]
            self.background = [
                [random.choice([BG_STONE, BG_SHIELD]) for _ in range(self.cols)]
                for _ in range(self.rows)
            ]
            # Add 1 fragment at top-middle (example)
            frag_col = self.cols // 2
            if (
                self.content[0][frag_col] != FRAGMENT
            ):  # Avoid overwriting if already there
                self.content[0][frag_col] = FRAGMENT
                self.fragments_on_board = 1
            self.target_fragments = 1

            # Ensure no initial matches (important!)
            while self._get_matches(self.content):
                log.debug(
                    "Regenerating initial board to remove pre-existing matches..."
                )
                self.content = [
                    [random.choice(GEM_TYPES) for _ in range(self.cols)]
                    for _ in range(self.rows)
                ]
                # Re-add fragment
                if self.content[0][frag_col] != FRAGMENT:
                    self.content[0][frag_col] = FRAGMENT
                    self.fragments_on_board = 1

        else:
            # --- Load from config ---
            self.rows = config.get("rows", DEFAULT_GRID_ROWS)
            self.cols = config.get("cols", DEFAULT_GRID_COLS)
            # Load content (needs robust parsing)
            self.content = config.get(
                "content",
                [
                    [random.choice(GEM_TYPES) for _ in range(self.cols)]
                    for _ in range(self.rows)
                ],
            )
            # Load background
            self.background = config.get(
                "background",
                [
                    [random.choice([BG_STONE, BG_SHIELD]) for _ in range(self.cols)]
                    for _ in range(self.rows)
                ],
            )
            self.target_fragments = config.get("target_fragments", 1)
            self.max_moves = config.get("max_moves", None)
            # Count initial fragments and background tiles
            self.fragments_on_board = sum(row.count(FRAGMENT) for row in self.content)
            self.total_initial_background_tiles = sum(
                1
                for r in range(self.rows)
                for c in range(self.cols)
                if self.background[r][c] != BG_NONE
                and self.background[r][c] != BG_BLOCKED
            )
            self.background_destroyed_count = 0  # Start fresh count

            # TODO: Add validation for loaded config (dimensions match, valid tiles)

        self.fragments_collected = 0
        self.current_score = 0
        self.moves_made = 0
        self.bonus_activation_counter = 0

    def reset(self, level_config: Optional[Dict] = None) -> np.ndarray:
        """Resets the simulator to the initial state of a level."""
        self._load_level(level_config)
        log.info("Game reset.")
        return self._get_state_representation()

    def _get_state_representation(self) -> np.ndarray:
        """
        Converts the current game state (content, background) into numerical format.
        Returns a NumPy array, e.g., shape (2, rows, cols) for 2 channels.
        Channel 0: Content (mapped to indices)
        Channel 1: Background (mapped to indices)
        """
        state_content = np.zeros((self.rows, self.cols), dtype=np.int8)
        state_background = np.zeros((self.rows, self.cols), dtype=np.int8)

        for r in range(self.rows):
            for c in range(self.cols):
                state_content[r, c] = MAP_FG.get(
                    self.content[r][c], 0
                )  # Default to 0 (empty) if unknown
                state_background[r, c] = MAP_BG.get(
                    self.background[r][c], 0
                )  # Default to 0 (none)

        # Stack channels: (channels, rows, cols)
        # Add more channels if needed (e.g., fragment locations, goal progress)
        return np.stack([state_content, state_background], axis=0)

    def _simulate_swap(
        self, grid: GridState, coord1: GridCoord, coord2: GridCoord
    ) -> GridState:
        """Creates a *new* grid with the swap applied. Does not modify original."""
        new_grid = [row[:] for row in grid]  # Create a shallow copy
        r1, c1 = coord1
        r2, c2 = coord2
        new_grid[r1][c1], new_grid[r2][c2] = new_grid[r2][c2], new_grid[r1][c1]
        return new_grid

    def _get_matches(self, grid: GridState) -> List[Set[GridCoord]]:
        """
        Finds all groups of 3+ matching tiles (non-empty, non-fragment)
        horizontally and vertically.
        Returns a list of sets, where each set contains the coordinates of a matched group.
        Merges overlapping matches (e.g., L or T shapes).
        """
        matches: List[Set[GridCoord]] = []
        if not grid:
            return matches
        rows, cols = len(grid), len(grid[0])
        matched_coords = set()  # Track coords already part of a match

        for r in range(rows):
            for c in range(cols):
                if (r, c) in matched_coords:
                    continue
                tile = grid[r][c]
                if tile == EMPTY or tile == FRAGMENT:
                    continue

                # Check horizontal
                h_match = {(r, c)}
                nc = c + 1
                while nc < cols and grid[r][nc] == tile:
                    h_match.add((r, nc))
                    nc += 1
                if len(h_match) >= 3:
                    matches.append(h_match)
                    matched_coords.update(h_match)

                # Check vertical
                v_match = {(r, c)}
                nr = r + 1
                while nr < rows and grid[nr][c] == tile:
                    v_match.add((nr, c))
                    nr += 1
                if len(v_match) >= 3:
                    # Avoid adding duplicate single-coord if already in horizontal
                    if (r, c) not in matched_coords or len(v_match) > 1:
                        matches.append(v_match)
                        matched_coords.update(v_match)

        # --- Merge overlapping matches ---
        if not matches:
            return []

        merged = True
        while merged:
            merged = False
            merged_matches: List[Set[GridCoord]] = []
            used = [False] * len(matches)
            for i in range(len(matches)):
                if used[i]:
                    continue
                current_merge = matches[i].copy()
                used[i] = True
                # Check against remaining matches for overlap
                for j in range(i + 1, len(matches)):
                    if not used[j] and not current_merge.isdisjoint(matches[j]):
                        current_merge.update(matches[j])
                        used[j] = True
                        merged = True  # Signal that a merge happened
                merged_matches.append(current_merge)
            matches = merged_matches

        log.debug(f"Found merged matches: {matches}")
        return matches

    def get_valid_swaps(self) -> List[Swap]:
        """
        Finds all valid swaps the player could make *now*.
        A swap is valid if:
        1. It involves two adjacent tiles.
        2. Neither tile is a FRAGMENT.
        3. The swap *results* in at least one match (3+ gems) OR activates a bonus tile.
        Returns a list of valid Swap tuples ((r1, c1), (r2, c2)).
        """
        valid_swaps: List[Swap] = []
        processed_swaps = set()  # Avoid checking (a,b) and (b,a)

        for r in range(self.rows):
            for c in range(self.cols):
                tile1 = self.content[r][c]
                if (
                    tile1 == FRAGMENT or tile1 == EMPTY
                ):  # Fragments/Empty cannot be swapped
                    continue

                # Check neighbors (right and down to avoid duplicates)
                for dr, dc in [(0, 1), (1, 0)]:
                    nr, nc = r + dr, c + dc
                    if not is_valid_coord(nr, nc, self.rows, self.cols):
                        continue

                    tile2 = self.content[nr][nc]
                    if (
                        tile2 == FRAGMENT or tile2 == EMPTY
                    ):  # Cannot swap with fragments/empty
                        continue

                    # Normalize swap coord pair for the processed check
                    coord1 = (r, c)
                    coord2 = (nr, nc)
                    current_swap_normalized = tuple(sorted((coord1, coord2)))
                    if current_swap_normalized in processed_swaps:
                        continue
                    processed_swaps.add(current_swap_normalized)

                    # Check for Bonus Activation
                    is_bonus_activation = False
                    # Can swap a bonus with any Gem or another Bonus
                    if (tile1 in BONUS_TYPES and tile2 in ALL_GEMS_AND_BONUSES) or (
                        tile2 in BONUS_TYPES and tile1 in ALL_GEMS_AND_BONUSES
                    ):
                        is_bonus_activation = True

                    # Simulate the swap and check for matches
                    swapped_grid = self._simulate_swap(self.content, coord1, coord2)
                    matches_after_swap = self._get_matches(swapped_grid)

                    if is_bonus_activation or matches_after_swap:
                        valid_swaps.append((coord1, coord2))
                        # log.debug(f"  Found valid swap: {(coord1, coord2)} -> Bonus: {is_bonus_activation}, Match: {bool(matches_after_swap)}")

        # log.info(f"Found {len(valid_swaps)} valid swaps.")
        return valid_swaps

    def _apply_gravity_and_fill(self):
        """
        Applies gravity to tiles and fragments, then fills empty top cells.
        Modifies self.content in place. Returns list of fallen fragment coords.
        """
        fallen_fragments: List[GridCoord] = []
        new_content = [[EMPTY for _ in range(self.cols)] for _ in range(self.rows)]

        for c in range(self.cols):
            write_row = self.rows - 1
            # Process column from bottom up
            for r in range(self.rows - 1, -1, -1):
                tile = self.content[r][c]
                if tile == EMPTY:
                    continue
                elif tile == FRAGMENT:
                    # Fragment falls as far as possible
                    fall_to_r = r
                    while (
                        fall_to_r + 1 < self.rows
                        and new_content[fall_to_r + 1][c] == EMPTY
                    ):
                        fall_to_r += 1
                    new_content[fall_to_r][c] = FRAGMENT
                    if fall_to_r != r:
                        fallen_fragments.append((fall_to_r, c))
                    # Since we fill bottom-up, this space is now claimed
                    write_row = min(
                        write_row, fall_to_r - 1
                    )  # Adjust next potential write pos
                else:  # Gems or Bonuses
                    # Find the lowest empty spot at or below current write_row
                    place_r = write_row
                    while (
                        place_r + 1 < self.rows and new_content[place_r + 1][c] == EMPTY
                    ):
                        place_r += 1
                    # Place tile if a valid spot is found within column bounds
                    while place_r >= 0 and new_content[place_r][c] != EMPTY:
                        place_r -= 1
                    if place_r >= 0:
                        new_content[place_r][c] = tile
                        write_row = place_r - 1  # next tile goes above this one

            # Fill remaining empty cells at the top with new random gems
            for r in range(self.rows):
                if new_content[r][c] == EMPTY:
                    new_content[r][c] = random.choice(GEM_TYPES)

        self.content = new_content
        return fallen_fragments

    def _handle_background_break(self, coords: Set[GridCoord]) -> int:
        """
        Breaks background tiles at given coordinates.
        Modifies self.background.
        Returns the number of background tiles broken/damaged *in this step*.
        """
        broken_count_step = 0
        for r, c in coords:
            if not is_valid_coord(r, c, self.rows, self.cols):
                continue

            bg = self.background[r][c]
            if bg == BG_SHIELD:
                self.background[r][c] = BG_STONE
                broken_count_step += 1  # Count shield break as progress
                log.debug(f"Background at {(r, c)}: Shield -> Stone")
            elif bg == BG_STONE:
                self.background[r][c] = BG_NONE
                self.background_destroyed_count += 1
                broken_count_step += 1
                log.debug(f"Background at {(r, c)}: Stone -> None")
                # Check for fragment spawning after breaking stone
                self._check_fragment_spawn()
        return broken_count_step

    def _apply_bonus_effect(
        self, bonus_type: str, r: int, c: int
    ) -> Tuple[Set[GridCoord], Set[GridCoord]]:
        """
        Applies the effect of a triggered bonus.
        Returns:
            - Set of coordinates cleared by the bonus (excluding the bonus tile itself).
            - Set of coordinates of other bonuses triggered by this bonus.
        """
        cleared_coords: Set[GridCoord] = set()
        triggered_bonus_coords: Set[GridCoord] = set()

        # Mark bonus tile itself as cleared (will be removed later)
        # cleared_coords.add((r, c)) # No, don't add self to cleared, it's handled separately

        if bonus_type == BONUS_0:  # Row Clear
            log.debug(f"Activating Bonus_0 (Row Clear) at {(r, c)}")
            self.bonus_activation_counter += 1
            for col_idx in range(self.cols):
                if (r, col_idx) != (
                    r,
                    c,
                ):  # don't clear self with own effect, avoid fragment
                    coord = (r, col_idx)
                    tile = self.content[r][col_idx]
                    if tile != FRAGMENT and tile != EMPTY:
                        cleared_coords.add(coord)
                        if tile in BONUS_TYPES:
                            triggered_bonus_coords.add(coord)

        elif bonus_type == BONUS_1:  # Plus Clear
            log.debug(f"Activating Bonus_1 (Plus Clear) at {(r, c)}")
            self.bonus_activation_counter += 1
            # Row
            for col_idx in range(self.cols):
                if (r, col_idx) != (r, c):
                    coord = (r, col_idx)
                    tile = self.content[r][col_idx]
                    if tile != FRAGMENT and tile != EMPTY:
                        cleared_coords.add(coord)
                        if tile in BONUS_TYPES:
                            triggered_bonus_coords.add(coord)
            # Column
            for row_idx in range(self.rows):
                if (row_idx, c) != (r, c):
                    coord = (row_idx, c)
                    tile = self.content[row_idx][c]
                    if tile != FRAGMENT and tile != EMPTY:
                        cleared_coords.add(coord)
                        if tile in BONUS_TYPES:
                            # Avoid double adding if row/col intersect on another bonus
                            triggered_bonus_coords.add(coord)

        elif bonus_type == BONUS_2:  # Random Clear
            log.debug(f"Activating Bonus_2 (Random Clear) at {(r, c)}")
            # (No increment for bonus_activation_counter for Bonus_2)
            num_to_clear = random.randint(15, 20)
            potential_targets = []
            for rr in range(self.rows):
                for cc in range(self.cols):
                    if (rr, cc) != (r, c):  # don't clear self
                        tile = self.content[rr][cc]
                        # Cannot clear fragments or other bonuses or empty
                        if (
                            tile != FRAGMENT
                            and tile not in BONUS_TYPES
                            and tile != EMPTY
                        ):
                            potential_targets.append((rr, cc))

            random.shuffle(potential_targets)
            cleared_coords.update(potential_targets[:num_to_clear])
            # Bonus_2 does not trigger other bonuses

        # Check if activating Bonus_0 or Bonus_1 triggers Bonus_2 creation
        if (
            bonus_type in [BONUS_0, BONUS_1]
            and self.bonus_activation_counter % 4 == 0
            and self.bonus_activation_counter > 0
        ):
            self._spawn_bonus(BONUS_2)  # Spawn AFTER current effects resolve

        log.debug(
            f"Bonus effect: Cleared {cleared_coords}, Triggered Bonuses {triggered_bonus_coords}"
        )
        return cleared_coords, triggered_bonus_coords

    def _spawn_bonus(self, bonus_type: str):
        """Spawns a bonus (usually Bonus_2) at a suitable top location."""
        spawn_options = []
        for c in range(self.cols):
            # Find the highest empty cell in the column
            highest_empty_r = -1
            for r in range(self.rows):
                if self.content[r][c] == EMPTY:
                    highest_empty_r = r
                    break  # Found the highest one
            if highest_empty_r != -1:
                spawn_options.append((highest_empty_r, c))

        if spawn_options:
            r, c = random.choice(spawn_options)
            self.content[r][c] = bonus_type
            log.info(f"Spawned {bonus_type} at {(r, c)}")
        else:
            log.warning(f"Could not find empty space to spawn {bonus_type}!")

    def _check_fragment_spawn(self):
        """Checks if conditions are met to spawn a new fragment."""
        if self.total_initial_background_tiles == 0:
            return  # Avoid division by zero

        if (
            self.background_destroyed_count / self.total_initial_background_tiles
        ) > 0.5:
            # Check if we *already* spawned for this threshold (simple flag for now)
            # A more robust way might be to track how many *should* have spawned vs how many *have*
            if (
                not hasattr(self, "_fragment_spawned_50pct")
                or not self._fragment_spawned_50pct
            ):
                log.info("Over 50% background cleared, attempting to spawn fragment.")
                self._spawn_fragment()
                self._fragment_spawned_50pct = (
                    True  # Mark as spawned for this threshold
                )

    def _spawn_fragment(self):
        """Spawns a fragment at a suitable top location."""
        spawn_options = []
        for c in range(self.cols):
            highest_empty_r = -1
            for r in range(self.rows):
                if self.content[r][c] == EMPTY:
                    highest_empty_r = r
                    break
            if highest_empty_r != -1:
                spawn_options.append((highest_empty_r, c))

        if spawn_options:
            r, c = random.choice(spawn_options)
            self.content[r][c] = FRAGMENT
            self.fragments_on_board += 1
            log.info(
                f"Spawned Fragment at {(r, c)}. Fragments on board: {self.fragments_on_board}"
            )
        else:
            log.warning("Could not find empty space to spawn Fragment!")

    def _check_completion_and_failure(self) -> Tuple[bool, int]:
        """
        Checks if the level is won (all fragments collected) or lost (no moves).
        Returns: (done, final_reward_penalty)
        """
        # 1. Check Win Condition
        if self.fragments_collected >= self.target_fragments:
            log.info(
                f"Level Complete! Collected {self.fragments_collected}/{self.target_fragments} fragments."
            )
            return True, 1000  # Large positive reward for winning

        # 2. Check Lose Condition - No Valid Moves
        # Note: Only check if not already won
        if not self.get_valid_swaps():
            log.info("Level Failed! No more valid moves.")
            return True, -500  # Large negative penalty for losing

        # 3. Check Lose Condition - Max Moves (if applicable)
        if self.max_moves is not None and self.moves_made >= self.max_moves:
            log.info(f"Level Failed! Exceeded max moves ({self.max_moves}).")
            return True, -500  # Large negative penalty

        # TODO: Add time limit check if needed

        # 4. Game continues
        return False, 0

    def step(self, swap_action: Swap) -> Tuple[np.ndarray, int, bool]:
        """
        Executes a swap action, resolves cascades, and updates the game state.
        Args:
            swap_action: A tuple ((r1, c1), (r2, c2)) representing the swap.
        Returns:
            - new_state: The state representation after the step.
            - step_reward: The reward accumulated during this step (swap + cascades).
            - done: Boolean indicating if the game ended (win or loss).
        """
        self.moves_made += 1
        step_reward = 0
        done = False
        final_reward_penalty = 0

        coord1, coord2 = swap_action
        r1, c1 = coord1
        r2, c2 = coord2

        # --- Validate Swap (Basic - should be pre-validated by agent ideally) ---
        # Is it a known valid swap according to get_valid_swaps?
        # Or at least, check basic rules here again? For robustness:
        if not (
            is_valid_coord(r1, c1, self.rows, self.cols)
            and is_valid_coord(r2, c2, self.rows, self.cols)
            and abs(r1 - r2) + abs(c1 - c2) == 1  # Adjacent
            and self.content[r1][c1] != FRAGMENT
            and self.content[r2][c2] != FRAGMENT
            and self.content[r1][c1] != EMPTY
            and self.content[r2][c2] != EMPTY
        ):
            log.error(f"Invalid swap attempted in step: {swap_action}")
            # Return current state, large penalty, and potentially end the game?
            # Or just penalty and no state change? Let's do penalty, no state change.
            return (
                self._get_state_representation(),
                -100,
                False,
            )  # Small penalty for invalid action attempt

        log.debug(f"\n--- Step {self.moves_made}: Applying Swap {swap_action} ---")

        # --- 1. Apply the Swap ---
        tile1_before = self.content[r1][c1]
        tile2_before = self.content[r2][c2]
        self.content[r1][c1], self.content[r2][c2] = (
            self.content[r2][c2],
            self.content[r1][c1],
        )
        step_reward -= 1  # Small cost per move

        # --- Cascade Loop ---
        cascade_level = 0
        while True:
            cascade_level += 1
            log.debug(f"--- Cascade Level {cascade_level} ---")

            # --- 2a. Find Matches and Activated Bonuses ---
            current_matches = self._get_matches(self.content)
            activated_bonuses: Dict[GridCoord, str] = (
                {}
            )  # Store coords and types of bonuses activated this cycle

            # Check if the initial swap activated a bonus directly
            # (Needs careful check to avoid double counting if swap *also* creates match)
            swap_activated_bonus = False
            if cascade_level == 1:  # Only check on the first cycle after the swap
                if tile1_before in BONUS_TYPES and tile2_before in ALL_GEMS_AND_BONUSES:
                    activated_bonuses[(r1, c1)] = (
                        tile1_before  # Bonus ended up at (r1, c1) after swap
                    )
                    swap_activated_bonus = True
                elif (
                    tile2_before in BONUS_TYPES and tile1_before in ALL_GEMS_AND_BONUSES
                ):
                    activated_bonuses[(r2, c2)] = (
                        tile2_before  # Bonus ended up at (r2, c2) after swap
                    )
                    swap_activated_bonus = True

            # Find coords involved in current gem matches
            matched_coords_this_cycle: Set[GridCoord] = (
                set().union(*current_matches) if current_matches else set()
            )

            # Identify bonuses *within* matched groups (they activate too)
            for r_match, c_match in matched_coords_this_cycle:
                tile = self.content[r_match][c_match]
                if tile in BONUS_TYPES:
                    if (
                        r_match,
                        c_match,
                    ) not in activated_bonuses:  # Avoid double activation
                        activated_bonuses[(r_match, c_match)] = tile

            # --- 2b. Exit if No Matches or Bonus Activations ---
            if not current_matches and not activated_bonuses:
                log.debug("No more matches or bonus activations found. Ending cascade.")
                break  # Exit cascade loop

            # --- Process Activations and Matches ---
            all_cleared_coords: Set[GridCoord] = set()  # All tiles removed this cycle
            all_triggered_bonuses: Set[GridCoord] = (
                set()
            )  # Bonuses activated by other bonuses

            # Apply bonus effects FIRST (they might clear gems needed for matches)
            bonuses_to_process = list(activated_bonuses.items())
            processed_bonus_coords = set()

            while bonuses_to_process:
                (br, bc), b_type = bonuses_to_process.pop(0)
                if (br, bc) in processed_bonus_coords:
                    continue  # Avoid reprocessing chained bonuses in same cycle

                log.debug(f"Processing activated bonus {b_type} at {(br, bc)}")
                step_reward += 25  # Reward for activating a bonus
                all_cleared_coords.add((br, bc))  # The bonus itself is cleared
                processed_bonus_coords.add((br, bc))

                cleared_by_bonus, triggered_by_bonus = self._apply_bonus_effect(
                    b_type, br, bc
                )
                all_cleared_coords.update(cleared_by_bonus)
                all_triggered_bonuses.update(triggered_by_bonus)

                # Add newly triggered bonuses to the processing list
                for tr, tc in triggered_by_bonus:
                    if (tr, tc) not in processed_bonus_coords and (tr, tc) not in [
                        p[0] for p in bonuses_to_process
                    ]:
                        triggered_type = self.content[tr][tc]
                        if triggered_type in BONUS_TYPES:  # Should always be true
                            bonuses_to_process.append(((tr, tc), triggered_type))

            # Now add gem match coordinates (excluding already cleared/activated bonuses)
            final_match_coords = matched_coords_this_cycle - processed_bonus_coords
            all_cleared_coords.update(final_match_coords)

            # --- 2c/f. Calculate Rewards (Matches, Background Breaks) ---
            # Reward for matched gems
            step_reward += len(final_match_coords) * 2  # Base reward per gem
            # Reward for background breaks (applies to all cleared coords)
            broken_count = self._handle_background_break(all_cleared_coords)
            step_reward += broken_count * 10

            # --- Create New Bonuses from Matches ---
            new_bonuses_coords: Dict[GridCoord, str] = {}
            if (
                not swap_activated_bonus
            ):  # Don't create bonus if swap just activated one
                for match_set in current_matches:
                    # Ensure the match involves coords from the *original* swap
                    # OR entstand purely from falling tiles (more complex check needed?)
                    # Simplification: Check match size, place at swap loc if relevant
                    match_involved_swap = coord1 in match_set or coord2 in match_set
                    if (
                        cascade_level == 1 and match_involved_swap
                    ):  # only first cascade level and involving swap
                        if len(match_set) >= 5:
                            # Check if coord1 is available and not already cleared by bonus
                            if coord1 not in all_cleared_coords:
                                new_bonuses_coords[coord1] = BONUS_1
                            # else maybe place at coord2? Needs careful logic
                            log.debug(
                                f"Creating Bonus_1 at {coord1} from {len(match_set)}-match"
                            )
                            step_reward += 50
                            break  # Only create one bonus per initiating match group? Assume yes.
                        elif len(match_set) == 4:
                            if coord1 not in all_cleared_coords:
                                new_bonuses_coords[coord1] = BONUS_0
                            log.debug(f"Creating Bonus_0 at {coord1} from 4-match")
                            step_reward += 30
                            break  # Only one bonus

            # --- 2d/e. Remove Tiles & Place New Bonuses ---
            log.debug(f"Clearing coords: {all_cleared_coords}")
            for r_clear, c_clear in all_cleared_coords:
                if (r_clear, c_clear) in new_bonuses_coords:
                    self.content[r_clear][c_clear] = new_bonuses_coords[
                        (r_clear, c_clear)
                    ]
                    log.debug(
                        f"Placed new {new_bonuses_coords[(r_clear, c_clear)]} at {(r_clear, c_clear)}"
                    )
                else:
                    self.content[r_clear][c_clear] = EMPTY

            # --- 2h/i. Apply Gravity and Fill ---
            fallen_fragments_coords = self._apply_gravity_and_fill()
            log.debug("Applied gravity and filled top rows.")

            # --- Check Fragment Collection ---
            fragments_collected_this_step = 0
            for r_frag, c_frag in fallen_fragments_coords:
                if (
                    r_frag == self.rows - 1
                    and self.background[r_frag][c_frag] == BG_NONE
                ):
                    # Fragment reached bottom on a cleared background cell
                    if (
                        self.content[r_frag][c_frag] == FRAGMENT
                    ):  # Verify it's still there
                        self.content[r_frag][
                            c_frag
                        ] = EMPTY  # Remove collected fragment
                        self.fragments_on_board -= 1
                        self.fragments_collected += 1
                        fragments_collected_this_step += 1
                        log.info(
                            f"Fragment collected at {(r_frag, c_frag)}! Total: {self.fragments_collected}/{self.target_fragments}"
                        )

            if fragments_collected_this_step > 0:
                step_reward += (
                    fragments_collected_this_step * 500
                )  # Large reward per fragment

            # --- Loop continues to check for new matches caused by falling tiles ---

        # --- End of Cascade Loop ---

        # --- 4/5. Check for Level Completion or Failure ---
        done, final_reward_penalty = self._check_completion_and_failure()
        step_reward += final_reward_penalty

        # --- 7. Get New State ---
        new_state = self._get_state_representation()

        log.debug(
            f"Step {self.moves_made} finished. Reward: {step_reward}, Done: {done}"
        )
        # log.debug(f"Board Content:\n{self._grid_to_string(self.content)}")
        # log.debug(f"Board Background:\n{self._grid_to_string(self.background)}")

        return new_state, step_reward, done

    def _grid_to_string(self, grid):
        """Helper for debugging: formats grid for printing."""
        if not grid:
            return "[]"
        return "\n".join(
            [" ".join([f"{str(cell):<9}" for cell in row]) for row in grid]
        )

    def render(self):
        """Optional: Basic text-based rendering for debugging."""
        print("\n--- Current State ---")
        print(f"Moves: {self.moves_made}/{self.max_moves if self.max_moves else 'inf'}")
        print(f"Score: {self.current_score}")  # Score not really used yet
        print(
            f"Fragments: {self.fragments_collected}/{self.target_fragments} (On Board: {self.fragments_on_board})"
        )
        print(
            f"Background Cleared: {self.background_destroyed_count}/{self.total_initial_background_tiles}"
        )
        print("Content:")
        print(self._grid_to_string(self.content))
        print("Background:")
        print(self._grid_to_string(self.background))
        print("-" * 20)


# Example Usage (for testing)
if __name__ == "__main__":
    sim = SevenWondersSimulator()
    sim.render()

    # Try getting valid swaps
    valid_swaps = sim.get_valid_swaps()
    print(f"Initial valid swaps: {valid_swaps}")

    if valid_swaps:
        # Take the first valid swap
        swap_to_try = valid_swaps[0]
        print(f"\nAttempting swap: {swap_to_try}")
        new_state, reward, done = sim.step(swap_to_try)

        print(f"\nAfter first step:")
        sim.render()
        print(f"Reward: {reward}, Done: {done}")

        # Simulate a few more random valid moves
        for i in range(5):
            if done:
                break
            print(f"\n--- Turn {i+2} ---")
            valid_swaps = sim.get_valid_swaps()
            if not valid_swaps:
                print("No more valid moves!")
                break
            swap_to_try = random.choice(valid_swaps)
            print(f"Attempting swap: {swap_to_try}")
            new_state, reward, done = sim.step(swap_to_try)
            sim.render()
            print(f"Reward: {reward}, Done: {done}")

    else:
        print("No valid moves possible on initial board.")
