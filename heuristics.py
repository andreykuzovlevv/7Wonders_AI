# heuristics_plus.py - Revised with dynamic priorities and objective focus
import random, math, copy, time, logging
import config  # Uses your existing constants
from typing import List, Tuple, Set, Dict, Optional

GridCoord = Tuple[int, int]
Swap = Tuple[GridCoord, GridCoord]
log = logging.getLogger(__name__)

# --------------------------- 1. GLOBAL WEIGHTS (Revised) ---------------------------
W = dict(
    # --- Base Match Scoring ---
    tiles_base=1.0,  # Score per tile in any match
    match4_bonus=2.0,  # Extra points for forming a 4-match line/shape
    match5_bonus=5.0,  # Extra points for forming a 5+ match line/shape/T/L
    # --- Bonus Creation ---
    bonus_creation_4=3.0,  # Reward for *creating* a bonus from a 4-match
    bonus_creation_5=6.0,  # Reward for *creating* a bonus from a 5+ match
    # --- Base Background Breaking Weights (Value increases dynamically) ---
    stone=5.0,  # Base value for breaking a stone (multiplied by scarcity)
    shield=10.0,  # Base value for breaking a shield (multiplied by scarcity)
    # --- Dynamic Weighting Control ---
    background_scarcity_multiplier=25.0,  # Factor controlling how much stone/shield value increases as they become rare (Higher = more value increase)
    # --- Fragment Path Clearing (Highest Priority) ---
    fragment_path_clear=50.0,  # VERY HIGH reward for clearing a tile DIRECTLY below a fragment (or in its path)
    # --- Penalties / Discouragement ---
    empty_above_penalty=-0.2,  # Minor penalty for matching high in already cleared columns (prevents wasteful top moves)
    bonus_destroys_bonus_penalty=-30.0,  # STRONG penalty for a bonus destroying another bonus tile
    bonus_hits_empty_penalty=-0.5,  # Penalty for bonus effect hitting already cleared background/empty cell
    # --- Bonus Activation Scoring ---
    bonus_base_tile_clear=0.5,  # Base value for a bonus clearing *any* regular gem tile (not background/path)
    # bonus_2 estimate is now more dynamic within estimate_bonus_score
    # --- Cascade Chaining ---
    chain_depth_decay=0.6,  # Multiplier for 1st cascade score, ^2 for 2nd etc. (Value < 1)
    # --- REMOVED/REPLACED ---
    # block=10.0              # Fragments aren't 'broken', path clearing is key
    # near_block_adj=0.8      # Replaced by direct fragment_path_clear
    # bottom_row_bonus=0.5    # Implicitly handled by fragment path focus
    # edge_bonus=0.3          # Less critical than path/background focus
    # bonus0/1_clear          # Replaced by dynamic bonus estimation
    # bonus2_estimated        # Replaced by dynamic bonus estimation
)

# --------------------------- 2. HELPER UTILITIES ---------------------------


def count_tiles(grid: List[List[str]], tile_type: str) -> int:
    """Counts occurrences of a specific tile type in content or background grid."""
    count = 0
    rows = len(grid)
    if rows == 0:
        return 0
    cols = len(grid[0])
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == tile_type:
                count += 1
    return count


def get_fragment_paths(content: List[List[str]]) -> Set[GridCoord]:
    """
    Identifies all grid cells located underneath the lowest 'block' (fragment)
    in each column, down to the bottom row. These are critical clear targets.
    Returns a set of (row, col) tuples.
    """
    rows, cols = config.GRID_ROWS, config.GRID_COLS
    paths: Set[GridCoord] = set()
    for c in range(cols):
        fragment_r = -1
        # Find the row index of the lowest fragment in this column
        for r in range(rows - 1, -1, -1):  # Scan bottom-up
            if content[r][c] == "block":
                fragment_r = r
                break  # Found the lowest one
        # If a fragment was found, add all cells below it to the path set
        if fragment_r != -1:
            for below_r in range(fragment_r + 1, rows):
                paths.add((below_r, c))
    return paths


def bonus_footprint(tile_type: str, r: int, c: int) -> Set[GridCoord]:
    """
    Returns the set of coordinates potentially affected by a bonus activation.
    Handles grid boundaries. (bonus_2 returns empty set - handled in estimate).
    """
    rows, cols = config.GRID_ROWS, config.GRID_COLS
    cells: Set[GridCoord] = set()
    if tile_type == "bonus_0":  # Horizontal line
        cells.update((r, col) for col in range(cols))
    elif tile_type == "bonus_1":  # Cross (+)
        cells.update((r, col) for col in range(cols))
        cells.update((row, c) for row in range(rows))
    elif tile_type == "bonus_2":  # Random - footprint determined by estimation logic
        pass
    # Filter out-of-bounds coordinates
    return {(rr, cc) for rr, cc in cells if 0 <= rr < rows and 0 <= cc < cols}


# --------------------------- 3. CORE SIMULATION UTILITIES ---------------------------


def simulate_swap(grid: List[List[str]], a: GridCoord, b: GridCoord) -> List[List[str]]:
    """Creates a new grid state after swapping tiles at coords a and b."""
    new_grid = [row[:] for row in grid]  # Deep copy
    (ra, ca), (rb, cb) = a, b
    new_grid[ra][ca], new_grid[rb][cb] = new_grid[rb][cb], new_grid[ra][ca]
    return new_grid


def get_matches(grid: List[List[str]]) -> List[Set[GridCoord]]:
    """
    Finds all groups of 3+ adjacent (H/V) identical, matchable tiles.
    Merges overlapping groups (e.g., T/L shapes).
    Ignores 'empty', 'block', and 'bonus_*' tiles for matching.
    """
    rows, cols = config.GRID_ROWS, config.GRID_COLS
    matchable_tiles = {f"gem_{i}" for i in range(8)}  # Only gems are matchable
    initial_matches: List[Set[GridCoord]] = []

    # Horizontal scan
    for r in range(rows):
        c = 0
        while c < cols - 2:
            tile = grid[r][c]
            if tile not in matchable_tiles:
                c += 1
                continue
            run_end = c + 1
            while run_end < cols and grid[r][run_end] == tile:
                run_end += 1
            if run_end - c >= 3:
                initial_matches.append({(r, x) for x in range(c, run_end)})
            c = run_end

    # Vertical scan
    for c in range(cols):
        r = 0
        while r < rows - 2:
            tile = grid[r][c]
            if tile not in matchable_tiles:
                r += 1
                continue
            run_end = r + 1
            while run_end < rows and grid[run_end][c] == tile:
                run_end += 1
            if run_end - r >= 3:
                initial_matches.append({(x, c) for x in range(r, run_end)})
            r = run_end

    # Merge overlapping matches of the same type (already guaranteed by scan)
    if not initial_matches:
        return []

    merged_matches: List[Set[GridCoord]] = []
    processed_indices = [False] * len(initial_matches)

    for i in range(len(initial_matches)):
        if processed_indices[i]:
            continue
        current_set = initial_matches[i].copy()
        processed_indices[i] = True
        merged_occurred = True
        while merged_occurred:
            merged_occurred = False
            for j in range(i + 1, len(initial_matches)):
                if not processed_indices[j] and current_set.intersection(
                    initial_matches[j]
                ):
                    current_set.update(initial_matches[j])
                    processed_indices[j] = True
                    merged_occurred = True
        merged_matches.append(current_set)

    return merged_matches


def apply_gravity(content: List[List[str]]) -> List[List[str]]:
    """
    Simulates simple vertical gravity after tile removals.
    Treats 'block' tiles as unmovable by gravity unless space below is empty
    (basic simulation - real game might differ). Fills top with 'empty'.
    Returns the *modified* grid (in-place modification for efficiency).

    NOTE: This is a simplified gravity model primarily for cascade *detection*
    in the lookahead. The *scoring* prioritizes based on the game state *before* gravity.
    It does NOT perfectly simulate fragment dropping physics, which is complex.
    Assumes 'block' tiles only fall one step if the cell below is 'empty'.
    """
    rows, cols = config.GRID_ROWS, config.GRID_COLS

    # Create a list of tiles that fall per column, excluding blocks initially
    for c in range(cols):
        falling_tiles = []
        original_block_rows = []
        for r in range(rows):
            if content[r][c] == "block":
                original_block_rows.append(r)
            elif content[r][c] != "empty":
                falling_tiles.append(content[r][c])

        # Clear the column first
        for r in range(rows):
            content[r][c] = "empty"

        # Place falling non-block tiles from the bottom up
        write_row = rows - 1
        for tile in reversed(falling_tiles):
            if write_row >= 0:
                content[write_row][c] = tile
                write_row -= 1

        # Place blocks back, potentially falling one step
        for r_orig in sorted(original_block_rows):
            r_target = r_orig
            # Check if space directly below original position IS NOW empty
            if r_orig + 1 < rows and content[r_orig + 1][c] == "empty":
                r_target = r_orig + 1  # Fall one step
            # Place block - find lowest available spot at or below r_target
            final_r = r_target
            while final_r + 1 < rows and content[final_r + 1][c] == "empty":
                final_r += 1

            if final_r >= 0:  # Should always be true if rows > 0
                content[final_r][c] = "block"

    return content


# ------------------- 4. BONUS ACTIVATION SCORING (Revised) -------------------


def estimate_bonus_score(
    tile_type: str,
    r: int,
    c: int,
    content: List[List[str]],
    background: List[List[str]],
    stone_mult: float,
    shield_mult: float,
    fragment_paths: Set[GridCoord],
) -> float:
    """
    Estimates the strategic value of ACTIVATING a bonus tile at (r, c).
    Considers fragment paths, scarce background tiles, and penalties.
    """
    score = 0.0

    if tile_type == "bonus_2":
        # Estimate value based on remaining high-value targets on the *entire board*
        # This is a rough heuristic for the random effect
        num_clears_expected = 18  # Average clears
        target_score = 0
        # Value from potential path clears
        if fragment_paths:
            # High value if any paths exist, scaled slightly by how many cells
            target_score += (
                W["fragment_path_clear"]
                * stone_mult
                * (len(fragment_paths) / 5.0 + 1.0)
            )  # Heuristic scaling
        # Value from potential stone/shield clears
        target_score += count_tiles(background, "stone") * W["stone"] * stone_mult
        target_score += count_tiles(background, "shield") * W["shield"] * shield_mult
        # Average value per clear based on total target value
        avg_value_per_clear = (
            target_score / (config.GRID_ROWS * config.GRID_COLS)
            if config.GRID_ROWS > 0
            else 0
        )
        score = (
            avg_value_per_clear * num_clears_expected * 0.5
        )  # Factor down estimate randomness
        # Ensure minimum score if targets exist but estimate is low
        if target_score > 0 and score < W["tiles_base"] * 5:
            score = W["tiles_base"] * 5
        log.debug(
            f"  Bonus_2 at ({r},{c}): Estimated score based on board state = {score:.2f}"
        )
        return score

    # --- Handle bonus_0 and bonus_1 ---
    cells_to_clear = bonus_footprint(tile_type, r, c)
    log.debug(f"  Bonus {tile_type} at ({r},{c}) affects {len(cells_to_clear)} cells.")

    for rr, cc in cells_to_clear:
        cell_score = 0
        current_content = content[rr][cc]
        current_background = background[rr][cc]

        # 1. Big Penalty: Destroying another bonus?
        if current_content.startswith("bonus_") and (rr, cc) != (r, c):
            cell_score += W["bonus_destroys_bonus_penalty"]
            score += cell_score  # Apply penalty and skip other scoring for this cell
            log.debug(
                f"    ({rr},{cc}): Hit other bonus! Penalty={W['bonus_destroys_bonus_penalty']:.2f}"
            )
            continue

        # 2. Highest Priority: Clearing a fragment path?
        if (rr, cc) in fragment_paths:
            # Significant reward, potentially reduced if it's just hitting empty space in path
            path_reward = W["fragment_path_clear"]
            if current_background == "none" and current_content == "empty":
                path_reward *= 0.1  # Much less value if path cell already clear
            cell_score += path_reward
            log.debug(f"    ({rr},{cc}): Hit fragment path! Score+{path_reward:.2f}")

        # 3. Priority: Clearing background tiles? (Apply dynamic multiplier)
        if current_background == "stone":
            bg_clear_score = W["stone"] * stone_mult
            cell_score += bg_clear_score
            log.debug(
                f"    ({rr},{cc}): Hit stone (mult={stone_mult:.2f})! Score+{bg_clear_score:.2f}"
            )
        elif current_background == "shield":
            bg_clear_score = W["shield"] * shield_mult
            cell_score += bg_clear_score
            log.debug(
                f"    ({rr},{cc}): Hit shield (mult={shield_mult:.2f})! Score+{bg_clear_score:.2f}"
            )
        elif current_background == "none" and current_content == "empty":
            # Penalize hitting already empty cells unless it's a needed path clear
            if (rr, cc) not in fragment_paths:
                cell_score += W["bonus_hits_empty_penalty"]
                log.debug(
                    f"    ({rr},{cc}): Hit empty background. Penalty={W['bonus_hits_empty_penalty']:.2f}"
                )

        # 4. Base value: Clearing any regular tile?
        if (
            current_content != "empty"
            and current_content != "block"
            and not current_content.startswith("bonus_")
        ):
            cell_score += W["bonus_base_tile_clear"]
            # log.debug(f"    ({rr},{cc}): Hit regular tile. Score+{W['bonus_base_tile_clear']:.2f}") # Can be verbose

        score += cell_score

    log.debug(f"  Bonus {tile_type} at ({r},{c}): Total Estimated Score = {score:.2f}")
    return score


# ----------------------- 5. SWAP SCORING (Revised Core Logic) -----------------------


def score_swap(
    swap: Swap,
    matches: List[Set[GridCoord]],  # Merged matches from get_matches
    content: List[List[str]],
    background: List[List[str]],
) -> float:
    """
    Scores a potential swap based on dynamic priorities:
    1. Is it activating a bonus effectively?
    2. Does it clear fragment paths?
    3. Does it break scarce background tiles?
    4. Does it create new bonuses?
    5. Basic match value and cascade potential.
    """
    rows, cols = config.GRID_ROWS, config.GRID_COLS
    score = 0.0
    (r1, c1), (r2, c2) = swap

    # --- Calculate Dynamic State ---
    num_remaining_stones = count_tiles(background, "stone")
    num_remaining_shields = count_tiles(background, "shield")
    # Add small epsilon to avoid division by zero if count is 0
    stone_multiplier = 1.0 + (
        W["background_scarcity_multiplier"] / (num_remaining_stones + 1.0)
    )
    shield_multiplier = 1.0 + (
        W["background_scarcity_multiplier"] / (num_remaining_shields + 1.0)
    )
    fragment_paths = get_fragment_paths(content)
    log.debug(
        f" Scoring Swap: {swap}. Paths: {len(fragment_paths)}, Stones: {num_remaining_stones}(x{stone_multiplier:.2f}), Shields: {num_remaining_shields}(x{shield_multiplier:.2f})"
    )

    # --- Check if this is primarily a Bonus Activation Swap ---
    tile1, tile2 = content[r1][c1], content[r2][c2]
    is_bonus_activation = False
    bonus_tile_type, bonus_r, bonus_c = None, -1, -1
    # Define swappable non-bonus targets (cannot swap bonus with empty/block/another bonus)
    swappable_target = (
        lambda t: t != "empty" and t != "block" and not t.startswith("bonus_")
    )

    if tile1.startswith("bonus_") and swappable_target(tile2):
        is_bonus_activation = True
        bonus_tile_type, bonus_r, bonus_c = (
            tile1,
            r1,
            c1,
        )  # Effect originates from bonus tile's position
    elif tile2.startswith("bonus_") and swappable_target(tile1):
        is_bonus_activation = True
        bonus_tile_type, bonus_r, bonus_c = (
            tile2,
            r2,
            c2,
        )  # Effect originates from bonus tile's position

    # --- Score Calculation Branch ---
    if is_bonus_activation:
        # --- A. Score Bonus Activation Swap ---
        log.debug(
            f"  Swap is BONUS ACTIVATION: {bonus_tile_type} at ({bonus_r},{bonus_c})"
        )
        score = estimate_bonus_score(
            bonus_tile_type,
            bonus_r,
            bonus_c,
            content,
            background,
            stone_multiplier,
            shield_multiplier,
            fragment_paths,
        )
        # Add a tiny base value to ensure it's preferred over doing nothing if estimate is zero
        score += 0.01

    else:
        # --- B. Score Regular Match Swap ---
        if not matches:
            # This might happen if a bonus swap was possible but didn't create a gem match
            # Or if find_valid_swaps logic allowed a non-match swap (shouldn't ideally)
            log.warning(
                f"  Swap {swap} is not bonus activation and has no matches? Score=0"
            )
            return 0.0  # Or -math.inf if this state indicates an error

        log.debug(f"  Swap is REGULAR MATCH. Matches found: {len(matches)}")
        total_tiles_matched: Set[GridCoord] = set().union(*matches)
        match_score = 0.0

        # B.1. Base Score, Size Bonus, Bonus Creation
        match_size = len(total_tiles_matched)
        match_score += match_size * W["tiles_base"]
        if match_size == 4:
            match_score += W["match4_bonus"] + W["bonus_creation_4"]
            log.debug(
                f"    4-Match bonus: +{W['match4_bonus'] + W['bonus_creation_4']:.2f}"
            )
        elif match_size >= 5:
            match_score += W["match5_bonus"] + W["bonus_creation_5"]
            log.debug(
                f"    5+ Match bonus: +{W['match5_bonus'] + W['bonus_creation_5']:.2f}"
            )

        # B.2. Evaluate Each Matched Tile's Contribution
        for r_match, c_match in total_tiles_matched:
            tile_contrib = 0
            # Highest prio: Is it clearing a fragment path?
            if (r_match, c_match) in fragment_paths:
                tile_contrib += W["fragment_path_clear"]
                log.debug(
                    f"    ({r_match},{c_match}): In fragment path! +{W['fragment_path_clear']:.2f}"
                )
                # Don't apply background score *in addition* if it's a path clear? Debatable.
                # Let's make path clear override background score for simplicity/focus.
            else:
                # Prio: Is it clearing background? (Apply dynamic multiplier)
                bg = background[r_match][c_match]
                if bg == "stone":
                    bg_score = W["stone"] * stone_multiplier
                    tile_contrib += bg_score
                    log.debug(
                        f"    ({r_match},{c_match}): Cleared stone (x{stone_multiplier:.2f})! +{bg_score:.2f}"
                    )
                elif bg == "shield":
                    bg_score = W["shield"] * shield_multiplier
                    tile_contrib += bg_score
                    log.debug(
                        f"    ({r_match},{c_match}): Cleared shield (x{shield_multiplier:.2f})! +{bg_score:.2f}"
                    )

            # Penalty: Is it high up in an already cleared column?
            if (
                background[r_match][c_match] == "none"
                and (r_match, c_match) not in fragment_paths
            ):
                # Check if cells below are mostly clear (more advanced check possible)
                is_high_clear = True
                for below_r in range(r_match + 1, rows):
                    if background[below_r][c_match] != "none":
                        is_high_clear = False
                        break
                if (
                    is_high_clear and r_match < rows // 2
                ):  # Only penalize top half clears?
                    tile_contrib += W["empty_above_penalty"]
                    log.debug(
                        f"    ({r_match},{c_match}): Empty above penalty. {W['empty_above_penalty']:.2f}"
                    )

            match_score += tile_contrib

        score += match_score

        # B.3. Cascade Lookahead (Simplified 1-step)
        if W.get("chain_depth_decay", 0) > 0:  # Check if cascade scoring is enabled
            log.debug("   Calculating cascade potential...")
            # Simulate the board *after* the initial match removals
            grid_after = simulate_swap(content, (r1, c1), (r2, c2))  # Get swapped grid
            bg_after = [row[:] for row in background]  # Copy background state

            # Remove matched tiles and update background for cascade scoring context
            for r_m, c_m in total_tiles_matched:
                grid_after[r_m][c_m] = "empty"
                if bg_after[r_m][c_m] == "shield":
                    bg_after[r_m][c_m] = "stone"
                elif bg_after[r_m][c_m] == "stone":
                    bg_after[r_m][c_m] = "none"

            # Apply gravity (using the simplified model)
            apply_gravity(grid_after)

            # Find matches in the cascaded state
            cascade_matches = get_matches(grid_after)
            if cascade_matches:
                log.debug(f"    Cascade detected! {len(cascade_matches)} groups.")
                cascade_gain = 0.0
                cascade_tiles = set().union(*cascade_matches)
                mult = W["chain_depth_decay"]  # Depth-1 cascade weight

                # Score cascade tiles similarly to primary match (path, background*mult)
                for r_cas, c_cas in cascade_tiles:
                    cas_contrib = mult * W["tiles_base"]  # Base score
                    if (r_cas, c_cas) in fragment_paths:  # Check original paths map
                        cas_contrib += (
                            mult * W["fragment_path_clear"] * 0.8
                        )  # Slightly less reward for cascade path clear?
                    else:
                        bg_cas = bg_after[r_cas][
                            c_cas
                        ]  # Use the updated background state
                        if bg_cas == "stone":
                            cas_contrib += mult * W["stone"] * stone_multiplier
                        elif bg_cas == "shield":
                            cas_contrib += mult * W["shield"] * shield_multiplier
                    cascade_gain += cas_contrib
                log.debug(f"    Cascade Score: +{cascade_gain:.2f}")
                score += cascade_gain
            else:
                log.debug("    No cascade detected.")

    log.info(f" Score for Swap {swap}: {score:.3f}")
    return score


# --------------------------- 6. FIND VALID SWAPS ---------------------------


def find_valid_swaps(
    content: List[List[str]],
    background: List[List[str]],  # Keep passing background if needed later
) -> List[Tuple[Swap, List[Set[GridCoord]]]]:
    """
    Finds all valid swaps:
    - Swapping two adjacent gems that create a match of 3+.
    - Swapping an adjacent bonus tile with a non-empty, non-block, non-bonus tile.
    Returns list of tuples: ((original_swap_coords), list_of_merged_matches_after_swap)
    """
    rows, cols = config.GRID_ROWS, config.GRID_COLS
    valid_swaps_info: List[Tuple[Swap, List[Set[GridCoord]]]] = []
    processed_swaps = (
        set()
    )  # Avoid checking same swap pair twice (e.g., (a,b) and (b,a))
    # Define tiles that cannot be part of a swap initiation or target (unless it's a bonus activating)
    non_swappable_base = {"empty", "block"}

    for r in range(rows):
        for c in range(cols):
            tile1 = content[r][c]
            if tile1 in non_swappable_base:
                continue

            # Check neighbors (right and down to avoid duplicates)
            for dr, dc in ((0, 1), (1, 0)):
                nr, nc = r + dr, c + dc
                if not (0 <= nr < rows and 0 <= nc < cols):
                    continue

                tile2 = content[nr][nc]
                if tile2 in non_swappable_base:
                    continue

                # Normalize swap for processed check
                current_swap_normalized = tuple(sorted(((r, c), (nr, nc))))
                if current_swap_normalized in processed_swaps:
                    continue
                processed_swaps.add(current_swap_normalized)

                # Original swap coordinates
                original_swap = ((r, c), (nr, nc))

                # --- Check Validity ---
                is_gem_match_swap = False
                is_bonus_activation_swap = False

                # Check for bonus activation: one is bonus, other is swappable target
                swappable_target = (
                    lambda t: t != "empty"
                    and t != "block"
                    and not t.startswith("bonus_")
                )
                if (tile1.startswith("bonus_") and swappable_target(tile2)) or (
                    tile2.startswith("bonus_") and swappable_target(tile1)
                ):
                    is_bonus_activation_swap = True

                # Check if swapping *gems* creates a match
                # (Don't bother simulating if it's a bonus activation, though it might also create match)
                matches_after_swap = []
                if not tile1.startswith("bonus_") and not tile2.startswith("bonus_"):
                    swapped_grid = simulate_swap(content, *original_swap)
                    matches_after_swap = get_matches(swapped_grid)
                    if matches_after_swap:
                        is_gem_match_swap = True

                # Add to list if valid
                if is_gem_match_swap or is_bonus_activation_swap:
                    # We need matches_after_swap even for bonus activation if it happens to create one
                    if is_bonus_activation_swap and not matches_after_swap:
                        swapped_grid = simulate_swap(content, *original_swap)
                        matches_after_swap = get_matches(
                            swapped_grid
                        )  # Check matches just in case

                    valid_swaps_info.append((original_swap, matches_after_swap))
                    log.debug(
                        f"  Found valid swap: {original_swap} (Bonus Activation: {is_bonus_activation_swap}, Gem Match: {is_gem_match_swap})"
                    )

    return valid_swaps_info


# --------------------------- 7. CHOOSE BEST SWAP (Wrapper) ---------------------------


def choose_best_swap(
    content: List[List[str]], background: List[List[str]], blacklist: Set[Swap] = set()
) -> Optional[Swap]:
    """
    Finds all valid swaps, scores them using the refined heuristics,
    and returns the highest-scoring swap (normalized) that isn't blacklisted.
    """
    swaps_info = find_valid_swaps(content, background)
    best_swap_normalized: Optional[Swap] = None
    best_score = -math.inf
    log.info(f"Evaluating {len(swaps_info)} potential valid swaps...")
    scored_swaps = []  # For logging top swaps

    if not swaps_info:
        log.warning("No valid swaps found!")
        return None

    for swap_coords, matches in swaps_info:
        # Normalize swap *only* for blacklist checking and the final return value
        norm_swap = tuple(sorted(swap_coords))

        if norm_swap in blacklist:
            log.warning(f"Skipping blacklisted swap: {norm_swap}")
            continue

        # Score the swap using its original coordinates
        current_score = score_swap(swap_coords, matches, content, background)
        scored_swaps.append(
            (current_score, swap_coords)
        )  # Store score with original coords for logging

        if current_score > best_score:
            best_score = current_score
            best_swap_normalized = norm_swap  # Store the normalized version

    # Log top N swaps for debugging
    scored_swaps.sort(key=lambda x: x[0], reverse=True)
    log.info("--- Top 5 Scored Swaps ---")
    for i, (s, sw) in enumerate(scored_swaps[:5]):
        is_blacklisted = tuple(sorted(sw)) in blacklist
        bl_marker = " (BLACKLISTED)" if is_blacklisted else ""
        log.info(f" {i+1}. Swap: {sw}, Score: {s:.3f}{bl_marker}")
    log.info("-------------------------")

    if best_swap_normalized:
        log.info(
            f"Selected Best Swap: {best_swap_normalized} with score {best_score:.3f}"
        )
        return best_swap_normalized
    else:
        log.warning(
            "No suitable swap found (all might be blacklisted or score <= -inf)."
        )
        return None
