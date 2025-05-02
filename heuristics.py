# heuristics_plus.py  ── paste or import in your bot ──────────────────────────
import random, math, copy, time, logging
import config  # ← uses your existing constants
from typing import List, Tuple, Set

GridCoord = Tuple[int, int]
Swap = Tuple[GridCoord, GridCoord]
log = logging.getLogger(__name__)

# ────────────────────────── 1. GLOBAL WEIGHTS  ───────────────────────────────
W = dict(
    tiles_base=1.0,  # per tile in *any* match
    match4_bonus=2.0,  # extra for every 4‑tile arm
    match5_bonus=5.0,  # extra for every 5+ tile arm / T / L
    stone=2.5,  # per stone slab broken
    shield=5.0,  # per shield broken
    block=10.0,  # per block (fragment blocker) removed
    near_block_adj=0.8,  # each matched tile orthogonal to a block
    bottom_row_bonus=0.5,  # extra per row counted from bottom (row 9 → 0, row 0 → −9/2)
    edge_bonus=0.3,  # reward for clearing near L/R edges or voids
    empty_above_penalty=0.4,  # discourage using an already‑open column (keeps play low)
    bonus0_clear=0.9,  # weight per background cell cleared by bonus_0
    bonus1_clear=1.1,  # weight per background cell cleared by bonus_1
    bonus2_estimated=18.0,  # average background clears expected from bonus_2
    chain_depth_decay=0.55,  # multiplier for 2nd cascade, ^2 for 3rd, …
)


# ────────────────────────── 2. CORE UTILITIES  ───────────────────────────────
def simulate_swap(grid: List[List[str]], a: GridCoord, b: GridCoord) -> List[List[str]]:
    new_grid = [row[:] for row in grid]
    (ra, ca), (rb, cb) = a, b
    new_grid[ra][ca], new_grid[rb][cb] = new_grid[rb][cb], new_grid[ra][ca]
    return new_grid


def get_matches(grid: List[List[str]]) -> List[Set[GridCoord]]:
    """Return list of all groups of 3+ identical tiles (after a hypothetical swap)."""
    rows, cols = config.GRID_ROWS, config.GRID_COLS
    matches: List[Set[GridCoord]] = []
    # Horizontal
    for r in range(rows):
        c = 0
        while c < cols - 2:
            if (
                grid[r][c] == "empty" or grid[r][c] == "block"
            ):  # Don't match empty or blocks
                c += 1
                continue
            run_val = grid[r][c]
            run_end = c + 1
            while run_end < cols and grid[r][run_end] == run_val:
                run_end += 1
            if run_end - c >= 3:
                matches.append({(r, x) for x in range(c, run_end)})
            c = run_end
    # Vertical
    for c in range(cols):
        r = 0
        while r < rows - 2:
            if (
                grid[r][c] == "empty" or grid[r][c] == "block"
            ):  # Don't match empty or blocks
                r += 1
                continue
            run_val = grid[r][c]
            run_end = r + 1
            while run_end < rows and grid[run_end][c] == run_val:
                run_end += 1
            if run_end - r >= 3:
                matches.append({(x, c) for x in range(r, run_end)})
            r = run_end
    return matches


def apply_gravity(content):
    """Drop tiles after removals (simple vertical gravity)."""
    rows, cols = config.GRID_ROWS, config.GRID_COLS
    for c in range(cols):
        write_row = rows - 1
        for read_row in range(rows - 1, -1, -1):
            if content[read_row][c] != "empty":
                content[write_row][c] = content[read_row][c]
                if write_row != read_row:
                    content[read_row][c] = "empty"
                write_row -= 1
        for r in range(write_row, -1, -1):
            content[r][c] = "empty"
    return content


# ───────────────────── 3. BONUS‑EFFECT SIMULATION  ───────────────────────────
def bonus_footprint(tile_type: str, r: int, c: int) -> Set[GridCoord]:
    """Return the set of coordinates a bonus tile would destroy – BOARD SIZE SAFE."""
    rows, cols = config.GRID_ROWS, config.GRID_COLS
    cells = set()
    if tile_type == "bonus_0":  # horizontal line
        cells.update((r, col) for col in range(cols))
    elif tile_type == "bonus_1":  # cross
        cells.update((r, col) for col in range(cols))
        cells.update((row, c) for row in range(rows))
    elif tile_type == "bonus_2":  # random 15‑20 ⇒ treat as expected value
        # we will not enumerate – scoring will use constant expectation
        pass
    return cells


def estimate_bonus_score(
    tile_type: str, r: int, c: int, background: List[List[str]]
) -> float:
    """Rough heuristic gain from detonating a bonus tile immediately."""
    if tile_type == "bonus_2":
        # assume ~18 tiles, half on background, ¼ stone, ¼ shield…
        exp_bg = W["bonus2_estimated"]
        exp_stn = exp_bg * 0.5
        exp_shld = exp_bg * 0.25
        return exp_bg * W["tiles_base"] + exp_stn * W["stone"] + exp_shld * W["shield"]
    else:
        score = 0.0
        for rr, cc in bonus_footprint(tile_type, r, c):
            bg = background[rr][cc]
            score += W["tiles_base"]
            if bg == "stone":
                score += W["stone"]
            elif bg == "shield":
                score += W["shield"]
        return score


# ─────────────────────────── 4. ADVANCED SCORE  ──────────────────────────────
def score_swap(
    swap: Swap,
    matches: List[Set[GridCoord]],
    content: List[List[str]],
    background: List[List[str]],
) -> float:
    """One‑ply + cascade estimate with extensive heuristics."""
    rows, cols = config.GRID_ROWS, config.GRID_COLS
    score = 0.0

    # ----- A. Immediate match quality -----
    for m in matches:
        size = len(m)
        score += size * W["tiles_base"]
        if size == 4:
            score += W["match4_bonus"]
        elif size >= 5:
            score += W["match5_bonus"]

        for r, c in m:
            bg = background[r][c]
            if bg == "stone":
                score += W["stone"]
            elif bg == "shield":
                score += W["shield"]

            # near‑block adjacency reward
            for dr, dc in ((0, 1), (1, 0), (-1, 0), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and content[nr][nc] == "block":
                    score += W["near_block_adj"]

            # positional weights
            score += (rows - 1 - r) * W["bottom_row_bonus"]
            if c in (0, cols - 1):
                score += W["edge_bonus"]
            if background[r][c] == "none":
                score -= W["empty_above_penalty"]

    # ----- B. Bonus tiles triggered directly by the swap -----
    for pos in swap:
        r, c = pos
        tile_type = content[r][c]
        if tile_type.startswith("bonus_"):
            score += estimate_bonus_score(tile_type, r, c, background)

    # ----- C. Look‑ahead: first cascade (very fast) -----
    # We copy‑simulate one step with gravity to catch obvious chains.
    grid_after = copy.deepcopy(content)
    for m in matches:
        for r, c in m:
            grid_after[r][c] = "empty"
    apply_gravity(grid_after)
    cascade = get_matches(grid_after)
    cascade_gain = 0.0
    if cascade:
        for chain in cascade:
            mult = W["chain_depth_decay"]  # depth‑1 cascade weight
            cascade_gain += mult * len(chain) * W["tiles_base"]
    score += cascade_gain

    # ----- D. Block clearing weight -----
    # (Block tiles are unswappable. Count how many get removed by this swap.)
    for m in matches:
        for r, c in m:
            if content[r][c] == "block":
                score += W["block"]

    return score


# ────────────────────────── 5. VALID SWAPS EXTENDED  ─────────────────────────
def find_valid_swaps(
    content: List[List[str]],
    background: List[List[str]],
) -> List[Tuple[Swap, List[Set[GridCoord]]]]:
    """All swaps that either (a) make a normal match or (b) trigger a bonus."""
    rows, cols = config.GRID_ROWS, config.GRID_COLS
    valid = []
    non_swappable = {"empty"}  # blocks AREN’T swappable, but can be removed
    for r in range(rows):
        for c in range(cols):
            if content[r][c] in non_swappable:
                continue
            for dr, dc in ((0, 1), (1, 0)):  # right & down neighbours
                nr, nc = r + dr, c + dc
                if nr >= rows or nc >= cols or content[nr][nc] in non_swappable:
                    continue
                swap = ((r, c), (nr, nc))
                swapped = simulate_swap(content, *swap)
                matches = get_matches(swapped)
                # Accept if match OR either end is a bonus
                if (
                    matches
                    or content[r][c].startswith("bonus_")
                    or content[nr][nc].startswith("bonus_")
                ):
                    valid.append((swap, matches))
    return valid


# ────────────────────────── 6. CHOICE WRAPPER  ───────────────────────────────
def choose_best_swap(content, background, blacklist=set()) -> Swap | None:
    swaps_info = find_valid_swaps(content, background)
    best_swap, best_score = None, -math.inf
    for swap, matches in swaps_info:
        norm_swap = tuple(sorted(swap))
        if norm_swap in blacklist:
            continue
        s = score_swap(swap, matches, content, background)
        if s > best_score:
            best_score, best_swap = s, norm_swap
    return best_swap
