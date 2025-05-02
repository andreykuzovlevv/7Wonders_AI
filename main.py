from __future__ import annotations

import hashlib
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import pyautogui
import torch
from PIL import Image
import pygetwindow as gw
import win32gui  # Required for accurate window client area coordinates

import config
from extract_tiles import (
    grab_board_tiles,
)  # Assuming this still works correctly for grabbing tiles relative to window
from tile_classifier import TileClassifier, classify_tile

# ---------------------------------------------------------------------------
# Logging & global constants
# ---------------------------------------------------------------------------
DEBUG_MODE = True
POST_FIRST_CLICK_DELAY = 0.4  # seconds between first and second click
POST_SWAP_DELAY = 0.6  # wait after second click for animation start
BOARD_CHANGE_TIMEOUT = 3.5  # seconds to wait for the board to differ
POLL_INTERVAL = 0.25
NO_MOVE_DELAY = 1.0
MODEL_PATH = "tile_classifier_model.pth"
WINDOW_TITLE_KEYWORD = "7 Wonders"  # Make sure this matches your game window title

# Simplified logging format
log_format = "%(message)s"
logging.basicConfig(
    level=logging.DEBUG if DEBUG_MODE else logging.INFO,
    format=log_format,
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("wonders_swap")
pyautogui.FAILSAFE = False  # Keep false for bot operation

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"Using device: {device}")

if not Path(MODEL_PATH).exists():
    log.error(f"Model file missing at {MODEL_PATH}")
    sys.exit(1)

model = TileClassifier(len(config.CONTENT_CLASSES), len(config.BACKGROUND_CLASSES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()
log.info(f"Classifier loaded from {MODEL_PATH}")

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
GridCoord = Tuple[int, int]
Swap = Tuple[GridCoord, GridCoord]
WindowOrigin = Tuple[int, int]  # (left, top) screen coordinates of window client area


def get_window_origin() -> Optional[WindowOrigin]:
    """Gets the screen coordinates of the top-left corner of the game window's client area."""
    try:
        windows = gw.getWindowsWithTitle(WINDOW_TITLE_KEYWORD)
        if not windows:
            log.error(
                f"Window with title containing '{WINDOW_TITLE_KEYWORD}' not found."
            )
            return None
        win = windows[0]
        hwnd = win._hWnd
        # Get the client area's top-left corner relative to the screen
        left, top = win32gui.ClientToScreen(hwnd, (0, 0))
        # Optional: Check if window is minimized
        if win.isMinimized:
            log.warning(f"Game window '{WINDOW_TITLE_KEYWORD}' is minimized.")
            return None
        # Optional: Check if window is active (might be useful)
        # if not win.isActive:
        #     log.warning(f"Game window '{WINDOW_TITLE_KEYWORD}' is not active.")
        #     # Consider activating: win.activate()
        return (left, top)
    except Exception as e:
        log.error(f"Error getting window coordinates: {e}")
        return None


def get_pixel_coords(r: int, c: int, window_origin: WindowOrigin) -> Tuple[int, int]:
    """Calculates the absolute screen pixel coordinates for the center of a grid cell."""
    window_left, window_top = window_origin

    # Calculate cell dimensions based on config values (relative to window client area)
    grid_pixel_width = config.GRID_PIXEL_RIGHT - config.GRID_PIXEL_LEFT
    grid_pixel_height = config.GRID_PIXEL_BOTTOM - config.GRID_PIXEL_TOP
    cell_w = grid_pixel_width / config.GRID_COLS
    cell_h = grid_pixel_height / config.GRID_ROWS

    # Calculate offset *within the grid* (relative to GRID_PIXEL_LEFT/TOP)
    offset_x = (c + 0.5) * cell_w
    offset_y = (r + 0.5) * cell_h

    # Calculate absolute screen coordinates
    # Start from window origin, add grid offset within window, add cell offset within grid
    screen_x = window_left + config.GRID_PIXEL_LEFT + offset_x
    screen_y = window_top + config.GRID_PIXEL_TOP + offset_y

    return int(screen_x), int(screen_y)


# ---------- board capture & representation ---------- #


def classify_board_tiles(
    tiles: List[Tuple[GridCoord, Image.Image]],
) -> Dict[GridCoord, Tuple[str, str]]:
    out: Dict[GridCoord, Tuple[str, str]] = {}
    for (r, c), img in tiles:
        try:
            content, background, *_ = classify_tile(model, img, device)
        except Exception as exc:
            log.warning(f"Classification failed at ({r},{c}): {exc}")
            content, background = "empty", "none"
        out[(r, c)] = (content, background)
    return out


def build_states(
    classif: Dict[GridCoord, Tuple[str, str]],
) -> Tuple[List[List[str]], List[List[str]]]:
    rows, cols = config.GRID_ROWS, config.GRID_COLS
    content = [["empty"] * cols for _ in range(rows)]
    background = [["none"] * cols for _ in range(rows)]
    for (r, c), (cont, back) in classif.items():
        # Ensure coordinates are within bounds (safety check)
        if 0 <= r < rows and 0 <= c < cols:
            content[r][c] = cont
            background[r][c] = back
        else:
            log.warning(
                f"Out-of-bounds coordinate ({r},{c}) in classification data ignored."
            )
    return content, background


# ---------- swap logic ---------- #


def board_hash(content: List[List[str]], background: List[List[str]]) -> str:
    s = (
        "|".join([",".join(row) for row in content])
        + "||"
        + "|".join([",".join(row) for row in background])
    )
    return hashlib.md5(s.encode()).hexdigest()


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


def simulate_swap(grid: List[List[str]], a: GridCoord, b: GridCoord) -> List[List[str]]:
    new_grid = [row[:] for row in grid]
    (ra, ca), (rb, cb) = a, b
    new_grid[ra][ca], new_grid[rb][cb] = new_grid[rb][cb], new_grid[ra][ca]
    return new_grid


def find_valid_swaps(
    content: List[List[str]],
) -> List[Tuple[Swap, List[Set[GridCoord]]]]:
    rows, cols = config.GRID_ROWS, config.GRID_COLS
    valid: List[Tuple[Swap, List[Set[GridCoord]]]] = []
    # Define non-swappable types explicitly
    non_swappable = {
        "empty",
        "block",
    }  # Add any other types like 'fragment' if they cannot be swapped
    for r in range(rows):
        for c in range(cols):
            if content[r][c] in non_swappable:
                continue
            # Check potential swaps right and down
            for dr, dc in ((0, 1), (1, 0)):
                nr, nc = r + dr, c + dc
                # Check bounds and if the neighbor is swappable
                if nr >= rows or nc >= cols or content[nr][nc] in non_swappable:
                    continue
                # Don't check swaps twice
                if (dr, dc) == (
                    1,
                    0,
                ):  # Only check down swap if not checked as right swap from above
                    pass  # Already checked (r-1, c) <-> (r, c)

                swapped_grid = simulate_swap(content, (r, c), (nr, nc))
                matches = get_matches(swapped_grid)
                if matches:
                    valid.append((((r, c), (nr, nc)), matches))
    return valid


def score_swap(
    swap: Swap, matches: List[Set[GridCoord]], background: List[List[str]]
) -> float:
    score = 0.0
    total_matched_tiles = 0
    background_cleared = 0
    shield_cleared = 0

    # Calculate score based on resulting matches
    for match_set in matches:
        total_matched_tiles += len(match_set)
        for r, c in match_set:
            # Check bounds just in case
            if 0 <= r < config.GRID_ROWS and 0 <= c < config.GRID_COLS:
                bg_type = background[r][c]
                if bg_type == "shield":
                    shield_cleared += 1
                elif bg_type == "stone":
                    background_cleared += 1

    # Scoring weights (tune these as needed)
    score += total_matched_tiles  # Base score for number of tiles matched
    score += background_cleared * 2  # Bonus for clearing stone
    score += shield_cleared * 4  # Higher bonus for clearing shield

    # --- Add future scoring logic here ---
    # e.g., bonus for matching near fragments, penalty for certain moves?
    # e.g., check if swap involves a 'bonus' tile type?

    return score


# ---------- clicking ---------- #


def click_coord(coord: GridCoord, window_origin: WindowOrigin):
    """Clicks the calculated screen coordinates for a grid cell."""
    x, y = get_pixel_coords(coord[0], coord[1], window_origin)
    pyautogui.moveTo(x, y, duration=0.1)
    time.sleep(0.1)
    log.debug(f"Clicking grid {coord} at screen ({x}, {y})")
    pyautogui.click(x, y)


def perform_swap(
    a: GridCoord, b: GridCoord, window_origin: WindowOrigin, content: List[List[str]]
):
    """Performs the two clicks for a swap, logging the tile types."""
    (r1, c1) = a
    (r2, c2) = b
    type1 = (
        content[r1][c1]
        if 0 <= r1 < config.GRID_ROWS and 0 <= c1 < config.GRID_COLS
        else "OOB"
    )
    type2 = (
        content[r2][c2]
        if 0 <= r2 < config.GRID_ROWS and 0 <= c2 < config.GRID_COLS
        else "OOB"
    )

    log.info(f"Clicking {a} ({type1}) then {b} ({type2})")
    click_coord(a, window_origin)
    time.sleep(POST_FIRST_CLICK_DELAY)
    click_coord(b, window_origin)
    time.sleep(POST_SWAP_DELAY)


def wait_for_change(prev_hash: str, window_origin: WindowOrigin) -> str:
    """Waits until the board state (hash) changes or timeout occurs."""
    end = time.time() + BOARD_CHANGE_TIMEOUT
    while time.time() < end:
        # We need the window origin to potentially capture the board correctly
        # Assuming grab_board_tiles uses the origin implicitly or is relative
        tiles = (
            grab_board_tiles()
        )  # This function needs to work correctly relative to the window origin
        if not tiles:
            log.warning("Board capture failed during wait.")
            time.sleep(POLL_INTERVAL)
            continue  # Maybe retry capture?

        classif = classify_board_tiles(tiles)
        content, background = build_states(classif)
        h = board_hash(content, background)
        if h != prev_hash:
            log.debug("Board change detected.")
            return h  # Changed
        time.sleep(POLL_INTERVAL)
    log.debug("Timeout waiting for board change.")
    return prev_hash  # Return previous hash if no change


# ---------- level completion ---------- #


def level_complete(content: List[List[str]], background: List[List[str]]) -> bool:
    """Checks if all background tiles are cleared. Assumes fragments drop automatically."""
    # Check if any 'block' tiles remain (assuming these represent fragments or objectives)
    # Or check if any breakable background tiles remain
    for r in range(config.GRID_ROWS):
        for c in range(config.GRID_COLS):
            # If fragments are represented by a 'block' type that needs to reach bottom:
            # You'd need a more complex check, e.g., ensure no 'block' tiles exist above row config.GRID_ROWS - 1
            # Simplified check: Are all breakable backgrounds gone?
            if background[r][c] in {"stone", "shield"}:
                return False
    # Add check for wonder fragments if they are distinct tile types that must reach bottom
    has_fragments = any(
        "block" in row for row in content
    )  # Example if 'block' represents fragments
    if has_fragments:
        # Check if all 'block' tiles are at the bottom row or gone
        for r in range(config.GRID_ROWS - 1):  # Check all rows except the last
            for c in range(config.GRID_COLS):
                if content[r][c] == "block":
                    return False  # Fragment found not at the bottom

    # If we passed all checks, level might be complete
    # Note: This might need adjustment based on exact win conditions (e.g., specific fragment collection)
    # For now, assume clearing background is sufficient if no 'block' tiles are stuck.
    log.info("Checking level completion: No stone/shield background found.")
    return True  # Simplified: No more breakable background tiles


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def main():
    log.info("--- Starting swap bot ---")
    log.info(f"Target window: '{WINDOW_TITLE_KEYWORD}'")
    log.info("Ensure game window is visible and not obstructed.")
    log.info("Press Ctrl+C in the console to stop.")
    time.sleep(2)  # Give user time to switch focus
    blacklist: Set[Swap] = set()

    while True:
        try:
            # --- Get Window Position ---
            window_origin = get_window_origin()
            if window_origin is None:
                log.error("Game window not found or inaccessible. Waiting...")
                time.sleep(5)
                continue

            # --- Capture and Classify Board ---
            log.debug("Capturing board...")
            tiles = grab_board_tiles()
            if not tiles:
                log.warning("Failed to capture board tiles. Retrying...")
                time.sleep(NO_MOVE_DELAY)
                continue

            log.debug("Classifying tiles...")
            classif = classify_board_tiles(tiles)
            content, background = build_states(classif)
            bhash = board_hash(content, background)
            log.debug(f"Current board hash: {bhash}")

            # --- Check Level Completion ---
            if level_complete(content, background):
                log.info("Level complete criteria met. Waiting...")
                blacklist.clear()  # Reset blacklist for next level
                time.sleep(5)  # Wait before checking again
                continue

            # --- Find and Score Swaps ---
            swaps_info = find_valid_swaps(content)
            log.debug(f"{len(swaps_info)} valid swaps found.")

            if not swaps_info:
                log.info("No valid swaps found. Waiting...")
                blacklist.clear()  # Clear blacklist if stuck with no moves
                time.sleep(NO_MOVE_DELAY)
                continue

            # --- Choose Best Swap (excluding blacklisted) ---
            best_swap, best_score = None, -float("inf")
            scored_swaps = []
            for swap, matches in swaps_info:
                # Normalize swap tuple to ensure ((r1,c1),(r2,c2)) where r1<=r2 or (r1==r2 and c1<c2)
                # This makes blacklist checking consistent.
                coord1, coord2 = swap
                normalized_swap = tuple(sorted(swap))

                if normalized_swap in blacklist:
                    log.debug(f"Skipping blacklisted swap: {normalized_swap}")
                    continue

                s = score_swap(swap, matches, background)
                scored_swaps.append(
                    (s, normalized_swap)
                )  # Store score and normalized swap
                if s > best_score:
                    best_score, best_swap = s, normalized_swap  # Use normalized swap

            if not best_swap:
                if blacklist:
                    log.info(
                        "All valid swaps are blacklisted. Clearing blacklist and retrying."
                    )
                    blacklist.clear()
                else:
                    log.info(
                        "No scoreable swaps found (maybe all had score <= -inf?). Waiting..."
                    )
                time.sleep(NO_MOVE_DELAY)
                continue

            # --- Perform Swap ---
            (coord1, coord2) = best_swap  # Use the chosen normalized swap
            log.debug(f"Best swap: {coord1} <-> {coord2} (Score: {best_score:.2f})")
            perform_swap(coord1, coord2, window_origin, content)  # Pass content grid

            # --- Wait for Board Change ---
            log.debug("Waiting for board change...")
            new_hash = wait_for_change(bhash, window_origin)

            if new_hash == bhash:
                log.info(f"Board unchanged after swap {best_swap}. Blacklisting.")
                blacklist.add(best_swap)  # Add the normalized swap to blacklist
            else:
                log.debug(f"Board changed. New hash: {new_hash}")
                if blacklist:
                    log.debug("Clearing blacklist after successful move.")
                    blacklist.clear()  # Clear blacklist on successful move

        except KeyboardInterrupt:
            log.info("... Stopping bot (Ctrl+C received).")
            break
        except Exception as exc:
            log.exception(f"An unexpected error occurred: {exc}")
            log.info("Attempting to recover in 3 seconds...")
            time.sleep(3)


if __name__ == "__main__":
    main()
