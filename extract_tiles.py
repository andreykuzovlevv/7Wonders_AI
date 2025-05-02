import mss
import pygetwindow as gw
import win32gui
from PIL import Image
from typing import List, Tuple
from config import *


def grab_board_tiles() -> List[Tuple[Tuple[int, int], Image.Image]]:
    screenshot = _take_screenshot()
    grid = _crop_to_grid(screenshot)
    tiles_with_coords = _separate_to_tiles(grid)
    return tiles_with_coords


def _get_window_bbox(title_keyword):
    windows = gw.getWindowsWithTitle(title_keyword)

    if not windows:
        raise Exception("Window not found.")
    win = windows[0]
    hwnd = win._hWnd

    # Get the actual drawable client area (no borders, title bar)
    rect = win32gui.GetClientRect(hwnd)
    left, top = win32gui.ClientToScreen(hwnd, (0, 0))
    width = rect[2]
    height = rect[3]
    return {"left": left, "top": top, "width": width, "height": height}


def _take_screenshot() -> Image.Image:
    # Set your window title keyword
    WINDOW_TITLE = "7 Wonders"

    # Grab bounding box
    bbox = _get_window_bbox(WINDOW_TITLE)

    # Capture with MSS
    with mss.mss() as sct:
        img = sct.grab(bbox)
        screenshot = Image.frombytes("RGB", (img.width, img.height), img.rgb)
        screenshot.save("screenshot.png")
        return screenshot


def _crop_to_grid(screenshot: Image.Image) -> Image.Image:
    """
    Crop the full-window screenshot down to just the grid rectangle.
    """
    return screenshot.crop(
        (GRID_PIXEL_LEFT, GRID_PIXEL_TOP, GRID_PIXEL_RIGHT, GRID_PIXEL_BOTTOM)
    )


def _separate_to_tiles(grid_img: Image.Image):
    """
    Split the cropped grid image into individual tile images.
    Returns a list of ((row, col), tile_image) tuples.
    """
    grid_img.save("screenshot_grid.png")
    tiles = []
    grid_w, grid_h = grid_img.size
    cell_w = grid_w / GRID_COLS
    cell_h = grid_h / GRID_ROWS

    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            left = c * cell_w
            top = r * cell_h
            right = left + cell_w
            bottom = top + cell_h

            tile_img = grid_img.crop((left, top, right, bottom))
            tiles.append(((r, c), tile_img))

    return tiles
