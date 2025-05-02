# config.py
# ───── CONFIGURATION ─────

GRID_ROWS = 10
GRID_COLS = 10

TILE_SIZE = 50

GRID_PIXEL_LEFT = 220
GRID_PIXEL_TOP = 20
GRID_PIXEL_RIGHT = GRID_PIXEL_LEFT + TILE_SIZE * 10
GRID_PIXEL_BOTTOM = GRID_PIXEL_TOP + 493

# ── label sets ──────────────────────────────────────────────────────
CONTENT_CLASSES = (
    ["empty"] +                        # 0
    [f"gem_{i}" for i in range(8)] +   # 1-8
    ["block"] +                        # 9
    ["bonus_0","bonus_1","bonus_2"]    # 10-12
)                                      # total 13
BACKGROUND_CLASSES = ["none","stone","shield"]   # 0-2

MAP_FG  = {c:i for i,c in enumerate(CONTENT_CLASSES)}
MAP_BG  = {c:i for i,c in enumerate(BACKGROUND_CLASSES)}