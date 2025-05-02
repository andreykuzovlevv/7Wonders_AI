# capture_tiles.py
import os, time
from extract_tiles import grab_board_tiles

os.makedirs("tiles", exist_ok=True)

def main():
    end_time = time.time() + 30*60          # 30 minutes
    while time.time() < end_time:
        tiles = grab_board_tiles()
        ts = time.time()
        for (r,c), img in tiles:
            img.save(f"tiles/tile_{r}_{c}_{ts:.0f}.png")
        time.sleep(10)                      # every 10 s

if __name__ == "__main__":
    main()