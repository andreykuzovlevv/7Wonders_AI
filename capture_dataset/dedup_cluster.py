# dedup_cluster.py
import os, shutil
from glob import glob
from PIL import Image
import imagehash

SRC, OUT = "tiles", "unique"
THRESH = 8

os.makedirs(OUT, exist_ok=True)
clusters = []  # list of [prototype_hash, [paths]]

def hdist(a, b):
    # Option A: use ImageHash subtraction
    return a - b
    # Option B: int-based XOR
    # return (int(a) ^ int(b)).bit_count()

for p in glob(f"{SRC}/*.png"):
    h = imagehash.phash(Image.open(p))
    for proto, paths in clusters:
        if hdist(h, proto) <= THRESH:
            paths.append(p)
            break
    else:
        clusters.append([h, [p]])

for i, (proto, paths) in enumerate(clusters):
    shutil.copy(paths[0], f"{OUT}/cluster_{i}.png")

print("wrote", len(clusters), "unique sprites to", OUT)
print("Now run label_gui.py to sort these into class folders.")
