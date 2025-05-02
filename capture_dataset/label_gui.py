# label_gui.py

import os, shutil, glob, tkinter as tk
from pathlib import Path
from PIL import Image, ImageTk
import config

SRC = "unique"
DST = "unique_labeled"
ICONS = "elements"

# ── just grab all sprites fresh every time ───────────────────────────
sprites = sorted(glob.glob(f"{SRC}/*.png"))
if not sprites:
    raise SystemExit("No sprites in 'unique/'. Run dedup first.")

# ── make all folders up front ────────────────────────────────────────
for fg in config.CONTENT_CLASSES:
    for bg in config.BACKGROUND_CLASSES:
        os.makedirs(f"{DST}/{fg}__{bg}", exist_ok=True)

root = tk.Tk()
root.title("Sprite Labeller")
root.configure(padx=12, pady=12)

# ── preview panel ────────────────────────────────────────────────────
cur_idx = 0
preview = tk.Label(root)
preview.grid(row=0, column=0, columnspan=8, pady=(0, 12))


def show():
    img = Image.open(sprites[cur_idx]).resize((128, 128), Image.NEAREST)
    preview.imgtk = ImageTk.PhotoImage(img)
    preview.config(image=preview.imgtk)


# ── load icons ───────────────────────────────────────────────────────
def load_icon(name):
    f = next(Path(ICONS).glob(f"{name}.*"), None)
    if not f:
        return ImageTk.PhotoImage(Image.new("RGB", (48, 48), "gray"))
    return ImageTk.PhotoImage(Image.open(f).resize((48, 48), Image.NEAREST))


icons = {c: load_icon(c) for c in config.CONTENT_CLASSES + config.BACKGROUND_CLASSES}


# ── special "empty" handler ──────────────────────────────────────────
def label_empty():
    global cur_idx
    dest = f"{DST}/empty__none"
    shutil.move(sprites[cur_idx], f"{dest}/{Path(sprites[cur_idx]).name}")
    cur_idx += 1
    if cur_idx >= len(sprites):
        root.quit()
    else:
        show()


tk.Button(
    root,
    image=icons["empty"],
    text="empty",
    compound="top",
    command=label_empty,
    width=96,
    height=96,
).grid(row=1, column=0, padx=4, pady=(0, 12))

# ── regular class selectors ──────────────────────────────────────────
sel_fg = tk.StringVar(value=config.CONTENT_CLASSES[1])
sel_bg = tk.StringVar(value=config.BACKGROUND_CLASSES[0])

# foreground (excluding empty)
fg_frame = tk.Frame(root)
fg_frame.grid(row=2, column=0, columnspan=8)
for i, fg in enumerate(config.CONTENT_CLASSES[1:]):
    tk.Radiobutton(
        fg_frame,
        image=icons[fg],
        text=fg,
        compound="top",
        value=fg,
        variable=sel_fg,
        indicatoron=False,
        width=96,
        height=96,
    ).grid(row=0, column=i, padx=4, pady=4)

# background
bg_frame = tk.Frame(root)
bg_frame.grid(row=3, column=0, columnspan=8)
for i, bg in enumerate(config.BACKGROUND_CLASSES):
    tk.Radiobutton(
        bg_frame,
        image=icons[bg],
        text=bg,
        compound="top",
        value=bg,
        variable=sel_bg,
        indicatoron=False,
        width=96,
        height=96,
    ).grid(row=0, column=i, padx=4, pady=4)


# commit
def commit(_=None):
    global cur_idx
    dst = f"{DST}/{sel_fg.get()}__{sel_bg.get()}"
    shutil.move(sprites[cur_idx], f"{dst}/{Path(sprites[cur_idx]).name}")
    cur_idx += 1
    if cur_idx >= len(sprites):
        root.quit()
    else:
        show()


# save/next button
tk.Button(
    root,
    text="Save / Next (Enter)",
    command=commit,
    font=("Segoe UI", 12, "bold"),
    width=20,
    height=2,
).grid(row=4, column=0, columnspan=4, pady=10)

root.bind("<Return>", commit)
root.bind("<space>", commit)

show()
root.mainloop()
