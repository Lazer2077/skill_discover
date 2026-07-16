#!/usr/bin/env python3
"""Assemble the three embodiment renders into one setup/teaser figure."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

PANELS = [("go2", "Go2 (quadruped)"), ("anymal", "ANYmal-D (quadruped)"), ("h1", "H1 (humanoid)")]
R = "paper/figures/renders"


def center_crop(im, ar=1.15):
    w, h = im.size
    tw = min(w, int(h * ar))
    left = (w - tw) // 2
    return im.crop((left, 0, left + tw, h))


fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.0))
for ax, (name, title) in zip(axes, PANELS):
    im = center_crop(Image.open(f"{R}/{name}.png"))
    ax.imshow(im)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
fig.tight_layout(pad=0.4)
out = "paper/figures/setup.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print("saved", out)
