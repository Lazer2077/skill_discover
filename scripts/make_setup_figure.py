#!/usr/bin/env python3
"""Assemble the three embodiment renders into one setup/teaser figure."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

# (render, title, crop left, crop top): a uniform window placed on the robot, which
# does not sit at the centre of frame, so a centre crop clips it.
CROP_W, CROP_H = 800, 600
PANELS = [
    ("go2",    "Go2 (quadruped)",      290, 120),
    ("anymal", "ANYmal-D (quadruped)", 225, 120),
    ("h1",     "H1 (humanoid)",        240,  40),
]
R = "paper/figures/renders"

fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.7))
for ax, (name, title, x, y) in zip(axes, PANELS):
    im = Image.open(f"{R}/{name}.png").crop((x, y, x + CROP_W, y + CROP_H))
    ax.imshow(im)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
fig.tight_layout(pad=0.4)
out = "paper/figures/setup.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print("saved", out)
