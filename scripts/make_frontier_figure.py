#!/usr/bin/env python3
"""COT vs completion-time frontier figure for the VGCC paper (E1, locked seeds 797-799)."""
import json
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FILES = [f"outputs/e1_isotime/frontier_seed{s}.json" for s in (794, 795, 796, 797, 798, 799)]

by = defaultdict(list)
for f in FILES:
    d = json.load(open(f))
    for tk, ms in d["records"].items():
        for m, rs in ms.items():
            by[m].extend(rs)


def g(m, k):
    return float(np.nanmean([r.get(k, np.nan) for r in by[m]]))


# method -> (label, color, marker, (dx,dy) text offset in points)
STYLE = {
    "direct_target_command": ("direct", "#444444", "o", (10, -4)),
    "scaled_target_command_75": ("fixed scale 0.75", "#1b9e77", "s", (10, -4)),
    "scaled_target_command_90": ("fixed scale 0.90", "#66c2a5", "s", (8, -26)),
    "reactive_governor_command": ("reactive governor", "#e6ab02", "^", (-8, 8)),
    "model_feedforward_command": ("VGCC (proxy ≈ power)", "#d95f02", "D", (10, 10)),
    "mechanical_feedforward_command": (None, "#7570b3", "D", None),
}

fig, ax = plt.subplots(figsize=(6.4, 4.4))

# fixed-scaling frontier line (direct -> 0.90 -> 0.75), sorted by time
scale_pts = sorted(
    (g(m, "elapsed_time_s"), g(m, "cost_of_transport"))
    for m in ["direct_target_command", "scaled_target_command_90", "scaled_target_command_75"]
)
ax.plot([p[0] for p in scale_pts], [p[1] for p in scale_pts], "--",
        color="#1b9e77", lw=1.4, alpha=0.75, zorder=1, label="fixed-scaling frontier")

for m, (lab, col, mk, off) in STYLE.items():
    t, cot, succ = g(m, "elapsed_time_s"), g(m, "cost_of_transport"), g(m, "success")
    ax.scatter([t], [cot], s=160, c=col, marker=mk, edgecolors="black",
               linewidths=0.9, zorder=3)
    if lab is not None:
        ax.annotate(f"{lab}\n(succ {succ:.2f})", (t, cot),
                    textcoords="offset points", xytext=off, fontsize=8.2, color="black")

ax.set_xlabel("completion time (s)  —  lower is better →", fontsize=10)
ax.set_ylabel("cost of transport  —  lower is better ↓", fontsize=10)
ax.set_title("Cost of transport vs completion time\n(Go2 harder targets, 6 locked seeds, $n{=}96$/method)",
             fontsize=9.5)
ax.grid(True, alpha=0.25)
ax.margins(x=0.16, y=0.16)
ax.legend(loc="lower left", fontsize=8.2, framealpha=0.9)
fig.tight_layout()
out = "paper/figures/frontier.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print("saved", out)
