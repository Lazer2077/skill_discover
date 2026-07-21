"""Adapter necessity audit pipeline schematic for the main paper figure.

Saves paper/figures/audit_pipeline.png.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK = "#111827"
ARR = "#374151"
SLATE_F, SLATE_E = "#f1f5f9", "#64748b"
BLUE_F, BLUE_E = "#dbeafe", "#2563eb"
GREEN_F, GREEN_E = "#dcfce7", "#15803d"
AMBER_F, AMBER_E = "#fef3c7", "#b45309"
ROSE_F, ROSE_E = "#ffe4e6", "#be123c"
VIOLET_F, VIOLET_E = "#ede9fe", "#6d28d9"

plt.rcParams.update({"font.size": 9})
fig, ax = plt.subplots(figsize=(12.8, 5.2))
ax.set_xlim(0, 12.8)
ax.set_ylim(0, 5.2)
ax.axis("off")


def box(x, y, w, h, text, fc, ec, lw=1.25, fs=8.4, bold=False):
    ax.add_patch(
        FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.10",
            fc=fc,
            ec=ec,
            lw=lw,
            zorder=2,
        )
    )
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        color=INK,
        fontsize=fs,
        fontweight="bold" if bold else "normal",
        zorder=3,
        linespacing=1.2,
    )


def arrow(x1, y1, x2, y2, color=ARR, lw=1.45, rad=0.0, ls="-"):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=12,
            lw=lw,
            color=color,
            linestyle=ls,
            connectionstyle=f"arc3,rad={rad}",
            zorder=1,
            shrinkA=2,
            shrinkB=2,
        )
    )


# Title strip
ax.text(
    0.35,
    4.95,
    "Adapter necessity audit",
    ha="left",
    va="center",
    fontsize=11,
    fontweight="bold",
    color=INK,
)
ax.text(
    12.45,
    4.95,
    "existence  ≠  recoverability  ≠  allocation",
    ha="right",
    va="center",
    fontsize=9,
    style="italic",
    color=SLATE_E,
)

# Stage band backgrounds
ax.add_patch(
    FancyBboxPatch(
        (0.25, 3.35),
        12.3,
        1.35,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        fc="#f8fafc",
        ec="#cbd5e1",
        lw=0.9,
        zorder=0,
    )
)
ax.add_patch(
    FancyBboxPatch(
        (0.25, 1.70),
        12.3,
        1.45,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        fc="#f8fafc",
        ec="#cbd5e1",
        lw=0.9,
        zorder=0,
    )
)
ax.add_patch(
    FancyBboxPatch(
        (0.25, 0.18),
        12.3,
        1.35,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        fc="#f8fafc",
        ec="#cbd5e1",
        lw=0.9,
        zorder=0,
    )
)

# Row 1: measure opportunity
ax.text(0.45, 4.50, "1  Measure opportunity", ha="left", va="center", fontsize=9, fontweight="bold", color=SLATE_E)
box(2.35, 3.95, 2.55, 0.78, "paired episodes /\nexact same-state replay", SLATE_F, SLATE_E, fs=8.2)
box(5.35, 3.95, 2.45, 0.78, r"$H_{\mathrm{global}}$" + "\nglobal operating point", BLUE_F, BLUE_E, fs=8.4)
box(8.25, 3.95, 2.45, 0.78, r"$H_{\mathrm{avail}}$" + "\nsame-state oracle", BLUE_F, BLUE_E, fs=8.4, bold=True)
box(11.05, 3.95, 2.15, 0.78, "local headroom?\n(existence)", AMBER_F, AMBER_E, fs=8.2, bold=True)
arrow(3.65, 3.95, 4.10, 3.95)
arrow(6.60, 3.95, 7.00, 3.95)
arrow(9.50, 3.95, 9.95, 3.95)

# Row 2: recoverability
ax.text(0.45, 2.95, "2  Test recoverability", ha="left", va="center", fontsize=9, fontweight="bold", color=SLATE_E)
box(2.35, 2.35, 2.55, 0.88, "cross-seed selector\nfrom $z(x)$ (obs ± model)", SLATE_F, SLATE_E, fs=8.1)
box(5.40, 2.35, 2.55, 0.88, "matched mixture\nsame action frequencies", VIOLET_F, VIOLET_E, fs=8.1)
box(8.35, 2.35, 2.40, 0.88, r"$H_{\mathrm{alloc}}$" + "\nstate-allocation gain", GREEN_F, GREEN_E, fs=8.3, bold=True)
box(11.05, 2.35, 2.15, 0.88, "beats mixture?\n(allocation)", AMBER_F, AMBER_E, fs=8.1, bold=True)
arrow(3.65, 2.35, 4.10, 2.35)
arrow(6.70, 2.35, 7.15, 2.35)
arrow(9.55, 2.35, 9.95, 2.35)
# optional features dashed from identification
arrow(2.35, 3.55, 2.35, 2.80, color=BLUE_E, lw=1.2, ls=(0, (3, 2)))
ax.text(1.15, 3.15, "optional\n$f_\\phi$ features", ha="center", va="center", fontsize=7.2, color=BLUE_E)

# Row 3: decide
ax.text(0.45, 1.35, "3  Decide under uncertainty", ha="left", va="center", fontsize=9, fontweight="bold", color=SLATE_E)
box(2.45, 0.78, 2.70, 0.85, "source-seed cluster\nbootstrap of $H_{\\mathrm{alloc}}$, $q$", SLATE_F, SLATE_E, fs=8.0)
box(5.45, 0.78, 2.40, 0.85, "thresholds\n$\\delta$ (value), $\\kappa$ (viol.)", SLATE_F, SLATE_E, fs=8.0)
box(7.95, 0.78, 1.55, 0.70, "GO", GREEN_F, GREEN_E, fs=9.5, bold=True, lw=1.6)
box(9.55, 0.78, 1.55, 0.70, "NO-GO", ROSE_F, ROSE_E, fs=9.0, bold=True, lw=1.6)
box(11.15, 0.78, 1.70, 0.70, "ABSTAIN", AMBER_F, AMBER_E, fs=8.8, bold=True, lw=1.6)
arrow(3.80, 0.78, 4.25, 0.78)
arrow(6.65, 0.78, 7.15, 0.78)
arrow(8.75, 0.78, 8.75, 0.43, color=GREEN_E, lw=1.0, ls=(0, (2, 2)))
ax.text(
    6.40,
    0.28,
    "controls: observable signal (sensitivity)  ·  hidden signal (specificity)",
    ha="center",
    va="center",
    fontsize=7.6,
    color=SLATE_E,
)

fig.tight_layout(pad=0.25)
out = "paper/figures/audit_pipeline.png"
fig.savefig(out, dpi=220, bbox_inches="tight")
print(f"saved {out}")
