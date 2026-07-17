"""VGCC method schematic: offline identification + online viability-gated compensation.

Regenerates paper/figures/vgfc_method.png to match the evaluated controller
(structured candidate set, raw-cost argmin behind the gate, bounded annealed
correction).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

GRAY_F, GRAY_E = "#e5e7eb", "#6b7280"
BLUE_F, BLUE_E = "#dbeafe", "#2563eb"
MODEL_F        = "#bfdbfe"
GATE_E         = "#b45309"
IDF, IDE       = "#f1f5f9", "#94a3b8"
INK, ARR       = "#111827", "#374151"

plt.rcParams.update({"font.size": 9})
fig, ax = plt.subplots(figsize=(13.0, 4.1))
ax.set_xlim(0, 13.0); ax.set_ylim(0, 4.1); ax.axis("off")

def box(x, y, w, h, text, fc, ec, lw=1.3, fs=9, bold=False, dash="solid"):
    ax.add_patch(FancyBboxPatch((x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.10", fc=fc, ec=ec, lw=lw,
        linestyle=dash, zorder=2))
    ax.text(x, y, text, ha="center", va="center", color=INK, fontsize=fs,
            fontweight="bold" if bold else "normal", zorder=3, linespacing=1.25)

def arrow(x1, y1, x2, y2, ls="-", color=ARR, lw=1.5, rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
        mutation_scale=13, lw=lw, color=color, linestyle=ls,
        connectionstyle=f"arc3,rad={rad}", zorder=1, shrinkA=2, shrinkB=2))

# ---------------- Identification band (top, offline) ----------------
ax.add_patch(FancyBboxPatch((0.25, 2.72), 12.5, 1.18, boxstyle="round,pad=0.02,rounding_size=0.12",
             fc=IDF, ec=IDE, lw=1.0, linestyle=(0, (4, 2)), zorder=0))
ax.text(0.55, 3.72, "Identification  (once, offline)", ha="left", va="center",
        fontsize=9.5, style="italic", color="#475569", fontweight="bold")
yid = 3.22
box(2.15, yid, 2.6, 0.62, "excite frozen plant\nwith random commands", BLUE_F, BLUE_E, fs=8.5)
box(5.75, yid, 2.7, 0.66, "frozen policy $\\pi_\\theta$ + robot\n(closed-loop plant)", GRAY_F, GRAY_E, fs=8.7)
box(8.95, yid, 2.3, 0.62, "regress short-horizon\nresponse", IDF, IDE, fs=8.5)
box(11.55, yid, 1.6, 0.66, "model $f_\\phi$", MODEL_F, BLUE_E, fs=9.6, bold=True)
arrow(3.45, yid, 4.40, yid)
arrow(7.10, yid, 7.80, yid)
arrow(10.10, yid, 10.75, yid)

# ---------------- Deployment band (bottom, per step) ----------------
y = 1.30
ax.text(0.55, 2.26, "Deployment  (per high-level step)", ha="left", va="center",
        fontsize=9.5, style="italic", color="#475569", fontweight="bold")
box(1.55, y, 2.5, 0.96, "task target $g$\n$\\Rightarrow$ direct command\n$c_{\\mathrm{goal}}$ (proportional)", "#ffffff", GRAY_E, fs=8.6)
box(4.20, y, 2.1, 0.96, "candidate set $\\mathcal{C}$:\nscalings, headings,\nyaw variants ($\\approx$70)", BLUE_F, BLUE_E, fs=8.4)
box(6.95, y, 2.6, 0.96, "$f_\\phi$: predict per\ncandidate: motion,\ncost, posture", MODEL_F, BLUE_E, fs=8.4)
box(10.00, y, 2.85, 0.96, "gate: restore $\\triangleright$ rescue $\\triangleright$\ncompensate ($\\beta$, floors, $\\epsilon$) $\\triangleright$ fallback\n$c = c_{\\mathrm{goal}} + \\alpha_t(c^\\ast - c_{\\mathrm{goal}})$", BLUE_F, GATE_E, lw=1.8, fs=7.9)
box(12.35, y, 1.3, 0.96, "frozen\n$\\pi_\\theta$ +\nrobot", GRAY_F, GRAY_E, fs=8.6)
arrow(2.80, y, 3.15, y)
arrow(5.25, y, 5.65, y)
arrow(8.25, y, 8.55, y)
arrow(11.45, y, 11.68, y)

# identified model f_phi supplies the deployment prediction block
arrow(11.35, 2.87, 7.35, y + 0.50, ls=(0, (4, 2)), color=BLUE_E, lw=1.5, rad=0.22)
ax.text(9.55, 2.30, "identified model $f_\\phi$", ha="center", va="center",
        fontsize=8.0, color=BLUE_E)

# observation feedback: robot -> prediction block and -> direct command
arrow(12.35, y - 0.50, 6.95, y - 0.62, ls=(0, (2, 2)), color=GRAY_E, lw=1.3, rad=0.28)
arrow(6.60, y - 0.62, 1.55, y - 0.50, ls=(0, (2, 2)), color=GRAY_E, lw=1.3, rad=0.28)
ax.text(6.75, 0.22, "observation $o_t$  (situational conditioning; body-frame target)",
        ha="center", va="center", fontsize=7.8, color=GRAY_E)

fig.tight_layout(pad=0.4)
fig.savefig("paper/figures/vgfc_method.png", dpi=200, bbox_inches="tight")
print("saved method schematic")
