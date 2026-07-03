"""Matplotlib plots for skill discovery diagnostics.

All plots share one fixed-order, colorblind-validated categorical palette:
skill i always gets CATEGORICAL_PALETTE[i], so colors are consistent across
the PCA scatter, the usage histogram, and the descriptor bars.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # headless-safe: no display needed on cluster/CI machines
import matplotlib.pyplot as plt
import numpy as np

# Fixed-order categorical palette (CVD-safe, validated). Assigned by skill id,
# never cycled: with more than 8 skills we fall back to tab20 for the overflow.
CATEGORICAL_PALETTE = [
    "#2a78d6",  # blue
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
    "#e87ba4",  # magenta
    "#eb6834",  # orange
]
_GRID_COLOR = "#e1e0d9"
_TEXT_COLOR = "#0b0b0b"
_MUTED_COLOR = "#898781"


def skill_colors(num_skills: int) -> List[str]:
    """Return one stable color per skill id."""
    if num_skills <= len(CATEGORICAL_PALETTE):
        return CATEGORICAL_PALETTE[:num_skills]
    cmap = plt.get_cmap("tab20")
    extra = [matplotlib.colors.to_hex(cmap(i % 20)) for i in range(num_skills - len(CATEGORICAL_PALETTE))]
    return CATEGORICAL_PALETTE + extra


def _style_axes(ax: plt.Axes) -> None:
    """Recessive grid and spines so the data marks dominate."""
    ax.grid(True, color=_GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(_MUTED_COLOR)
    ax.tick_params(colors=_MUTED_COLOR, labelcolor=_TEXT_COLOR)


def _save(fig: plt.Figure, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_descriptor_pca(
    descriptor_matrix: np.ndarray,
    skill_ids: np.ndarray,
    path: str | Path,
    title: str = "Skill clusters (PCA of behavior descriptors)",
) -> None:
    """2D PCA scatter of descriptors, colored by skill id."""
    from sklearn.decomposition import PCA

    proj = PCA(n_components=2).fit_transform(descriptor_matrix)
    unique_ids = np.unique(skill_ids)
    colors = skill_colors(int(unique_ids.max()) + 1 if len(unique_ids) else 1)

    fig, ax = plt.subplots(figsize=(8, 6))
    _style_axes(ax)
    for sid in unique_ids:
        mask = skill_ids == sid
        label = f"skill {sid}" if sid >= 0 else "noise"
        color = colors[sid] if sid >= 0 else _MUTED_COLOR
        ax.scatter(
            proj[mask, 0], proj[mask, 1],
            s=14, color=color, alpha=0.7, linewidths=0, label=label, zorder=3,
        )
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title(title, color=_TEXT_COLOR)
    ax.legend(frameon=False, fontsize=9, markerscale=1.6)
    _save(fig, path)


def plot_skill_histogram(
    skill_ids: np.ndarray,
    path: str | Path,
    title: str = "Segments per skill",
) -> None:
    """Bar chart of how many segments each skill absorbed."""
    unique_ids, counts = np.unique(skill_ids, return_counts=True)
    colors = skill_colors(int(unique_ids.max()) + 1 if len(unique_ids) else 1)
    bar_colors = [colors[sid] if sid >= 0 else _MUTED_COLOR for sid in unique_ids]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    _style_axes(ax)
    bars = ax.bar(unique_ids.astype(str), counts, color=bar_colors, width=0.7, zorder=3)
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height(),
            str(count), ha="center", va="bottom", fontsize=9, color=_TEXT_COLOR,
        )
    ax.set_xlabel("skill id")
    ax.set_ylabel("num segments")
    ax.set_title(title, color=_TEXT_COLOR)
    _save(fig, path)


def plot_skill_descriptors(
    center_descriptors: np.ndarray,
    descriptor_names: Sequence[str],
    path: str | Path,
    title: str = "Mean descriptor values per skill",
) -> None:
    """Small-multiples bar panels: one panel per descriptor, one bar per skill."""
    num_skills, num_desc = center_descriptors.shape
    colors = skill_colors(num_skills)
    ncols = 3
    nrows = int(np.ceil(num_desc / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 2.6 * nrows), squeeze=False)
    for d in range(num_desc):
        ax = axes[d // ncols][d % ncols]
        _style_axes(ax)
        ax.bar(np.arange(num_skills).astype(str), center_descriptors[:, d], color=colors, width=0.7, zorder=3)
        ax.axhline(0.0, color=_MUTED_COLOR, linewidth=0.8)
        ax.set_title(descriptor_names[d], fontsize=10, color=_TEXT_COLOR)
        ax.tick_params(labelsize=8)
    for d in range(num_desc, nrows * ncols):
        axes[d // ncols][d % ncols].axis("off")
    fig.suptitle(title, color=_TEXT_COLOR)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save(fig, path)


def plot_skill_displacements(
    skill_means: Dict[int, Dict[str, float]],
    path: str | Path,
    title: str = "Average per-segment displacement per skill (body frame)",
) -> None:
    """Arrow plot of each skill's mean (delta_x, delta_y), annotated with yaw."""
    fig, ax = plt.subplots(figsize=(7, 7))
    _style_axes(ax)
    colors = skill_colors(max(skill_means) + 1 if skill_means else 1)
    lim = 1.15 * max(
        (abs(m["mean_delta_x"]) + abs(m["mean_delta_y"]) for m in skill_means.values()), default=1.0
    )
    lim = max(lim, 1e-3)
    for i, (sid, means) in enumerate(sorted(skill_means.items())):
        dx, dy = means["mean_delta_x"], means["mean_delta_y"]
        ax.annotate(
            "", xy=(dx, dy), xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color=colors[sid], lw=2),
        )
        # Stagger labels of short arrows vertically so they don't collide near origin.
        label_dy = 0.05 * lim * (i + 1) if np.hypot(dx, dy) < 0.15 * lim else 0.0
        ax.text(
            dx, dy + label_dy, f" {sid} (yaw {means['mean_delta_yaw']:+.2f})",
            fontsize=9, color=_TEXT_COLOR,
        )
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("delta x [m]")
    ax.set_ylabel("delta y [m]")
    ax.set_title(title, color=_TEXT_COLOR)
    _save(fig, path)


def plot_composition_trajectories(
    trajectories: List[np.ndarray],
    target_xy: np.ndarray,
    path: str | Path,
    threshold: float = 0.5,
    title: str = "Skill-composition rollouts",
) -> None:
    """XY base-position traces for each evaluation trial plus the target circle."""
    fig, ax = plt.subplots(figsize=(7, 7))
    _style_axes(ax)
    colors = skill_colors(len(trajectories))
    for i, traj in enumerate(trajectories):
        ax.plot(traj[:, 0], traj[:, 1], color=colors[i], lw=1.5, alpha=0.85, label=f"trial {i}", zorder=3)
        ax.scatter(traj[0, 0], traj[0, 1], color=colors[i], s=25, marker="o", zorder=4)
    circle = plt.Circle(tuple(target_xy), threshold, color=_MUTED_COLOR, fill=False, ls="--")
    ax.add_patch(circle)
    ax.scatter([target_xy[0]], [target_xy[1]], color=_TEXT_COLOR, marker="*", s=140, zorder=5, label="target")
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title, color=_TEXT_COLOR)
    if len(trajectories) <= 10:
        ax.legend(frameon=False, fontsize=8)
    _save(fig, path)
