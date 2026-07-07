"""Plot grouped metrics from summarize_skill_eval_results.py output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_METRICS = "success,final_distance,energy_proxy,final_height"
LABELS = {
    "success": "Success",
    "final_distance": "Final distance",
    "energy_proxy": "Energy proxy",
    "final_height": "Final height",
    "terminated_early": "Termination",
    "num_commands_used": "Archive commands",
}
COLORS = {
    "direct_target_command": "#4c78a8",
    "skill_command": "#f58518",
    "guarded_skill_command": "#54a24b",
    "learned_guarded_skill_command": "#b279a2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot skill eval aggregate metrics.")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--section", default="episode_weighted", choices=["episode_weighted", "target_balanced"])
    parser.add_argument("--metrics", default=DEFAULT_METRICS)
    parser.add_argument("--methods", default="")
    parser.add_argument("--title", default="")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = json.loads(Path(args.summary).read_text())
    section = data[args.section]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()] or list(section)
    methods = [m for m in methods if m in section]
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]

    fig, axes = plt.subplots(1, len(metrics), figsize=(4.0 * len(metrics), 3.4), squeeze=False)
    x = np.arange(len(methods))
    for ax, metric in zip(axes[0], metrics):
        means = np.asarray([section[method][metric]["mean"] for method in methods], dtype=np.float64)
        lows = np.asarray([section[method][metric]["ci_low"] for method in methods], dtype=np.float64)
        highs = np.asarray([section[method][metric]["ci_high"] for method in methods], dtype=np.float64)
        yerr = np.vstack([np.maximum(0.0, means - lows), np.maximum(0.0, highs - means)])
        colors = [COLORS.get(method, "#777777") for method in methods]
        ax.bar(x, means, color=colors, width=0.68, edgecolor="black", linewidth=0.5)
        ax.errorbar(x, means, yerr=yerr, fmt="none", ecolor="black", elinewidth=1.0, capsize=3)
        ax.set_title(LABELS.get(metric, metric))
        ax.set_xticks(x)
        ax.set_xticklabels([method.replace("_", "\n") for method in methods], fontsize=8)
        ax.grid(axis="y", alpha=0.25)
        if metric in {"success", "terminated_early"}:
            ax.set_ylim(0.0, 1.05)
    if args.title:
        fig.suptitle(args.title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94 if args.title else 1))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
