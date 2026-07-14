"""Plot the exact replay MPC DAgger distribution shift audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dagger_inputs", nargs="+", required=True)
    parser.add_argument("--validations", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    replay_errors: list[float] = []
    terminal_errors: list[float] = []
    for path in args.dagger_inputs:
        data = json.loads(Path(path).read_text())
        for sample in data["mpc_dagger_samples"]:
            replay_errors.append(float(sample["replay_observation_l2"]))
            if sample["actual_terminal_input"] is None:
                continue
            predicted = np.asarray(sample["predicted_terminal_inputs"], dtype=np.float64)
            actual = np.asarray(sample["actual_terminal_input"], dtype=np.float64)
            terminal_errors.extend(np.linalg.norm(predicted - actual[None, :], axis=1))

    colors = {
        "completion_mpc_command": "#0072B2",
        "scaled_target_command_90": "#D55E00",
        "direct_target_command": "#444444",
    }
    labels = {
        "completion_mpc_command": "completion MPC",
        "scaled_target_command_90": "fixed 0.90",
        "direct_target_command": "direct",
    }
    methods = list(colors)

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.75))
    ax = axes[0]
    bins = np.linspace(0.0, max(16.0, max(terminal_errors)), 28)
    ax.hist(
        terminal_errors,
        bins=bins,
        density=True,
        color="#56B4E9",
        edgecolor="white",
        linewidth=0.5,
        label="macro terminal prediction",
    )
    ax.axvline(np.median(terminal_errors), color="#0072B2", linewidth=1.5, label="median")
    ax.scatter(
        [max(replay_errors)],
        [0.02],
        marker="x",
        s=42,
        linewidth=1.7,
        color="#D55E00",
        label="max replay mismatch",
        zorder=4,
    )
    ax.set_xlabel(r"terminal input error $\|\hat{x}-x\|_2$")
    ax.set_ylabel("density")
    ax.set_title("(a) Exact replay isolates model shift")
    ax.legend(frameon=False, fontsize=7, loc="upper right")

    ax = axes[1]
    markers = ["o", "s"]
    for validation_index, path in enumerate(args.validations):
        data = json.loads(Path(path).read_text())
        flat = {method: [] for method in methods}
        for method_records in data["records"].values():
            for method in methods:
                flat[method].extend(method_records[method])
        for method in methods:
            work = np.mean([record["mechanical_energy_j"] for record in flat[method]])
            duration = np.mean([record["elapsed_time_s"] for record in flat[method]])
            success = np.mean([record["success"] for record in flat[method]])
            ax.scatter(
                duration,
                work,
                s=35 + 75 * success,
                marker=markers[validation_index],
                color=colors[method],
                edgecolor="white",
                linewidth=0.6,
                zorder=3,
            )
        points = [
            (
                np.mean([record["elapsed_time_s"] for record in flat[method]]),
                np.mean([record["mechanical_energy_j"] for record in flat[method]]),
            )
            for method in methods
        ]
        ax.plot(
            [point[0] for point in points],
            [point[1] for point in points],
            color="#BBBBBB",
            linewidth=0.8,
            zorder=1,
        )
    for method in methods:
        ax.scatter([], [], color=colors[method], label=labels[method])
    ax.scatter([], [], marker="o", color="#888888", label="DAgger iter. 1")
    ax.scatter([], [], marker="s", color="#888888", label="DAgger iter. 2")
    ax.set_xlabel("elapsed task time (s)")
    ax.set_ylabel("absolute mechanical work (J)")
    ax.set_title("(b) Two unseen seed validations")
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper right")

    for ax in axes:
        ax.grid(alpha=0.2, linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(w_pad=1.6)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=250, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
