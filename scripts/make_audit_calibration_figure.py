"""Calibration figure for the adapter necessity audit.

Left: observable-signal sweep of H_alloc and violation rate with decisions at δ=1%.
Right: hidden-signal false-GO counts across ten mappings.

Saves paper/figures/audit_calibration.png.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/home/thing1/skill_discover")
OBS = ROOT / "outputs/response_model/adapter_value_audit_observable_curve_10seed_summary.json"
HID = ROOT / "outputs/response_model/adapter_value_audit_hidden_05pct_repeats_10seed_summary.json"
REAL = ROOT / "outputs/response_model/adapter_value_audit_10seed_summary.json"
OUT = ROOT / "paper/figures/audit_calibration.png"

INK = "#111827"
BLUE = "#2563eb"
ROSE = "#be123c"
GREEN = "#15803d"
AMBER = "#b45309"
GRAY = "#64748b"


def load_obs():
    payload = json.loads(OBS.read_text())
    strengths, h_mean, h_lo, h_hi, q_mean, q_lo, q_hi, dec = [], [], [], [], [], [], [], []
    for row in payload["results"]:
        strengths.append(100.0 * row["semi_synthetic_control"]["strength"])
        g = row["allocation_gain_percent"]
        q = row["constraint_violation_rate"]
        h_mean.append(g["mean"])
        h_lo.append(g["ci95"][0])
        h_hi.append(g["ci95"][1])
        q_mean.append(100.0 * q["mean"])
        q_lo.append(100.0 * q["ci95"][0])
        q_hi.append(100.0 * q["ci95"][1])
        dec.append(row["decisions"]["1.00%"])
    real = json.loads(REAL.read_text())["results"][0]
    return {
        "strength": np.asarray(strengths),
        "h_mean": np.asarray(h_mean),
        "h_lo": np.asarray(h_lo),
        "h_hi": np.asarray(h_hi),
        "q_mean": np.asarray(q_mean),
        "q_lo": np.asarray(q_lo),
        "q_hi": np.asarray(q_hi),
        "dec": dec,
        "real_h": real["allocation_gain_percent"]["mean"],
        "real_h_ci": real["allocation_gain_percent"]["ci95"],
        "real_q": 100.0 * real["constraint_violation_rate"]["mean"],
        "real_q_ci": [100.0 * x for x in real["constraint_violation_rate"]["ci95"]],
        "real_dec": real["decisions"]["1.00%"],
    }


def load_hidden():
    payload = json.loads(HID.read_text())
    counts = {"0.25%": 0, "0.50%": 0, "1.00%": 0}
    n = 0
    for row in payload["results"]:
        n += 1
        for thr, decision in row["decisions"].items():
            if decision == "GO":
                counts[thr] += 1
    return counts, n


def main():
    obs = load_obs()
    hidden_counts, n_hidden = load_hidden()

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.9), gridspec_kw={"width_ratios": [1.55, 1.0]})

    ax = axes[0]
    ax2 = ax.twinx()
    x = obs["strength"]
    ax.fill_between(x, obs["h_lo"], obs["h_hi"], color=BLUE, alpha=0.18, linewidth=0)
    ax.plot(x, obs["h_mean"], "-o", color=BLUE, lw=1.8, ms=5.5, label=r"$H_{\mathrm{alloc}}$")
    ax2.fill_between(x, obs["q_lo"], obs["q_hi"], color=ROSE, alpha=0.14, linewidth=0)
    ax2.plot(x, obs["q_mean"], "-s", color=ROSE, lw=1.6, ms=5.0, label="violation rate $q$")

    # real outcome marker at gamma=0
    ax.errorbar(
        [0.0],
        [obs["real_h"]],
        yerr=[[obs["real_h"] - obs["real_h_ci"][0]], [obs["real_h_ci"][1] - obs["real_h"]]],
        fmt="D",
        color=AMBER,
        ms=6,
        capsize=3,
        label="real outcomes",
    )
    ax2.errorbar(
        [0.0],
        [obs["real_q"]],
        yerr=[[obs["real_q"] - obs["real_q_ci"][0]], [obs["real_q_ci"][1] - obs["real_q"]]],
        fmt="D",
        color=AMBER,
        ms=6,
        capsize=3,
        alpha=0.85,
    )

    ax.axhline(1.0, color=BLUE, ls="--", lw=1.0, alpha=0.7)
    ax2.axhline(5.0, color=ROSE, ls="--", lw=1.0, alpha=0.7)
    ax.text(10.2, 1.12, r"$\delta=1\%$", color=BLUE, fontsize=8)
    ax2.text(10.2, 5.25, r"$\kappa=5\%$", color=ROSE, fontsize=8)

    for xi, decision in zip(x, obs["dec"]):
        color = {"GO": GREEN, "NO-GO": ROSE, "ABSTAIN": AMBER}[decision]
        ax.text(xi, obs["h_hi"][list(x).index(xi)] + 0.18, decision, ha="center", va="bottom", fontsize=7.2, color=color, fontweight="bold")

    ax.set_xlim(-0.6, 10.8)
    ax.set_ylim(0, 3.8)
    ax2.set_ylim(0, 9.0)
    ax.set_xlabel(r"injected observable work benefit $\gamma$ (%)")
    ax.set_ylabel(r"allocation gain $H_{\mathrm{alloc}}$ (%)", color=BLUE)
    ax2.set_ylabel(r"constraint violation $q$ (%)", color=ROSE)
    ax.set_title(r"Observable-signal calibration (joint rule at $\delta=1\%$, $\kappa=5\%$)", fontsize=9.5, pad=8)
    ax.tick_params(axis="y", colors=BLUE)
    ax2.tick_params(axis="y", colors=ROSE)
    ax.legend(loc="upper left", frameon=False, fontsize=8)

    # annotate formal gain-only GO: L_H > δ first at γ=9%
    ax.annotate(
        r"gain-only GO ($L_H>\delta$)" + "\nfirst at " + r"$\gamma=9\%$",
        xy=(9.0, 2.32),
        xytext=(5.8, 3.25),
        fontsize=7.5,
        color=GRAY,
        arrowprops=dict(arrowstyle="->", color=GRAY, lw=0.9),
    )
    ax.annotate(
        r"mean $H_{\mathrm{alloc}}>\delta$" + "\nat " + r"$\gamma=5\%$",
        xy=(5.0, 1.26),
        xytext=(2.2, 2.55),
        fontsize=7.4,
        color=GRAY,
        arrowprops=dict(arrowstyle="->", color=GRAY, lw=0.9),
    )

    axb = axes[1]
    thrs = ["0.25%", "0.50%", "1.00%"]
    vals = [hidden_counts[t] for t in thrs]
    colors = [ROSE if v > 0 else GREEN for v in vals]
    bars = axb.bar(thrs, vals, color=colors, edgecolor=INK, width=0.62, alpha=0.85)
    for bar, v in zip(bars, vals):
        axb.text(bar.get_x() + bar.get_width() / 2, v + 0.08, f"{v}/{n_hidden}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    axb.set_ylim(0, max(vals + [1]) + 1.2)
    axb.set_ylabel(f"false GO counts / {n_hidden} hidden mappings")
    axb.set_xlabel(r"practical-value threshold $\delta$")
    axb.set_title(r"Hidden-signal specificity ($\gamma=5\%$)", fontsize=9.5, pad=8)
    axb.axhline(0, color=INK, lw=0.6)
    axb.text(0.5, -0.22, "zero false GO at " + r"$\delta=1\%$", transform=axb.transAxes, ha="center", va="top", fontsize=8, color=GREEN)

    fig.tight_layout(pad=0.6, w_pad=1.4)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=220, bbox_inches="tight")
    print(f"saved {OUT}")
    print("observable decisions@1%:", obs["dec"], "real:", obs["real_dec"])
    print("hidden false GO:", hidden_counts)


if __name__ == "__main__":
    main()
