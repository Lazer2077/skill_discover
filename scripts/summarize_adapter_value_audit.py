"""Cluster-bootstrap adapter allocation value and issue a three-way decision.

Each input is a leave-one-source-seed-out treatment-effect audit.  The source
seed, not the query, is the resampling unit.  GO requires the lower confidence
bound on state-allocation gain to clear a practical-value threshold and the
upper bound on realized constraint violations to stay below tolerance.  NO-GO
requires the gain upper bound to fall below the threshold (or violations to be
provably excessive); all other cases ABSTAIN.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--thresholds", default="0.25,0.50,1.00")
    parser.add_argument("--max-violation-rate", type=float, default=0.05)
    parser.add_argument("--bootstrap-replicates", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def interval(values: np.ndarray, indices: np.ndarray) -> dict:
    boot = values[indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95": [float(value) for value in np.quantile(boot, [0.025, 0.975])],
        "fold_values": [float(value) for value in values],
    }


def main() -> None:
    args = parse_args()
    thresholds = [float(value) for value in args.thresholds.split(",") if value.strip()]
    payloads = [json.loads(Path(path).read_text()) for path in args.inputs]
    rng = np.random.default_rng(args.seed)
    results = []
    for path, payload in zip(args.inputs, payloads):
        folds = payload["folds"]
        n = len(folds)
        indices = rng.integers(0, n, size=(args.bootstrap_replicates, n))
        allocation_gain = np.asarray(
            [
                -100.0
                * fold["matched_randomized_mixture"]["selector_relative_work_vs_mixture"]
                for fold in folds
            ]
        )
        total_gain = np.asarray(
            [-100.0 * fold["selection"]["mean_relative_work_vs_direct"] for fold in folds]
        )
        violation_rate = np.asarray(
            [fold["selection"]["constraint_violation_rate"] for fold in folds]
        )
        allocation_summary = interval(allocation_gain, indices)
        total_summary = interval(total_gain, indices)
        violation_summary = interval(violation_rate, indices)
        decisions = {}
        gain_low, gain_high = allocation_summary["ci95"]
        violation_low, violation_high = violation_summary["ci95"]
        for threshold in thresholds:
            if (
                gain_low > threshold
                and violation_high <= args.max_violation_rate
            ):
                decision = "GO"
            elif (
                gain_high < threshold
                or violation_low > args.max_violation_rate
            ):
                decision = "NO-GO"
            else:
                decision = "ABSTAIN"
            decisions[f"{threshold:.2f}%"] = decision
        results.append(
            {
                "input": path,
                "feature_set": payload.get("feature_set", "unspecified"),
                "semi_synthetic_control": payload.get("semi_synthetic_control", {}),
                "num_source_seeds": n,
                "allocation_gain_percent": allocation_summary,
                "total_gain_vs_direct_percent": total_summary,
                "constraint_violation_rate": violation_summary,
                "decisions": decisions,
            }
        )
    output = {
        "description": (
            "Source-seed cluster bootstrap of state-allocation value relative to a "
            "frequency-matched randomized mixture."
        ),
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seed": args.seed,
        "practical_gain_thresholds_percent": thresholds,
        "max_constraint_violation_rate": args.max_violation_rate,
        "decision_rule": {
            "GO": "gain CI lower bound > threshold and violation CI upper bound <= tolerance",
            "NO-GO": "gain CI upper bound < threshold or violation CI lower bound > tolerance",
            "ABSTAIN": "otherwise",
        },
        "results": results,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
