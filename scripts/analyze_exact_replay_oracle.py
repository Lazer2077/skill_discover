"""Measure the state-adaptive headroom in exact-replay MPC interventions.

The diagnostic compares constant gain prefixes that start from the *same*
replayed query state and subsequently use the same direct-control continuation.
It is an oracle analysis, not an online controller evaluation: it asks whether
state-conditioned selection could in principle improve on all-direct under
locked success and remaining-time constraints.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--time-ratios", default="1.00,1.05,1.10,1.20")
    parser.add_argument("--scales", default="0.75,0.90,1.00")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def parse_floats(raw: str) -> list[float]:
    return [round(float(value), 2) for value in raw.split(",") if value.strip()]


def constant_scale(sample: dict) -> float | None:
    sequence = np.asarray(sample["candidate_sequence"], dtype=np.float64)
    if len(sequence) == 0 or not np.allclose(sequence, sequence[0], atol=1e-5):
        return None
    return round(float(sequence[0]), 2)


def outcome(sample: dict) -> dict:
    return {
        "work_j": float(sample["prefix_work_j"] + sample["remaining_work_j"]),
        "time_s": float(sample["prefix_time_s"] + sample["remaining_time_s"]),
        "success": bool(sample["success"]),
        "prefix_min_height": float(sample["prefix_min_relative_height"]),
        "prefix_max_tilt": float(sample["prefix_max_tilt"]),
        "prefix_max_ang_speed": float(sample["prefix_max_ang_speed"]),
    }


def mean_metric(rows: list[dict], key: str) -> float:
    return float(np.mean([row[key] for row in rows]))


def summarize_groups(
    groups: list[dict[float, dict]], scales: list[float], time_ratio: float
) -> dict:
    direct_scale = max(scales)
    direct_rows, oracle_rows, choices = [], [], []
    for group in groups:
        direct = group[direct_scale]
        eligible = [
            (scale, row)
            for scale, row in group.items()
            if row["success"] >= direct["success"]
            and row["time_s"] <= time_ratio * direct["time_s"] + 1e-9
        ]
        if not eligible:
            raise RuntimeError("The all-direct candidate must be eligible.")
        scale, oracle = min(eligible, key=lambda item: (item[1]["work_j"], item[1]["time_s"]))
        direct_rows.append(direct)
        oracle_rows.append(oracle)
        choices.append(scale)
    direct_work = mean_metric(direct_rows, "work_j")
    oracle_work = mean_metric(oracle_rows, "work_j")
    return {
        "num_query_states": len(groups),
        "time_ratio": time_ratio,
        "direct": {
            "mean_work_j": direct_work,
            "mean_time_s": mean_metric(direct_rows, "time_s"),
            "success_rate": mean_metric(direct_rows, "success"),
        },
        "oracle": {
            "mean_work_j": oracle_work,
            "mean_time_s": mean_metric(oracle_rows, "time_s"),
            "success_rate": mean_metric(oracle_rows, "success"),
            "relative_work_change": float(oracle_work / direct_work - 1.0),
            "choice_counts": {f"{scale:.2f}": choices.count(scale) for scale in scales},
        },
    }


def main() -> None:
    args = parse_args()
    scales = parse_floats(args.scales)
    ratios = parse_floats(args.time_ratios)
    if len(scales) < 2 or max(scales) not in scales:
        raise ValueError("Provide at least two scales including the all-direct scale.")

    source_groups: dict[str, dict[tuple, dict[float, tuple[bool, dict]]]] = {}
    for path_string in args.inputs:
        path = Path(path_string)
        samples = json.loads(path.read_text()).get("mpc_dagger_samples", [])
        groups: dict[tuple, dict[float, tuple[bool, dict]]] = defaultdict(dict)
        for sample in samples:
            scale = constant_scale(sample)
            if scale not in scales:
                continue
            key = (tuple(sample["target"]), int(sample["trial"]), int(sample["query_step"]))
            explicit = sample["candidate_label"] != "selected"
            prior = groups[key].get(scale)
            if prior is None or (explicit and not prior[0]):
                groups[key][scale] = (explicit, outcome(sample))
        source_groups[str(path)] = groups

    complete_by_source = {}
    all_groups = []
    for path, groups in source_groups.items():
        complete = [
            {scale: value[1] for scale, value in group.items()}
            for group in groups.values()
            if all(scale in group for scale in scales)
        ]
        if not complete:
            raise ValueError(f"No complete constant-gain query groups in {path}.")
        complete_by_source[path] = complete
        all_groups.extend(complete)

    payload = {
        "description": (
            "Exact-replay single-intervention oracle. Every compared gain prefix starts "
            "from the same replayed query state and then shares direct-control continuation."
        ),
        "scales": scales,
        "time_ratios": ratios,
        "per_source": {
            path: [summarize_groups(groups, scales, ratio) for ratio in ratios]
            for path, groups in complete_by_source.items()
        },
        "pooled": [summarize_groups(all_groups, scales, ratio) for ratio in ratios],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
