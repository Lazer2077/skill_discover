"""Train an episode-level 0.9-vs-1.0 scale selector from paired rollouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--summary", default="")
    p.add_argument("--command_slice", default="9:12")
    p.add_argument("--height_scan_slice", default="48:235")
    p.add_argument("--time_ratio", type=float, default=1.10)
    p.add_argument("--c", type=float, default=0.1)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=2026)
    return p.parse_args()


def selector_features(obs: np.ndarray, local_target: np.ndarray, distance: float, scan_slice):
    scan = obs[scan_slice[0] : scan_slice[1]]
    scan_stats = np.asarray(
        [
            scan.mean(),
            scan.std(),
            scan.min(),
            scan.max(),
            *np.quantile(scan, [0.10, 0.25, 0.50, 0.75, 0.90]),
        ],
        dtype=np.float32,
    )
    return np.concatenate(
        [
            np.asarray(local_target[:2], dtype=np.float32),
            np.asarray([distance], dtype=np.float32),
            obs[:9].astype(np.float32),
            obs[12:48].astype(np.float32),
            scan_stats,
        ]
    )


def aggregate(records):
    def mean(key):
        return float(np.mean([float(record[key]) for record in records]))

    return {
        "n": len(records),
        "success": mean("success"),
        "work_j": mean("mechanical_energy_j"),
        "time_s": mean("elapsed_time_s"),
        "cot": mean("cost_of_transport"),
    }


def main() -> None:
    args = parse_args()
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scan_slice = tuple(int(value) for value in args.height_scan_slice.split(":"))
    rows = []
    for source_index, path in enumerate(args.inputs):
        data = json.loads(Path(path).read_text())
        for target, methods in data["records"].items():
            if "scaled_target_command_90" not in methods or "direct_target_command" not in methods:
                continue
            scaled_records = methods["scaled_target_command_90"]
            direct_records = methods["direct_target_command"]
            for scaled, direct in zip(scaled_records, direct_records):
                scaled_sample = scaled["value_samples"][0]
                direct_sample = direct["value_samples"][0]
                obs = np.asarray(direct_sample["observation"], dtype=np.float32)
                features = selector_features(
                    obs,
                    np.asarray(direct_sample["local_target"], dtype=np.float32),
                    float(direct_sample["distance"]),
                    scan_slice,
                )
                feasible = (
                    scaled["elapsed_time_s"] <= args.time_ratio * direct["elapsed_time_s"] + 1e-9
                    and (scaled["success"] or not direct["success"])
                )
                label = int(feasible and scaled["mechanical_energy_j"] < direct["mechanical_energy_j"])
                pair_diff = float(
                    np.max(
                        np.abs(
                            np.asarray(scaled_sample["observation"], dtype=np.float32)
                            - np.asarray(direct_sample["observation"], dtype=np.float32)
                        )
                    )
                )
                rows.append(
                    {
                        "source": source_index,
                        "target": target,
                        "features": features,
                        "label": label,
                        "scaled": scaled,
                        "direct": direct,
                        "pair_diff": pair_diff,
                    }
                )
    if not rows:
        raise ValueError("No paired scale-0.9/direct episodes found.")
    x = np.stack([row["features"] for row in rows])
    y = np.asarray([row["label"] for row in rows], dtype=np.int64)
    selected = []
    probabilities = np.zeros(len(rows), dtype=np.float32)
    fold_details = []
    for source in sorted(set(row["source"] for row in rows)):
        train = np.asarray([row["source"] != source for row in rows])
        test = ~train
        scaler = StandardScaler().fit(x[train])
        model = LogisticRegression(
            C=args.c, class_weight="balanced", max_iter=2000, random_state=args.seed
        ).fit(scaler.transform(x[train]), y[train])
        probability = model.predict_proba(scaler.transform(x[test]))[:, 1]
        probabilities[test] = probability
        indices = np.flatnonzero(test)
        fold_selected = []
        for index, prob in zip(indices, probability):
            record = rows[index]["scaled"] if prob >= args.threshold else rows[index]["direct"]
            selected.append(record)
            fold_selected.append(record)
        fold_details.append(
            {
                "held_out_input": args.inputs[source],
                "n": int(test.sum()),
                "classification_accuracy": float(
                    np.mean((probability >= args.threshold) == (y[test] > 0))
                ),
                "selected_scale_09": int(np.sum(probability >= args.threshold)),
                "outcomes": aggregate(fold_selected),
            }
        )
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(
        C=args.c, class_weight="balanced", max_iter=2000, random_state=args.seed
    ).fit(scaler.transform(x), y)
    payload = {
        "model_type": "paired_scale_selector",
        "x_mean": scaler.mean_.tolist(),
        "x_std": scaler.scale_.tolist(),
        "coef": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
        "threshold": args.threshold,
        "selected_scale": 0.90,
        "fallback_scale": 1.00,
        "height_scan_slice": list(scan_slice),
        "feature_dim": int(x.shape[1]),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2))
    direct_records = [row["direct"] for row in rows]
    scaled_records = [row["scaled"] for row in rows]
    oracle_records = [
        row["scaled"] if row["label"] else row["direct"]
        for row in rows
    ]
    summary = {
        "inputs": args.inputs,
        "output": args.output,
        "num_pairs": len(rows),
        "positive_fraction": float(y.mean()),
        "loso_classification_accuracy": float(
            np.mean((probabilities >= args.threshold) == (y > 0))
        ),
        "loso_selected_scale_09": int(np.sum(probabilities >= args.threshold)),
        "loso_selector": aggregate(selected),
        "direct": aggregate(direct_records),
        "fixed_scale_09": aggregate(scaled_records),
        "paired_oracle": aggregate(oracle_records),
        "max_initial_observation_pair_difference": max(row["pair_diff"] for row in rows),
        "folds": fold_details,
    }
    summary_path = args.summary or str(Path(args.output).with_suffix("")) + "_summary.json"
    Path(summary_path).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
