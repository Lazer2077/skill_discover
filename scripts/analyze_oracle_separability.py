"""Test whether exact-replay oracle choices are linearly separable across seeds.

This is a diagnostic probe, not a controller.  Oracle labels use realized
same-state intervention outcomes from ``analyze_exact_replay_oracle.py``'s
protocol.  A fixed L2 multinomial logistic probe is evaluated leave-one-source-
seed-out, once with state/target features and once after adding the macro
model's candidate predictions.  No hyperparameter sweep is performed.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


SCALES = (0.75, 0.90, 1.00)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--time-ratio", type=float, default=1.10)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def constant_scale(sample: dict) -> float | None:
    sequence = np.asarray(sample["candidate_sequence"], dtype=np.float64)
    if len(sequence) == 0 or not np.allclose(sequence, sequence[0], atol=1e-5):
        return None
    scale = round(float(sequence[0]), 2)
    return scale if scale in SCALES else None


def outcome(sample: dict) -> dict:
    return {
        "work_j": float(sample["prefix_work_j"] + sample["remaining_work_j"]),
        "time_s": float(sample["prefix_time_s"] + sample["remaining_time_s"]),
        "success": bool(sample["success"]),
    }


def scan_summary(scan: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            scan.mean(),
            scan.std(),
            scan.min(),
            scan.max(),
            *np.quantile(scan, [0.10, 0.25, 0.50, 0.75, 0.90]),
        ],
        dtype=np.float32,
    )


def build_examples(paths: list[str], time_ratio: float) -> list[dict]:
    examples = []
    for source_index, path_string in enumerate(paths):
        samples = json.loads(Path(path_string).read_text())["mpc_dagger_samples"]
        groups: dict[tuple, dict[float, tuple[bool, dict]]] = defaultdict(dict)
        for sample in samples:
            scale = constant_scale(sample)
            if scale is None:
                continue
            key = (tuple(sample["target"]), int(sample["trial"]), int(sample["query_step"]))
            explicit = sample["candidate_label"] != "selected"
            prior = groups[key].get(scale)
            if prior is None or (explicit and not prior[0]):
                groups[key][scale] = (explicit, sample)
        for group in groups.values():
            if not all(scale in group for scale in SCALES):
                continue
            samples_by_scale = {scale: group[scale][1] for scale in SCALES}
            outcomes = {scale: outcome(sample) for scale, sample in samples_by_scale.items()}
            direct = outcomes[1.00]
            eligible = [
                scale
                for scale, candidate in outcomes.items()
                if candidate["success"] >= direct["success"]
                and candidate["time_s"] <= time_ratio * direct["time_s"] + 1e-9
            ]
            oracle_scale = min(
                eligible,
                key=lambda scale: (outcomes[scale]["work_j"], outcomes[scale]["time_s"]),
            )
            reference = samples_by_scale[1.00]
            observation = np.asarray(reference["query_observation"], dtype=np.float32)
            target = np.asarray(reference["query_local_target"], dtype=np.float32)
            state_features = np.concatenate(
                [
                    observation[:48],
                    scan_summary(observation[48:235]),
                    target,
                    np.asarray(
                        [np.linalg.norm(target), reference["query_start_relative_height"]],
                        dtype=np.float32,
                    ),
                ]
            )
            macro_features = []
            for scale in SCALES:
                predicted = samples_by_scale[scale]["predicted_prefix"]
                for name in (
                    "work_j",
                    "progress_m",
                    "min_height",
                    "max_tilt",
                    "max_ang_speed",
                ):
                    values = np.asarray(predicted[name], dtype=np.float32)
                    macro_features.extend([float(values.mean()), float(values.std())])
            examples.append(
                {
                    "source_index": source_index,
                    "state_features": state_features,
                    "state_macro_features": np.concatenate(
                        [state_features, np.asarray(macro_features, dtype=np.float32)]
                    ),
                    "oracle_scale": int(round(100 * oracle_scale)),
                    "outcomes": outcomes,
                }
            )
    return examples


def evaluate_probe(examples: list[dict], feature_key: str, time_ratio: float) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    source_indices = sorted({example["source_index"] for example in examples})
    folds = []
    predictions = []
    for held_out in source_indices:
        train = [example for example in examples if example["source_index"] != held_out]
        test = [example for example in examples if example["source_index"] == held_out]
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=2000, random_state=0),
        )
        model.fit(
            np.stack([example[feature_key] for example in train]),
            np.asarray([example["oracle_scale"] for example in train]),
        )
        selected = model.predict(np.stack([example[feature_key] for example in test]))
        correct = []
        relative_work = []
        violations = []
        for example, scale_int in zip(test, selected):
            chosen = example["outcomes"][scale_int / 100.0]
            direct = example["outcomes"][1.00]
            correct.append(scale_int == example["oracle_scale"])
            relative_work.append(chosen["work_j"] / direct["work_j"] - 1.0)
            violations.append(
                not (
                    chosen["success"] >= direct["success"]
                    and chosen["time_s"] <= time_ratio * direct["time_s"] + 1e-9
                )
            )
        predictions.extend(int(value) for value in selected)
        folds.append(
            {
                "held_out_source_index": held_out,
                "num_query_states": len(test),
                "oracle_accuracy": float(np.mean(correct)),
                "mean_relative_work_vs_direct": float(np.mean(relative_work)),
                "constraint_violation_rate": float(np.mean(violations)),
                "choice_counts": dict(Counter(str(value) for value in selected)),
            }
        )
    return {
        "feature_set": feature_key,
        "probe": "standardized L2 multinomial logistic regression, C=1.0 (fixed)",
        "folds": folds,
        "mean_oracle_accuracy": float(np.mean([fold["oracle_accuracy"] for fold in folds])),
        "mean_relative_work_vs_direct": float(
            np.mean([fold["mean_relative_work_vs_direct"] for fold in folds])
        ),
        "mean_constraint_violation_rate": float(
            np.mean([fold["constraint_violation_rate"] for fold in folds])
        ),
        "choice_counts": dict(Counter(str(value) for value in predictions)),
    }


def main() -> None:
    args = parse_args()
    examples = build_examples(args.inputs, args.time_ratio)
    if len({example["source_index"] for example in examples}) < 3:
        raise ValueError("This diagnostic requires at least three source seeds.")
    payload = {
        "description": (
            "Cross-source-seed linear separability probe for exact-replay oracle labels. "
            "It is a mechanism diagnostic, not an online controller."
        ),
        "inputs": args.inputs,
        "time_ratio": args.time_ratio,
        "num_query_states": len(examples),
        "oracle_label_counts": dict(
            Counter(str(example["oracle_scale"]) for example in examples)
        ),
        "results": [
            evaluate_probe(examples, "state_features", args.time_ratio),
            evaluate_probe(examples, "state_macro_features", args.time_ratio),
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
