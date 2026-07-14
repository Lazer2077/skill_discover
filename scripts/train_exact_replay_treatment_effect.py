"""Evaluate a direct treatment-effect model on exact-replay interventions.

The model predicts each constant prefix's outcome *relative to the all-direct
prefix from the same query state*: relative work, relative remaining time, and
whether the candidate meets the realized success/time constraint.  This avoids
learning absolute return across terrain seeds.  It is a fixed-mechanism probe:
architecture, loss, confidence rule, and leave-one-source-seed-out evaluation
are fixed here; no online controller is tuned from its results.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


SCALES = (0.75, 0.90, 1.00)
OUTCOME_NAMES = ("work_j", "progress_m", "min_height", "max_tilt", "max_ang_speed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--time-ratio", type=float, default=1.10)
    parser.add_argument("--ensemble-size", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=4050)
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


def feature(sample: dict, direct: dict, scale: float) -> np.ndarray:
    observation = np.asarray(sample["query_observation"], dtype=np.float32)
    target = np.asarray(sample["query_local_target"], dtype=np.float32)
    state = np.concatenate(
        [
            observation[:48],
            scan_summary(observation[48:235]),
            target,
            np.asarray(
                [np.linalg.norm(target), sample["query_start_relative_height"]],
                dtype=np.float32,
            ),
        ]
    )
    comparison = []
    for name in OUTCOME_NAMES:
        candidate_values = np.asarray(sample["predicted_prefix"][name], dtype=np.float32)
        direct_values = np.asarray(direct["predicted_prefix"][name], dtype=np.float32)
        comparison.extend(
            [
                float(candidate_values.mean() - direct_values.mean()),
                float(candidate_values.std()),
                float(direct_values.std()),
            ]
        )
    return np.concatenate(
        [state, np.asarray([scale], dtype=np.float32), np.asarray(comparison, dtype=np.float32)]
    )


def load_queries(paths: list[str], time_ratio: float) -> list[dict]:
    queries = []
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
        for group_index, group in enumerate(groups.values()):
            if not all(scale in group for scale in SCALES):
                continue
            samples_by_scale = {scale: group[scale][1] for scale in SCALES}
            outcomes = {scale: outcome(sample) for scale, sample in samples_by_scale.items()}
            direct = outcomes[1.00]
            candidates = []
            for scale in SCALES[:-1]:
                candidate = outcomes[scale]
                candidates.append(
                    {
                        "scale": scale,
                        "feature": feature(samples_by_scale[scale], samples_by_scale[1.00], scale),
                        "work_rel": candidate["work_j"] / direct["work_j"] - 1.0,
                        "time_rel": candidate["time_s"] / direct["time_s"] - 1.0,
                        "eligible": float(
                            candidate["success"] >= direct["success"]
                            and candidate["time_s"] <= time_ratio * direct["time_s"] + 1e-9
                        ),
                    }
                )
            queries.append(
                {
                    "source_index": source_index,
                    "group_index": group_index,
                    "candidates": candidates,
                    "outcomes": outcomes,
                }
            )
    return queries


def make_model(torch, input_dim: int, hidden_dim: int):
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, 3),
    )


def build_arrays(queries: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features, targets, group_ids = [], [], []
    for query_position, query in enumerate(queries):
        for candidate in query["candidates"]:
            features.append(candidate["feature"])
            targets.append(
                [candidate["work_rel"], candidate["time_rel"], candidate["eligible"]]
            )
            group_ids.append(query_position)
    return np.stack(features), np.asarray(targets, dtype=np.float32), np.asarray(group_ids)


def train_ensemble(
    torch,
    train_queries: list[dict],
    test_queries: list[dict],
    args: argparse.Namespace,
) -> tuple[list, dict]:
    train_x, train_y, train_groups = build_arrays(train_queries)
    test_x, test_y, _ = build_arrays(test_queries)
    x_mean = train_x.mean(0)
    x_std = np.maximum(train_x.std(0), 1e-6)
    regression_mean = train_y[:, :2].mean(0)
    regression_std = np.maximum(train_y[:, :2].std(0), 1e-6)
    train_xn = (train_x - x_mean) / x_std
    test_xn = (test_x - x_mean) / x_std
    train_yn = train_y.copy()
    train_yn[:, :2] = (train_yn[:, :2] - regression_mean) / regression_std

    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = []
    query_groups = np.unique(train_groups)
    for member in range(args.ensemble_size):
        torch.manual_seed(args.seed + member)
        rng = np.random.default_rng(args.seed + member)
        boot_groups = rng.choice(query_groups, size=len(query_groups), replace=True)
        indices = np.concatenate([np.flatnonzero(train_groups == group) for group in boot_groups])
        x = torch.from_numpy(train_xn[indices]).float().to(device)
        y = torch.from_numpy(train_yn[indices]).float().to(device)
        model = make_model(torch, train_x.shape[1], args.hidden_dim).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        for _ in range(args.epochs):
            prediction = model(x)
            regression_loss = torch.nn.functional.smooth_l1_loss(prediction[:, :2], y[:, :2])
            eligibility_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                prediction[:, 2], y[:, 2]
            )
            loss = regression_loss + eligibility_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        models.append(model)

    predictions = []
    tensor = torch.from_numpy(test_xn).float().to(device)
    with torch.inference_mode():
        for model in models:
            raw = model(tensor).cpu().numpy()
            raw[:, :2] = raw[:, :2] * regression_std + regression_mean
            raw[:, 2] = 1.0 / (1.0 + np.exp(-raw[:, 2]))
            predictions.append(raw)
    return models, {
        "prediction_members": np.stack(predictions),
        "test_targets": test_y,
        "x_mean": x_mean,
        "x_std": x_std,
        "regression_mean": regression_mean,
        "regression_std": regression_std,
    }


def evaluate_fold(test_queries: list[dict], output: dict, time_ratio: float) -> dict:
    predictions = output["prediction_members"]
    targets = output["test_targets"]
    mean = predictions.mean(0)
    std = predictions.std(0)
    work_error = mean[:, 0] - targets[:, 0]
    time_error = mean[:, 1] - targets[:, 1]
    eligible_prediction = mean[:, 2] - std[:, 2] >= 0.5
    eligibility_accuracy = float(
        np.mean((mean[:, 2] >= 0.5) == (targets[:, 2] >= 0.5))
    )

    selected_outcomes = []
    direct_outcomes = []
    oracle_correct = []
    activations = 0
    position = 0
    for query in test_queries:
        candidate_indices = list(range(position, position + len(query["candidates"])))
        position += len(candidate_indices)
        feasible_indices = [
            index
            for index in candidate_indices
            if eligible_prediction[index]
            and mean[index, 1] + std[index, 1] <= time_ratio - 1.0
            and mean[index, 0] + std[index, 0] < 0.0
        ]
        direct = query["outcomes"][1.00]
        selected_scale = 1.00
        if feasible_indices:
            selected_index = min(feasible_indices, key=lambda index: mean[index, 0] + std[index, 0])
            selected_scale = query["candidates"][candidate_indices.index(selected_index)]["scale"]
            activations += 1
        selected = query["outcomes"][selected_scale]
        eligible_actual = [
            scale
            for scale, candidate in query["outcomes"].items()
            if candidate["success"] >= direct["success"]
            and candidate["time_s"] <= time_ratio * direct["time_s"] + 1e-9
        ]
        oracle_scale = min(
            eligible_actual,
            key=lambda scale: (
                query["outcomes"][scale]["work_j"],
                query["outcomes"][scale]["time_s"],
            ),
        )
        selected_outcomes.append(selected)
        direct_outcomes.append(direct)
        oracle_correct.append(selected_scale == oracle_scale)

    selected_work = float(np.mean([item["work_j"] for item in selected_outcomes]))
    direct_work = float(np.mean([item["work_j"] for item in direct_outcomes]))
    violations = [
        not (
            selected["success"] >= direct["success"]
            and selected["time_s"] <= time_ratio * direct["time_s"] + 1e-9
        )
        for selected, direct in zip(selected_outcomes, direct_outcomes)
    ]
    return {
        "candidate_regression": {
            "work_rel_mae": float(np.mean(np.abs(work_error))),
            "work_rel_bias": float(np.mean(work_error)),
            "time_rel_mae": float(np.mean(np.abs(time_error))),
            "eligibility_accuracy": eligibility_accuracy,
        },
        "selection": {
            "activation_rate": float(activations / len(test_queries)),
            "oracle_choice_accuracy": float(np.mean(oracle_correct)),
            "mean_work_j": selected_work,
            "mean_relative_work_vs_direct": float(selected_work / direct_work - 1.0),
            "mean_time_s": float(np.mean([item["time_s"] for item in selected_outcomes])),
            "success_rate": float(np.mean([item["success"] for item in selected_outcomes])),
            "constraint_violation_rate": float(np.mean(violations)),
        },
    }


def main() -> None:
    args = parse_args()
    import torch

    queries = load_queries(args.inputs, args.time_ratio)
    source_indices = sorted({query["source_index"] for query in queries})
    if len(source_indices) < 3:
        raise ValueError("At least three source seeds are required for this diagnostic.")
    folds = []
    for held_out in source_indices:
        train = [query for query in queries if query["source_index"] != held_out]
        test = [query for query in queries if query["source_index"] == held_out]
        _, output = train_ensemble(torch, train, test, args)
        fold = evaluate_fold(test, output, args.time_ratio)
        fold["held_out_input"] = args.inputs[held_out]
        fold["num_train_queries"] = len(train)
        fold["num_test_queries"] = len(test)
        folds.append(fold)
    selection_keys = (
        "activation_rate",
        "oracle_choice_accuracy",
        "mean_relative_work_vs_direct",
        "mean_time_s",
        "success_rate",
        "constraint_violation_rate",
    )
    candidate_keys = ("work_rel_mae", "work_rel_bias", "time_rel_mae", "eligibility_accuracy")
    payload = {
        "description": (
            "Fixed direct treatment-effect ensemble on exact-replay constant-prefix data. "
            "All outputs are leave-one-source-seed-out diagnostics, not tuned online results."
        ),
        "inputs": args.inputs,
        "time_ratio": args.time_ratio,
        "architecture": {
            "ensemble_size": args.ensemble_size,
            "hidden_dim": args.hidden_dim,
            "epochs": args.epochs,
            "selection_rule": (
                "candidate work UCB < 0, time UCB <= ratio-1, and eligibility probability LCB >= 0.5"
            ),
        },
        "num_queries": len(queries),
        "eligible_candidate_count": int(
            sum(candidate["eligible"] for query in queries for candidate in query["candidates"])
        ),
        "folds": folds,
        "mean_candidate_regression": {
            key: float(np.mean([fold["candidate_regression"][key] for fold in folds]))
            for key in candidate_keys
        },
        "mean_selection": {
            key: float(np.mean([fold["selection"][key] for fold in folds]))
            for key in selection_keys
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
