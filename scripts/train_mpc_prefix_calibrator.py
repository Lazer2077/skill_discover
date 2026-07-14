"""Train an exact-replay calibrator for recursive MPC prefix outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


OUTCOME_NAMES = [
    "work_j",
    "progress_m",
    "min_height",
    "max_tilt",
    "max_ang_speed",
]
TARGET_KEYS = [
    "prefix_work_j",
    "prefix_progress_m",
    "prefix_min_relative_height",
    "prefix_max_tilt",
    "prefix_max_ang_speed",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", default="")
    parser.add_argument("--validation_input", type=int, default=-1)
    parser.add_argument("--height_scan_slice", default="48:235")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--ensemble_size", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ranking_weight", type=float, default=0.5)
    parser.add_argument(
        "--target_mode",
        choices=("absolute", "residual"),
        default="residual",
        help="Predict absolute outcomes or residuals relative to the macro-model mean.",
    )
    parser.add_argument("--seed", type=int, default=3050)
    return parser.parse_args()


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


def sample_feature(sample: dict, scan_slice: tuple[int, int]) -> np.ndarray:
    obs = np.asarray(sample["query_observation"], dtype=np.float32)
    local_target = np.asarray(sample["query_local_target"], dtype=np.float32)
    sequence = np.asarray(sample["candidate_sequence"], dtype=np.float32)
    lo, hi = scan_slice
    predicted = sample["predicted_prefix"]
    predicted_stats = []
    for name in OUTCOME_NAMES:
        values = np.asarray(predicted[name], dtype=np.float32)
        predicted_stats.extend([float(values.mean()), float(values.std())])
    return np.concatenate(
        [
            obs[:48],
            scan_summary(obs[lo:hi]),
            local_target,
            np.asarray(
                [np.linalg.norm(local_target), sample["query_start_relative_height"]],
                dtype=np.float32,
            ),
            sequence,
            np.asarray(predicted_stats, dtype=np.float32),
        ]
    )


def make_model(torch, input_dim: int, hidden_dim: int, output_dim: int):
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, output_dim),
    )


def make_work_pairs(indices: np.ndarray, groups: list[str], targets: np.ndarray):
    local = {int(global_index): position for position, global_index in enumerate(indices)}
    grouped: dict[str, list[int]] = {}
    for global_index in indices:
        grouped.setdefault(groups[int(global_index)], []).append(int(global_index))
    left, right, signs = [], [], []
    for values in grouped.values():
        for position, i in enumerate(values):
            for j in values[position + 1 :]:
                difference = float(targets[i, 0] - targets[j, 0])
                if abs(difference) < 1e-8:
                    continue
                left.append(local[i])
                right.append(local[j])
                signs.append(np.sign(difference))
    return np.asarray(left), np.asarray(right), np.asarray(signs, dtype=np.float32)


def main() -> None:
    args = parse_args()
    import torch

    lo, hi = (int(value) for value in args.height_scan_slice.split(":"))
    validation_index = args.validation_input % len(args.inputs)
    xs, ys, baselines, source_indices, ranking_groups = [], [], [], [], []
    source_counts = {}
    for source_index, path in enumerate(args.inputs):
        data = json.loads(Path(path).read_text())
        samples = data.get("mpc_dagger_samples", [])
        source_counts[path] = len(samples)
        for sample in samples:
            if "predicted_prefix" not in sample:
                continue
            xs.append(sample_feature(sample, (lo, hi)))
            ys.append(np.asarray([sample[key] for key in TARGET_KEYS], dtype=np.float32))
            baselines.append(
                np.asarray(
                    [np.mean(sample["predicted_prefix"][name]) for name in OUTCOME_NAMES],
                    dtype=np.float32,
                )
            )
            source_indices.append(source_index)
            ranking_groups.append(
                f"{source_index}|{sample['target']}|{sample['trial']}|{sample['query_step']}"
            )
    if not xs:
        raise ValueError("No prefix-labeled DAgger samples were found.")
    X = np.stack(xs)
    Y = np.stack(ys)
    baseline = np.stack(baselines)
    source_indices = np.asarray(source_indices)
    train_mask = source_indices != validation_index
    validation_mask = source_indices == validation_index
    if not train_mask.any() or not validation_mask.any():
        raise ValueError("Both training and validation inputs must contain samples.")

    x_mean = X[train_mask].mean(0)
    x_std = np.maximum(X[train_mask].std(0), 1e-6)
    learning_target = Y - baseline if args.target_mode == "residual" else Y
    y_mean = learning_target[train_mask].mean(0)
    y_std = np.maximum(learning_target[train_mask].std(0), 1e-6)
    Xn = (X - x_mean) / x_std
    Yn = (learning_target - y_mean) / y_std

    device = "cuda" if torch.cuda.is_available() else "cpu"
    xv = torch.from_numpy(Xn[validation_mask]).float().to(device)
    yv = torch.from_numpy(Yn[validation_mask]).float().to(device)
    train_indices = np.flatnonzero(train_mask)
    validation_indices = np.flatnonzero(validation_mask)
    val_left, val_right, val_sign = make_work_pairs(
        validation_indices, ranking_groups, Y
    )
    val_left_t = torch.from_numpy(val_left).long().to(device)
    val_right_t = torch.from_numpy(val_right).long().to(device)
    val_sign_t = torch.from_numpy(val_sign).float().to(device)
    state_dicts = []
    validation_losses = []
    for member in range(args.ensemble_size):
        member_seed = args.seed + member
        torch.manual_seed(member_seed)
        rng = np.random.default_rng(member_seed)
        retained = np.sort(
            rng.choice(
                train_indices,
                size=max(2, int(0.9 * len(train_indices))),
                replace=False,
            )
        )
        train_left, train_right, train_sign = make_work_pairs(
            retained, ranking_groups, Y
        )
        xt = torch.from_numpy(Xn[retained]).float().to(device)
        yt = torch.from_numpy(Yn[retained]).float().to(device)
        train_baseline_work_t = torch.from_numpy(baseline[retained, 0]).float().to(device)
        train_left_t = torch.from_numpy(train_left).long().to(device)
        train_right_t = torch.from_numpy(train_right).long().to(device)
        train_sign_t = torch.from_numpy(train_sign).float().to(device)
        model = make_model(torch, X.shape[1], args.hidden_dim, Y.shape[1]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        best_loss = float("inf")
        best_state = None
        stale = 0
        for _epoch in range(args.epochs):
            model.train()
            prediction = model(xt)
            regression_loss = torch.nn.functional.smooth_l1_loss(prediction, yt)
            predicted_work = (
                prediction[:, 0] * float(y_std[0])
                + float(y_mean[0])
                + (
                    train_baseline_work_t
                    if args.target_mode == "residual"
                    else 0.0
                )
            ) / float(np.maximum(Y[train_mask, 0].std(), 1e-6))
            ranking_loss = torch.nn.functional.softplus(
                -train_sign_t
                * (predicted_work[train_left_t] - predicted_work[train_right_t])
            ).mean()
            loss = regression_loss + args.ranking_weight * ranking_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.eval()
            with torch.inference_mode():
                validation_prediction = model(xv)
                validation_regression = torch.nn.functional.smooth_l1_loss(
                    validation_prediction, yv
                )
                validation_predicted_work = (
                    validation_prediction[:, 0] * float(y_std[0])
                    + float(y_mean[0])
                    + (
                        torch.from_numpy(baseline[validation_mask, 0]).float().to(device)
                        if args.target_mode == "residual"
                        else 0.0
                    )
                ) / float(np.maximum(Y[validation_mask, 0].std(), 1e-6))
                validation_ranking = torch.nn.functional.softplus(
                    -val_sign_t
                    * (
                        validation_predicted_work[val_left_t]
                        - validation_predicted_work[val_right_t]
                    )
                ).mean()
                validation_loss = float(
                    (
                        validation_regression
                        + args.ranking_weight * validation_ranking
                    ).item()
                )
            if validation_loss < best_loss - 1e-5:
                best_loss = validation_loss
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
            if stale >= 50:
                break
        state_dicts.append(best_state)
        validation_losses.append(best_loss)
        print(f"member {member}: validation loss={best_loss:.4f}")

    predictions = []
    for state_dict in state_dicts:
        model = make_model(torch, X.shape[1], args.hidden_dim, Y.shape[1]).to(device)
        model.load_state_dict(state_dict)
        model.eval()
        with torch.inference_mode():
            raw = model(xv).cpu().numpy()
        prediction = raw * y_std + y_mean
        if args.target_mode == "residual":
            prediction = prediction + baseline[validation_mask]
        predictions.append(prediction)
    ensemble_predictions = np.stack(predictions)
    prediction_mean = ensemble_predictions.mean(0)
    prediction_std = ensemble_predictions.std(0)
    truth = Y[validation_mask]
    baseline_truth = baseline[validation_mask]

    def regression_metrics(prediction: np.ndarray, include_std: bool) -> dict:
        result = {}
        for index, name in enumerate(OUTCOME_NAMES):
            error = prediction[:, index] - truth[:, index]
            result[name] = {
                "mae": float(np.mean(np.abs(error))),
                "rmse": float(np.sqrt(np.mean(error**2))),
                "bias": float(np.mean(error)),
                "r2": float(
                    1.0
                    - np.sum(error**2)
                    / max(
                        np.sum((truth[:, index] - truth[:, index].mean()) ** 2),
                        1e-12,
                    )
                ),
            }
            if include_std:
                result[name]["mean_ensemble_std"] = float(
                    prediction_std[:, index].mean()
                )
        return result

    metrics = regression_metrics(prediction_mean, include_std=True)
    baseline_metrics = regression_metrics(baseline_truth, include_std=False)

    validation_groups = np.asarray(ranking_groups)[validation_mask]
    def ranking_metrics(prediction: np.ndarray) -> dict:
        pair_correct = pair_total = best_correct = group_total = 0
        for group in dict.fromkeys(validation_groups.tolist()):
            indices = np.flatnonzero(validation_groups == group)
            if len(indices) < 2:
                continue
            group_total += 1
            best_correct += int(
                indices[np.argmin(prediction[indices, 0])]
                == indices[np.argmin(truth[indices, 0])]
            )
            for i_position, i in enumerate(indices):
                for j in indices[i_position + 1 :]:
                    true_difference = truth[i, 0] - truth[j, 0]
                    if abs(true_difference) < 1e-8:
                        continue
                    pair_total += 1
                    pair_correct += int(
                        np.sign(true_difference)
                        == np.sign(prediction[i, 0] - prediction[j, 0])
                    )
        return {
            "work_pairwise_accuracy": float(pair_correct / max(pair_total, 1)),
            "work_best_sequence_accuracy": float(best_correct / max(group_total, 1)),
            "num_pairs": pair_total,
            "num_query_groups": group_total,
        }

    ranking = ranking_metrics(prediction_mean)
    baseline_ranking = ranking_metrics(baseline_truth)
    print(
        json.dumps(
            {
                "baseline_metrics": baseline_metrics,
                "metrics": metrics,
                "baseline_ranking": baseline_ranking,
                "ranking": ranking,
            },
            indent=2,
        )
    )

    payload = {
        "model_type": "mpc_prefix_calibrator_ensemble",
        "state_dicts": state_dicts,
        "input_dim": int(X.shape[1]),
        "output_names": OUTCOME_NAMES,
        "hidden_dim": args.hidden_dim,
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "height_scan_slice": [lo, hi],
        "target_mode": args.target_mode,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    summary = {
        "inputs": args.inputs,
        "validation_input": args.inputs[validation_index],
        "source_sample_counts": source_counts,
        "num_train": int(train_mask.sum()),
        "num_validation": int(validation_mask.sum()),
        "validation_losses": validation_losses,
        "ranking_weight": args.ranking_weight,
        "target_mode": args.target_mode,
        "baseline_metrics": baseline_metrics,
        "metrics": metrics,
        "baseline_ranking": baseline_ranking,
        "ranking": ranking,
        "output": str(output),
    }
    summary_path = args.summary or str(output.with_suffix("")) + "_summary.json"
    Path(summary_path).write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
