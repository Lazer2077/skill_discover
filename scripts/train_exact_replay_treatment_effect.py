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
    parser.add_argument(
        "--feature-set",
        choices=("observation", "observation_fphi", "observation_macro", "observation_fphi_macro"),
        default="observation_macro",
        help="Deployment-time representation used by the treatment-effect ensemble.",
    )
    parser.add_argument(
        "--response-model",
        default="",
        help="Command-response ensemble used for feature sets containing fphi.",
    )
    parser.add_argument(
        "--control-mode",
        choices=("none", "observable", "hidden"),
        default="none",
        help="Semi-synthetic calibration: inject equal work benefits using an observable or hidden preferred scale.",
    )
    parser.add_argument(
        "--control-strength",
        type=float,
        default=0.0,
        help="Fractional work reduction injected into the preferred non-direct scale.",
    )
    parser.add_argument("--control-seed", type=int, default=20260717)
    parser.add_argument("--time-ratio", type=float, default=1.10)
    parser.add_argument("--ensemble-size", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=4050)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--height-scan-slice",
        default="48:235",
        help="Observation slice for terrain-relative height scan (Go2 48:235, H1 69:256).",
    )
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


def load_response_predictor(torch, path: str):
    if not path:
        raise ValueError("--response-model is required for feature sets containing fphi.")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if "state_dicts" not in payload:
        raise ValueError(f"{path} is not a command-response ensemble.")
    models = []
    for state_dict in payload["state_dicts"]:
        model = torch.nn.Sequential(
            torch.nn.Linear(payload["obs_dim"], payload["hidden_dim"]),
            torch.nn.ReLU(),
            torch.nn.Dropout(payload.get("dropout", 0.0)),
            torch.nn.Linear(payload["hidden_dim"], payload["hidden_dim"]),
            torch.nn.ReLU(),
            torch.nn.Dropout(payload.get("dropout", 0.0)),
            torch.nn.Linear(payload["hidden_dim"], len(payload["output_names"])),
        )
        model.load_state_dict(state_dict)
        model.eval()
        models.append(model)
    x_mean = np.asarray(payload["x_mean"], dtype=np.float32)
    x_std = np.asarray(payload["x_std"], dtype=np.float32)
    y_mean = np.asarray(payload["y_mean"], dtype=np.float32)
    y_std = np.asarray(payload["y_std"], dtype=np.float32)
    command_slice = tuple(int(value) for value in payload["command_slice"])

    def predict(observation: np.ndarray, command: np.ndarray) -> np.ndarray:
        value = np.asarray(observation, dtype=np.float32).copy()
        lo, hi = command_slice
        value[lo:hi] = command[: hi - lo]
        tensor = torch.from_numpy(((value - x_mean) / x_std)[None, :]).float()
        with torch.inference_mode():
            members = np.stack([model(tensor).numpy()[0] for model in models])
        return members * y_std[None, :] + y_mean[None, :]

    return predict


def target_command(target: np.ndarray, scale: float) -> np.ndarray:
    command = np.zeros(3, dtype=np.float32)
    command[:2] = np.clip(target[:2], -1.0, 1.0)
    command[2] = np.clip(
        np.arctan2(target[1], max(float(target[0]), 1e-6)), -1.0, 1.0
    )
    return np.clip(command * scale, -1.0, 1.0)


def parse_slice(raw: str) -> tuple[int, int]:
    start_s, end_s = raw.split(":")
    start, end = int(start_s), int(end_s)
    if end <= start:
        raise ValueError(f"Invalid slice '{raw}'")
    return start, end


def feature(
    sample: dict,
    direct: dict,
    scale: float,
    feature_set: str,
    response_predictor=None,
    height_scan_slice: tuple[int, int] = (48, 235),
) -> np.ndarray:
    observation = np.asarray(sample["query_observation"], dtype=np.float32)
    target = np.asarray(sample["query_local_target"], dtype=np.float32)
    scan_lo, scan_hi = height_scan_slice
    if observation.shape[0] < scan_hi:
        raise ValueError(
            f"Observation dim {observation.shape[0]} is shorter than scan end {scan_hi}."
        )
    state = np.concatenate(
        [
            observation[:scan_lo],
            scan_summary(observation[scan_lo:scan_hi]),
            target,
            np.asarray(
                [np.linalg.norm(target), sample["query_start_relative_height"]],
                dtype=np.float32,
            ),
        ]
    )
    components = [state, np.asarray([scale], dtype=np.float32)]
    if "fphi" in feature_set:
        candidate_response = response_predictor(
            observation, target_command(target, scale)
        )
        direct_response = response_predictor(
            observation, target_command(target, 1.0)
        )
        fphi_comparison = np.stack(
            [
                candidate_response.mean(0) - direct_response.mean(0),
                candidate_response.std(0),
                direct_response.std(0),
            ],
            axis=1,
        ).reshape(-1)
        components.append(fphi_comparison.astype(np.float32))
    if "macro" in feature_set:
        macro_comparison = []
        for name in OUTCOME_NAMES:
            candidate_values = np.asarray(sample["predicted_prefix"][name], dtype=np.float32)
            direct_values = np.asarray(direct["predicted_prefix"][name], dtype=np.float32)
            macro_comparison.extend(
                [
                    float(candidate_values.mean() - direct_values.mean()),
                    float(candidate_values.std()),
                    float(direct_values.std()),
                ]
            )
        components.append(np.asarray(macro_comparison, dtype=np.float32))
    return np.concatenate(components)


def load_queries(
    paths: list[str],
    time_ratio: float,
    feature_set: str,
    response_predictor=None,
    control_mode: str = "none",
    control_strength: float = 0.0,
    control_seed: int = 20260717,
    height_scan_slice: tuple[int, int] = (48, 235),
) -> list[dict]:
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
            if control_mode != "none" and control_strength > 0.0:
                if control_mode == "observable":
                    target = np.asarray(samples_by_scale[1.00]["query_local_target"])
                    preferred_scale = 0.90 if float(target[1]) >= 0.0 else 0.75
                else:
                    control_rng = np.random.default_rng(
                        control_seed + source_index * 100_003 + group_index
                    )
                    preferred_scale = float(control_rng.choice(SCALES[:-1]))
                outcomes[preferred_scale] = dict(outcomes[preferred_scale])
                outcomes[preferred_scale]["work_j"] *= 1.0 - control_strength
            direct = outcomes[1.00]
            candidates = []
            for scale in SCALES[:-1]:
                candidate = outcomes[scale]
                candidates.append(
                    {
                        "scale": scale,
                        "feature": feature(
                            samples_by_scale[scale],
                            samples_by_scale[1.00],
                            scale,
                            feature_set,
                            response_predictor,
                            height_scan_slice=height_scan_slice,
                        ),
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
    selected_scales = []
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
        selected_scales.append(selected_scale)
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
    choice_counts = Counter(selected_scales)
    choice_probabilities = {
        scale: choice_counts.get(scale, 0) / len(test_queries) for scale in SCALES
    }
    # A matched randomized mixture uses exactly the selector's marginal action
    # frequencies but allocates them independently of state.  Any improvement of
    # the learned selector over this expectation is therefore attributable to
    # state-conditioned allocation, rather than merely slowing more often.
    mixture_work = float(
        sum(
            probability
            * np.mean([query["outcomes"][scale]["work_j"] for query in test_queries])
            for scale, probability in choice_probabilities.items()
        )
    )
    mixture_time = float(
        sum(
            probability
            * np.mean([query["outcomes"][scale]["time_s"] for query in test_queries])
            for scale, probability in choice_probabilities.items()
        )
    )
    mixture_success = float(
        sum(
            probability
            * np.mean([query["outcomes"][scale]["success"] for query in test_queries])
            for scale, probability in choice_probabilities.items()
        )
    )
    mixture_violation = float(
        sum(
            probability
            * np.mean(
                [
                    not (
                        query["outcomes"][scale]["success"]
                        >= query["outcomes"][1.00]["success"]
                        and query["outcomes"][scale]["time_s"]
                        <= time_ratio * query["outcomes"][1.00]["time_s"] + 1e-9
                    )
                    for query in test_queries
                ]
            )
            for scale, probability in choice_probabilities.items()
        )
    )
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
            "choice_counts": {f"{scale:.2f}": choice_counts.get(scale, 0) for scale in SCALES},
        },
        "matched_randomized_mixture": {
            "definition": "Same marginal scale frequencies as the learned selector, allocated independently of state.",
            "mean_work_j": mixture_work,
            "mean_relative_work_vs_direct": float(mixture_work / direct_work - 1.0),
            "mean_time_s": mixture_time,
            "success_rate": mixture_success,
            "constraint_violation_rate": mixture_violation,
            "selector_relative_work_vs_mixture": float(selected_work / mixture_work - 1.0),
        },
    }


def main() -> None:
    args = parse_args()
    import torch

    response_predictor = (
        load_response_predictor(torch, args.response_model)
        if "fphi" in args.feature_set
        else None
    )
    queries = load_queries(
        args.inputs,
        args.time_ratio,
        args.feature_set,
        response_predictor,
        args.control_mode,
        args.control_strength,
        args.control_seed,
        height_scan_slice=parse_slice(args.height_scan_slice),
    )
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
    mixture_keys = (
        "mean_relative_work_vs_direct",
        "mean_time_s",
        "success_rate",
        "constraint_violation_rate",
        "selector_relative_work_vs_mixture",
    )
    payload = {
        "description": (
            "Fixed direct treatment-effect ensemble on exact-replay constant-prefix data. "
            "All outputs are leave-one-source-seed-out diagnostics, not tuned online results."
        ),
        "inputs": args.inputs,
        "time_ratio": args.time_ratio,
        "feature_set": args.feature_set,
        "response_model": args.response_model if "fphi" in args.feature_set else "",
        "semi_synthetic_control": {
            "mode": args.control_mode,
            "strength": args.control_strength,
            "seed": args.control_seed,
            "definition": (
                "Preferred scale is 0.90 for nonnegative target-y and 0.75 otherwise."
                if args.control_mode == "observable"
                else (
                    "Preferred scale is deterministic-random per query and absent from features."
                    if args.control_mode == "hidden"
                    else "No outcome modification."
                )
            ),
        },
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
        "mean_matched_randomized_mixture": {
            key: float(np.mean([fold["matched_randomized_mixture"][key] for fold in folds]))
            for key in mixture_keys
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
