"""Train a recursive closed-loop macro-transition ensemble for command-sequence MPC."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--summary", default="")
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--members", type=int, default=5)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--test_fraction", type=float, default=0.15)
    p.add_argument("--validation_fraction", type=float, default=0.15)
    p.add_argument("--max_rollout_samples", type=int, default=2000)
    p.add_argument("--seed", type=int, default=2026)
    return p.parse_args()


def compose_motion(xy: np.ndarray, yaw: float, local_delta: np.ndarray):
    c, s = np.cos(yaw), np.sin(yaw)
    xy = xy + np.asarray(
        [c * local_delta[0] - s * local_delta[1], s * local_delta[0] + c * local_delta[1]]
    )
    yaw = float(np.arctan2(np.sin(yaw + local_delta[2]), np.cos(yaw + local_delta[2])))
    return xy, yaw


def main() -> None:
    args = parse_args()
    import torch

    data = np.load(args.dataset)
    required = {"X", "X_next", "Y", "G", "output_names", "step_index", "env_index"}
    missing = required - set(data.files)
    if missing:
        raise ValueError(f"Dataset is missing recursive-transition fields: {sorted(missing)}")
    x = data["X"].astype(np.float32)
    x_next = data["X_next"].astype(np.float32)
    y_response = data["Y"].astype(np.float32)
    groups = data["G"].astype(np.int64)
    output_names = data["output_names"].tolist()
    response_keep = [
        name
        for name in (
            "delta_x",
            "delta_y",
            "delta_yaw",
            "mechanical_power",
            "min_height",
            "max_tilt",
            "max_ang_speed",
        )
        if name in output_names
    ]
    required_response = {"delta_x", "delta_y", "delta_yaw", "mechanical_power", "min_height"}
    if not required_response.issubset(response_keep):
        raise ValueError(f"Dataset response labels must include {sorted(required_response)}")
    response_indices = [output_names.index(name) for name in response_keep]
    obs_delta = x_next - x
    target = np.concatenate([obs_delta, y_response[:, response_indices]], axis=1).astype(np.float32)

    rng = np.random.default_rng(args.seed)
    unique_groups = np.unique(groups)
    rng.shuffle(unique_groups)
    n_test = max(1, int(len(unique_groups) * args.test_fraction))
    n_val = max(1, int(len(unique_groups) * args.validation_fraction))
    test_groups = unique_groups[:n_test]
    val_groups = unique_groups[n_test : n_test + n_val]
    train_groups = unique_groups[n_test + n_val :]
    train_mask = np.isin(groups, train_groups)
    val_mask = np.isin(groups, val_groups)
    test_mask = np.isin(groups, test_groups)
    x_mean = x[train_mask].mean(0)
    x_std = np.maximum(x[train_mask].std(0), 1e-6)
    y_mean = target[train_mask].mean(0)
    y_std = np.maximum(target[train_mask].std(0), 1e-6)

    def normalize_x(values):
        return (values - x_mean) / x_std

    def normalize_y(values):
        return (values - y_mean) / y_std

    device = "cuda" if torch.cuda.is_available() else "cpu"

    def make_model():
        return torch.nn.Sequential(
            torch.nn.Linear(x.shape[1], args.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(args.hidden_dim, args.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(args.hidden_dim, target.shape[1]),
        ).to(device)

    xv = torch.from_numpy(normalize_x(x[val_mask])).float().to(device)
    yv = torch.from_numpy(normalize_y(target[val_mask])).float().to(device)
    train_group_values = np.unique(groups[train_mask])
    state_dicts = []
    best_losses = []
    for member in range(args.members):
        member_seed = args.seed + 1009 * member
        torch.manual_seed(member_seed)
        member_rng = np.random.default_rng(member_seed)
        bootstrap_groups = member_rng.choice(
            train_group_values, size=len(train_group_values), replace=True
        )
        bootstrap_indices = np.concatenate(
            [np.flatnonzero(groups == group) for group in bootstrap_groups]
        )
        xt = torch.from_numpy(normalize_x(x[bootstrap_indices])).float().to(device)
        yt = torch.from_numpy(normalize_y(target[bootstrap_indices])).float().to(device)
        model = make_model()
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        best_loss = float("inf")
        best_state = None
        for epoch in range(args.epochs):
            model.train()
            perm = torch.randperm(len(xt), device=device)
            for start in range(0, len(xt), args.batch_size):
                idx = perm[start : start + args.batch_size]
                pred = model(xt[idx])
                loss = torch.nn.functional.mse_loss(pred, yt[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
            model.eval()
            with torch.inference_mode():
                val_loss = float(torch.nn.functional.mse_loss(model(xv), yv).item())
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = copy.deepcopy(
                    {key: value.detach().cpu() for key, value in model.state_dict().items()}
                )
        state_dicts.append(best_state)
        best_losses.append(best_loss)
        print(f"member {member}: best validation loss {best_loss:.4f}")

    models = []
    for state in state_dicts:
        model = make_model()
        model.load_state_dict(state)
        model.eval()
        models.append(model)

    def predict_members(states: np.ndarray):
        tensor = torch.from_numpy(normalize_x(states).astype(np.float32)).to(device)
        with torch.inference_mode():
            normalized = np.stack([model(tensor).cpu().numpy() for model in models])
        return normalized * y_std[None, None, :] + y_mean[None, None, :]

    one_step_members = predict_members(x[test_mask])
    one_step_pred = one_step_members.mean(0)
    one_step_metrics = {}
    for index, name in enumerate(
        [f"next_obs_delta_{i}" for i in range(x.shape[1])] + response_keep
    ):
        error = one_step_pred[:, index] - target[test_mask, index]
        if name.startswith("next_obs_delta_"):
            continue
        true = target[test_mask, index]
        one_step_metrics[name] = {
            "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(error**2))),
            "r2": float(
                1.0 - np.sum(error**2) / max(np.sum((true - true.mean()) ** 2), 1e-12)
            ),
        }
    one_step_metrics["next_observation_normalized_rmse"] = float(
        np.sqrt(
            np.mean(
                ((one_step_pred[:, : x.shape[1]] - obs_delta[test_mask]) / y_std[: x.shape[1]])
                ** 2
            )
        )
    )

    # Open-loop evaluation with the actual future command sequence.  Each
    # ensemble member recursively advances its own predicted observation.
    step_index = data["step_index"].astype(np.int64)
    env_index = data["env_index"].astype(np.int64)
    horizon = int(data["horizon"])
    command_slice = data["command_slice"].astype(int).tolist()
    lookup = {
        (int(env_index[i]), int(step_index[i]), int(groups[i])): i for i in range(len(x))
    }
    test_indices = np.flatnonzero(test_mask)
    rng.shuffle(test_indices)
    test_indices = test_indices[: args.max_rollout_samples]
    response_offset = x.shape[1]
    response_map = {name: response_offset + i for i, name in enumerate(response_keep)}
    duration = horizon * float(data["control_dt"])
    rollout_metrics = {}
    for rollout_steps in (1, 2, 4, 8):
        displacement_errors = []
        yaw_errors = []
        work_errors = []
        min_height_errors = []
        state_errors = []
        for start_idx in test_indices:
            indices = []
            for step in range(rollout_steps):
                key = (
                    int(env_index[start_idx]),
                    int(step_index[start_idx] + step * horizon),
                    int(groups[start_idx]),
                )
                if key not in lookup:
                    indices = []
                    break
                indices.append(lookup[key])
            if not indices or not all(test_mask[index] for index in indices):
                continue
            member_states = np.repeat(x[start_idx][None, :], args.members, axis=0)
            pred_xy = np.zeros((args.members, 2), dtype=np.float64)
            pred_yaw = np.zeros(args.members, dtype=np.float64)
            pred_work = np.zeros(args.members, dtype=np.float64)
            pred_min_height = np.full(args.members, np.inf, dtype=np.float64)
            true_xy = np.zeros(2, dtype=np.float64)
            true_yaw = 0.0
            true_work = 0.0
            true_min_height = np.inf
            for index in indices:
                lo, hi = command_slice
                member_states[:, lo:hi] = x[index, lo:hi]
                member_outputs = []
                for member, model in enumerate(models):
                    tensor = torch.from_numpy(normalize_x(member_states[member : member + 1])).float().to(device)
                    with torch.inference_mode():
                        value = model(tensor).cpu().numpy()[0] * y_std + y_mean
                    member_outputs.append(value)
                member_outputs = np.stack(member_outputs)
                member_states = member_states + member_outputs[:, : x.shape[1]]
                for member in range(args.members):
                    local = np.asarray(
                        [
                            member_outputs[member, response_map["delta_x"]],
                            member_outputs[member, response_map["delta_y"]],
                            member_outputs[member, response_map["delta_yaw"]],
                        ]
                    )
                    pred_xy[member], pred_yaw[member] = compose_motion(
                        pred_xy[member], pred_yaw[member], local
                    )
                pred_work += member_outputs[:, response_map["mechanical_power"]] * duration
                pred_min_height = np.minimum(
                    pred_min_height, member_outputs[:, response_map["min_height"]]
                )
                true_response = target[index]
                true_local = np.asarray(
                    [
                        true_response[response_map["delta_x"]],
                        true_response[response_map["delta_y"]],
                        true_response[response_map["delta_yaw"]],
                    ]
                )
                true_xy, true_yaw = compose_motion(true_xy, true_yaw, true_local)
                true_work += true_response[response_map["mechanical_power"]] * duration
                true_min_height = min(
                    true_min_height, true_response[response_map["min_height"]]
                )
            pred_xy_mean = pred_xy.mean(0)
            pred_yaw_mean = float(np.arctan2(np.sin(pred_yaw).mean(), np.cos(pred_yaw).mean()))
            displacement_errors.append(float(np.linalg.norm(pred_xy_mean - true_xy)))
            yaw_errors.append(
                abs(float(np.arctan2(np.sin(pred_yaw_mean - true_yaw), np.cos(pred_yaw_mean - true_yaw))))
            )
            work_errors.append(abs(float(pred_work.mean() - true_work)))
            min_height_errors.append(abs(float(pred_min_height.mean() - true_min_height)))
            true_final = x_next[indices[-1]]
            state_errors.append(
                float(np.sqrt(np.mean(((member_states.mean(0) - true_final) / x_std) ** 2)))
            )
        rollout_metrics[str(rollout_steps)] = {
            "n": len(displacement_errors),
            "duration_s": rollout_steps * duration,
            "displacement_mae_m": float(np.mean(displacement_errors)),
            "yaw_mae_rad": float(np.mean(yaw_errors)),
            "work_mae_j": float(np.mean(work_errors)),
            "min_height_mae_m": float(np.mean(min_height_errors)),
            "state_normalized_rmse": float(np.mean(state_errors)),
        }

    payload = {
        "model_type": "macro_transition_ensemble",
        "state_dicts": state_dicts,
        "input_dim": int(x.shape[1]),
        "hidden_dim": args.hidden_dim,
        "output_dim": int(target.shape[1]),
        "response_names": response_keep,
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "horizon": horizon,
        "control_dt": float(data["control_dt"]),
        "command_slice": command_slice,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    summary = {
        "dataset": args.dataset,
        "output": args.output,
        "num_samples": int(len(x)),
        "num_episode_groups": int(len(unique_groups)),
        "num_train": int(train_mask.sum()),
        "num_validation": int(val_mask.sum()),
        "num_test": int(test_mask.sum()),
        "member_best_validation_losses": best_losses,
        "one_step_metrics": one_step_metrics,
        "rollout_metrics": rollout_metrics,
    }
    summary_path = args.summary or str(Path(args.output).with_suffix("")) + "_summary.json"
    Path(summary_path).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
