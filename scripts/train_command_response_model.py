"""Train a command-response model of a converged policy's closed-loop behavior.

The model plays the role of the plant model in classical feedforward
compensation: given the current observation and a candidate velocity command,
it predicts the short-horizon closed-loop response of (policy + robot):
body-frame displacement, yaw change, per-step energy, and height statistics.
At deployment the compensator inverts this model over a candidate command set
to pre-correct the direct target command.

Training data comes from the same converged-policy rollouts that populate the
online action archive: sliding windows over archived segments in which the
sampled training command is constant.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

OUTPUT_NAMES = ["delta_x", "delta_y", "delta_yaw", "energy", "min_height", "mean_height"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train command-response model from archived rollouts.")
    parser.add_argument("--online_action_set", type=str, default="")
    parser.add_argument("--dataset", type=str, default="",
                        help="npz dataset from collect_response_dataset.py (takes precedence).")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--data_fraction", type=float, default=1.0,
                        help="Train on this fraction of episode groups (for identification data-efficiency studies).")
    parser.add_argument("--horizon", type=int, default=16, help="Prediction horizon in low-level steps.")
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--command_slice", type=str, default="9:12")
    parser.add_argument("--command_tol", type=float, default=1e-5,
                        help="Max command variation within a window for it to count as constant.")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--test_fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--summary", type=str, default="")
    return parser.parse_args()


def quat_to_yaw(quat: np.ndarray) -> float:
    w, x, y, z = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def build_dataset(action_set, horizon: int, stride: int, cmd_lo: int, cmd_hi: int, tol: float):
    inputs: List[np.ndarray] = []
    outputs: List[np.ndarray] = []
    groups: List[int] = []
    skipped_cmd = 0
    for seg in action_set._archive_segments:
        obs_seq = np.asarray(seg["obs_seq"], dtype=np.float32)
        act_seq = np.asarray(seg["action_seq"], dtype=np.float32)
        pos_seq = np.asarray(seg["base_pos_seq"], dtype=np.float32)
        quat_seq = np.asarray(seg["base_quat_seq"], dtype=np.float32)
        jvel_seq = np.asarray(seg["joint_vel_seq"], dtype=np.float32)
        T = obs_seq.shape[0]
        group = int(seg.get("episode_id", 0)) * 100000 + int(seg.get("env_id", 0))
        for t0 in range(0, T - horizon, stride):
            window_cmd = obs_seq[t0 : t0 + horizon, cmd_lo:cmd_hi]
            if float(np.max(np.abs(window_cmd - window_cmd[0]))) > tol:
                skipped_cmd += 1
                continue
            yaw0 = quat_to_yaw(quat_seq[t0])
            yaw1 = quat_to_yaw(quat_seq[t0 + horizon])
            d_world = pos_seq[t0 + horizon, :2] - pos_seq[t0, :2]
            c, s = np.cos(-yaw0), np.sin(-yaw0)
            dx = c * d_world[0] - s * d_world[1]
            dy = s * d_world[0] + c * d_world[1]
            dyaw = float(np.arctan2(np.sin(yaw1 - yaw0), np.cos(yaw1 - yaw0)))
            a = act_seq[t0 : t0 + horizon]
            jv = jvel_seq[t0 : t0 + horizon]
            m = min(a.shape[1], jv.shape[1])
            energy = float(np.mean(np.abs(a[:, :m] * jv[:, :m])))
            heights = pos_seq[t0 : t0 + horizon + 1, 2]
            inputs.append(obs_seq[t0])
            outputs.append(
                np.asarray([dx, dy, dyaw, energy, float(np.min(heights)), float(np.mean(heights))], dtype=np.float32)
            )
            groups.append(group)
    X = np.stack(inputs)
    Y = np.stack(outputs)
    G = np.asarray(groups)
    return X, Y, G, skipped_cmd


def main() -> None:
    args = parse_args()
    import torch

    from skill_discovery.online.online_action_set import OnlineActionSet

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cmd_lo, cmd_hi = (int(v) for v in args.command_slice.split(":"))
    if args.dataset:
        data = np.load(args.dataset)
        X, Y, G = data["X"], data["Y"], data["G"]
        output_names = data["output_names"].tolist() if "output_names" in data else OUTPUT_NAMES
        skipped = 0
        args.horizon = int(data["horizon"]) if "horizon" in data else args.horizon
    else:
        if not args.online_action_set:
            raise ValueError("Provide --dataset or --online_action_set.")
        action_set = OnlineActionSet.load(args.online_action_set)
        X, Y, G, skipped = build_dataset(action_set, args.horizon, args.stride, cmd_lo, cmd_hi, args.command_tol)
        output_names = OUTPUT_NAMES
    print(f"dataset: {X.shape[0]} samples, {X.shape[1]}-dim obs, {len(np.unique(G))} episode groups, "
          f"{skipped} windows skipped (command changed)")

    rng = np.random.default_rng(args.seed)
    unique_groups = np.unique(G)
    rng.shuffle(unique_groups)
    n_test_groups = max(1, int(len(unique_groups) * args.test_fraction))
    test_groups = set(unique_groups[:n_test_groups].tolist())
    train_groups = unique_groups[n_test_groups:]
    if args.data_fraction < 1.0:
        n_keep = max(1, int(len(train_groups) * args.data_fraction))
        train_groups = train_groups[:n_keep]
        print(f"data_fraction={args.data_fraction}: {n_keep}/{len(unique_groups)-n_test_groups} train episode groups")
    train_group_set = set(train_groups.tolist())
    test_mask = np.asarray([g in test_groups for g in G])
    train_mask = np.asarray([g in train_group_set for g in G])

    x_mean = X[train_mask].mean(axis=0)
    x_std = np.maximum(X[train_mask].std(axis=0), 1e-6)
    y_mean = Y[train_mask].mean(axis=0)
    y_std = np.maximum(Y[train_mask].std(axis=0), 1e-6)

    Xn = (X - x_mean) / x_std
    Yn = (Y - y_mean) / y_std

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.nn.Sequential(
        torch.nn.Linear(X.shape[1], args.hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Dropout(args.dropout),
        torch.nn.Linear(args.hidden_dim, args.hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Dropout(args.dropout),
        torch.nn.Linear(args.hidden_dim, Y.shape[1]),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    xt = torch.from_numpy(Xn[train_mask]).float().to(device)
    yt = torch.from_numpy(Yn[train_mask]).float().to(device)
    xv = torch.from_numpy(Xn[test_mask]).float().to(device)
    yv = torch.from_numpy(Yn[test_mask]).float().to(device)

    best_val = float("inf")
    best_state = None
    n = xt.shape[0]
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        total = 0.0
        for i in range(0, n, args.batch_size):
            idx = perm[i : i + args.batch_size]
            pred = model(xt[idx])
            loss = torch.nn.functional.mse_loss(pred, yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(idx)
        model.eval()
        with torch.no_grad():
            val = float(torch.nn.functional.mse_loss(model(xv), yv).item())
        if val < best_val:
            best_val = val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 20 == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch}: train {total / n:.4f} val {val:.4f} (best {best_val:.4f})")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_n = model(xv).cpu().numpy()
    pred = pred_n * y_std + y_mean
    true = Y[test_mask]
    metrics: Dict[str, Dict[str, float]] = {}
    if len(output_names) != Y.shape[1]:
        raise ValueError(f"{len(output_names)} output names for Y with {Y.shape[1]} columns")
    for j, name in enumerate(output_names):
        err = pred[:, j] - true[:, j]
        ss_res = float(np.sum(err ** 2))
        ss_tot = float(np.sum((true[:, j] - true[:, j].mean()) ** 2))
        metrics[name] = {
            "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "r2": 1.0 - ss_res / max(ss_tot, 1e-12),
        }
        print(f"{name}: mae {metrics[name]['mae']:.4f} rmse {metrics[name]['rmse']:.4f} r2 {metrics[name]['r2']:.4f}")

    payload = {
        "state_dict": {k: v for k, v in model.cpu().state_dict().items()},
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "obs_dim": int(X.shape[1]),
        "output_names": output_names,
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "horizon": args.horizon,
        "command_slice": [cmd_lo, cmd_hi],
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    import torch as _torch

    _torch.save(payload, args.output)
    print(f"saved model to {args.output}")

    summary = {
        "online_action_set": args.online_action_set,
        "num_samples": int(X.shape[0]),
        "num_train": int(train_mask.sum()),
        "num_test": int(test_mask.sum()),
        "num_episode_groups": int(len(unique_groups)),
        "num_test_groups": int(n_test_groups),
        "horizon": args.horizon,
        "stride": args.stride,
        "skipped_windows_command_change": int(skipped),
        "best_val_mse_normalized": best_val,
        "test_metrics": metrics,
        "output": args.output,
    }
    summary_path = args.summary or (os.path.splitext(args.output)[0] + "_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved summary to {summary_path}")


if __name__ == "__main__":
    main()
