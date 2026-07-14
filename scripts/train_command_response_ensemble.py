"""Train a deep ensemble command-response model with conformal height bounds.

The train/calibration/test split is episode-grouped. Ensemble members share the
split and normalization but differ in initialization and minibatch order.  A
one-sided scale factor is calibrated so

    ensemble_mean(min_height) - k * ensemble_std(min_height)

is a lower prediction bound with the requested marginal coverage on held-out
episode groups.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


OUTPUT_NAMES = ["delta_x", "delta_y", "delta_yaw", "energy", "min_height", "mean_height"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--summary", default="")
    p.add_argument("--ensemble_size", type=int, default=5)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--calibration_fraction", type=float, default=0.1)
    p.add_argument("--validation_fraction", type=float, default=0.1)
    p.add_argument("--test_fraction", type=float, default=0.1)
    p.add_argument("--coverage", type=float, default=0.95)
    p.add_argument("--split_seed", type=int, default=20260712)
    p.add_argument("--seed", type=int, default=1000)
    return p.parse_args()


def make_model(torch, obs_dim: int, hidden: int, dropout: float, out_dim: int):
    return torch.nn.Sequential(
        torch.nn.Linear(obs_dim, hidden),
        torch.nn.ReLU(),
        torch.nn.Dropout(dropout),
        torch.nn.Linear(hidden, hidden),
        torch.nn.ReLU(),
        torch.nn.Dropout(dropout),
        torch.nn.Linear(hidden, out_dim),
    )


def metrics(true: np.ndarray, pred: np.ndarray, output_names: list[str]) -> dict:
    result = {}
    for j, name in enumerate(output_names):
        err = pred[:, j] - true[:, j]
        ss_res = float(np.sum(err**2))
        ss_tot = float(np.sum((true[:, j] - true[:, j].mean()) ** 2))
        result[name] = {
            "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err**2))),
            "r2": 1.0 - ss_res / max(ss_tot, 1e-12),
        }
    return result


def main() -> None:
    args = parse_args()
    import torch

    data = np.load(args.dataset)
    X = np.asarray(data["X"], dtype=np.float32)
    Y = np.asarray(data["Y"], dtype=np.float32)
    output_names = data["output_names"].tolist() if "output_names" in data else OUTPUT_NAMES
    if len(output_names) != Y.shape[1]:
        raise ValueError(f"{len(output_names)} output names for Y with {Y.shape[1]} columns")
    G = np.asarray(data["G"])
    groups = np.unique(G)
    rng = np.random.default_rng(args.split_seed)
    rng.shuffle(groups)
    n_test = max(1, round(len(groups) * args.test_fraction))
    n_cal = max(1, round(len(groups) * args.calibration_fraction))
    n_val = max(1, round(len(groups) * args.validation_fraction))
    test_groups = groups[:n_test]
    cal_groups = groups[n_test : n_test + n_cal]
    val_groups = groups[n_test + n_cal : n_test + n_cal + n_val]
    train_groups = groups[n_test + n_cal + n_val :]
    test_mask = np.isin(G, test_groups)
    cal_mask = np.isin(G, cal_groups)
    val_mask = np.isin(G, val_groups)
    train_mask = np.isin(G, train_groups)

    x_mean = X[train_mask].mean(0)
    x_std = np.maximum(X[train_mask].std(0), 1e-6)
    y_mean = Y[train_mask].mean(0)
    y_std = np.maximum(Y[train_mask].std(0), 1e-6)
    Xn = (X - x_mean) / x_std
    Yn = (Y - y_mean) / y_std
    device = "cuda" if torch.cuda.is_available() else "cpu"
    xt = torch.from_numpy(Xn[train_mask]).float().to(device)
    yt = torch.from_numpy(Yn[train_mask]).float().to(device)
    xc = torch.from_numpy(Xn[cal_mask]).float().to(device)
    xval = torch.from_numpy(Xn[val_mask]).float().to(device)
    yval = torch.from_numpy(Yn[val_mask]).float().to(device)
    xtest = torch.from_numpy(Xn[test_mask]).float().to(device)

    state_dicts = []
    cal_members = []
    test_members = []
    member_best_val = []
    for member in range(args.ensemble_size):
        member_seed = args.seed + member
        torch.manual_seed(member_seed)
        np.random.seed(member_seed)
        model = make_model(torch, X.shape[1], args.hidden_dim, args.dropout, Y.shape[1]).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        best_loss = float("inf")
        best_state = None
        for epoch in range(args.epochs):
            model.train()
            perm = torch.randperm(len(xt), device=device)
            for i in range(0, len(xt), args.batch_size):
                idx = perm[i : i + args.batch_size]
                loss = torch.nn.functional.mse_loss(model(xt[idx]), yt[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
            model.eval()
            with torch.inference_mode():
                val_loss = float(torch.nn.functional.mse_loss(model(xval), yval).item())
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if epoch % 40 == 0 or epoch == args.epochs - 1:
                print(f"member {member + 1}/{args.ensemble_size} epoch {epoch}: val {val_loss:.4f} best {best_loss:.4f}")
        model.load_state_dict(best_state)
        model.eval()
        with torch.inference_mode():
            cal_n = model(xc).cpu().numpy()
            test_n = model(xtest).cpu().numpy()
        cal_members.append(cal_n * y_std + y_mean)
        test_members.append(test_n * y_std + y_mean)
        state_dicts.append(best_state)
        member_best_val.append(best_loss)

    cal_stack = np.stack(cal_members)
    test_stack = np.stack(test_members)
    cal_mean = cal_stack.mean(0)
    cal_std = np.maximum(cal_stack.std(0, ddof=1), 1e-6)
    test_mean = test_stack.mean(0)
    test_std = np.maximum(test_stack.std(0, ddof=1), 1e-6)
    height_idx = output_names.index("min_height")
    scores = (cal_mean[:, height_idx] - Y[cal_mask, height_idx]) / cal_std[:, height_idx]
    # Higher interpolation is conservative for a lower prediction bound.
    conformal_k = float(np.quantile(scores, args.coverage, method="higher"))
    lower = test_mean[:, height_idx] - conformal_k * test_std[:, height_idx]
    coverage = float(np.mean(Y[test_mask, height_idx] >= lower))
    mean_width = float(np.mean(test_mean[:, height_idx] - lower))
    test_metrics = metrics(Y[test_mask], test_mean, output_names)
    print(f"calibrated k={conformal_k:.4f}; test lower-bound coverage={coverage:.4f}; mean width={mean_width:.4f} m")
    for name, vals in test_metrics.items():
        print(f"{name}: MAE {vals['mae']:.4f}, RMSE {vals['rmse']:.4f}, R2 {vals['r2']:.4f}")

    payload = {
        "state_dicts": state_dicts,
        "ensemble_size": args.ensemble_size,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "obs_dim": int(X.shape[1]),
        "output_names": output_names,
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "horizon": int(data["horizon"]) if "horizon" in data else 16,
        "command_slice": data["command_slice"].tolist() if "command_slice" in data else [9, 12],
        "height_conformal_k": conformal_k,
        "height_coverage_target": args.coverage,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    summary = {
        "dataset": args.dataset,
        "output": args.output,
        "ensemble_size": args.ensemble_size,
        "num_samples": int(len(X)),
        "num_train": int(train_mask.sum()),
        "num_calibration": int(cal_mask.sum()),
        "num_validation": int(val_mask.sum()),
        "num_test": int(test_mask.sum()),
        "num_train_groups": int(len(train_groups)),
        "num_calibration_groups": int(len(cal_groups)),
        "num_validation_groups": int(len(val_groups)),
        "num_test_groups": int(len(test_groups)),
        "coverage_target": args.coverage,
        "height_conformal_k": conformal_k,
        "height_test_coverage": coverage,
        "height_mean_bound_width_m": mean_width,
        "member_best_cal_mse_normalized": member_best_val,
        "test_metrics": test_metrics,
    }
    summary_path = args.summary or os.path.splitext(args.output)[0] + "_summary.json"
    Path(summary_path).write_text(json.dumps(summary, indent=2))
    print(f"saved {args.output} and {summary_path}")


if __name__ == "__main__":
    main()
