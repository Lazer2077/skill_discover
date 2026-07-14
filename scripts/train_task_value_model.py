"""Train a task-level remaining-outcome model from mixed-controller rollouts."""

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
    p.add_argument(
        "--methods",
        default="",
        help="Optional comma-separated controller methods to retain (for policy-consistent terminal values).",
    )
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--test_fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=2026)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    import torch
    from sklearn.metrics import roc_auc_score

    lo, hi = (int(x) for x in args.command_slice.split(":"))
    retained_methods = {value.strip() for value in args.methods.split(",") if value.strip()}
    xs, ys, groups = [], [], []
    method_counts: dict[str, int] = {}
    episode_index = 0
    for raw in args.inputs:
        data = json.loads(Path(raw).read_text())
        for target, method_records in data["records"].items():
            for method, records in method_records.items():
                if retained_methods and method not in retained_methods:
                    continue
                for record in records:
                    samples = record.get("value_samples", [])
                    if not samples:
                        continue
                    group = episode_index
                    episode_index += 1
                    method_counts[method] = method_counts.get(method, 0) + len(samples)
                    for sample in samples:
                        obs = np.asarray(sample["observation"], dtype=np.float32).copy()
                        command = np.asarray(sample["command"], dtype=np.float32)
                        obs[lo : min(hi, len(obs))] = command[: max(0, min(hi, len(obs)) - lo)]
                        x = np.concatenate(
                            [
                                obs,
                                np.asarray(sample["local_target"], dtype=np.float32),
                                np.asarray([sample["distance"]], dtype=np.float32),
                            ]
                        )
                        y = np.asarray(
                            [sample["remaining_work_j"], sample["remaining_time_s"], float(sample["success"])],
                            dtype=np.float32,
                        )
                        xs.append(x)
                        ys.append(y)
                        groups.append(group)
        for sample in data.get("mpc_dagger_samples", []):
            predicted_inputs = np.asarray(
                sample["predicted_terminal_inputs"], dtype=np.float32
            )
            if predicted_inputs.ndim == 1:
                predicted_inputs = predicted_inputs[None, :]
            group = episode_index
            episode_index += 1
            label = np.asarray(
                [
                    sample["remaining_work_j"],
                    sample["remaining_time_s"],
                    float(sample["success"]),
                ],
                dtype=np.float32,
            )
            for predicted_input in predicted_inputs:
                xs.append(predicted_input)
                ys.append(label.copy())
                groups.append(group)
            method_counts["mpc_dagger"] = method_counts.get("mpc_dagger", 0) + len(
                predicted_inputs
            )
    if not xs:
        raise ValueError("No task-value samples were found in the supplied inputs.")
    X = np.stack(xs)
    Y = np.stack(ys)
    G = np.asarray(groups)
    rng = np.random.default_rng(args.seed)
    unique_groups = np.unique(G)
    rng.shuffle(unique_groups)
    n_test = max(1, int(len(unique_groups) * args.test_fraction))
    test_groups = unique_groups[:n_test]
    train_mask = ~np.isin(G, test_groups)
    test_mask = ~train_mask

    x_mean = X[train_mask].mean(0)
    x_std = np.maximum(X[train_mask].std(0), 1e-6)
    y_mean = Y[train_mask, :2].mean(0)
    y_std = np.maximum(Y[train_mask, :2].std(0), 1e-6)
    Xn = (X - x_mean) / x_std
    Yn = (Y[:, :2] - y_mean) / y_std

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    model = torch.nn.Sequential(
        torch.nn.Linear(X.shape[1], args.hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(args.hidden_dim, args.hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(args.hidden_dim, 3),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    xt = torch.from_numpy(Xn[train_mask]).float().to(device)
    yr = torch.from_numpy(Yn[train_mask]).float().to(device)
    ysuc = torch.from_numpy(Y[train_mask, 2]).float().to(device)
    xv = torch.from_numpy(Xn[test_mask]).float().to(device)
    best_loss, best_state = float("inf"), None
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(xt), device=device)
        for i in range(0, len(xt), args.batch_size):
            idx = perm[i : i + args.batch_size]
            pred = model(xt[idx])
            loss_reg = torch.nn.functional.mse_loss(pred[:, :2], yr[idx])
            loss_cls = torch.nn.functional.binary_cross_entropy_with_logits(pred[:, 2], ysuc[idx])
            loss = loss_reg + 0.25 * loss_cls
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.inference_mode():
            pv = model(xv)
            val_reg = torch.nn.functional.mse_loss(
                pv[:, :2], torch.from_numpy(Yn[test_mask]).float().to(device)
            )
            val_cls = torch.nn.functional.binary_cross_entropy_with_logits(
                pv[:, 2], torch.from_numpy(Y[test_mask, 2]).float().to(device)
            )
            val = float((val_reg + 0.25 * val_cls).item())
        if val < best_loss:
            best_loss = val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 40 == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch}: val={val:.4f} best={best_loss:.4f}")

    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        pred_raw = model(xv).cpu().numpy()
    pred_reg = pred_raw[:, :2] * y_std + y_mean
    pred_prob = 1.0 / (1.0 + np.exp(-pred_raw[:, 2]))
    metrics = {}
    for j, name in enumerate(["remaining_work_j", "remaining_time_s"]):
        true = Y[test_mask, j]
        err = pred_reg[:, j] - true
        metrics[name] = {
            "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err**2))),
            "r2": float(1.0 - np.sum(err**2) / max(np.sum((true - true.mean()) ** 2), 1e-12)),
        }
    try:
        auc = float(roc_auc_score(Y[test_mask, 2], pred_prob))
    except ValueError:
        auc = float("nan")
    metrics["success"] = {
        "auc": auc,
        "brier": float(np.mean((pred_prob - Y[test_mask, 2]) ** 2)),
        "accuracy_05": float(np.mean((pred_prob >= 0.5) == (Y[test_mask, 2] > 0.5))),
    }
    print(json.dumps(metrics, indent=2))
    payload = {
        "state_dict": best_state,
        "input_dim": int(X.shape[1]),
        "obs_dim": int(X.shape[1] - 3),
        "hidden_dim": args.hidden_dim,
        "command_slice": [lo, hi],
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    summary = {
        "inputs": args.inputs,
        "output": args.output,
        "num_samples": int(len(X)),
        "num_episodes": int(len(unique_groups)),
        "num_train": int(train_mask.sum()),
        "num_test": int(test_mask.sum()),
        "num_test_episodes": int(n_test),
        "method_sample_counts": method_counts,
        "retained_methods": sorted(retained_methods),
        "best_validation_loss": best_loss,
        "metrics": metrics,
    }
    summary_path = args.summary or str(Path(args.output).with_suffix("")) + "_summary.json"
    Path(summary_path).write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
