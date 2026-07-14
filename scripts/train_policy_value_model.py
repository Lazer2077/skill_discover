"""Train a policy-conditioned remaining-outcome ensemble and audit fixed-gain selection."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np


SCALE_BY_METHOD = {
    "scaled_target_command_60": 0.60,
    "scaled_target_command_75": 0.75,
    "scaled_target_command_90": 0.90,
    "direct_target_command": 1.00,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train_inputs", nargs="+", required=True)
    p.add_argument("--test_inputs", nargs="+", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--summary", default="")
    p.add_argument("--command_slice", default="9:12")
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--members", type=int, default=5)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--validation_fraction", type=float, default=0.15)
    p.add_argument("--success_margin", type=float, default=0.02)
    p.add_argument("--time_ratio", type=float, default=1.10)
    p.add_argument("--work_margin", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=2026)
    return p.parse_args()


def load_episodes(paths: list[str], lo: int, hi: int):
    episodes = []
    for path in paths:
        data = json.loads(Path(path).read_text())
        for target, method_records in data["records"].items():
            for method, records in method_records.items():
                if method not in SCALE_BY_METHOD:
                    continue
                scale = SCALE_BY_METHOD[method]
                for record in records:
                    samples = record.get("value_samples", [])
                    if not samples:
                        continue
                    xs, ys = [], []
                    for sample in samples:
                        obs = np.asarray(sample["observation"], dtype=np.float32).copy()
                        obs[lo:hi] = 0.0
                        x = np.concatenate(
                            [
                                obs,
                                np.asarray(sample["local_target"], dtype=np.float32),
                                np.asarray([sample["distance"], scale], dtype=np.float32),
                            ]
                        )
                        y = np.asarray(
                            [
                                sample["remaining_work_j"],
                                sample["remaining_time_s"],
                                float(sample["success"]),
                            ],
                            dtype=np.float32,
                        )
                        xs.append(x)
                        ys.append(y)
                    episodes.append(
                        {
                            "source": path,
                            "target": target,
                            "trial": int(record.get("trial", 0)),
                            "method": method,
                            "scale": scale,
                            "x": np.stack(xs),
                            "y": np.stack(ys),
                            "record": record,
                        }
                    )
    if not episodes:
        raise ValueError("No fixed-scale episodes with value_samples were found.")
    return episodes


def flatten(episodes, indices):
    x = np.concatenate([episodes[i]["x"] for i in indices], axis=0)
    y = np.concatenate([episodes[i]["y"] for i in indices], axis=0)
    return x, y


def metrics(true: np.ndarray, raw: np.ndarray, y_mean: np.ndarray, y_std: np.ndarray):
    reg = raw[:, :2] * y_std + y_mean
    prob = 1.0 / (1.0 + np.exp(-raw[:, 2]))
    out = {}
    for j, name in enumerate(("remaining_work_j", "remaining_time_s")):
        err = reg[:, j] - true[:, j]
        out[name] = {
            "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err**2))),
            "r2": float(
                1.0
                - np.sum(err**2)
                / max(np.sum((true[:, j] - true[:, j].mean()) ** 2), 1e-12)
            ),
        }
    positives = true[:, 2] > 0.5
    n_pos = int(positives.sum())
    n_neg = int((~positives).sum())
    if n_pos and n_neg:
        order = np.argsort(prob, kind="mergesort")
        sorted_prob = prob[order]
        ranks = np.empty(len(prob), dtype=np.float64)
        start = 0
        while start < len(prob):
            stop = start + 1
            while stop < len(prob) and sorted_prob[stop] == sorted_prob[start]:
                stop += 1
            ranks[order[start:stop]] = 0.5 * (start + 1 + stop)
            start = stop
        auc = float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
    else:
        auc = float("nan")
    out["success"] = {
        "auc": auc,
        "brier": float(np.mean((prob - true[:, 2]) ** 2)),
        "accuracy_05": float(np.mean((prob >= 0.5) == (true[:, 2] > 0.5))),
    }
    return out


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
    import torch

    lo, hi = (int(value) for value in args.command_slice.split(":"))
    train_episodes = load_episodes(args.train_inputs, lo, hi)
    test_episodes = load_episodes(args.test_inputs, lo, hi)
    rng = np.random.default_rng(args.seed)
    episode_order = np.arange(len(train_episodes))
    rng.shuffle(episode_order)
    n_val = max(1, int(len(episode_order) * args.validation_fraction))
    val_indices = episode_order[:n_val]
    fit_indices = episode_order[n_val:]
    x_fit, y_fit = flatten(train_episodes, fit_indices)
    x_val, y_val = flatten(train_episodes, val_indices)
    x_test, y_test = flatten(test_episodes, np.arange(len(test_episodes)))

    x_mean = x_fit.mean(0)
    x_std = np.maximum(x_fit.std(0), 1e-6)
    y_mean = y_fit[:, :2].mean(0)
    y_std = np.maximum(y_fit[:, :2].std(0), 1e-6)

    def normalize_x(x):
        return (x - x_mean) / x_std

    def normalize_y(y):
        return (y[:, :2] - y_mean) / y_std

    device = "cuda" if torch.cuda.is_available() else "cpu"

    def make_model():
        return torch.nn.Sequential(
            torch.nn.Linear(x_fit.shape[1], args.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(args.hidden_dim, args.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(args.hidden_dim, 3),
        ).to(device)

    xv = torch.from_numpy(normalize_x(x_val)).float().to(device)
    yv_reg = torch.from_numpy(normalize_y(y_val)).float().to(device)
    yv_succ = torch.from_numpy(y_val[:, 2]).float().to(device)
    state_dicts = []
    member_best_losses = []
    for member in range(args.members):
        member_seed = args.seed + 1009 * member
        torch.manual_seed(member_seed)
        member_rng = np.random.default_rng(member_seed)
        boot_episode_indices = member_rng.choice(fit_indices, size=len(fit_indices), replace=True)
        xb, yb = flatten(train_episodes, boot_episode_indices)
        xt = torch.from_numpy(normalize_x(xb)).float().to(device)
        yt_reg = torch.from_numpy(normalize_y(yb)).float().to(device)
        yt_succ = torch.from_numpy(yb[:, 2]).float().to(device)
        model = make_model()
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        best_loss = float("inf")
        best_state = None
        for _epoch in range(args.epochs):
            model.train()
            perm = torch.randperm(len(xt), device=device)
            for start in range(0, len(xt), args.batch_size):
                idx = perm[start : start + args.batch_size]
                pred = model(xt[idx])
                loss = torch.nn.functional.mse_loss(pred[:, :2], yt_reg[idx])
                loss += 0.25 * torch.nn.functional.binary_cross_entropy_with_logits(
                    pred[:, 2], yt_succ[idx]
                )
                opt.zero_grad()
                loss.backward()
                opt.step()
            model.eval()
            with torch.inference_mode():
                pred = model(xv)
                val_loss = torch.nn.functional.mse_loss(pred[:, :2], yv_reg)
                val_loss += 0.25 * torch.nn.functional.binary_cross_entropy_with_logits(
                    pred[:, 2], yv_succ
                )
            value = float(val_loss.item())
            if value < best_loss:
                best_loss = value
                best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        state_dicts.append(best_state)
        member_best_losses.append(best_loss)
        print(f"member {member}: best validation loss {best_loss:.4f}")

    models = []
    for state in state_dicts:
        model = make_model()
        model.load_state_dict(state)
        model.eval()
        models.append(model)

    def predict(x):
        tensor = torch.from_numpy(normalize_x(x).astype(np.float32)).to(device)
        with torch.inference_mode():
            raw_members = np.stack([model(tensor).cpu().numpy() for model in models])
        reg_members = raw_members[:, :, :2] * y_std + y_mean
        prob_members = 1.0 / (1.0 + np.exp(-raw_members[:, :, 2]))
        mean = np.concatenate([reg_members.mean(0), prob_members.mean(0)[:, None]], axis=1)
        std = np.concatenate([reg_members.std(0), prob_members.std(0)[:, None]], axis=1)
        raw_mean = raw_members.mean(0)
        return mean, std, raw_mean

    test_mean, test_std, test_raw = predict(x_test)
    test_metrics = metrics(y_test, test_raw, y_mean, y_std)
    # Replace success metrics with the mean ensemble probability rather than
    # sigmoid(mean logit), matching deployment.
    test_metrics["success"]["brier"] = float(np.mean((test_mean[:, 2] - y_test[:, 2]) ** 2))
    test_metrics["success"]["accuracy_05"] = float(
        np.mean((test_mean[:, 2] >= 0.5) == (y_test[:, 2] > 0.5))
    )

    grouped = {}
    for episode in test_episodes:
        grouped.setdefault((episode["source"], episode["target"], episode["trial"]), {})[
            episode["method"]
        ] = episode
    decision_audit = {}
    for uncertainty_k in (0.0, 1.0):
        selected_records = []
        selected_scales = []
        oracle_records = []
        pairing_max_diffs = []
        candidate_diagnostics = {
            f"{scale:.2f}": {
                "predicted_work_ratio": [],
                "predicted_time_ratio": [],
                "predicted_success_delta": [],
                "eligible": [],
            }
            for scale in SCALE_BY_METHOD.values()
        }
        for method_episodes in grouped.values():
            if set(method_episodes) != set(SCALE_BY_METHOD):
                continue
            direct_episode = method_episodes["direct_target_command"]
            base_obs = direct_episode["x"][0].copy()
            candidate_x = []
            candidate_episodes = []
            for method, scale in SCALE_BY_METHOD.items():
                x = base_obs.copy()
                x[-1] = scale
                candidate_x.append(x)
                candidate_episodes.append(method_episodes[method])
                pairing_max_diffs.append(
                    float(np.max(np.abs(method_episodes[method]["x"][0][:-1] - base_obs[:-1])))
                )
            pred_mean, pred_std, _ = predict(np.stack(candidate_x))
            direct_idx = len(candidate_episodes) - 1
            success_lcb = pred_mean[:, 2] - uncertainty_k * pred_std[:, 2]
            time_ucb = pred_mean[:, 1] + uncertainty_k * pred_std[:, 1]
            eligible = (success_lcb >= success_lcb[direct_idx] - args.success_margin) & (
                time_ucb <= args.time_ratio * time_ucb[direct_idx]
            )
            eligible[direct_idx] = True
            for idx, episode in enumerate(candidate_episodes):
                diag = candidate_diagnostics[f"{episode['scale']:.2f}"]
                diag["predicted_work_ratio"].append(
                    float(pred_mean[idx, 0] / max(pred_mean[direct_idx, 0], 1e-9))
                )
                diag["predicted_time_ratio"].append(
                    float(pred_mean[idx, 1] / max(pred_mean[direct_idx, 1], 1e-9))
                )
                diag["predicted_success_delta"].append(
                    float(pred_mean[idx, 2] - pred_mean[direct_idx, 2])
                )
                diag["eligible"].append(float(eligible[idx]))
            best_idx = int(np.argmin(np.where(eligible, pred_mean[:, 0], np.inf)))
            if pred_mean[best_idx, 0] >= (1.0 - args.work_margin) * pred_mean[direct_idx, 0]:
                best_idx = direct_idx
            selected_records.append(candidate_episodes[best_idx]["record"])
            selected_scales.append(candidate_episodes[best_idx]["scale"])

            actual_direct = direct_episode["record"]
            oracle_eligible = [
                episode["record"]
                for episode in candidate_episodes
                if episode["record"]["elapsed_time_s"]
                <= args.time_ratio * actual_direct["elapsed_time_s"] + 1e-9
                and (episode["record"]["success"] or not actual_direct["success"])
            ]
            if not oracle_eligible:
                oracle_eligible = [actual_direct]
            oracle_records.append(min(oracle_eligible, key=lambda record: record["mechanical_energy_j"]))
        scale_counts = {f"{scale:.2f}": selected_scales.count(scale) for scale in sorted(set(selected_scales))}
        decision_audit[f"uncertainty_k_{uncertainty_k:g}"] = {
            "selector": aggregate(selected_records),
            "oracle": aggregate(oracle_records),
            "scale_counts": scale_counts,
            "max_initial_pairing_difference": max(pairing_max_diffs, default=0.0),
            "candidate_diagnostics": {
                scale: {name: float(np.mean(values)) for name, values in diag.items()}
                for scale, diag in candidate_diagnostics.items()
            },
        }
    baselines = {}
    for method in SCALE_BY_METHOD:
        baselines[method] = aggregate(
            [method_episodes[method]["record"] for method_episodes in grouped.values()]
        )

    payload = {
        "model_type": "policy_conditioned_value",
        "state_dicts": state_dicts,
        "input_dim": int(x_fit.shape[1]),
        "obs_dim": int(x_fit.shape[1] - 4),
        "hidden_dim": args.hidden_dim,
        "command_slice": [lo, hi],
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "scales": sorted(SCALE_BY_METHOD.values()),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    summary = {
        "train_inputs": args.train_inputs,
        "test_inputs": args.test_inputs,
        "output": args.output,
        "num_train_episodes": len(train_episodes),
        "num_test_episodes": len(test_episodes),
        "num_fit_samples": int(len(x_fit)),
        "num_validation_samples": int(len(x_val)),
        "num_test_samples": int(len(x_test)),
        "member_best_validation_losses": member_best_losses,
        "test_metrics": test_metrics,
        "baselines": baselines,
        "decision_audit": decision_audit,
    }
    summary_path = args.summary or str(Path(args.output).with_suffix("")) + "_summary.json"
    Path(summary_path).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
