"""Train a lightweight viability classifier from guarded-skill eval records.

The current guarded controller uses hand-tuned thresholds.  This script turns
recorded decisions into a supervised dataset for the next research iteration:

    candidate decision -> episode-level viable / not viable

The model is intentionally small.  It is meant to audit whether the guard
features contain enough signal before wiring a learned discriminator into the
Isaac evaluator.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


FEATURE_NAMES = [
    "current_distance",
    "predicted_distance",
    "predicted_progress",
    "state_distance",
    "applicability",
    "utility",
    "energy",
    "stability",
    "guard_alignment",
    "guard_energy_z",
    "guard_target_norm",
    "guard_skill_norm",
    "guard_blend",
    "guard_accepted",
    "guard_low_height",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a decision-level skill viability model.")
    parser.add_argument("--records", nargs="+", required=True, help="Evaluation JSON files with decision records.")
    parser.add_argument("--methods", type=str, default="guarded_skill_command,skill_command")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--test_fraction", type=float, default=0.25)
    parser.add_argument(
        "--drop_features",
        type=str,
        default="guard_blend,guard_accepted,guard_low_height",
        help="Comma-separated feature names to remove before training.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="outputs/skill_viability_model.pt")
    parser.add_argument("--summary", type=str, default="outputs/skill_viability_model_summary.json")
    return parser.parse_args()


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        out = float(value)
        if not np.isfinite(out):
            return default
        return out
    except Exception:
        return default


def decision_feature(decision: Dict[str, Any]) -> np.ndarray | None:
    guard = decision.get("guard", {})
    if not guard and not any(k in decision for k in ("predicted_progress", "energy", "stability")):
        return None
    return np.asarray(
        [
            _float(decision.get("current_distance")),
            _float(decision.get("predicted_distance")),
            _float(decision.get("predicted_progress")),
            _float(decision.get("state_distance")),
            _float(decision.get("applicability")),
            _float(decision.get("utility")),
            _float(decision.get("energy")),
            _float(decision.get("stability")),
            _float(guard.get("alignment")),
            _float(guard.get("energy_z")),
            _float(guard.get("target_norm")),
            _float(guard.get("skill_norm")),
            _float(guard.get("blend")),
            1.0 if guard.get("accepted") else 0.0,
            1.0 if guard.get("low_height") else 0.0,
        ],
        dtype=np.float32,
    )


def load_dataset(
    paths: Iterable[str],
    methods: Iterable[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    method_set = set(methods)
    features: List[np.ndarray] = []
    labels: List[float] = []
    groups: List[str] = []
    meta: List[Dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path)
        with path.open() as f:
            data = json.load(f)
        for target_key, target_records in data.get("records", {}).items():
            for method, records in target_records.items():
                if method not in method_set:
                    continue
                for record in records:
                    label = 1.0 if record.get("success") else 0.0
                    group = f"{path}|{target_key}|{method}|{record.get('trial')}"
                    for decision_index, decision in enumerate(record.get("decisions", [])):
                        feature = decision_feature(decision)
                        if feature is None:
                            continue
                        features.append(feature)
                        labels.append(label)
                        groups.append(group)
                        meta.append(
                            {
                                "path": str(path),
                                "target": target_key,
                                "method": method,
                                "trial": record.get("trial"),
                                "decision_index": decision_index,
                            }
                        )
    if not features:
        raise ValueError("No decision features found in the requested records/methods.")
    return np.stack(features), np.asarray(labels, dtype=np.float32), np.asarray(groups), meta


def auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    pos = y_true >= 0.5
    neg = ~pos
    if not np.any(pos) or not np.any(neg):
        return None
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1)
    pos_ranks = np.sum(ranks[pos])
    n_pos = float(np.sum(pos))
    n_neg = float(np.sum(neg))
    return float((pos_ranks - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg))


def main() -> None:
    args = parse_args()

    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    X, y, groups, meta = load_dataset(args.records, methods)
    drop_features = {item.strip() for item in args.drop_features.split(",") if item.strip()}
    unknown_drop_features = sorted(drop_features - set(FEATURE_NAMES))
    if unknown_drop_features:
        raise ValueError(f"Unknown --drop_features entries: {unknown_drop_features}")
    keep_indices = [i for i, name in enumerate(FEATURE_NAMES) if name not in drop_features]
    selected_feature_names = [FEATURE_NAMES[i] for i in keep_indices]
    X = X[:, keep_indices]
    rng = np.random.default_rng(args.seed)
    unique_groups = np.unique(groups)
    rng.shuffle(unique_groups)
    test_group_count = max(1, int(round(len(unique_groups) * float(args.test_fraction))))
    test_groups = set(unique_groups[:test_group_count])
    test_idx = np.asarray([i for i, group in enumerate(groups) if group in test_groups], dtype=np.int64)
    train_idx = np.asarray([i for i, group in enumerate(groups) if group not in test_groups], dtype=np.int64)
    if len(train_idx) == 0:
        raise ValueError("Not enough samples for a train/test split.")

    mean = X[train_idx].mean(axis=0, keepdims=True)
    std = X[train_idx].std(axis=0, keepdims=True) + 1e-6
    Xn = ((X - mean) / std).astype(np.float32)

    torch.manual_seed(args.seed)
    model = nn.Sequential(
        nn.Linear(X.shape[1], args.hidden_dim),
        nn.ReLU(),
        nn.Linear(args.hidden_dim, args.hidden_dim),
        nn.ReLU(),
        nn.Linear(args.hidden_dim, 1),
    )
    train_y = y[train_idx]
    pos = float(np.sum(train_y >= 0.5))
    neg = float(np.sum(train_y < 0.5))
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    ds = TensorDataset(torch.from_numpy(Xn[train_idx]), torch.from_numpy(train_y.reshape(-1, 1)))
    loader = DataLoader(ds, batch_size=min(256, len(ds)), shuffle=True)

    final_loss = 0.0
    for _ in range(max(1, args.epochs)):
        for xb, yb in loader:
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            final_loss = float(loss.detach())

    with torch.no_grad():
        logits = model(torch.from_numpy(Xn)).squeeze(-1)
        probs = torch.sigmoid(logits).numpy()
    test_probs = probs[test_idx]
    test_y = y[test_idx]
    test_pred = test_probs >= 0.5
    test_accuracy = float(np.mean(test_pred == (test_y >= 0.5)))
    test_auc = auc_score(test_y, test_probs)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "feature_names": selected_feature_names,
            "state_dict": model.state_dict(),
            "hidden_dim": args.hidden_dim,
            "mean": mean,
            "std": std,
            "seed": args.seed,
        },
        out,
    )

    summary = {
        "records": args.records,
        "methods": methods,
        "num_samples": int(len(y)),
        "num_train": int(len(train_idx)),
        "num_test": int(len(test_idx)),
        "num_episode_groups": int(len(unique_groups)),
        "num_train_episode_groups": int(len(unique_groups) - len(test_groups)),
        "num_test_episode_groups": int(len(test_groups)),
        "split": "episode_group",
        "positive_rate": float(np.mean(y >= 0.5)),
        "train_positive_rate": float(np.mean(y[train_idx] >= 0.5)),
        "test_positive_rate": float(np.mean(test_y >= 0.5)),
        "final_train_loss": final_loss,
        "test_accuracy": test_accuracy,
        "test_auc": test_auc,
        "feature_names": selected_feature_names,
        "dropped_features": sorted(drop_features),
        "output": str(out),
    }
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
