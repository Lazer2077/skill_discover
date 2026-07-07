"""Train a closed-loop skill-conditioned behavior cloning policy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train skill-conditioned BC policy.")
    parser.add_argument("--online_action_set", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max_samples", type=int, default=200000)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--embedding_dim", type=int, default=16)
    parser.add_argument("--zero_obs_slices", type=str, default="")
    parser.add_argument("--label_mode", choices=["skill", "archive"], default="skill")
    parser.add_argument("--use_phase", action="store_true")
    parser.add_argument("--no_balance_skills", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--summary", type=str, default=None)
    return parser.parse_args()


def _jsonable(obj: Any) -> Any:
    if hasattr(obj, "__dict__"):
        return {k: _jsonable(v) for k, v in obj.__dict__.items() if not k.startswith("_")}
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def main() -> None:
    args = parse_args()

    from skill_discovery.learning.skill_policy import SkillPolicy, parse_obs_slices
    from skill_discovery.online.online_action_set import OnlineActionSet

    action_set = OnlineActionSet.load(args.online_action_set)
    policy = SkillPolicy(
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        zero_obs_slices=parse_obs_slices(args.zero_obs_slices),
        label_mode=args.label_mode,
        use_phase=args.use_phase,
        device=args.device,
        seed=args.seed,
    )
    stats = policy.fit(
        action_set,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_samples=args.max_samples if args.max_samples > 0 else None,
        balance_skills=not args.no_balance_skills,
    )
    policy.save(args.output)
    summary = {
        "online_action_set": args.online_action_set,
        "output": args.output,
        "zero_obs_slices": policy.zero_obs_slices,
        "label_mode": policy.label_mode,
        "use_phase": policy.use_phase,
        "train": _jsonable(stats),
    }
    summary_path = Path(args.summary) if args.summary else Path(args.output).with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
