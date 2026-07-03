"""V2 online skill/action-set learning and discriminator-guided evaluation.

This is the online counterpart to the V1 offline pipeline.  It repeatedly:

1. runs exploration in Isaac Lab,
2. cuts fresh trajectories into short action chunks,
3. updates an online skill/action set by novelty + utility,
4. trains a learning-based state-skill discriminator,
5. finally evaluates by applying only dynamically applicable skills.

Run through Isaac Lab:

    TERM=xterm CONDA_PREFIX=/path/to/env_isaaclab ./isaaclab.sh -p \
      /path/to/skill_discover/scripts/online_skill_learning.py \
      --task Isaac-Ant-v0 --num_envs 64 --online_iterations 5 \
      --steps_per_iter 8192 --headless
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_targets(raw: str) -> List[List[float]]:
    targets: List[List[float]] = []
    for item in raw.split(";"):
        item = item.strip()
        if not item:
            continue
        x, y = item.split(",")
        targets.append([float(x), float(y)])
    return targets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V2 online skill/action-set learning.")
    parser.add_argument("--task", type=str, default="Isaac-Ant-v0")
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--online_iterations", type=int, default=5)
    parser.add_argument("--steps_per_iter", type=int, default=8192)
    parser.add_argument("--policy", type=str, choices=["random", "latent"], default="latent")
    parser.add_argument("--action_std", type=float, default=0.5)
    parser.add_argument("--action_smoothing", type=float, default=0.2)
    parser.add_argument("--segment_horizon", type=int, default=32)
    parser.add_argument("--segment_stride", type=int, default=16)
    parser.add_argument("--min_segment_length", type=int, default=16)
    parser.add_argument("--max_skills", type=int, default=16)
    parser.add_argument("--clustering_method", type=str, default="weighted_kmeans",
                        choices=["weighted_kmeans", "nearest"])
    parser.add_argument("--novelty_threshold", type=float, default=3.0)
    parser.add_argument("--replace_margin", type=float, default=0.05)
    parser.add_argument("--max_archive_size", type=int, default=5000)
    parser.add_argument("--min_stability_for_archive", type=float, default=0.2)
    parser.add_argument("--discriminator_epochs", type=int, default=5)
    parser.add_argument("--discriminator_batch_size", type=int, default=256)
    parser.add_argument("--discriminator_lr", type=float, default=1e-3)
    parser.add_argument("--discriminator_threshold", type=float, default=0.55)
    parser.add_argument("--discriminator_hybrid_alpha", type=float, default=0.5)
    parser.add_argument("--discriminator_rbf_temperature", type=float, default=1.5)
    parser.add_argument("--discriminator_device", type=str, default="cpu")
    parser.add_argument("--negative_ratio", type=int, default=2)
    parser.add_argument("--train_outcome_model", action="store_true", default=True)
    parser.add_argument("--outcome_epochs", type=int, default=5)
    parser.add_argument("--outcome_batch_size", type=int, default=256)
    parser.add_argument("--outcome_lr", type=float, default=1e-3)
    parser.add_argument("--outcome_success_weight", type=float, default=0.5)
    parser.add_argument("--outcome_device", type=str, default="cpu")
    parser.add_argument("--eval_planner", type=str, default="learned_mpc",
                        choices=["learned_mpc", "discriminator_greedy"])
    parser.add_argument("--mpc_horizon", type=int, default=3)
    parser.add_argument("--eval_targets", type=str, default="0.5,0;0.5,0.5;-0.5,0")
    parser.add_argument("--eval_trials", type=int, default=3)
    parser.add_argument("--target_threshold", type=float, default=0.3)
    parser.add_argument("--max_high_level_steps", type=int, default=8)
    parser.add_argument("--lambda_energy", type=float, default=0.05)
    parser.add_argument("--lambda_discriminator", type=float, default=1.0)
    parser.add_argument("--lambda_reliability", type=float, default=0.25)
    parser.add_argument("--lambda_no_progress", type=float, default=1.0)
    parser.add_argument("--min_predicted_progress", type=float, default=0.02)
    parser.add_argument("--strict_applicability", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="outputs/online_v2")
    return parser.parse_args()


def _jsonable_stats(obj: Any) -> Dict[str, Any]:
    if hasattr(obj, "__dict__"):
        return {k: _jsonable_stats(v) for k, v in obj.__dict__.items() if not k.startswith("_")}
    if isinstance(obj, dict):
        return {k: _jsonable_stats(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable_stats(v) for v in obj]
    try:
        import numpy as np

        if isinstance(obj, (np.generic,)):
            return obj.item()
    except Exception:
        pass
    return obj


def main() -> None:
    args = parse_args()

    from skill_discovery.envs.isaac_env_wrapper import IsaacEnvWrapper, launch_app

    sim_app = launch_app(headless=args.headless)

    try:
        import numpy as np

        from skill_discovery.control.discriminator_skill_composer import DiscriminatorGuidedSkillComposer
        from skill_discovery.control.learned_mpc_composer import LearnedMPCSkillComposer
        from skill_discovery.descriptors.locomotion_descriptors import LocomotionDescriptorExtractor
        from skill_discovery.exploration.latent_policy import LatentExplorationPolicy
        from skill_discovery.exploration.random_policy import RandomExplorationPolicy
        from skill_discovery.exploration.rollout_collector import RolloutCollector
        from skill_discovery.learning.state_skill_discriminator import StateSkillDiscriminator
        from skill_discovery.learning.skill_outcome_model import SkillOutcomeModel
        from skill_discovery.online.online_action_set import OnlineActionSet, OnlineActionSetConfig
        from skill_discovery.segmentation.fixed_horizon_segmenter import FixedHorizonSegmenter
        from skill_discovery.utils.logging import get_logger
        from skill_discovery.utils.math_utils import set_global_seed

        logger = get_logger("online_v2")
        set_global_seed(args.seed)
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        env = IsaacEnvWrapper.create(args.task, num_envs=args.num_envs, device=args.device)
        policy_cls = LatentExplorationPolicy if args.policy == "latent" else RandomExplorationPolicy
        policy = policy_cls(
            num_envs=env.num_envs,
            action_dim=env.action_dim,
            action_std=args.action_std,
            action_smoothing=args.action_smoothing,
            seed=args.seed,
        )

        collector = RolloutCollector(env, policy)
        segmenter = FixedHorizonSegmenter(
            segment_horizon=args.segment_horizon,
            segment_stride=args.segment_stride,
            min_segment_length=args.min_segment_length,
        )
        extractor = LocomotionDescriptorExtractor()
        action_set = OnlineActionSet(
            OnlineActionSetConfig(
                max_skills=args.max_skills,
                clustering_method=args.clustering_method,
                novelty_threshold=args.novelty_threshold,
                replace_margin=args.replace_margin,
                max_archive_size=args.max_archive_size,
                min_stability_for_archive=args.min_stability_for_archive,
            )
        )
        action_set.metadata = {
            "task": args.task,
            "seed": args.seed,
            "policy": args.policy,
            "online_iterations": args.online_iterations,
            "steps_per_iter": args.steps_per_iter,
        }
        discriminator = StateSkillDiscriminator(
            threshold=args.discriminator_threshold,
            hybrid_alpha=args.discriminator_hybrid_alpha,
            rbf_temperature=args.discriminator_rbf_temperature,
            device=args.discriminator_device,
            seed=args.seed,
        )
        outcome_model = SkillOutcomeModel(device=args.outcome_device, seed=args.seed)

        history: List[Dict[str, Any]] = []
        for iteration in range(args.online_iterations):
            logger.info("V2 online iteration %d/%d", iteration + 1, args.online_iterations)
            trajectories = collector.collect(args.steps_per_iter)
            segments = segmenter.segment_all(trajectories)
            if not segments:
                logger.warning("No segments produced at iteration %d; skipping update.", iteration)
                continue
            descriptor_matrix, descriptor_dicts = extractor.compute_matrix(segments)
            update_stats = action_set.update(
                segments,
                descriptor_matrix,
                descriptor_dicts,
                iteration=iteration,
            )
            train_stats = None
            outcome_stats = None
            if action_set.state_feature_dim is not None and action_set.skills:
                train_stats = discriminator.fit(
                    action_set,
                    epochs=args.discriminator_epochs,
                    batch_size=args.discriminator_batch_size,
                    lr=args.discriminator_lr,
                    negative_ratio=args.negative_ratio,
                )
                if args.train_outcome_model:
                    outcome_stats = outcome_model.fit(
                        action_set,
                        epochs=args.outcome_epochs,
                        batch_size=args.outcome_batch_size,
                        lr=args.outcome_lr,
                        success_weight=args.outcome_success_weight,
                    )

            row = {
                "iteration": iteration,
                "num_trajectories": len(trajectories),
                "num_segments": len(segments),
                "update": _jsonable_stats(update_stats),
                "discriminator": _jsonable_stats(train_stats) if train_stats is not None else None,
                "outcome_model": _jsonable_stats(outcome_stats) if outcome_stats is not None else None,
                "skill_summary": action_set.summary(),
            }
            history.append(row)
            (out_dir / "online_history.json").write_text(json.dumps(history, indent=2))
            action_set.save(str(out_dir / "online_action_set.pkl"))
            action_set.to_skill_library().save(str(out_dir / "skill_library_v2.pkl"))
            if discriminator.model is not None:
                discriminator.save(out_dir / "state_skill_discriminator.pt")
            if outcome_model.model is not None:
                outcome_model.save(out_dir / "skill_outcome_model.pt")

        eval_summary: Dict[str, Any] = {}
        if not args.skip_eval and action_set.skills:
            targets = parse_targets(args.eval_targets)
            if args.eval_planner == "learned_mpc" and outcome_model.is_fitted:
                composer = LearnedMPCSkillComposer(
                    action_set=action_set,
                    discriminator=discriminator,
                    outcome_model=outcome_model,
                    horizon=args.mpc_horizon,
                    target_threshold=args.target_threshold,
                    max_high_level_steps=args.max_high_level_steps,
                    lambda_energy=args.lambda_energy,
                    lambda_discriminator=args.lambda_discriminator,
                    lambda_success=1.0,
                    lambda_reliability=args.lambda_reliability,
                )
            else:
                composer = DiscriminatorGuidedSkillComposer(
                    action_set=action_set,
                    discriminator=discriminator,
                    target_threshold=args.target_threshold,
                    max_high_level_steps=args.max_high_level_steps,
                    lambda_energy=args.lambda_energy,
                    lambda_discriminator=args.lambda_discriminator,
                    lambda_reliability=args.lambda_reliability,
                    lambda_no_progress=args.lambda_no_progress,
                    min_predicted_progress=args.min_predicted_progress,
                    applicability_threshold=args.discriminator_threshold,
                    strict_applicability=args.strict_applicability,
                )
            for target in targets:
                key = f"{target[0]:g},{target[1]:g}"
                target_xy = np.asarray(target, dtype=np.float64)
                records = []
                for trial in range(args.eval_trials):
                    res = composer.rollout(env, target_xy)
                    records.append(
                        {
                            "trial": trial,
                            "success": bool(res.success),
                            "final_distance": float(res.final_distance),
                            "energy_proxy": float(res.energy_proxy),
                            "num_skills_used": int(res.num_skills_used),
                            "skill_sequence": [int(s) for s in res.skill_sequence],
                            "applicability": [
                                _jsonable_stats(d)
                                for d in composer.last_decisions
                            ],
                        }
                    )
                eval_summary[key] = {
                    "success_rate": float(np.mean([r["success"] for r in records])),
                    "average_final_distance": float(np.mean([r["final_distance"] for r in records])),
                    "average_energy_proxy": float(np.mean([r["energy_proxy"] for r in records])),
                    "average_num_skills_used": float(np.mean([r["num_skills_used"] for r in records])),
                    "records": records,
                }

        result = {
            "task": args.task,
            "online_iterations": args.online_iterations,
            "steps_per_iter": args.steps_per_iter,
            "num_skills": len(action_set.skills),
            "skill_summary": action_set.summary(),
            "eval_summary": eval_summary,
        }
        (out_dir / "online_v2_summary.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        env.close()
    finally:
        sim_app.close()


if __name__ == "__main__":
    main()
