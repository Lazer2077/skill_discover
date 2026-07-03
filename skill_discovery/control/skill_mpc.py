"""Skill-level model-predictive planning by brute-force search.

This is a *skill-space optimal control approximation*, not a solution of the
full Hamilton-Jacobi-Bellman equation: the continuous control problem is
reduced to a discrete search over skill sequences, using each skill's
*average* recorded outcome as a deterministic dynamics model

    (x, y, yaw)  --skill k-->  (x, y, yaw) + averaged body-frame outcome of k.

Cost per step:  distance_to_goal + lambda_energy * energy + lambda_yaw * yaw_error.

FUTURE (V2+):
    * learn a stochastic skill-level dynamics model instead of cluster means,
    * HJB-inspired value function over (x, y, yaw) for infinite-horizon planning,
    * receding-horizon MPC with replanning from observed state mismatch,
    * diffusion model sampling action chunks conditioned on desired outcome.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Tuple

import numpy as np

from skill_discovery.library.skill_library import SkillLibrary
from skill_discovery.utils.math_utils import wrap_angle


@dataclass
class SkillMPC:
    """Brute-force multi-step search over discrete skill sequences."""

    skill_library: SkillLibrary
    horizon: int = 3
    lambda_energy: float = 0.05
    lambda_yaw: float = 0.1

    def plan(
        self,
        current_state: Dict[str, float],
        target_state: Dict[str, float],
        skill_library: SkillLibrary | None = None,
    ) -> Tuple[List[int], float]:
        """Return (best skill-id sequence, its cost).

        `current_state`: keys x, y, yaw. `target_state`: keys x, y and
        optionally yaw (goal heading; if omitted, yaw error is measured
        against the direction pointing at the goal).
        """
        library = skill_library or self.skill_library
        skill_ids = library.skill_ids
        if not skill_ids:
            raise RuntimeError("Skill library is empty; cannot plan.")

        goal = np.array([target_state["x"], target_state["y"]])
        goal_yaw = target_state.get("yaw")

        best_seq: List[int] = []
        best_cost = np.inf
        # Exhaustive enumeration: |skills|^horizon rollouts of the mean model.
        # Fine for V1 scales (e.g. 8^3 = 512); replace with sampling/graph
        # search if the library grows.
        for seq in product(skill_ids, repeat=self.horizon):
            cost = self._evaluate_sequence(seq, current_state, goal, goal_yaw, library)
            if cost < best_cost:
                best_cost, best_seq = cost, list(seq)
        return best_seq, float(best_cost)

    def _evaluate_sequence(
        self,
        seq: Tuple[int, ...],
        current_state: Dict[str, float],
        goal: np.ndarray,
        goal_yaw: float | None,
        library: SkillLibrary,
    ) -> float:
        x, y, yaw = current_state["x"], current_state["y"], current_state["yaw"]
        cost = 0.0
        for sid in seq:
            skill = library.get_skill(sid)
            x, y, yaw = skill.predict_outcome(x, y, yaw)
            distance = float(np.linalg.norm(goal - np.array([x, y])))
            desired_yaw = goal_yaw if goal_yaw is not None else float(
                np.arctan2(goal[1] - y, goal[0] - x)
            )
            yaw_error = abs(float(wrap_angle(yaw - desired_yaw))) if distance > 1e-3 else 0.0
            cost += distance + self.lambda_energy * skill.mean_energy + self.lambda_yaw * yaw_error
        return cost
