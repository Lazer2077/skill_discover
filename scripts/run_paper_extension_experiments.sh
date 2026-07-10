#!/usr/bin/env bash
# Sequential experiment chain for the paper extension:
#  1) two extra seeds (104, 105) on Go2 harder targets, 10 trials each
#  2) archive-size ablation (500 / 2000 / full 3904) at seed 106, 5 trials
set -uo pipefail

PY=/home/thing1/miniconda3/envs/env_isaaclab/bin/python
CKPT=.pretrained_checkpoints/rsl_rl/Isaac-Velocity-Rough-Unitree-Go2-v0/checkpoint.pt
OUT=outputs/isaac_go2_rough_rsl_rl_skills_long
ARCH=$OUT/online_action_set.pkl
TARGETS="2.0,0;1.5,1.0;0,1.25;-0.75,0;1.0,-1.0"

COMMON_ARGS=(
  --task Isaac-Velocity-Rough-Unitree-Go2-v0
  --checkpoint "$CKPT"
  --targets "$TARGETS"
  --max_high_level_steps 80 --execution_horizon 4
  --lambda_state 0.0 --lambda_energy 0.0 --lambda_stability 0.0
  --lambda_no_progress 2.0 --lambda_utility 0.0 --lambda_progress 2.0
  --k_nearest 0 --skill_target_command_blend 0.5
  --guard_recovery_height_fraction 0.72
  --relative_targets --store_positions --headless
)

run() {
  echo "=== [$(date +%H:%M:%S)] $* ==="
  "$PY" scripts/evaluate_rsl_skill_command_control.py "$@" || echo "!!! FAILED: $*"
}

# --- extra seeds ---
run "${COMMON_ARGS[@]}" --online_action_set "$ARCH" \
  --methods guarded_skill_command,direct_target_command \
  --num_trials 10 --seed 104 \
  --output $OUT/rsl_guarded_skill_command_eval_v1_recovery_harder_10trials_seed104_positions.json

run "${COMMON_ARGS[@]}" --online_action_set "$ARCH" \
  --methods guarded_skill_command,direct_target_command \
  --num_trials 10 --seed 105 \
  --output $OUT/rsl_guarded_skill_command_eval_v1_recovery_harder_10trials_seed105_positions.json

# --- archive-size ablation (same seed, guarded only + one direct reference) ---
run "${COMMON_ARGS[@]}" --online_action_set $OUT/online_action_set_arch500.pkl \
  --methods guarded_skill_command \
  --num_trials 5 --seed 106 \
  --output $OUT/rsl_archive_ablation_arch500_harder_5trials_seed106_positions.json

run "${COMMON_ARGS[@]}" --online_action_set $OUT/online_action_set_arch2000.pkl \
  --methods guarded_skill_command \
  --num_trials 5 --seed 106 \
  --output $OUT/rsl_archive_ablation_arch2000_harder_5trials_seed106_positions.json

run "${COMMON_ARGS[@]}" --online_action_set "$ARCH" \
  --methods guarded_skill_command,direct_target_command \
  --num_trials 5 --seed 106 \
  --output $OUT/rsl_archive_ablation_archfull_harder_5trials_seed106_positions.json

echo "=== [$(date +%H:%M:%S)] ALL DONE ==="
