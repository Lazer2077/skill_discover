#!/usr/bin/env bash
# Systematic experiments for model-feedforward compensation (v2):
#   Go2 harder   : seeds 201-205 x 10 trials, 3 methods
#   Go2 moderate : seeds 211-213 x 10 trials, 3 methods
#   Go2 holdout  : seeds 221-223 x 5 trials,  3 methods
#   H1           : seeds 231-233 x 5 trials,  3 methods (collect+train model first)
set -uo pipefail

PY=/home/thing1/miniconda3/envs/env_isaaclab/bin/python
GO2_CKPT=.pretrained_checkpoints/rsl_rl/Isaac-Velocity-Rough-Unitree-Go2-v0/checkpoint.pt
H1_CKPT=.pretrained_checkpoints/rsl_rl/Isaac-Velocity-Rough-H1-v0/checkpoint.pt
GO2_OUT=outputs/isaac_go2_rough_rsl_rl_skills_long
H1_OUT=outputs/isaac_h1_rough_rsl_rl_skills
GO2_ARCH=$GO2_OUT/online_action_set.pkl
H1_ARCH=$H1_OUT/online_action_set.pkl
GO2_RM=outputs/response_model/go2_command_response_model.pt
H1_RM=outputs/response_model/h1_command_response_model_v2.pt

METHODS=model_feedforward_command,guarded_skill_command,direct_target_command

GO2_ARGS=(
  --task Isaac-Velocity-Rough-Unitree-Go2-v0
  --checkpoint "$GO2_CKPT"
  --online_action_set "$GO2_ARCH"
  --response_model "$GO2_RM"
  --max_high_level_steps 80 --execution_horizon 4
  --lambda_state 0.0 --lambda_energy 0.0 --lambda_stability 0.0
  --lambda_no_progress 2.0 --lambda_utility 0.0 --lambda_progress 2.0
  --k_nearest 0 --skill_target_command_blend 0.5
  --guard_recovery_height_fraction 0.72
  --relative_targets --store_positions --headless
  --methods "$METHODS"
)

H1_ARGS=(
  --task Isaac-Velocity-Rough-H1-v0
  --checkpoint "$H1_CKPT"
  --online_action_set "$H1_ARCH"
  --response_model "$H1_RM"
  --ff_height_scan_slice 69:256
  --max_high_level_steps 80 --execution_horizon 4
  --lambda_state 0.0 --lambda_energy 0.0 --lambda_stability 0.0
  --lambda_no_progress 2.0 --lambda_utility 0.0 --lambda_progress 2.0
  --k_nearest 0 --skill_target_command_blend 0.5
  --relative_targets --store_positions --headless
  --methods "$METHODS"
)

run() {
  echo "=== [$(date +%H:%M:%S)] $1 ==="
  shift
  "$PY" "$@" || echo "!!! FAILED: $1"
}

# --- H1 response model (collect + train) ---
if [ ! -f "$H1_RM" ]; then
  run "h1 collect" scripts/collect_response_dataset.py \
    --task Isaac-Velocity-Rough-H1-v0 --checkpoint "$H1_CKPT" \
    --num_envs 64 --steps_per_env 8000 --seed 7 --headless \
    --command_slice 9:12 --height_scan_slice 69:256 \
    --output outputs/response_model/h1_response_dataset_v2.npz
  run "h1 train" scripts/train_command_response_model.py \
    --dataset outputs/response_model/h1_response_dataset_v2.npz \
    --output "$H1_RM"
fi

# --- Go2 harder: 5 seeds x 10 trials ---
for SEED in 201 202 203 204 205; do
  run "go2 harder seed $SEED" scripts/evaluate_rsl_skill_command_control.py \
    "${GO2_ARGS[@]}" \
    --targets "2.0,0;1.5,1.0;0,1.25;-0.75,0;1.0,-1.0" \
    --num_trials 10 --seed "$SEED" \
    --output $GO2_OUT/ff_v3_harder_10trials_seed${SEED}_positions.json
done

# --- Go2 moderate: 3 seeds x 10 trials ---
for SEED in 211 212 213; do
  run "go2 moderate seed $SEED" scripts/evaluate_rsl_skill_command_control.py \
    "${GO2_ARGS[@]}" \
    --targets "0.75,0;0.75,0.75;1.5,0;0,0.75" \
    --num_trials 10 --seed "$SEED" \
    --output $GO2_OUT/ff_v3_moderate_10trials_seed${SEED}_positions.json
done

# --- Go2 holdout: 3 seeds x 5 trials ---
for SEED in 221 222 223; do
  run "go2 holdout seed $SEED" scripts/evaluate_rsl_skill_command_control.py \
    "${GO2_ARGS[@]}" \
    --targets "1.25,0.5;0.5,1.0;-0.5,0.5;1.25,-0.5" \
    --num_trials 5 --seed "$SEED" \
    --output $GO2_OUT/ff_v3_holdout_5trials_seed${SEED}_positions.json
done

# --- H1: 3 seeds x 5 trials ---
for SEED in 231 232 233; do
  run "h1 seed $SEED" scripts/evaluate_rsl_skill_command_control.py \
    "${H1_ARGS[@]}" \
    --ff_energy_margin_frac 0.25 --targets "0.5,0;0.5,0.5;1.0,0" \
    --num_trials 5 --seed "$SEED" \
    --output $H1_OUT/ff_v3_h1_5trials_seed${SEED}_positions.json
done

echo "=== [$(date +%H:%M:%S)] ALL DONE ==="
