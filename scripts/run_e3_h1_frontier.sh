#!/usr/bin/env bash
# E3/E5: H1 safety comparison. fixed-scaling (0.75/0.90) vs VGCC vs direct, paired.
# Tests whether an aggressive fixed scale collapses humanoid posture where VGCC does not.
# H1 VGCC config replicated from ff_v7_h1solved (morphology-adapted posture floors).
set -uo pipefail
cd /home/thing1/skill_discover
PY=/home/thing1/miniconda3/envs/env_isaaclab/bin/python
CKPT=.pretrained_checkpoints/rsl_rl/Isaac-Velocity-Rough-H1-v0/checkpoint.pt
ARCH=outputs/isaac_h1_rough_rsl_rl_skills/online_action_set.pkl
RM=outputs/response_model/h1_command_response_model_v2.pt
TARGETS="2.0,0;1.5,1.0;0,1.25;-0.75,0;1.0,-1.0;0,-1.25;-1.0,0.75;1.75,-0.75"
export CUDA_VISIBLE_DEVICES=1
mkdir -p outputs/e3_h1_frontier

for SEED in "$@"; do
  echo "=== [$(date +%H:%M:%S)] H1 seed $SEED ==="
  $PY scripts/evaluate_rsl_skill_command_control.py \
    --task Isaac-Velocity-Rough-H1-v0 \
    --checkpoint "$CKPT" --online_action_set "$ARCH" --response_model "$RM" \
    --targets "$TARGETS" --num_trials 5 --seed "$SEED" \
    --max_high_level_steps 80 --execution_horizon 4 \
    --command_gain 1.0 --command_max 1.0 --yaw_command_gain 1.0 \
    --ff_height_scan_slice 69:256 \
    --ff_min_height_fraction 0.93 --ff_rescue_height_fraction 0.90 \
    --ff_current_height_fraction 0.96 --ff_max_height_drop 0.008 \
    --ff_energy_margin_frac 0.1 --ff_progress_floor 0.9 \
    --lambda_no_progress 2.0 --lambda_progress 2.0 --k_nearest 0 --skill_target_command_blend 0.5 \
    --relative_targets --store_positions --headless --paired_method_resets \
    --methods direct_target_command,scaled_target_command_75,scaled_target_command_90,model_feedforward_command \
    --output outputs/e3_h1_frontier/e3_h1_seed${SEED}.json > outputs/e3_h1_frontier/e3_h1_seed${SEED}.log 2>&1
  echo "=== [$(date +%H:%M:%S)] H1 seed $SEED exit=$? ==="
done
echo "=== E3/E5 H1 DONE: $* ==="
