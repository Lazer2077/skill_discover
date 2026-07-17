#!/usr/bin/env bash
# Heterogeneous-terrain paired frontier: direct / fixed-0.90 / fixed-0.75 / VGCC
# from identical initial states, on the all-flat and all-rough Go2 tasks.
# VGCC = model_feedforward_command with the hetero-identified response model and
# the paper's frozen quadruped gate defaults (no re-tuning).
# Usage: run_hetero_frontier.sh <seed> [seed...]
set -uo pipefail
cd /home/thing1/skill_discover
PY=/home/thing1/miniconda3/envs/env_isaaclab/bin/python
CKPT=.pretrained_checkpoints/rsl_rl/Isaac-Velocity-Rough-Unitree-Go2-v0/checkpoint.pt
ARCH=outputs/isaac_go2_rough_rsl_rl_skills_long/online_action_set.pkl
RM=/home/thing1/storage/skill_discover/outputs/hetero/hetero_go2_response_model.pt
TARGETS="2.0,0;1.5,1.0;0,1.25;-0.75,0;1.0,-1.0;0,-1.25;-1.0,0.75;1.75,-0.75"
OUT=/home/thing1/storage/skill_discover/outputs/hetero
export CUDA_VISIBLE_DEVICES=1
export XDG_CACHE_HOME=/home/thing1/storage/skill_discover/cache
export TMPDIR=/home/thing1/storage/skill_discover/tmp

for SEED in "$@"; do
  for T in Flat Rough; do
    low=$(echo "$T" | tr 'A-Z' 'a-z')
    echo "=== [$(date +%H:%M:%S)] seed $SEED $T ==="
    $PY scripts/eval_hetero.py scripts/evaluate_rsl_skill_command_control.py \
      --task Isaac-Velocity-Hetero${T}-Unitree-Go2-v0 \
      --checkpoint "$CKPT" --online_action_set "$ARCH" --response_model "$RM" \
      --targets "$TARGETS" --num_trials 2 --seed "$SEED" \
      --max_high_level_steps 80 --execution_horizon 4 \
      --command_gain 1.0 --command_max 1.0 --yaw_command_gain 1.0 \
      --ff_height_scan_slice 48:235 \
      --lambda_state 2.0 --lambda_energy 0.05 --lambda_stability 0.5 \
      --lambda_no_progress 1.5 --lambda_utility 0.1 --lambda_progress 0.25 \
      --k_nearest 16 --skill_target_command_blend 1.0 \
      --relative_targets --store_positions --headless --paired_method_resets \
      --freeze_terrain_curriculum \
      --methods direct_target_command,scaled_target_command_75,scaled_target_command_90,model_feedforward_command \
      --output "$OUT/frontier_${low}_seed${SEED}.json" \
      > "$OUT/frontier_${low}_seed${SEED}.log" 2>&1
    echo "=== [$(date +%H:%M:%S)] seed $SEED $T exit=$? ==="
  done
done
echo "=== HETERO FRONTIER DONE: $* ==="
