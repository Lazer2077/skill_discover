#!/usr/bin/env bash
# Evaluate the improved gate objective (proxy_efficiency_feedforward_command =
# cost-per-progress objective + execution-aligned scoring; everything else equal
# to the evaluated model_feedforward_command) as single-method paired runs:
#   1. hetero terrains (flat/mid/rough) x seeds 811-813, hetero response model
#   2. the paper's uniform-rough benchmark x seeds 797-799, original response
#      model, paired against the existing E1 frontier records
set -uo pipefail
cd /home/thing1/skill_discover
PY=/home/thing1/miniconda3/envs/env_isaaclab/bin/python
CKPT=.pretrained_checkpoints/rsl_rl/Isaac-Velocity-Rough-Unitree-Go2-v0/checkpoint.pt
ARCH=outputs/isaac_go2_rough_rsl_rl_skills_long/online_action_set.pkl
HRM=/home/thing1/storage/skill_discover/outputs/hetero/hetero_go2_response_model.pt
ORM=outputs/response_model/go2_command_response_model.pt
TARGETS="2.0,0;1.5,1.0;0,1.25;-0.75,0;1.0,-1.0;0,-1.25;-1.0,0.75;1.75,-0.75"
OUT=/home/thing1/storage/skill_discover/outputs/hetero
export CUDA_VISIBLE_DEVICES=1
export XDG_CACHE_HOME=/home/thing1/storage/skill_discover/cache
export TMPDIR=/home/thing1/storage/skill_discover/tmp

run_eval () { # task rm output seed extra_flag
  $PY scripts/eval_hetero.py scripts/evaluate_rsl_skill_command_control.py \
    --task "$1" \
    --checkpoint "$CKPT" --online_action_set "$ARCH" --response_model "$2" \
    --targets "$TARGETS" --num_trials 2 --seed "$4" \
    --max_high_level_steps 80 --execution_horizon 4 \
    --command_gain 1.0 --command_max 1.0 --yaw_command_gain 1.0 \
    --ff_height_scan_slice 48:235 \
    --lambda_state 2.0 --lambda_energy 0.05 --lambda_stability 0.5 \
    --lambda_no_progress 1.5 --lambda_utility 0.1 --lambda_progress 0.25 \
    --k_nearest 16 --skill_target_command_blend 1.0 \
    --relative_targets --store_positions --headless --paired_method_resets \
    ${5:-} \
    --methods proxy_efficiency_feedforward_command \
    --output "$3" > "${3%.json}.log" 2>&1
}

for SEED in 811 812 813; do
  for T in Flat Mid Rough; do
    low=$(echo "$T" | tr 'A-Z' 'a-z')
    echo "=== [$(date +%H:%M:%S)] pe $T seed $SEED ==="
    run_eval Isaac-Velocity-Hetero${T}-Unitree-Go2-v0 "$HRM" \
      "$OUT/pe_${low}_seed${SEED}.json" "$SEED" --freeze_terrain_curriculum
    echo "=== [$(date +%H:%M:%S)] pe $T seed $SEED exit=$? ==="
  done
done

for SEED in 797 798 799; do
  echo "=== [$(date +%H:%M:%S)] pe uniform-rough seed $SEED ==="
  run_eval Isaac-Velocity-Rough-Unitree-Go2-v0 "$ORM" \
    "$OUT/pe_uniform_seed${SEED}.json" "$SEED" ""
  echo "=== [$(date +%H:%M:%S)] pe uniform seed $SEED exit=$? ==="
done
echo "=== PROXY-EFFICIENCY RUNS DONE ==="
