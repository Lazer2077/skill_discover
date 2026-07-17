#!/usr/bin/env bash
# Supplement to run_hetero_frontier.sh:
#   1. Mid-difficulty rough (0.6) x 5 methods: healthy-success rough basis.
#   2. scale-0.60 column added to the existing Flat and Rough runs (paired resets
#      are deterministic per (seed, target, trial), verified by the check below).
#   3. pairing check: re-run direct on Flat seed 811 alone; records must match the
#      4-method run exactly.
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

run_eval () { # task methods output seed
  $PY scripts/eval_hetero.py scripts/evaluate_rsl_skill_command_control.py \
    --task "$1" \
    --checkpoint "$CKPT" --online_action_set "$ARCH" --response_model "$RM" \
    --targets "$TARGETS" --num_trials 2 --seed "$4" \
    --max_high_level_steps 80 --execution_horizon 4 \
    --command_gain 1.0 --command_max 1.0 --yaw_command_gain 1.0 \
    --ff_height_scan_slice 48:235 \
    --lambda_state 2.0 --lambda_energy 0.05 --lambda_stability 0.5 \
    --lambda_no_progress 1.5 --lambda_utility 0.1 --lambda_progress 0.25 \
    --k_nearest 16 --skill_target_command_blend 1.0 \
    --relative_targets --store_positions --headless --paired_method_resets \
    --freeze_terrain_curriculum \
    --methods "$2" \
    --output "$3" > "${3%.json}.log" 2>&1
}

echo "=== [$(date +%H:%M:%S)] pairing check: Flat 811 direct-only ==="
run_eval Isaac-Velocity-HeteroFlat-Unitree-Go2-v0 direct_target_command \
  "$OUT/paircheck_flat_seed811.json" 811
echo "=== [$(date +%H:%M:%S)] exit=$? ==="

for SEED in 811 812 813; do
  echo "=== [$(date +%H:%M:%S)] seed $SEED Mid x5 ==="
  run_eval Isaac-Velocity-HeteroMid-Unitree-Go2-v0 \
    direct_target_command,scaled_target_command_60,scaled_target_command_75,scaled_target_command_90,model_feedforward_command \
    "$OUT/frontier_mid_seed${SEED}.json" "$SEED"
  echo "=== [$(date +%H:%M:%S)] seed $SEED Mid exit=$? ==="
  for T in Flat Rough; do
    low=$(echo "$T" | tr 'A-Z' 'a-z')
    echo "=== [$(date +%H:%M:%S)] seed $SEED $T scale-0.60 ==="
    run_eval Isaac-Velocity-Hetero${T}-Unitree-Go2-v0 scaled_target_command_60 \
      "$OUT/s60_${low}_seed${SEED}.json" "$SEED"
    echo "=== [$(date +%H:%M:%S)] seed $SEED $T s60 exit=$? ==="
  done
done
echo "=== HETERO SUPPLEMENT DONE ==="
