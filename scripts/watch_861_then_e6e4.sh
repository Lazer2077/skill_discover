#!/usr/bin/env bash
# Wait for the OTHER project's run (seed 861) to finish, then run E6 (MPC) and
# E4 (extra Go2 frontier seeds). All output + temp on ~/storage; no store_positions
# (keeps files tiny so the full root disk is never touched for writes).
set -uo pipefail
cd /home/thing1/skill_discover
PY=/home/thing1/miniconda3/envs/env_isaaclab/bin/python
CKPT=.pretrained_checkpoints/rsl_rl/Isaac-Velocity-Rough-Unitree-Go2-v0/checkpoint.pt
ARCH=outputs/isaac_go2_rough_rsl_rl_skills_long/online_action_set.pkl
RM=outputs/response_model/go2_command_response_power_ensemble5.pt
TARGETS="2.0,0;1.5,1.0;0,1.25;-0.75,0;1.0,-1.0;0,-1.25;-1.0,0.75;1.75,-0.75"
export CUDA_VISIBLE_DEVICES=1
export TMPDIR="$HOME/storage/skill_discover/tmp"; mkdir -p "$TMPDIR"
LOG="$HOME/storage/skill_discover/outputs/e6e4_driver.log"
mkdir -p "$HOME/storage/skill_discover/outputs/e1_isotime"

echo "=== [$(date +%F_%H:%M:%S)] waiting for other project's seed 861 to finish ===" | tee -a "$LOG"
while pgrep -f "seed 861" >/dev/null 2>&1; do sleep 60; done
echo "=== [$(date +%F_%H:%M:%S)] seed 861 finished; starting E6 then E4 ===" | tee -a "$LOG"

# --- E6: sampling-MPC baseline ---
bash scripts/run_e6_mpc.sh 797 798 799 >> "$LOG" 2>&1
echo "=== [$(date +%F_%H:%M:%S)] E6 done ===" | tee -a "$LOG"

# --- E4: extra Go2 frontier seeds (no store_positions; metrics only) ---
for SEED in 794 795 796; do
  echo "=== [$(date +%F_%H:%M:%S)] E4 frontier seed $SEED ===" | tee -a "$LOG"
  $PY scripts/evaluate_rsl_skill_command_control.py \
    --task Isaac-Velocity-Rough-Unitree-Go2-v0 \
    --checkpoint "$CKPT" --online_action_set "$ARCH" --response_model "$RM" \
    --targets "$TARGETS" --num_trials 2 --seed "$SEED" \
    --max_high_level_steps 80 --execution_horizon 4 \
    --command_gain 1.0 --command_max 1.0 --yaw_command_gain 1.0 \
    --ff_height_scan_slice 48:235 \
    --lambda_state 2.0 --lambda_energy 0.05 --lambda_stability 0.5 \
    --lambda_no_progress 1.5 --lambda_utility 0.1 --lambda_progress 0.25 \
    --k_nearest 16 --skill_target_command_blend 1.0 \
    --relative_targets --headless --paired_method_resets \
    --methods direct_target_command,scaled_target_command_75,scaled_target_command_90,reactive_governor_command,model_feedforward_command,mechanical_feedforward_command \
    --output outputs/e1_isotime/frontier_seed${SEED}.json >> "$LOG" 2>&1
  echo "=== [$(date +%F_%H:%M:%S)] E4 seed $SEED exit=$? ===" | tee -a "$LOG"
done

touch "$HOME/storage/skill_discover/outputs/E6E4_DONE"
echo "=== [$(date +%F_%H:%M:%S)] E6+E4 ALL DONE ===" | tee -a "$LOG"
