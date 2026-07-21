#!/usr/bin/env bash
# Collect exact-replay constant-scale prefix data for the adapter value audit.
# Writes large JSON under ~/storage and symlinks into the repo outputs tree so
# the nearly-full root disk is not used for Isaac temporary files or results.
set -uo pipefail
cd /home/thing1/skill_discover
PY=/home/thing1/miniconda3/envs/env_isaaclab/bin/python
CKPT=.pretrained_checkpoints/rsl_rl/Isaac-Velocity-Rough-Unitree-Go2-v0/checkpoint.pt
ARCH=outputs/isaac_go2_rough_rsl_rl_skills_long/online_action_set.pkl
MACRO=outputs/response_model/go2_macro_transition_h4_ensemble5.pt
TV=outputs/response_model/go2_direct_terminal_value_dagger_iter2.pt
TARGETS="2.0,0;1.5,1.0;0,1.25;-0.75,0;1.0,-1.0;0,-1.25;-1.0,0.75;1.75,-0.75"
OUT_DIR="$HOME/storage/skill_discover/outputs/isaac_go2_rough_rsl_rl_skills_long"
LINK_DIR=outputs/isaac_go2_rough_rsl_rl_skills_long
LOG_DIR="$HOME/storage/skill_discover/outputs/exact_replay_prefix_logs"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export TMPDIR="$HOME/storage/skill_discover/tmp"
mkdir -p "$TMPDIR" "$OUT_DIR" "$LOG_DIR" "$LINK_DIR"

if [[ $# -lt 1 ]]; then
  echo "usage: $0 SEED [SEED ...]" >&2
  exit 2
fi

for SEED in "$@"; do
  OUT="$OUT_DIR/completion_mpc_prefix_train_seed${SEED}.json"
  LINK="$LINK_DIR/completion_mpc_prefix_train_seed${SEED}.json"
  LOG="$LOG_DIR/seed${SEED}.log"
  if [[ -f "$OUT" && -s "$OUT" ]]; then
    echo "=== [$(date +%F_%H:%M:%S)] skip existing seed $SEED ==="
    ln -sfn "$OUT" "$LINK"
    continue
  fi
  echo "=== [$(date +%F_%H:%M:%S)] collecting exact-replay prefix seed $SEED ===" | tee -a "$LOG_DIR/driver.log"
  $PY scripts/evaluate_rsl_skill_command_control.py \
    --task Isaac-Velocity-Rough-Unitree-Go2-v0 \
    --checkpoint "$CKPT" \
    --online_action_set "$ARCH" \
    --macro_transition_model "$MACRO" \
    --task_value_model "$TV" \
    --methods completion_mpc_command \
    --mpc_scales "0.75,0.90,1.00" \
    --mpc_planning_steps 4 \
    --mpc_uncertainty_k 1.0 \
    --mpc_prefix_uncertainty_k 1.0 \
    --mpc_work_margin_frac 0.02 \
    --collect_mpc_dagger \
    --dagger_queries_per_episode 3 \
    --dagger_candidate_sequences "selected;0.75,0.75,0.75,0.75;0.90,0.90,0.90,0.90;1.00,1.00,1.00,1.00" \
    --targets "$TARGETS" \
    --num_trials 2 \
    --seed "$SEED" \
    --max_high_level_steps 80 \
    --execution_horizon 4 \
    --command_gain 1.0 \
    --command_max 1.0 \
    --yaw_command_gain 1.0 \
    --ff_height_scan_slice 48:235 \
    --relative_targets \
    --freeze_terrain_curriculum \
    --paired_method_resets \
    --headless \
    --output "$OUT" > "$LOG" 2>&1
  status=$?
  echo "=== [$(date +%F_%H:%M:%S)] seed $SEED exit=$status ===" | tee -a "$LOG_DIR/driver.log"
  if [[ $status -eq 0 && -s "$OUT" ]]; then
    ln -sfn "$OUT" "$LINK"
  else
    echo "seed $SEED failed; see $LOG" >&2
    exit "$status"
  fi
done
echo "=== [$(date +%F_%H:%M:%S)] ALL DONE: $* ===" | tee -a "$LOG_DIR/driver.log"
