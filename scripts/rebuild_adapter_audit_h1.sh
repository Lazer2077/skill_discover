#!/usr/bin/env bash
# Rebuild H1 Stages 2–3 adapter necessity audit from exact-replay prefixes.
set -euo pipefail
cd /home/thing1/skill_discover
OUT=outputs/response_model
PREFIX=outputs/isaac_h1_rough_exact_replay
SEEDS=(870 871 872 873 874 875 876 877 878 879)
INPUTS=()
for seed in "${SEEDS[@]}"; do
  path="$PREFIX/completion_mpc_prefix_train_seed${seed}.json"
  if [[ -s "$path" ]]; then
    INPUTS+=("$path")
  fi
done
if [[ ${#INPUTS[@]} -lt 5 ]]; then
  echo "need at least 5 H1 prefix files; found ${#INPUTS[@]}" >&2
  exit 1
fi
N=${#INPUTS[@]}
echo "Using $N H1 source seeds:"
printf '  %s\n' "${INPUTS[@]}"

python scripts/analyze_exact_replay_oracle.py \
  --inputs "${INPUTS[@]}" \
  --output "$PREFIX/exact_replay_prefix_oracle_${N}seed.json"

python scripts/train_exact_replay_treatment_effect.py \
  --inputs "${INPUTS[@]}" \
  --feature-set observation \
  --height-scan-slice 69:256 \
  --output "$OUT/exact_replay_adapter_audit_h1_observation_${N}seed.json"

python scripts/summarize_adapter_value_audit.py \
  --inputs "$OUT/exact_replay_adapter_audit_h1_observation_${N}seed.json" \
  --output "$OUT/adapter_value_audit_h1_${N}seed_summary.json"

# Lightweight calibration on whatever seed count we have.
for strength in 0.05 0.09; do
  label=${strength/0./}
  python scripts/train_exact_replay_treatment_effect.py \
    --inputs "${INPUTS[@]}" \
    --feature-set observation \
    --height-scan-slice 69:256 \
    --control-mode observable \
    --control-strength "$strength" \
    --output "$OUT/adapter_audit_h1_control_observable_${label}pct_${N}seed.json"
done

python scripts/summarize_adapter_value_audit.py \
  --inputs "$OUT"/adapter_audit_h1_control_observable_*pct_${N}seed.json \
  --output "$OUT/adapter_value_audit_h1_observable_curve_${N}seed_summary.json"

echo "H1 ${N}-seed audit artifacts written under $OUT and $PREFIX"
