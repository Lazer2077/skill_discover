#!/usr/bin/env bash
# Master driver for E3/E4/E5/E6 on GPU 1, sequential. Waits for the seed-901 H1
# test to free the GPU first.
set -uo pipefail
cd /home/thing1/skill_discover
export CUDA_VISIBLE_DEVICES=1

echo "=== [$(date +%H:%M:%S)] waiting for seed-901 H1 eval to finish ==="
while pgrep -f "e3_h1_seed901" >/dev/null 2>&1; do sleep 20; done

# E3 (remaining originals) + E5 (fresh confirmation seeds)
echo "=== [$(date +%H:%M:%S)] E3/E5 H1 seeds 902 903 904 905 906 ==="
bash scripts/run_e3_h1_frontier.sh 902 903 904 905 906

# E6 sampling-MPC baseline
echo "=== [$(date +%H:%M:%S)] E6 MPC seeds 797 798 799 ==="
bash scripts/run_e6_mpc.sh 797 798 799

# E4 extra Go2 frontier seeds (statistical strengthening)
echo "=== [$(date +%H:%M:%S)] E4 Go2 frontier seeds 794 795 796 ==="
bash scripts/run_e1_frontier.sh 794 795 796

echo "=== [$(date +%H:%M:%S)] MASTER E3-E6 ALL DONE ==="
