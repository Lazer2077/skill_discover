#!/usr/bin/env python3
"""C(ii) go/no-go probe: is there STATE-DEPENDENT adaptivity headroom that a single
global fixed scale cannot capture, and does VGCC capture any of it?

Uses existing PAIRED frontier data (all methods share initial state per target×trial).
For each episode we compare the available scales {direct=1.0, 0.90, 0.75} and VGCC.
Reports, over successful-in-all-methods episodes:
  - best SINGLE GLOBAL scale (min mean COT over the pool)
  - ORACLE per-episode best scale COT (upper bound on adaptivity among these scales)
  - VGCC COT
  - headroom = best_global - oracle ; captured = best_global - VGCC
Also: how often each scale is the per-episode winner (if concentrated on one scale,
adaptivity is useless here).
"""
import json
import sys
from collections import defaultdict

import numpy as np

FILES = sys.argv[1:]
SCALES = {
    "direct_target_command": "1.00",
    "scaled_target_command_90": "0.90",
    "scaled_target_command_75": "0.75",
}
VGCC = "model_feedforward_command"


def load_paired(files):
    # key: (file, target, trial) -> {method: record}
    ep = defaultdict(dict)
    for f in files:
        d = json.load(open(f))
        for tk, ms in d["records"].items():
            for m, rs in ms.items():
                for r in rs:
                    ep[(f, tk, r["trial"])][m] = r
    return ep


def main():
    ep = load_paired(FILES)
    scale_methods = list(SCALES)
    # keep episodes where all scale methods AND vgcc have a record and all succeed
    rows = []
    for key, md in ep.items():
        if not all(m in md for m in scale_methods + [VGCC]):
            continue
        if not all(md[m]["success"] for m in scale_methods + [VGCC]):
            continue
        cots = {m: md[m]["cost_of_transport"] for m in scale_methods}
        rows.append((cots, md[VGCC]["cost_of_transport"]))
    n = len(rows)
    if n == 0:
        print("no fully-successful paired episodes")
        return
    # mean COT per global scale
    mean_scale = {m: np.mean([r[0][m] for r in rows]) for m in scale_methods}
    best_global_m = min(mean_scale, key=mean_scale.get)
    best_global = mean_scale[best_global_m]
    # oracle per-episode best among the 3 scales
    oracle = np.mean([min(r[0].values()) for r in rows])
    vgcc = np.mean([r[1] for r in rows])
    # winner distribution
    win = defaultdict(int)
    for r in rows:
        win[min(r[0], key=r[0].get)] += 1

    print(f"n fully-successful paired episodes = {n}")
    print("mean COT by single global scale:")
    for m in scale_methods:
        print(f"   {SCALES[m]:>5} : {mean_scale[m]:.4f}")
    print(f"best SINGLE GLOBAL scale        = {SCALES[best_global_m]} (COT {best_global:.4f})")
    print(f"ORACLE per-episode best scale   = COT {oracle:.4f}")
    print(f"VGCC                            = COT {vgcc:.4f}")
    head = best_global - oracle
    print(f"\nADAPTIVITY HEADROOM (best_global - oracle) = {head:.4f} "
          f"({100*head/best_global:.1f}% of best global)")
    print(f"VGCC vs best_global = {best_global - vgcc:+.4f} "
          f"({100*(best_global-vgcc)/best_global:+.1f}%)  "
          f"[positive = VGCC beats the best single scale]")
    print("per-episode winning scale distribution:")
    for m in scale_methods:
        print(f"   {SCALES[m]:>5} wins {win[m]:>3}/{n} ({100*win[m]/n:.0f}%)")
    print("\nVERDICT:", end=" ")
    if head / best_global < 0.02:
        print("NO-GO — a single global scale is ~optimal; adaptivity headroom < 2%.")
    elif best_global - vgcc > 0:
        print("PROMISING — headroom exists AND VGCC already beats the best single scale.")
    else:
        print("HEADROOM EXISTS but VGCC does not capture it here; needs finer scales / "
              "true heterogeneous terrain to test whether an adaptive gate can.")


if __name__ == "__main__":
    main()
