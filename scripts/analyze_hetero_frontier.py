#!/usr/bin/env python3
"""Heterogeneous-terrain oracle analysis (the paper's decisive open experiment).

Reads the paired frontier JSONs from run_hetero_frontier.sh (filenames carry the
terrain: frontier_flat_seed*.json / frontier_rough_seed*.json) and reports:
  1. per-terrain frontier: mean COT / work / power / time / success per method;
  2. per-episode winning-scale distribution, per terrain and pooled;
  3. pooled per-episode oracle vs best single global scale (adaptivity headroom),
     on episodes where all methods succeed (paired basis, as in the paper);
  4. VGCC's position: does one frozen configuration beat the best single fixed
     scale on the pooled heterogeneous deployment?

Usage: analyze_hetero_frontier.py <frontier_*.json ...>
"""
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

SCALES = {
    "direct_target_command": "1.00",
    "scaled_target_command_90": "0.90",
    "scaled_target_command_75": "0.75",
    "scaled_target_command_60": "0.60",
}
VGCC = os.environ.get("VGCC_METHOD", "model_feedforward_command")
ALL = list(SCALES) + [VGCC]


def terrain_of(path):
    for t in ("flat", "mid", "rough"):
        if t in path.lower():
            return t
    return "?"


def seed_of(path):
    m = re.search(r"seed(\d+)", path)
    return m.group(1) if m else "?"


def load(files):
    ep = defaultdict(dict)  # (terrain, seed, target, trial) -> {method: rec}
    for f in files:
        d = json.load(open(f))
        for tk, ms in d["records"].items():
            for m, rs in ms.items():
                for r in rs:
                    ep[(terrain_of(f), seed_of(f), tk, r["trial"])][m] = r
    return ep


def fmt_row(name, recs):
    return (f"   {name:<12} succ {np.mean([r['success'] for r in recs]):.3f}"
            f" | COT {np.mean([r['cost_of_transport'] for r in recs]):.4f}"
            f" | work {np.mean([r['mechanical_energy_j'] for r in recs]):7.2f} J"
            f" | power {np.mean([r['mean_mechanical_power_w'] for r in recs]):6.2f} W"
            f" | time {np.mean([r['elapsed_time_s'] for r in recs]):.3f} s"
            f" | n={len(recs)}")


def main():
    ep = load(sys.argv[1:])
    terrains = sorted({k[0] for k in ep})

    print("=" * 78)
    print("1) PER-TERRAIN FRONTIER (all episodes, per method)")
    for t in terrains:
        print(f" {t}:")
        for m in ALL:
            recs = [md[m] for k, md in ep.items() if k[0] == t and m in md]
            if recs:
                label = f"fixed {SCALES[m]}" if m in SCALES else "VGCC"
                if m == "direct_target_command":
                    label = "direct 1.00"
                print(fmt_row(label, recs))

    # paired-successful basis
    have = set()
    for md in ep.values():
        have.update(md)
    scale_present = [m for m in SCALES if m in have]
    needed = scale_present + [VGCC]
    rows = []  # (terrain, {scale_method: cot}, vgcc_cot)
    for k, md in ep.items():
        if not all(m in md for m in needed):
            continue
        if not all(md[m]["success"] for m in needed):
            continue
        rows.append((k[0], {m: md[m]["cost_of_transport"] for m in scale_present},
                     md[VGCC]["cost_of_transport"]))
    n = len(rows)
    print("=" * 78)
    print(f"2) PAIRED-SUCCESSFUL EPISODES: n={n} "
          f"({', '.join(f'{t}: {sum(1 for r in rows if r[0]==t)}' for t in terrains)})")
    for t in terrains + ["pooled"]:
        sel = [r for r in rows if t == "pooled" or r[0] == t]
        if not sel:
            continue
        win = defaultdict(int)
        for r in sel:
            win[min(r[1], key=r[1].get)] += 1
        dist = ", ".join(f"{SCALES[m]}: {win[m]}/{len(sel)} ({100*win[m]/len(sel):.0f}%)"
                         for m in scale_present)
        print(f"   winners [{t}]  {dist}")

    print("=" * 78)
    print("3) POOLED ORACLE ANALYSIS (the paper's headroom quantity)")
    mean_scale = {m: np.mean([r[1][m] for r in rows]) for m in scale_present}
    for m in scale_present:
        print(f"   single global {SCALES[m]:>5} : COT {mean_scale[m]:.4f}")
    best_m = min(mean_scale, key=mean_scale.get)
    best = mean_scale[best_m]
    oracle = np.mean([min(r[1].values()) for r in rows])
    # terrain-wise fixed scale (per-terrain best constant): the deployment-relevant reference
    tw = np.mean([min(np.mean([q[1][m] for q in rows if q[0] == r[0]]) for m in scale_present)
                  for r in rows])
    vgcc = np.mean([r[2] for r in rows])
    head = best - oracle
    print(f"   best SINGLE global scale     = {SCALES[best_m]}  COT {best:.4f}")
    print(f"   PER-EPISODE oracle           = COT {oracle:.4f}")
    print(f"   ADAPTIVITY HEADROOM          = {head:.4f}  ({100*head/best:.1f}% of best global)")
    print(f"   VGCC (one frozen config)     = COT {vgcc:.4f}  "
          f"(vs best global {100*(best-vgcc)/best:+.1f}%; positive = VGCC wins)")

    print("=" * 78)
    print("4) VGCC MECHANISM: does the gate behave differently per terrain?")
    for t in terrains:
        acts, scales = [], []
        for k, md in ep.items():
            if k[0] != t or VGCC not in md:
                continue
            dec = md[VGCC].get("decisions") or []
            for d in dec:
                if not isinstance(d, dict) or "ff_active" not in d:
                    continue
                acts.append(bool(d["ff_active"]))
                if d["ff_active"] and d.get("command") is not None and d.get("ff_mode") == "follow":
                    # executed-command magnitude relative to a unit direct command is
                    # not recoverable without the direct command; report predicted
                    # progress ratio instead (executed vs direct)
                    pd = d.get("pred_progress_direct") or 0.0
                    pb = d.get("pred_progress_best")
                    if pb is not None and pd and pd > 1e-6:
                        scales.append(pb / pd)
        if acts:
            r = 100.0 * np.mean(acts)
            msg = f"   [{t}] compensation active {r:.0f}% of decisions (n={len(acts)})"
            if scales:
                msg += (f"; when active, predicted progress of the chosen candidate is "
                        f"{np.mean(scales):.2f}x direct's (median {np.median(scales):.2f})")
            print(msg)
    print("=" * 78)
    print("5) reference: the paper's uniform-terrain headroom was 1.0-1.9%")


if __name__ == "__main__":
    main()
