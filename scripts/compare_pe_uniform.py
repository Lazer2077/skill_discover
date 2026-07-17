#!/usr/bin/env python3
"""Non-degradation check on the paper's uniform-rough benchmark: compare the
improved gate objective (proxy_efficiency_feedforward_command, run paired on
seeds 797-799) against the archived E1 frontier records (direct, fixed scales,
VGCC-as-evaluated) episode by episode.

Usage: compare_pe_uniform.py <pe_uniform_seed*.json ...> --e1 <frontier_seed*.json ...>
"""
import json
import sys
from collections import defaultdict

import numpy as np

args = sys.argv[1:]
split = args.index("--e1")
PE_FILES, E1_FILES = args[:split], args[split + 1:]

BASELINES = [
    ("direct_target_command", "direct 1.00"),
    ("scaled_target_command_90", "fixed 0.90"),
    ("scaled_target_command_75", "fixed 0.75"),
    ("model_feedforward_command", "VGCC (evaluated)"),
    ("mechanical_feedforward_command", "VGCC (power)"),
]
PE = "proxy_efficiency_feedforward_command"


def load(files):
    ep = defaultdict(dict)
    for f in files:
        d = json.load(open(f))
        seed = d.get("controller", {}).get("seed") if isinstance(d.get("controller"), dict) else None
        if seed is None:
            import re
            m = re.search(r"seed(\d+)", f)
            seed = m.group(1) if m else "?"
        for tk, ms in d["records"].items():
            for meth, rs in ms.items():
                for r in rs:
                    ep[(str(seed), tk, r["trial"])][meth] = r
    return ep


ep = defaultdict(dict)
for k, md in load(PE_FILES).items():
    ep[k].update(md)
for k, md in load(E1_FILES).items():
    ep[k].update(md)

pairs = {k: md for k, md in ep.items() if PE in md and "direct_target_command" in md}
print(f"paired episodes: {len(pairs)}")

# pairing sanity: identical initial states imply identical target positions
mism = sum(
    1 for md in pairs.values()
    if not np.allclose(md[PE]["target_position"], md["direct_target_command"]["target_position"], atol=1e-6)
)
print(f"target-position mismatches (should be 0): {mism}")


def stats(sel, m):
    rs = [md[m] for md in sel if m in md]
    if not rs:
        return None
    return (np.mean([r["success"] for r in rs]),
            np.mean([r["cost_of_transport"] for r in rs]),
            np.mean([r["mechanical_energy_j"] for r in rs]),
            np.mean([r["mean_mechanical_power_w"] for r in rs]),
            np.mean([r["elapsed_time_s"] for r in rs]),
            len(rs))


sel = list(pairs.values())
print("\nALL paired episodes (same initial states):")
print(f"   {'method':<18} {'succ':>6} {'COT':>8} {'work J':>8} {'power W':>8} {'time s':>7}")
for m, label in BASELINES + [(PE, "pe (improved)")]:
    s = stats(sel, m)
    if s:
        print(f"   {label:<18} {s[0]:6.3f} {s[1]:8.4f} {s[2]:8.2f} {s[3]:8.2f} {s[4]:7.3f}  n={s[5]}")

# paired-successful basis for COT comparability
methods_present = [m for m, _ in BASELINES if any(m in md for md in sel)] + [PE]
ok = [md for md in sel if all(m in md and md[m]["success"] for m in methods_present)]
print(f"\npaired-successful basis (all methods succeed): n={len(ok)}")
for m, label in BASELINES + [(PE, "pe (improved)")]:
    s = stats(ok, m)
    if s:
        print(f"   {label:<18} {s[0]:6.3f} {s[1]:8.4f} {s[2]:8.2f} {s[3]:8.2f} {s[4]:7.3f}")

# per-seed paired deltas pe vs direct and pe vs VGCC
print("\nper-seed paired deltas (all episodes, pe - baseline):")
for base in ["direct_target_command", "model_feedforward_command"]:
    for seed in sorted({k[0] for k in pairs}):
        sub = [md for k, md in pairs.items() if k[0] == seed and base in md]
        if not sub:
            continue
        dW = np.mean([md[PE]["mechanical_energy_j"] - md[base]["mechanical_energy_j"] for md in sub])
        dS = np.mean([md[PE]["success"] - md[base]["success"] for md in sub])
        dC = np.mean([md[PE]["cost_of_transport"] - md[base]["cost_of_transport"] for md in sub])
        print(f"   vs {base:<28} seed {seed}: dWork {dW:+7.2f} J  dCOT {dC:+7.4f}  dSucc {dS:+6.3f}")
