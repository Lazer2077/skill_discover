#!/usr/bin/env python3
"""Pool per-method records across targets and seeds for the iso-time frontier (E1/E6).

Usage: aggregate_e1_frontier.py file1.json [file2.json ...]
Prints a per-method table pooled over ALL episodes in ALL given files.
"""
import json
import sys
from collections import defaultdict

import numpy as np


def pool(files):
    by_method = defaultdict(list)
    for f in files:
        d = json.load(open(f))
        for tk, methods in d["records"].items():
            for m, recs in methods.items():
                by_method[m].extend(recs)
    return by_method


def stat(recs, key, default=float("nan")):
    vals = [r.get(key, default) for r in recs]
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def main():
    by_method = pool(sys.argv[1:])
    hdr = f"{'method':<28}{'n':>5}{'succ':>8}{'work_J':>9}{'power_W':>9}{'time_s':>8}{'J/m':>8}{'COT':>8}{'proxy':>8}"
    print(hdr)
    print("-" * len(hdr))
    order = sorted(by_method, key=lambda m: stat(by_method[m], "cost_of_transport"))
    for m in order:
        recs = by_method[m]
        print(
            f"{m:<28}{len(recs):>5}"
            f"{stat(recs,'success'):>8.3f}"
            f"{stat(recs,'mechanical_energy_j'):>9.2f}"
            f"{stat(recs,'mean_mechanical_power_w'):>9.2f}"
            f"{stat(recs,'elapsed_time_s'):>8.3f}"
            f"{stat(recs,'energy_per_meter_j_m'):>8.2f}"
            f"{stat(recs,'cost_of_transport'):>8.3f}"
            f"{stat(recs,'energy_proxy'):>8.3f}"
        )


if __name__ == "__main__":
    main()
