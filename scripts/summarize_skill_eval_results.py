"""Aggregate Isaac skill-control evaluation JSON files.

The evaluator writes per-target/per-method records.  This helper combines one
or more eval files and reports both target-balanced and episode-weighted
metrics, with simple bootstrap intervals for paper tables.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np


METRICS = (
    "success",
    "final_distance",
    "energy_proxy",
    "final_height",
    "terminated_early",
    "num_commands_used",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize skill-control evaluation files.")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--methods", type=str, default="")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()


def _record_value(record: Dict[str, Any], metric: str) -> float:
    if metric == "success":
        return 1.0 if record.get("success") else 0.0
    if metric == "terminated_early":
        return 1.0 if record.get("terminated_early") else 0.0
    return float(record.get(metric, 0.0))


def _mean_ci(values: np.ndarray, rng: np.random.Generator, bootstrap: int) -> Dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    mean = float(np.mean(values))
    if len(values) == 1 or bootstrap <= 0:
        return {"mean": mean, "ci_low": mean, "ci_high": mean, "n": int(len(values))}
    samples = np.empty(bootstrap, dtype=np.float64)
    for i in range(bootstrap):
        idx = rng.integers(0, len(values), size=len(values))
        samples[i] = float(np.mean(values[idx]))
    return {
        "mean": mean,
        "ci_low": float(np.percentile(samples, 2.5)),
        "ci_high": float(np.percentile(samples, 97.5)),
        "n": int(len(values)),
    }


def _records_by_method(files: Iterable[str], method_filter: set[str]) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    out: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for raw in files:
        path = Path(raw)
        data = json.loads(path.read_text())
        for target_key, target_records in data.get("records", {}).items():
            out.setdefault(target_key, {})
            for method, records in target_records.items():
                if method_filter and method not in method_filter:
                    continue
                enriched = []
                for record in records:
                    item = dict(record)
                    item["_source_file"] = str(path)
                    item["_target"] = target_key
                    item["_method"] = method
                    enriched.append(item)
                out[target_key].setdefault(method, []).extend(enriched)
    return out


def summarize(
    records_by_target: Dict[str, Dict[str, List[Dict[str, Any]]]],
    bootstrap: int,
    seed: int,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    methods = sorted({method for target in records_by_target.values() for method in target})
    per_target: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for target_key, target_records in sorted(records_by_target.items()):
        per_target[target_key] = {}
        for method in methods:
            records = target_records.get(method, [])
            if not records:
                continue
            per_target[target_key][method] = {
                metric: _mean_ci(np.asarray([_record_value(r, metric) for r in records]), rng, bootstrap)
                for metric in METRICS
            }

    episode_weighted: Dict[str, Dict[str, Any]] = {}
    target_balanced: Dict[str, Dict[str, Any]] = {}
    for method in methods:
        all_records = [
            record
            for target_records in records_by_target.values()
            for record in target_records.get(method, [])
        ]
        if all_records:
            episode_weighted[method] = {
                metric: _mean_ci(np.asarray([_record_value(r, metric) for r in all_records]), rng, bootstrap)
                for metric in METRICS
            }
        target_metric_means: Dict[str, List[float]] = {metric: [] for metric in METRICS}
        for target_records in records_by_target.values():
            records = target_records.get(method, [])
            if not records:
                continue
            for metric in METRICS:
                target_metric_means[metric].append(float(np.mean([_record_value(r, metric) for r in records])))
        if any(target_metric_means.values()):
            target_balanced[method] = {
                metric: _mean_ci(np.asarray(values), rng, bootstrap)
                for metric, values in target_metric_means.items()
                if values
            }

    return {
        "metrics": list(METRICS),
        "per_target": per_target,
        "target_balanced": target_balanced,
        "episode_weighted": episode_weighted,
    }


def main() -> None:
    args = parse_args()
    method_filter = {item.strip() for item in args.methods.split(",") if item.strip()}
    records = _records_by_method(args.inputs, method_filter)
    summary = summarize(records, bootstrap=args.bootstrap, seed=args.seed)
    summary["inputs"] = args.inputs
    summary["methods"] = sorted(method_filter) if method_filter else sorted(
        {method for target in records.values() for method in target}
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["target_balanced"], indent=2))


if __name__ == "__main__":
    main()
