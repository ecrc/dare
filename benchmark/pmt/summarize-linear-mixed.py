#!/usr/bin/env python3
"""Summarize independent one-GPU snapshot replays; count shared LU only once."""
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median


def summarize(directory):
    runs = []
    for path in sorted(directory.glob("it-*-repeat-*.csv")):
        rows = list(csv.DictReader(path.open()))
        if not rows:
            raise ValueError(f"Empty replay: {path}")
        for row in rows:
            for key in row.keys() - {"snapshot", "rhs", "precision"}:
                row[key] = float(row[key])
        refs = [r for r in rows if r["precision"] == "FP64"]
        if {r["rhs"] for r in refs} != {"A", "G"} or len(refs) != 2:
            raise ValueError(f"Incomplete FP64 reference: {path}")
        baseline = refs[0]["factor_s"] + sum(r["solve_update_s"] for r in refs)
        run = {
            "file": path.name, "iteration": int(path.name.split("-")[1]),
            "n": int(rows[0]["n"]), "fp64_core_seconds": baseline,
            "fp64_max_relative_residual": max(r["relative_residual_f"] for r in refs),
            "fp64_factor_seconds": refs[0]["factor_s"], "stages": [],
        }
        mixed = [r for r in rows if r["precision"] == "FP32_IR"]
        for step in sorted({int(r["corrections"]) for r in mixed}):
            group = [r for r in mixed if r["corrections"] == step]
            if {r["rhs"] for r in group} != {"A", "G"}:
                continue  # An earlier RHS may have become nonfinite.
            solve = group[0]["factor_s"] + sum(r["solve_update_s"] for r in group)
            checks = sum(r["checks_s"] for r in group)
            run["stages"].append({
                "corrections": step, "core_seconds_including_checks": solve + checks,
                "factor_solve_update_seconds": solve, "checks_seconds": checks,
                "ratio_to_fp64_core": (solve + checks) / baseline,
                "raw_fp32_ratio_to_fp64_core": solve / baseline if step == 0 else None,
                "rhs": {r["rhs"]: {k: r[k] for k in
                    ("backward_error_f", "relative_residual_f", "relative_x_error_f")}
                    for r in group},
            })
        runs.append(run)
    if not runs:
        raise ValueError("No replay CSVs")
    groups = []
    for iteration in sorted({r["iteration"] for r in runs}):
        group = [r for r in runs if r["iteration"] == iteration]
        summary = {
            "iteration": iteration, "repeats": len(group),
            "fp64_core_seconds_median": median(r["fp64_core_seconds"] for r in group),
            "fp64_factor_seconds_median": median(r["fp64_factor_seconds"] for r in group),
            "stages": [],
        }
        for step in range(4):
            stages = [s for r in group for s in r["stages"] if s["corrections"] == step]
            if not stages:
                continue
            summary["stages"].append({
                "corrections": step, "samples": len(stages),
                "seconds_median": median(s["core_seconds_including_checks"] for s in stages),
                "factor_solve_update_seconds_median": median(s["factor_solve_update_seconds"] for s in stages),
                "checks_seconds_median": median(s["checks_seconds"] for s in stages),
                "worst_relative_x_error": max(v["relative_x_error_f"] for s in stages for v in s["rhs"].values()),
                "worst_relative_residual": max(v["relative_residual_f"] for s in stages for v in s["rhs"].values()),
            })
        groups.append(summary)
    return {
        "scope": "One GPU per snapshot, full dense RHS, FP64-origin snapshots, no-pivot LU, strict FP32 TRSM",
        "timing": "Factor once for both RHS. Disk/layout/allocation/diagnostic reference comparisons excluded. Mixed includes all residual checks; FP64 baseline excludes diagnostic residual checks.",
        "groups": groups, "runs": runs,
    }


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(json_safe(summarize(args.directory)), indent=2, allow_nan=False))
