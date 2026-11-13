#!/usr/bin/env python3
"""Summarize tuning logs, keeping instrumented runs separate from timing runs."""

import argparse
import csv
import re
import sys
from pathlib import Path
from statistics import median


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    writer = csv.writer(sys.stdout)
    writer.writerow(("states", "case", "cpus", "workers", "tile", "streams",
                     "scheduler", "mode", "gpus", "variant", "completed_runs",
                     "expected_runs", "median_seconds", "median_tflops",
                     "median_wall_seconds", "iterations", "max_final_residual"))
    for config in sorted(args.directory.glob("*/config.txt")):
        fields = config.read_text().split()
        states, label, cpus, workers, tile, streams, scheduler, mode, repeats, gpus = fields[:10]
        variant = fields[10] if len(fields) > 10 else "original"
        times, rates, walls, iterations, residuals = [], [], [], [], []
        for run in range(1, int(repeats) + 1):
            path = config.parent / f"run-{run}.log"
            if not path.exists() or path.with_suffix(".exclude").exists():
                continue
            log = path.read_text()
            timing = re.search(r"^Time:\s+(\S+) GFlops:\s+(\S+)", log, re.M)
            wall = re.search(r"^Wall seconds: (\S+);", log, re.M)
            convergence = re.findall(r"it (\d+) norm (\S+) tol (\S+)", log)
            if not timing or not wall or not convergence:
                continue
            if re.search(r"sequence failed|CANCELLED|Exited with exit code|Command exited", log):
                continue
            step, norm, tolerance = convergence[-1]
            if not 0 <= float(norm) < float(tolerance):
                continue
            times.append(float(timing[1]))
            rates.append(float(timing[2]) / 1000)
            walls.append(float(wall[1]))
            iterations.append(step)
            residual = re.findall(r"\|\|LHS-RHS\|\|f = (\S+)", log)
            if residual:
                residuals.append(float(residual[-1]))
        writer.writerow((states, label, cpus, workers, tile, streams, scheduler,
                         mode, gpus, variant, len(times), repeats,
                         median(times) if times else "",
                         median(rates) if rates else "",
                         median(walls) if walls else "",
                         " ".join(sorted(set(iterations))),
                         max(residuals) if residuals else ""))


if __name__ == "__main__":
    main()
