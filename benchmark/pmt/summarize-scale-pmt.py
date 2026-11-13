#!/usr/bin/env python3
"""Summarize fixed-size Perlmutter solves from archived per-rank logs."""

import argparse
import csv
from pathlib import Path
import re
from statistics import median
import sys


TIME = re.compile(r"^Time:\s+(\S+) GFlops:\s+(\S+)", re.M)
NORM = re.compile(r"^it (\d+) norm (\S+) tol (\S+)", re.M)
GRID = re.compile(r"^DARE MPI rank (\d+)/(\d+) grid (\d+x\d+) A tiles (\d+)/(\d+)", re.M)
BUS = re.compile(r"^\s*(CUDA|NUMA) \d+ -> (CUDA|NUMA) \d+\s+([0-9.]+) GB", re.M)
MPI = re.compile(r"^\[starpu_comm_stats\]\[\d+\] TOTAL:\s+([0-9.]+) B", re.M)
PHASE = re.compile(r"^PHASE \d+ (\w+) ([0-9.]+)", re.M)


def summarize(case):
    nodes = int(case.name.split("node", 1)[0])
    times, rates = [], []
    states = iterations = None
    for leader in sorted(case.glob("[0-9]*-rank-0.log")):
        run = leader.name.split("-", 1)[0]
        logs = [(case / f"{run}-rank-{rank}.log").read_text() for rank in range(nodes)]
        timings = [TIME.search(log) for log in logs]
        convergence = [NORM.findall(log) for log in logs]
        if not all(timings) or not all(convergence):
            raise ValueError(f"{case}, run {run}: incomplete rank logs")
        norms = [steps[-1] for steps in convergence]
        if (len({iteration for iteration, _, _ in norms}) != 1
                or any(not 0 <= float(norm) < float(tol) for _, norm, tol in norms)):
            raise ValueError(f"{case}, run {run}: unconverged rank")
        if any(float(timing[2]) <= 0 for timing in timings):
            raise ValueError(f"{case}, run {run}: solver failure")
        if nodes > 1:
            ownership = [GRID.search(log) for log in logs]
            if (not all(ownership)
                    or [int(row[1]) for row in ownership] != list(range(nodes))
                    or any(int(row[2]) != nodes for row in ownership)
                    or sum(int(row[4]) for row in ownership) != int(ownership[0][5])):
                raise ValueError(f"{case}, run {run}: invalid tile ownership")
        slowest = max(timings, key=lambda timing: float(timing[1]))
        times.append(float(slowest[1]))
        rates.append(float(slowest[2]) / 1000)
        states = int(re.search(r"^# N:\s+(\d+)", logs[0], re.M)[1])
        iterations = int(norms[0][0])

    warmup = [log.read_text() for log in sorted(case.glob("warmup-rank-*.log"))]
    bus = [item for log in warmup for item in BUS.findall(log)]
    host_gpu = sum(float(amount) for a, b, amount in bus if a != b)
    gpu_gpu = sum(float(amount) for a, b, amount in bus if a == b == "CUDA")
    mpi = [float(amount) for log in warmup for amount in MPI.findall(log)]
    phases = {}
    for path in case.glob("phase-rank-0.log"):
        for name, seconds in PHASE.findall(path.read_text()):
            phases[name] = phases.get(name, 0.0) + float(seconds)
    return {
        "job": case.parent.name, "states": states, "nodes": nodes, "gpus": nodes * 4,
        "iterations": iterations, "samples": len(times), "median_solver_s": median(times),
        "median_tflops": median(rates), "warmup_host_gpu_gib": host_gpu,
        "warmup_gpu_gpu_gib": gpu_gpu, "warmup_mpi_send_gib": sum(mpi) / 2**30 if mpi else "",
        **{f"phase_{name}_s": seconds for name, seconds in phases.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", nargs="+", type=Path, help="archived <nodes>node directories")
    args = parser.parse_args()
    records = [summarize(case) for case in args.cases]
    baseline = next((row for row in records if row["nodes"] == 1), None)
    if baseline is not None:
        for row in records:
            if row["states"] == baseline["states"]:
                speedup = baseline["median_solver_s"] / row["median_solver_s"]
                row.update(speedup_vs_4gpu=speedup, efficiency_vs_4gpu=speedup / row["nodes"])
    columns = list(dict.fromkeys(key for row in records for key in row))
    writer = csv.DictWriter(sys.stdout, fieldnames=columns)
    writer.writeheader()
    writer.writerows(records)


if __name__ == "__main__":
    main()
