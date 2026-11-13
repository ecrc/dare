#!/usr/bin/env python3
"""Summarize successful Lair DARE benchmark jobs from their saved logs."""

import argparse
import csv
import re
import sys
from math import isfinite
from pathlib import Path
from statistics import median


def read_run(path, gpu_count):
    log = path.read_text()

    def header_int(label):
        match = re.search(rf"^# {label}:\s+(\d+)$", log, re.M)
        if match is None:
            raise ValueError(f"{path}: missing {label}")
        return int(match[1])

    timing = re.search(r"^Time:\s+(\S+) GFlops:\s+(\S+)$", log, re.M)
    wall = re.search(r"^Wall seconds: (\S+);", log, re.M)
    iterations = re.findall(r"^it (\d+) norm (\S+) tol (\S+)$", log, re.M)
    if timing is None or wall is None or not iterations:
        raise ValueError(f"{path}: missing timing or convergence data")
    step, norm, tolerance = iterations[-1]
    if not 0 <= float(norm) < float(tolerance) or header_int("Nb gpus") != gpu_count:
        raise ValueError(f"{path}: solver did not converge or GPU count differs")
    metrics = tuple(map(float, (*timing.groups(), wall[1])))
    if not all(isfinite(value) and value > 0 for value in metrics):
        raise ValueError(f"{path}: invalid timing or throughput")
    workload = (header_int("N"), header_int("NINSTR"), header_int("Nb threads"), int(step))
    return workload, metrics


def summarize_case(case, job_id):
    gpu_count = int(case.name.removesuffix("gpu"))
    workload, _ = read_run(case / "run-warmup.log", gpu_count)
    measured = []
    for run in (1, 2, 3):
        run_workload, metrics = read_run(case / f"run-{run}.log", gpu_count)
        if run_workload != workload:
            raise ValueError(f"{case}: workload changed between runs")
        measured.append(metrics)

    names = set()
    peaks = {}
    with (case / "gpu.csv").open(newline="") as stream:
        for row in csv.DictReader(stream, skipinitialspace=True):
            names.add(row["name"].removeprefix("NVIDIA "))
            index = int(row["index"])
            utilization = int(row["utilization.gpu [%]"].split()[0])
            peaks[index] = max(peaks.get(index, 0), utilization)
    if len(names) != 1 or len(peaks) != gpu_count or min(peaks.values()) <= 0:
        raise ValueError(f"{case}: GPU telemetry does not match the allocation")

    states, measurements, workers, iterations = workload
    times, rates, walls = zip(*measured)
    return (
        names.pop(), gpu_count, job_id, states, measurements, workers, len(times),
        *times, median(times), median(rates), median(walls),
        " ".join(str(peaks[index]) for index in sorted(peaks)), iterations,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", type=Path, help="directory containing successful job folders")
    args = parser.parse_args()
    try:
        jobs = sorted(args.logs.glob("[0-9]*-*"), key=lambda job: job.name.split("-", 1)[1])
        rows = [
            summarize_case(case, job.name.split("-", 1)[0])
            for job in jobs
            for case in sorted(job.glob("[0-9]*gpu"), key=lambda case: int(case.name[:-3]))
        ]
        if not rows:
            raise ValueError(f"{args.logs}: no benchmark cases found")
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))

    writer = csv.writer(sys.stdout)
    writer.writerow((
        "GPU", "GPUs", "Job", "States", "Measurements", "CPU workers", "Measured runs",
        "Solver run 1 seconds", "Solver run 2 seconds", "Solver run 3 seconds",
        "Median solver seconds", "Median GFLOP/s", "Median wall seconds",
        "Peak GPU utilization percent by device", "Iterations",
    ))
    writer.writerows(rows)


if __name__ == "__main__":
    main()
