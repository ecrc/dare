#!/usr/bin/env python3
"""Summarize alternating four-GPU repeats without hiding per-run tail latency."""
import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def summarize(directory):
    rows = list(csv.DictReader((directory / "summary.csv").open()))
    results = {}
    for host in ("0", "1"):
        scopes = {}
        for mode in ("baseline", "selected"):
            runs = [r for r in rows if r["host"] == host and r["run"].endswith(mode)]
            if len(runs) != 3:
                raise ValueError(f"Expected three {mode} runs for host={host}, got {len(runs)}")
            raw = []
            for run in runs:
                for key, threshold in (("relative_l2", 5e-6), ("replay_l2", 5e-6),
                                       ("cpu_fp64_scaled", 1e-4), ("replay_cpu", 1e-4)):
                    value = float(run[key])
                    if not math.isfinite(value) or value > threshold:
                        raise ValueError(f"Invalid numerical check: {run['run']} {key}={value}")
                samples = [float(r["latency_us"]) for r in
                           csv.DictReader((directory / f"{run['run']}-raw.csv").open())]
                if len(samples) != int(run["samples"]) or any(not math.isfinite(x) or x <= 0 for x in samples):
                    raise ValueError(f"Invalid raw samples for {run['run']}")
                raw.extend(samples)
            raw.sort()
            scopes[mode] = {
                "runs": [r["run"] for r in runs],
                "samples": len(raw),
                "configuration": {k: runs[0][k] for k in
                                  ("split", "graph", "column_groups", "peer_direct")},
                "median_run_p50_us": statistics.median(float(r["p50_us"]) for r in runs),
                "run_p50_us": [float(r["p50_us"]) for r in runs],
                "run_p99_us": [float(r["p99_us"]) for r in runs],
                "run_rms_us": [float(r["rms_us"]) for r in runs],
                "pooled_p50_us": raw[math.ceil(.5 * len(raw)) - 1],
                "pooled_p99_us": raw[math.ceil(.99 * len(raw)) - 1],
                "max_us": raw[-1],
                "misses_500us": sum(x > 500 for x in raw),
            }
        base = scopes["baseline"]["median_run_p50_us"]
        selected = scopes["selected"]["median_run_p50_us"]
        scopes["p50_reduction_percent"] = 100 * (1 - selected / base)
        results["host_roundtrip" if host == "1" else "device_resident_wall"] = scopes
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = summarize(args.directory)
    output = json.dumps(result, indent=2) + "\n"
    (args.directory / "comparison.json").write_text(output)
    print(output, end="")
