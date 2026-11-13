#!/usr/bin/env python3
"""Validate and summarize same-allocation dependent-controller repeats."""
import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def summarize(root):
    rows = list(csv.DictReader((root / 'repeat' / 'summary.csv').open()))
    result = {}
    for host in ('0', '1'):
        modes = {}
        for mode in ('row', 'hybrid', 'selected'):
            runs = [r for r in rows if r['host'] == host and r['run'].endswith('-' + mode)]
            if len(runs) != 3:
                raise ValueError(f'Expected three {mode} repeats for host={host}')
            samples = []
            configs = set()
            for r in runs:
                if tuple(int(r[k]) for k in ('gpus', 'states', 'measurements', 'actuators')) != (4, 30000, 20000, 5000):
                    raise ValueError('Not the requested four-GPU full-size problem')
                for key, limit in (('state_l2', 1e-5), ('command_l2', 1e-5), ('cpu_scaled', 1e-3)):
                    value = float(r[key])
                    if not math.isfinite(value) or value > limit:
                        raise ValueError(f'Validation failed: {r["run"]} {key}')
                configs.add((r['split'], r.get('layout', r.get('hybrid')), r['graph'], r.get('rows_per_block', '8')))
                raw = [float(s['latency_us']) for s in csv.DictReader((root / 'repeat' / (r['run'] + '-raw.csv')).open())]
                if len(raw) != int(r['samples']) or any(not math.isfinite(x) or x <= 0 for x in raw):
                    raise ValueError('Invalid raw latency samples')
                samples.extend(raw)
            if len(configs) != 1:
                raise ValueError('Repeat configuration changed')
            config = next(iter(configs))
            samples.sort()
            modes[mode] = {
                'configuration': dict(zip(('split', 'layout', 'graph', 'rows_per_block'), config)),
                'samples': len(samples),
                'median_run_p50_us': statistics.median(float(r['p50_us']) for r in runs),
                'run_p50_us': [float(r['p50_us']) for r in runs],
                'run_p99_us': [float(r['p99_us']) for r in runs],
                'run_rms_us': [float(r['rms_us']) for r in runs],
                'pooled_p99_us': samples[math.ceil(.99 * len(samples)) - 1],
                'max_us': samples[-1],
                'misses_500us': sum(x > 500 for x in samples),
                'max_state_l2': max(float(r['state_l2']) for r in runs),
                'max_command_l2': max(float(r['command_l2']) for r in runs),
            }
        base = modes['row']['median_run_p50_us']
        modes['distribution_only_reduction_percent'] = 100 * (1 - modes['hybrid']['median_run_p50_us'] / base)
        modes['selected_reduction_percent'] = 100 * (1 - modes['selected']['median_run_p50_us'] / base)
        result['host_roundtrip' if host == '1' else 'device_resident_wall'] = modes
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    text = json.dumps(summarize(args.directory), indent=2) + '\n'
    (args.directory / 'comparison.json').write_text(text)
    print(text, end='')
