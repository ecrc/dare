#!/usr/bin/env bash
set -euo pipefail
exe=$(realpath "${1:?executable}")
out=${2:?output directory}
samples=${3:-1000}
column_groups=${4:-1}
mkdir -p "$out"
{
    date -u
    hostname
    nvidia-smi --query-gpu=name,driver_version,pci.bus_id,memory.total,clocks.sm,clocks.mem --format=csv
    nvidia-smi topo -m
    taskset -pc $$
    sha256sum "$exe"
} > "$out/hardware.txt" 2>&1
for cg in $column_groups; do
for host in ${MVM_HOST_SCOPES:-0 1}; do
    for graph in ${MVM_GRAPH_MODES:-0 1}; do
        for split in ${MVM_SPLITS:-0 1 2 4 8 16}; do
            label="s${split}-g${graph}-h${host}-c${cg}"
            printf '%s %s\n' "$(date -u +%FT%TZ)" "$label"
            "$exe" 4 "$samples" "$split" "$graph" "$host" "$out/$label-raw.csv" "$cg" > "$out/$label.csv" 2> "$out/$label.err"
        done
    done
done
done
python3 - "$out" <<'PY'
import csv, pathlib, sys
p=pathlib.Path(sys.argv[1]); rows=[]
for f in sorted(p.glob('s*-g*-h*.csv')):
    if f.name.endswith('-raw.csv'): continue
    rows.extend(csv.DictReader(f.open()))
with (p/'summary.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
for r in sorted(rows,key=lambda r:float(r['p50_us'])):
    print(r)
PY
