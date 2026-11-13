#!/usr/bin/env bash
set -euo pipefail
# Supply PREFIX-r4, PREFIX-r8, PREFIX-r16, built from the same source snapshot.
prefix=$(realpath "${1:?executable prefix}"-r8)
prefix=${prefix%-r8}
out=${2:?output directory}
samples=${3:-1000}
repeat_samples=${4:-6000}
mkdir -p "$out/screen" "$out/repeat"
{
    date -u
    hostname
    nvidia-smi --query-gpu=name,driver_version,pci.bus_id,memory.total,clocks.sm,clocks.mem --format=csv
    nvidia-smi topo -m
    taskset -pc $$
    sha256sum "$prefix"-r4 "$prefix"-r8 "$prefix"-r16
} > "$out/hardware.txt" 2>&1
for host in 0 1; do
    for layout in 0 1; do
        label="baseline-h${host}-l${layout}"
        printf '%s %s\n' "$(date -u +%FT%TZ)" "$label"
        "$prefix-r8" 4 "$samples" 0 1 "$host" "$layout" "$out/screen/$label-raw.csv" \
            > "$out/screen/$label.csv" 2> "$out/screen/$label.err"
    done
    for rows in 4 8 16; do
        for layout in 1 2; do
            for split in 1 2 4; do
                label="b${rows}-h${host}-l${layout}-s${split}"
                printf '%s %s\n' "$(date -u +%FT%TZ)" "$label"
                "$prefix-r$rows" 4 "$samples" "$split" 1 "$host" "$layout" "$out/screen/$label-raw.csv" \
                    > "$out/screen/$label.csv" 2> "$out/screen/$label.err"
            done
        done
    done
done
python3 - "$out" <<'PY'
import csv, pathlib, sys
p=pathlib.Path(sys.argv[1]); rows=[]
for f in sorted((p/'screen').glob('*.csv')):
    if not f.name.endswith('-raw.csv'): rows.extend(csv.DictReader(f.open()))
with (p/'screen'/'summary.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
with (p/'selected.txt').open('w') as f:
    for host in ('0','1'):
        r=min((r for r in rows if r['host']==host),key=lambda r:(float(r['p99_us']),float(r['p50_us'])))
        print(host,r['rows_per_block'],r['split'],r['layout'],file=f)
PY
if [[ ${MVM_SCREEN_ONLY:-0} == 1 ]]; then
    exit 0
fi
while read -r host rows split layout; do
    for round in 0 1 2; do
        for i in 0 1 2; do
            case $(((round+i)%3)) in
                0) label=row; b=8; s=0; y=0;;
                1) label=hybrid; b=8; s=0; y=1;;
                2) label=selected; b=$rows; s=$split; y=$layout;;
            esac
            run="h${host}-r${round}-${label}"
            printf '%s %s\n' "$(date -u +%FT%TZ)" "$run"
            "$prefix-r$b" 4 "$repeat_samples" "$s" 1 "$host" "$y" "$out/repeat/$run-raw.csv" \
                > "$out/repeat/$run.csv" 2> "$out/repeat/$run.err"
        done
    done
done < "$out/selected.txt"
python3 - "$out" <<'PY'
import csv,pathlib,sys
p=pathlib.Path(sys.argv[1]); rows=[]
for f in sorted((p/'repeat').glob('h*.csv')):
    if not f.name.endswith('-raw.csv'):
        for r in csv.DictReader(f.open()): rows.append({'run':f.stem,**r})
with (p/'repeat'/'summary.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
PY
