#!/usr/bin/env bash
set -euo pipefail
exe=$(realpath "${1:?executable}")
screen=${2:?screening summary CSV}
out=${3:?output directory}
samples=${4:-6000}
mkdir -p "$out"
python3 - "$screen" > "$out/selected.txt" <<'PY'
import csv,sys
rows=list(csv.DictReader(open(sys.argv[1])))
for host in ['0','1']:
    options=[r for r in rows if r['host']==host]
    r=min(options,key=lambda r:(float(r['p99_us']),float(r['p50_us'])))
    print(host,r['split'],r['column_groups'],r.get('peer_direct','0'),r['graph'])
PY
while read -r host split cg peer graph; do
    # Balanced AB/BA/AB ordering, same allocation and executable for both modes.
    for i in 0 1 2 3 4 5; do
        case "$i" in 0|3|4) mode=baseline; s=0; c=1; p=0; g=1;;
            *) mode=selected; s=$split; c=$cg; p=$peer; g=$graph;; esac
        label="h${host}-${i}-${mode}"
        printf '%s %s\n' "$(date -u +%FT%TZ)" "$label"
        MVM_PEER_DIRECT=$p "$exe" 4 "$samples" "$s" "$g" "$host" "$out/$label-raw.csv" "$c" \
            > "$out/$label.csv" 2> "$out/$label.err"
    done
done < "$out/selected.txt"
python3 - "$out" <<'PY'
import csv,pathlib,sys
p=pathlib.Path(sys.argv[1]); rows=[]
for f in sorted(p.glob('h*.csv')):
    if f.name.endswith('-raw.csv'): continue
    for r in csv.DictReader(f.open()): rows.append({'run':f.stem,**r})
with (p/'summary.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
PY
