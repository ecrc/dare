#!/usr/bin/env bash
set -euo pipefail

executable=$1
output=$2
gpus=${3:-4}
samples=${4:-3000}
mkdir -p "$output"
nvidia-smi --query-gpu=name,pci.bus_id,memory.total,driver_version --format=csv > "$output/hardware.csv"

result=$(mktemp)
trap 'rm -f "$result"' EXIT
csv="$output/results.csv"
: > "$csv"

# These are Jesse's nominal output-by-input dimensions. Each case is measured
# independently with synthetic weights and the same timing scopes as the hot path.
for shape in '30000 30000' '30000 5000' '5000 30000' '30000 20000'; do
    read -r rows cols <<< "$shape"
    for layout in row col; do
        printf 'Cold-path dense FP32 MVM %s x %s, %s, %s GPUs\n' "$rows" "$cols" "$layout" "$gpus" >&2
        "$executable" "$rows" "$cols" "$gpus" fp32 "$layout" "$samples" 0 > "$result"
        if [[ -s $csv ]]; then
            tail -n +2 "$result" >> "$csv"
        else
            cat "$result" >> "$csv"
        fi
    done
done
