#!/usr/bin/env bash
# Resume/download/validate/evaluate one queued DAI job. Safe to rerun.
# Usage from workspace: bash dare/benchmark/ao/finish_dai.sh 3576001
set -euo pipefail
job=${1:?DAI job ID required}
[[ $job =~ ^[0-9]+$ ]] || exit 2
workspace=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$workspace"
suite=log/ao-size-sweep
out="$suite/dai/$job"
mkdir -p "$out"
trap 'rc=$?; if ((rc)); then printf "exit=%s at=%s\n" "$rc" "$(date -Is)" > "$out/POSTPROCESS_FAILED"; fi' EXIT
remote="/work/hdd/bhwj/yhong4/dare-dai-28210/ao-size-sweep/results/$job"
for attempt in $(seq 1 4320); do
    if ssh -o BatchMode=yes dai "test -f '$remote/COMPUTE_OK'"; then break; fi
    state=$(ssh -o BatchMode=yes dai "sacct -j '$job' -X -n -o State | head -1" | tr -d ' ')
    printf '%s %s\n' "$(date -Is)" "$state" > "$out/queue-status.txt"
    case "$state" in COMPLETED*|FAILED*|CANCELLED*|TIMEOUT*|OUT_OF_MEMORY*) exit 1;; esac
    sleep 30
done
ssh -o BatchMode=yes dai "test -f '$remote/COMPUTE_OK'"
rsync -a --exclude=starpu "dai:$remote/" "$out/"
source "$suite/runtime.sh"
for n in 7090 12570 28210; do
    input="datasets/$n"
    [[ $n != 28210 ]] || input="$suite/28210/input"
    "$compass_py" dare/benchmark/ao/check_residual.py --inputs "$input" --solution "$out/$n"
    : > "$out/$n/vs-perlmutter.jsonl"
    for matrix in P K; do
        "$compass_py" dare/benchmark/pmt/compare-fits.py \
            "$suite/perlmutter/61125702/$n/result-$matrix.fits" "$out/$n/result-$matrix.fits" \
            --rtol 1e-9 >> "$out/$n/vs-perlmutter.jsonl"
    done
done
# Avoid overlapping this evaluation with the initial Perlmutter science sweep.
for attempt in $(seq 1 4320); do
    [[ ! -f "$suite/LOCAL_EVALUATION_DONE" ]] || break
    sleep 30
done
test -f "$suite/LOCAL_EVALUATION_DONE"
for n in 7090 12570 28210; do
    input="datasets/$n"
    [[ $n != 28210 ]] || input="$suite/28210/input"
    if [[ ! -f "$out/$n/controller/controller.json" ]]; then
        "$compass_py" -u dare/benchmark/ao/build_controller.py --inputs "$input" \
            --model "$suite/$n/model" --gain "$out/$n/result-K.fits" \
            --out "$out/$n/controller" > "$out/$n/assembly.log" 2>&1
    fi
    for seed in 1100 2300 3700; do
        [[ ! -f "$out/$n/ao-$seed/metrics.json" ]] || continue
        "$compass_py" -u dare/benchmark/ao/run_native.py --ata ata \
            --params dare/benchmark/ao/mavis_archive.py --controller "$out/$n/controller" \
            --out "$out/$n/ao-$seed" --frames 1000 --burn-in 200 --seed "$seed" \
            --backend gpu --allow-model-mismatch > "$out/$n/ao-$seed.log" 2>&1
    done
done
"$compass_py" dare/benchmark/ao/summarize.py "$suite" > "$out/summary.json"
"$compass_py" dare/benchmark/ao/plot_results.py "$suite"
touch "$out/POSTPROCESS_OK"
