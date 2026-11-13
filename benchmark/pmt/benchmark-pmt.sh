#!/usr/bin/env bash
set -eo pipefail
dare_root=${1:?Pass the deployment directory}
shift
if [[ ${1:-} == --run ]]; then
    set -u
    export MPICH_GPU_SUPPORT_ENABLED=0
    gpu_count=$2
    run=$3
    case_dir=$4
    states=$5
    if [[ $run == warmup ]]; then
        printf 'Node: %s\n' "$(hostname)"
        lscpu | grep -E '^Model name:|^Socket\(s\):'
    fi
    gpu_info=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits)
    printf '%s\n' "$gpu_info"
    awk -F, -v expected="$gpu_count" '
        $1 !~ /A100/ || $2 + 0 < 39000 { bad = 1 }
        END { exit (bad || NR != expected) }
    ' <<< "$gpu_info"
    printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:-unset}"
    taskset -pc "$$"

    nvidia-smi --query-gpu=timestamp,index,name,uuid,utilization.gpu,memory.used,power.draw \
        --format=csv -l 1 > "$case_dir/gpu-$run.csv" &
    monitor_pid=$!
    trap 'kill "$monitor_pid" 2>/dev/null || true; wait "$monitor_pid" 2>/dev/null || true' EXIT

    export STARPU_HOME="$case_dir/starpu-home"
    mkdir -p "$STARPU_HOME"
    /usr/bin/time -f 'Wall seconds: %e; Max RSS KiB: %M' \
        stdbuf -oL -eL "$dare_root/build/dare/ddarechameleon" \
        --threads=5 --n="$states" --ninstr=18694 --nb=720 --ib=180 \
        --gpus="$gpu_count" --datapath="$dare_root/datasets/$states" \
        > "$case_dir/run-$run.log" 2>&1 || {
            tail -40 "$case_dir/run-$run.log"
            exit 1
        }
    grep -E '^it [0-9]+ norm |^Time:|^Wall seconds:' "$case_dir/run-$run.log" | tail -3
    exit
fi

source "$dare_root/env.sh"
set -u
states=${1:-7090}
if [[ ! $states =~ ^[1-9][0-9]*$ ]]; then
    printf 'Expected a positive number of states, got %s\n' "$states" >&2
    exit 2
fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export STARPU_SCHED=dmda STARPU_SILENT=1 STARPU_CALIBRATE=1
export STARPU_CUDA_PIPELINE=4 STARPU_NWORKER_PER_CUDA=4 STARPU_WORKERS_GETBIND=1

result_dir="$dare_root/results/${SLURM_JOB_ID}-A100"
mkdir -p "$result_dir"
printf 'Allocation: %s, states: %s\n' "$SLURM_JOB_NODELIST" "$states"
for gpu_count in 1 2 4; do
    case_dir="$result_dir/${gpu_count}gpu"
    mkdir -p "$case_dir"
    for run in warmup 1 2 3; do
        printf 'A100 %s GPU(s), run %s at %s\n' "$gpu_count" "$run" "$(date -Is)"
        srun --mpi=cray_shasta --exact --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
            --gpus-per-task="$gpu_count" --cpu-bind=cores \
            "$dare_root/benchmark-pmt.sh" "$dare_root" --run "$gpu_count" "$run" "$case_dir" "$states"
    done
    awk 'FNR > 1 || NR == 1' "$case_dir"/gpu-{warmup,1,2,3}.csv > "$case_dir/gpu.csv"
done
