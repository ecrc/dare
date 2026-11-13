#!/usr/bin/env bash
set -euo pipefail

dare_root=${1:?Pass the deployment directory}
gpu_type=${2:?Pass the Slurm GPU type}
shift 2

if [[ ${1:-} == --case ]]; then
    gpu_count=$2
    result_dir=$3
    source "$dare_root/env.sh"
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    export STARPU_SCHED=dmda STARPU_SILENT=1 STARPU_CALIBRATE=1
    export STARPU_CUDA_PIPELINE=4 STARPU_NWORKER_PER_CUDA=4 STARPU_WORKERS_GETBIND=1
    export STARPU_HOME="$result_dir/starpu-home"
    mkdir -p "$STARPU_HOME"
    gpu_names=$(nvidia-smi --query-gpu=name --format=csv,noheader)
    printf '%s\n' "$gpu_names"
    [[ $(wc -l <<< "$gpu_names") -eq $gpu_count ]]
    [[ $(grep -Fc "$gpu_type" <<< "$gpu_names") -eq $gpu_count ]]
    printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:-unset}"
    taskset -pc "$$"

    nvidia-smi --query-gpu=timestamp,index,name,uuid,utilization.gpu,memory.used,power.draw \
        --format=csv -l 1 > "$result_dir/gpu.csv" &
    monitor_pid=$!
    trap 'kill "$monitor_pid" 2>/dev/null || true; wait "$monitor_pid" 2>/dev/null || true' EXIT

    for run in warmup 1 2 3; do
        printf '%s %s GPU(s), run %s at %s\n' "$gpu_type" "$gpu_count" "$run" "$(date -Is)"
        /usr/bin/time -f 'Wall seconds: %e; Max RSS KiB: %M' \
            mpirun -np 1 --bind-to none --mca ras "^slurm" --mca plm isolated \
            stdbuf -oL -eL "$dare_root/build/dare/ddarechameleon" \
            --threads=5 --n=7090 --ninstr=18694 --nb=720 --ib=180 \
            --gpus="$gpu_count" --datapath="$dare_root/datasets/7090" \
            > "$result_dir/run-$run.log" 2>&1 || {
                tail -40 "$result_dir/run-$run.log"
                exit 1
            }
        grep -E '^it [0-9]+ norm |^Time:|^Wall seconds:' "$result_dir/run-$run.log" | tail -3
    done
    exit
fi

result_dir="$dare_root/results/${SLURM_JOB_ID}-$gpu_type"
mkdir -p "$result_dir"
printf 'Node: %s\n' "$(hostname)"
lscpu | grep -E '^Model name:|^Socket\(s\):'
for gpu_count in "$@"; do
    case_dir="$result_dir/${gpu_count}gpu"
    mkdir -p "$case_dir"
    srun --mpi=none --exact --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
        --gres="gpu:$gpu_type:$gpu_count" --cpu-bind=cores \
        "$dare_root/benchmark-lair.sh" "$dare_root" "$gpu_type" --case "$gpu_count" "$case_dir"
done
