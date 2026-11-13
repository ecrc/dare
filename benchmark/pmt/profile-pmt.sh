#!/usr/bin/env bash
# Run a tuning plan inside a single-node Slurm allocation; preserve baseline results.
# Plan: states label cpus workers tile streams scheduler mode repeats gpus [variant]
# mode: timing, counters, phase, check, profile, nsys; variant: original, reference, gpu-copy.
# Each case has an unmeasured warmup.
# StarPU 1.3.11 profile mode polls NVML per task and can severely perturb execution.
# A scheduler such as dmda:4 sets STARPU_SCHED_BETA=4 (transfer-cost weight).
# A tile such as 720:32 selects outer tile 720 and inner block 32 (default 180).
# A third tile field selects a task-window maximum or stage barriers: 720:32:10000,
# 720:32:barriers. The minimum task-window size is half the maximum.
# A fourth field, e.g. 720:32::export3, exports P and K after timed run 3.
# A case-directory file named nsight-warmup enables a delayed five-second capture
# during the unmeasured warmup. Some Nsight versions return before the application
# finishes despite --wait=all; an incomplete solve is rejected below.
set -eo pipefail
dare_root=${1:?Pass the deployment directory}
source "$dare_root/env.sh"
set -u

if [[ ${2:-} == --worker ]]; then
    case_dir=$3 states=$4 workers=$5 tile=$6 streams=$7 scheduler=$8 mode=$9
    run=${10} gpus=${11}
    variant=${12:-original}
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPICH_GPU_SUPPORT_ENABLED=0
    export STARPU_HOME="$case_dir/starpu-home" STARPU_CALIBRATE=1
    export STARPU_SILENT=1 STARPU_SCHED="${scheduler%%:*}" STARPU_WORKERS_GETBIND=1
    unset STARPU_SCHED_BETA
    if [[ $scheduler == *:* ]]; then export STARPU_SCHED_BETA="${scheduler#*:}"; fi
    export STARPU_CUDA_PIPELINE=4 STARPU_NWORKER_PER_CUDA="$streams"
    unset STARPU_PROFILING STARPU_WORKER_STATS STARPU_BUS_STATS
    unset DARE_GPU_COPY DARE_PHASE_PROFILE DARE_OUTPUT_PREFIX DARE_STAGE_BARRIERS
    unset STARPU_LIMIT_MAX_SUBMITTED_TASKS STARPU_LIMIT_MIN_SUBMITTED_TASKS
    executable="$dare_root/build/dare/ddarechameleon"
    if [[ $variant != original ]]; then
        executable="$dare_root/build/dare-profile/ddarechameleon"
    fi
    if [[ $variant == gpu-copy ]]; then export DARE_GPU_COPY=1; fi
    mkdir -p "$STARPU_HOME"
    printf 'Node: %s; run: %s; mode: %s\n' "$(hostname)" "$run" "$mode"
    taskset -pc "$$"
    nvidia-smi --query-gpu=name,memory.total,uuid --format=csv
    if [[ $run == warmup ]]; then nvidia-smi topo -m; fi
    nvidia-smi --query-gpu=timestamp,index,name,uuid,utilization.gpu,memory.used,power.draw \
        --format=csv -l 1 > "$case_dir/gpu-$run.csv" &
    monitor_pid=$!
    trap 'kill "$monitor_pid" 2>/dev/null || true; wait "$monitor_pid" 2>/dev/null || true' EXIT
    IFS=: read -r outer inner flow output <<< "$tile"
    inner=${inner:-180}
    flow=${flow:-}
    output=${output:-}
    if [[ $output == "export$run" ]]; then
        [[ $variant != original ]] || { echo 'Export requires the diagnostic executable'; exit 2; }
        export DARE_OUTPUT_PREFIX="$case_dir/result-$run"
    fi
    if [[ $flow == barriers ]]; then
        [[ $variant != original ]] || { echo 'Stage barriers require the diagnostic executable'; exit 2; }
        export DARE_STAGE_BARRIERS=1
    elif [[ -n $flow ]]; then
        [[ $flow =~ ^[0-9]+$ && $flow -ge 2 ]] || exit 2
        export STARPU_LIMIT_MAX_SUBMITTED_TASKS="$flow"
        export STARPU_LIMIT_MIN_SUBMITTED_TASKS="$((flow / 2))"
    fi
    args=(--threads="$workers" --n="$states" --ninstr=18694 --nb="$outer" --ib="$inner"
          --gpus="$gpus" --datapath="$dare_root/datasets/$states")
    prefix=()
    if [[ $run == warmup && -f "$case_dir/nsight-warmup" ]]; then
        nsys_binary=${DARE_NSYS:-nsys}
        "$nsys_binary" --version
        prefix=("$nsys_binary" profile --trace=cuda,nvtx --sample=none --cpuctxsw=none
                --delay=60 --duration=5 --kill=none --wait=all
                --force-overwrite=true --output="$case_dir/trace-warmup")
        if "$nsys_binary" profile --help | grep -- '--cuda-trace-all-apis=' >/dev/null; then
            prefix+=(--cuda-trace-all-apis=true)
        fi
    elif [[ $run != warmup && $mode == profile ]]; then
        export STARPU_PROFILING=1 STARPU_WORKER_STATS=1 STARPU_BUS_STATS=1
        args+=(--profile)
    elif [[ $run != warmup && $mode == nsys ]]; then
        nsys_binary=${DARE_NSYS:-/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/profilers/Nsight_Systems/bin/nsys}
        "$nsys_binary" --version
        prefix=("$nsys_binary" profile --trace=cuda,nvtx --sample=none --cpuctxsw=none
                --force-overwrite=true --output="$case_dir/trace-$run")
    elif [[ $run != warmup && $mode == counters ]]; then
        export STARPU_WORKER_STATS=1 STARPU_BUS_STATS=1
    elif [[ $run != warmup && $mode == phase ]]; then
        export DARE_PHASE_PROFILE=1
    elif [[ $run != warmup && $mode == check ]]; then
        args+=(--check)
        export DARE_OUTPUT_PREFIX="$case_dir/result-$run"
    fi
    launch=(stdbuf -oL -eL "$executable" "${args[@]}")
    # Profile the solver directly, avoiding a preload/exec wrapper in CUDA traces.
    if (( ${#prefix[@]} )); then launch=("$executable" "${args[@]}"); fi
    /usr/bin/time -f 'Wall seconds: %e; Max RSS KiB: %M' \
        "${prefix[@]}" "${launch[@]}"
    exit
fi

plan=${2:?Pass the tuning plan TSV}
result_dir="$dare_root/profiling/${SLURM_JOB_ID}"
mkdir -p "$result_dir"
cp "$plan" "$result_dir/plan.tsv"
failed=0
while read -r states label cpus workers tile streams scheduler mode repeats gpus variant; do
    [[ -z "$states" || $states == \#* ]] && continue
    [[ $mode == timing || $mode == counters || $mode == phase || $mode == check || $mode == profile || $mode == nsys ]] || exit 2
    variant=${variant:-original}
    [[ $variant == original || $variant == reference || $variant == gpu-copy ]] || exit 2
    case_dir="$result_dir/$states-$label"
    mkdir -p "$case_dir"
    printf '%s\n' "$states $label $cpus $workers $tile $streams $scheduler $mode $repeats $gpus $variant" > "$case_dir/config.txt"
    for run in warmup $(seq 1 "$repeats"); do
        printf 'START %s %s %s %s\n' "$(date -Is)" "$states" "$label" "$run"
        if srun --mpi=cray_shasta --input=none --exact --ntasks=1 --cpus-per-task="$cpus" \
            --gpus-per-task="$gpus" --cpu-bind=cores \
            bash "$dare_root/profile-pmt.sh" "$dare_root" --worker \
            "$case_dir" "$states" "$workers" "$tile" "$streams" "$scheduler" "$mode" "$run" "$gpus" "$variant" \
            > "$case_dir/run-$run.log" 2>&1; then
            if ! grep -q '^Time:' "$case_dir/run-$run.log"; then
                printf 'INCOMPLETE %s %s %s: no completed solver timing\n' "$states" "$label" "$run"
                failed=1
                break
            fi
            grep -E '^it 17 |^Time:|^Wall seconds:' "$case_dir/run-$run.log"
        else
            printf 'FAILED %s %s %s\n' "$states" "$label" "$run"
            tail -20 "$case_dir/run-$run.log"
            failed=1
            break
        fi
    done
done < "$result_dir/plan.tsv"
exit "$failed"
