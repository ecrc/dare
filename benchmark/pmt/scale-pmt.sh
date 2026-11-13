#!/usr/bin/env bash
# Fixed-size DARE strong scaling on Perlmutter, one rank per GPU node.
set -eo pipefail
root=${1:?Pass the deployment directory}
source "$root/env.sh"
set -u
states=${DARE_SCALE_STATES:-28210}
repeats=${DARE_SCALE_REPEATS:-3}
executable=${DARE_SCALE_BINARY:-"$root/build/dare-profile/ddarechameleon"}
overlay="$root/build/${DARE_SCALE_RUNTIME:-chameleon-cusolver}/runtime/starpu"
out="$root/scaling/${SLURM_JOB_ID:?Run inside a Slurm allocation}"
mkdir -p "$out"

export LD_LIBRARY_PATH="$overlay:$LD_LIBRARY_PATH"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPICH_GPU_SUPPORT_ENABLED=0
export STARPU_SCHED=dmda STARPU_SCHED_BETA=4 STARPU_SILENT=1 STARPU_CALIBRATE=1
export STARPU_CUDA_PIPELINE=4 STARPU_NWORKER_PER_CUDA=2 STARPU_WORKERS_GETBIND=1
export STARPU_LIMIT_MAX_SUBMITTED_TASKS=10000 STARPU_LIMIT_MIN_SUBMITTED_TASKS=5000
export DARE_GPU_COPY=${DARE_SCALE_GPU_COPY:-1} DARE_CUSOLVER_LU=${DARE_SCALE_GPU_LU:-1}
unset DARE_GPU_NORM DARE_SWAP_ALPHA DARE_OUTPUT_PREFIX DARE_PHASE_PROFILE DARE_STAGE_BARRIERS
unset STARPU_BUS_STATS STARPU_WORKER_STATS STARPU_COMM_STATS
shift

for nodes in "$@"; do
    case_dir="$out/${nodes}node"
    mkdir -p "$case_dir"
    export STARPU_HOME="$case_dir/starpu-home"
    mkdir -p "$STARPU_HOME"
    # Perlmutter counts hardware threads, so reserve 24 for 12 physical cores.
    launch=(srun --mpi=cray_shasta --input=none --exact --kill-on-bad-exit=1
            --nodes="$nodes" --ntasks="$nodes" --ntasks-per-node=1 --cpus-per-task="${DARE_PMT_STEP_CPUS:-24}"
            --gpus-per-node=4 --gpu-bind=none --cpu-bind=cores)
    "${launch[@]}" --label bash -ec '
        hostname
        gpu_info=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits)
        printf "%s\n" "$gpu_info"
        awk -F, '\''$1 !~ /A100/ || $2+0 < 79000 { bad=1 } END { exit (bad || NR != 4) }'\'' <<< "$gpu_info"
        ldd "$1" | grep -F "$2/libchameleon_starpu.so"
    ' bash "$executable" "$overlay" > "$case_dir/hardware.log"

    runs=(warmup $(seq 1 "$repeats"))
    if [[ ${DARE_SCALE_PHASE:-1} == 1 ]]; then runs+=(phase); fi
    for run in "${runs[@]}"; do
        unset STARPU_BUS_STATS STARPU_COMM_STATS DARE_PHASE_PROFILE DARE_OUTPUT_PREFIX
        if [[ $run == warmup ]]; then export STARPU_BUS_STATS=1 STARPU_COMM_STATS=1; fi
        if [[ $run == phase ]]; then export DARE_PHASE_PROFILE=1; fi
        if [[ $run == "$repeats" && ${DARE_SCALE_EXPORT:-1} == 1 ]]; then
            export DARE_OUTPUT_PREFIX="$case_dir/result"
        fi
        if [[ -n ${DARE_SCALE_MPI_SAMPLE:-} ]]; then
            export DARE_TILE_SAMPLE_DIR="$case_dir/$run-tiles"
            mkdir -p "$DARE_TILE_SAMPLE_DIR"
        fi
        printf '%s states=%s nodes=%s run=%s\n' "$(date -Is)" "$states" "$nodes" "$run"
        /usr/bin/time -f 'Step wall seconds: %e' -o "$case_dir/$run.wall" \
            "${launch[@]}" --output="$case_dir/$run-rank-%t.log" --error="$case_dir/$run-rank-%t.log" \
            bash -ec '
                sample_prefix=$1
                shift
                nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw \
                    --format=csv -l 1 > "$sample_prefix-gpu-$SLURM_PROCID.csv" &
                monitor=$!
                trap "kill $monitor 2>/dev/null || true; wait $monitor 2>/dev/null || true" EXIT
                if [[ -n ${DARE_SCALE_MPI_SAMPLE:-} ]]; then
                    export LD_PRELOAD="$DARE_SCALE_MPI_SAMPLE${LD_PRELOAD:+:$LD_PRELOAD}"
                fi
                if [[ ${DARE_SCALE_GDB:-0} == 1 ]]; then
                    gdb --batch --return-child-result -ex "set startup-with-shell off" \
                        -ex "set environment MPICH_GPU_SUPPORT_ENABLED 0" \
                        -ex run -ex "thread apply all bt 8" --args "$@"
                else
                    stdbuf -oL -eL "$@"
                fi
            ' bash "$case_dir/$run" "$executable" --threads=5 --n="$states" --ninstr=18694 \
                --nb=720 --ib=180 --gpus=4 --datapath="$root/datasets/$states"
        grep '^Time:' "$case_dir/$run-rank-0.log"
    done
done
