#!/usr/bin/env bash
# Compare CPU and cuSOLVER diagonal LU on the same Slurm allocation.
# Additional plans test GPU copies, tile norms, and descriptor swapping.
set -eo pipefail
root=${1:?Pass the deployment directory}
states=${2:?Pass the state count}
gpus=${3:?Pass the GPU count}
plan=${4:-lu}
runtime=chameleon-cusolver
case $plan in
    lu) modes=(cpu cuda) ;;
    combined) modes=(cpu copy cuda combined combined-2workers) ;;
    norm) modes=(combined-2workers gpu-norm); runtime=chameleon-gpu-norm ;;
    swap) modes=(gpu-norm gpu-norm-swap); runtime=chameleon-gpu-norm ;;
    *) printf 'Unknown benchmark plan %s\n' "$plan" >&2; exit 2 ;;
esac
source "$root/env.sh"
set -u

overlay="$root/build/$runtime/runtime/starpu"
profile=dare-profile
if [[ $plan == swap ]]; then profile=dare-profile-swap; fi
executable=${DARE_BENCHMARK_BINARY:-"$root/build/$profile/ddarechameleon"}
results="$root/cusolver-lu/results/${SLURM_JOB_ID:?Run inside a Slurm allocation}"
mkdir -p "$results"
cd "$results"

export LD_LIBRARY_PATH="$overlay:$LD_LIBRARY_PATH"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPICH_GPU_SUPPORT_ENABLED=0
export STARPU_SCHED=dmda STARPU_SILENT=1 STARPU_CALIBRATE=1 STARPU_WORKERS_GETBIND=1
export STARPU_CUDA_PIPELINE=4
unset DARE_PHASE_PROFILE DARE_STAGE_BARRIERS DARE_OUTPUT_PREFIX
unset STARPU_BUS_STATS STARPU_WORKER_STATS

if [[ -n ${dare_math:-} ]]; then
    # Perlmutter counts hardware threads, so reserve 24 for 12 physical cores.
    launch=(srun --mpi=cray_shasta --input=none --exact --ntasks=1
            --cpus-per-task="${DARE_PMT_STEP_CPUS:-24}" --gpus-per-task="$gpus" --cpu-bind=cores)
else
    launch=(srun --mpi=none --input=none --exact --ntasks=1 --cpu-bind=cores
            mpirun -np 1 --bind-to none --mca ras '^slurm' --mca plm isolated)
fi
"${launch[@]}" bash -ec '
    hostname > node.txt
    nvidia-smi --query-gpu=name,memory.total,uuid --format=csv > hardware.csv
    ldd "$1" > libraries.txt
' bash "$executable"
grep -F "$overlay/libchameleon_starpu.so" libraries.txt
[[ $(awk 'END { print NR-1 }' hardware.csv) == "$gpus" ]]

reference_iterations=
for run in warmup 1 2 3; do
    offset=0
    if [[ $plan != lu && $run != warmup ]]; then offset=$run; fi
    for ((index=0; index<${#modes[@]}; index++)); do
        mode=${modes[(index+offset)%${#modes[@]}]}
        export STARPU_HOME="$results/$mode-starpu"
        mkdir -p "$STARPU_HOME"
        export DARE_CUSOLVER_LU=0 DARE_GPU_COPY=0 DARE_GPU_NORM=0 DARE_SWAP_ALPHA=0 STARPU_NWORKER_PER_CUDA=4
        unset STARPU_SCHED_BETA STARPU_LIMIT_MAX_SUBMITTED_TASKS STARPU_LIMIT_MIN_SUBMITTED_TASKS
        case $mode in
            cuda|combined*|gpu-norm*) export DARE_CUSOLVER_LU=1 ;;
        esac
        case $mode in
            copy|combined*|gpu-norm*)
                export DARE_GPU_COPY=1 STARPU_SCHED_BETA=4
                export STARPU_LIMIT_MAX_SUBMITTED_TASKS=10000 STARPU_LIMIT_MIN_SUBMITTED_TASKS=5000 ;;
        esac
        case $mode in
            combined-2workers|gpu-norm*) export STARPU_NWORKER_PER_CUDA=2 ;;
        esac
        if [[ $mode == gpu-norm* ]]; then export DARE_GPU_NORM=1; fi
        if [[ $mode == gpu-norm-swap ]]; then export DARE_SWAP_ALPHA=1; fi
        if [[ $plan == norm && $run == warmup ]]; then export STARPU_BUS_STATS=1; fi
        if [[ $run != warmup ]]; then unset STARPU_BUS_STATS; fi
        if [[ $run == 3 ]]; then export DARE_OUTPUT_PREFIX="$results/$mode"; fi
        log="$mode-$run.log"
        printf '%s %s %s\n' "$(date -Is)" "$mode" "$run"
        if ! "${launch[@]}" /usr/bin/time -f 'Wall seconds: %e; Max RSS KiB: %M' \
            stdbuf -oL -eL "$executable" --threads=5 --n="$states" --ninstr=18694 \
            --nb=720 --ib=180 --gpus="$gpus" --datapath="$root/datasets/$states" \
            > "$log" 2>&1; then
            tail -40 "$log"
            exit 1
        fi
        grep '^Time:' "$log"
        iterations=$(awk '/^it [0-9]+ norm / { n++ } END { print n+0 }' "$log")
        (( iterations > 0 ))
        if [[ -z $reference_iterations ]]; then reference_iterations=$iterations; fi
        (( iterations == reference_iterations ))
        tiles=$(awk '/^DARE cuSOLVER LU device / { n+=$NF } END { print n+0 }' "$log")
        (( tiles == DARE_CUSOLVER_LU * iterations * ((states + 719) / 720) ))
    done
done

status=0
for mode in "${modes[@]:1}"; do
    for matrix in P K; do
        python3 "$root/cusolver-lu/compare-fits.py" "${modes[0]}-$matrix.fits" "$mode-$matrix.fits" || status=1
    done
done > accuracy.jsonl
cat accuracy.jsonl
exit "$status"
