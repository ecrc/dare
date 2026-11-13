#!/usr/bin/env bash
#SBATCH -A bhwj-dtai-gh
#SBATCH -p ghx4-interactive
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 288
#SBATCH --gpus-per-node=4
#SBATCH -t 00:25:00
#SBATCH -J dare-ao-fp64
set -eo pipefail
root=/work/hdd/bhwj/yhong4/dare-dai-28210
source "$root/env.sh"
set -u
suite="$root/ao-size-sweep"
out="$suite/results/$SLURM_JOB_ID"
mkdir -p "$out/starpu"
export LD_LIBRARY_PATH="$suite/bin:$LD_LIBRARY_PATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPICH_GPU_SUPPORT_ENABLED=0
export STARPU_SCHED=dmda STARPU_SCHED_BETA=4 STARPU_SILENT=1 STARPU_CALIBRATE=1
export STARPU_CUDA_PIPELINE=4 STARPU_NWORKER_PER_CUDA=2 STARPU_WORKERS_GETBIND=1
export STARPU_WORKERS_CPUID="4 76 148 220 5 77 149 221 6 78 150 222 7 79 151 223"
export STARPU_LIMIT_MAX_SUBMITTED_TASKS=10000 STARPU_LIMIT_MIN_SUBMITTED_TASKS=5000
export DARE_CUSOLVER_LU=1 DARE_GPU_COPY=1 DARE_GPU_NORM=1 DARE_SWAP_ALPHA=0
export DARE_DIRECT_BETA=1 DARE_DIAGONAL_IDENTITY=1 DARE_SPLIT_UPDATE_WORK=0
export DARE_STATIC_GEMM=1 STARPU_HOME="$out/starpu"
export DARE_FITS_READERS_PER_FILE=4
unset DARE_TILE_OUTPUT_DIR DARE_SNAPSHOT_DIR DARE_ASYNC_NORM
launch=(srun --mpi=cray_shasta --input=none --exact -N1 -n1 -c288 --gpus-per-node=4 --gpu-bind=none --cpu-bind=none)
"${launch[@]}" bash -c 'hostname; nvidia-smi topo -m; nvidia-smi --query-gpu=name,memory.total --format=csv' > "$out/hardware.txt"
cp "$suite/build-sha256.txt" "$out/"
run_case() {
    local n=$1 nb=$2 name=$3 save=$4 async=$5
    test -f "$root/datasets/$n/.verified"
    mkdir -p "$out/$name"
    unset DARE_OUTPUT_PREFIX
    if ((save)); then export DARE_OUTPUT_PREFIX="$out/$name/result"; fi
    export DARE_ASYNC_NORM="$async"
    "${launch[@]}" /usr/bin/time -f 'Wall seconds: %e' numactl --interleave=0-3 \
        "$suite/bin/ddarechameleon" --threads=12 --n="$n" --ninstr=18694 --nb="$nb" --ib=180 \
        --gpus=4 --datapath="$root/datasets/$n" > "$out/$name/solver.log" 2>&1
    awk '/^it / {step=$2; norm=$4; tol=$6} END{exit !(step>0 && norm>=0 && norm<tol)}' "$out/$name/solver.log"
    grep -E '^Time:|^Wall seconds:' "$out/$name/solver.log"
}
run_case 7090 720 7090 1 0
run_case 12570 720 12570 1 0
run_case 28210 720 28210 1 0
run_case 28210 1440 28210-nb1440 1 0
run_case 28210 1440 28210-nb1440-async 0 1
run_case 28210 1440 28210-nb1440-repeat 0 0
touch "$out/COMPUTE_OK"
