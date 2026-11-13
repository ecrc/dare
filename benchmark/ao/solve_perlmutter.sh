#!/usr/bin/env bash
#SBATCH -A m1982_g
#SBATCH -q debug
#SBATCH -C gpu&a100&hbm80g
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH -t 00:20:00
#SBATCH -J dare-ao-fp64
set -eo pipefail
root="$SCRATCH/dare-a100-7090"
source "$root/env.sh"
set -u
out="$root/ao-size-sweep/$SLURM_JOB_ID"
mkdir -p "$out/starpu"
binary="$root/cusolver-lu/28210-tiles720/build-lazy-io/ddarechameleon"
export LD_LIBRARY_PATH="$root/build/chameleon-gpu-static/runtime/starpu:$LD_LIBRARY_PATH"
export STARPU_HOME="$out/starpu" STARPU_SCHED=dmda STARPU_SCHED_BETA=4
export STARPU_SILENT=1 STARPU_CALIBRATE=1 STARPU_CUDA_PIPELINE=4 STARPU_NWORKER_PER_CUDA=2
export STARPU_LIMIT_MAX_SUBMITTED_TASKS=10000 STARPU_LIMIT_MIN_SUBMITTED_TASKS=5000
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPICH_GPU_SUPPORT_ENABLED=0
export DARE_CUSOLVER_LU=1 DARE_GPU_COPY=1 DARE_GPU_NORM=1 DARE_SWAP_ALPHA=0
export DARE_DIRECT_BETA=1 DARE_DIAGONAL_IDENTITY=1 DARE_SPLIT_UPDATE_WORK=0 DARE_STATIC_GEMM=1
export DARE_FITS_READERS_PER_FILE=4
unset DARE_TILE_OUTPUT_DIR DARE_SNAPSHOT_DIR DARE_ASYNC_NORM DARE_STATIC_TRSM DARE_STATIC_LU
sha256sum "$binary" > "$out/binary-sha256.txt"
srun --mpi=cray_shasta --input=none --exact -N1 -n1 -c64 --gpus-per-node=4 --gpu-bind=none --cpu-bind=cores \
    bash -c 'hostname; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv; nvidia-smi topo -m' \
    > "$out/hardware.txt"
for n in 7090 12570 28210; do
    mkdir -p "$out/$n"
    export DARE_OUTPUT_PREFIX="$out/$n/result"
    srun --mpi=cray_shasta --input=none --exact -N1 -n1 -c64 --gpus-per-node=4 --gpu-bind=none --cpu-bind=cores \
        /usr/bin/time -f 'Wall seconds: %e' "$binary" --threads=5 --n="$n" --ninstr=18694 \
        --nb=720 --ib=180 --gpus=4 --datapath="$root/datasets/$n" > "$out/$n/solver.log" 2>&1
    test -s "$out/$n/result-K.fits"
done
touch "$out/COMPUTE_OK"
