#!/bin/bash
#SBATCH -A m1982_g
#SBATCH -C gpu&a100&hbm80g
#SBATCH -q debug
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH -t 00:15:00
#SBATCH -J dare-io28210
set -eo pipefail
root="$SCRATCH/dare-a100-7090"
suite="$root/cusolver-lu/28210-io-breakdown"
source "$root/env.sh"
set -u
out="$suite/results/$SLURM_JOB_ID"
mkdir -p "$out/files" "$out/starpu"
lfs setstripe -c 4 -S 8M "$out/files"
export LD_LIBRARY_PATH="$root/build/chameleon-gpu-static/runtime/starpu:$LD_LIBRARY_PATH" STARPU_HOME="$out/starpu"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPICH_GPU_SUPPORT_ENABLED=0
export STARPU_SCHED=dmda STARPU_SCHED_BETA=4 STARPU_SILENT=1 STARPU_CALIBRATE=1
export STARPU_CUDA_PIPELINE=4 STARPU_NWORKER_PER_CUDA=2 STARPU_WORKERS_GETBIND=1
export STARPU_LIMIT_MAX_SUBMITTED_TASKS=10000 STARPU_LIMIT_MIN_SUBMITTED_TASKS=5000
export DARE_CUSOLVER_LU=1 DARE_GPU_COPY=1 DARE_GPU_NORM=1 DARE_SWAP_ALPHA=0
export DARE_DIRECT_BETA=1 DARE_DIAGONAL_IDENTITY=1 DARE_SPLIT_UPDATE_WORK=0 DARE_STATIC_GEMM=1
export DARE_TILE_READERS_PER_FILE=2 DARE_TILE_WRITERS_PER_FILE=4
unset DARE_OUTPUT_PREFIX DARE_TILE_PINNED_INPUT DARE_RICCATI_ONLY DARE_PHASE_PROFILE DARE_STAGE_BARRIERS STARPU_BUS_STATS STARPU_COMM_STATS DARE_SNAPSHOT_DIR DARE_SNAPSHOT_STEPS DARE_ASYNC_NORM DARE_STATIC_TRSM DARE_STATIC_LU
launch=(srun --mpi=cray_shasta --input=none --exact -N1 -n1 -c64 --gpus-per-node=4 --gpu-bind=none --cpu-bind=cores)
"${launch[@]}" bash -c 'hostname; nvidia-smi --query-gpu=name,memory.total --format=csv; nvidia-smi topo -m; grep Cpus_allowed_list /proc/self/status' > "$out/hardware.txt"
sha256sum "$suite/build/ddarechameleon" "$suite/src/dare/ddarechameleon.cpp" > "$out/provenance.sha256"
for run in 1 2; do
  export DARE_TILE_OUTPUT_DIR="$out/files/run-$run"
  "${launch[@]}" /usr/bin/time -f 'Wall seconds: %e\nMaximum host RSS (KiB): %M' stdbuf -oL -eL "$suite/build/ddarechameleon" --threads=5 --n=28210 --ninstr=18694 --nb=720 --ib=180 --gpus=4 --tilepath="$root/datasets/28210-tiles720" > "$out/run-$run.log" 2>&1
  awk '/^it / {step=$2; norm=$4; tol=$6} END{exit !(step==17 && norm>=0 && norm<tol)}' "$out/run-$run.log"
  grep -E 'WALL_|seconds:|^Time:|^it 17 ' "$out/run-$run.log"
done
for run in 1 2; do
  for matrix in P K; do
    "${launch[@]}" python3 "$suite/compare-packed-tiles.py" "$root/cusolver-lu/28210-tiles720/results/61077914/files/tiles-4w/$matrix.tiles" "$out/files/run-$run/$matrix.tiles" --rtol 1e-9 >> "$out/accuracy.jsonl"
  done
done
touch "$out/COMPLETE"
