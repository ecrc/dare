#!/usr/bin/env bash
# Build a StarPU library overlay for an existing Lair or Perlmutter installation.
set -eo pipefail
root=$(cd "${1:?Pass the DARE deployment directory}" && pwd)
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$root/env.sh"
set -u

build="$root/build/chameleon-cusolver"
patches=("$script_dir/chameleon-cusolver-lu.patch")
cuda_options=()
case ${2:-lu} in
    lu) ;;
    mpi)
        build="$root/build/chameleon-cusolver-mpi"
        cuda_options=(-DCHAMELEON_USE_MPI_DATATYPES=OFF) ;;
    gpu-norm|gpu-static|gpu-static-solve)
        build="$root/build/chameleon-${2}"
        patches+=("$script_dir/chameleon-gpu-norm.patch")
        cuda_options=(-DCHAMELEON_DARE_GPU_NORM=ON -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc"
                      -DCMAKE_CUDA_ARCHITECTURES="${dare_cuda_arch:-80}")
        if [[ $2 == gpu-static || $2 == gpu-static-solve ]]; then
            cuda_options+=(-DPYTHON_EXECUTABLE="${DARE_BUILD_PYTHON:-$(command -v python3)}")
        fi ;;
    *) printf 'Unknown build variant %s\n' "$2" >&2; exit 2 ;;
esac
source_dir="$build/source"
mkdir -p "$build"
if [[ ! -d "$source_dir" ]]; then
    cp -a "$root/src/chameleon-1.1.0" "$source_dir"
fi
# The norm patch is layered on the LU patch and touches some of the same lines.
if [[ ${2:-lu} == gpu-norm || ${2:-lu} == gpu-static || ${2:-lu} == gpu-static-solve ]] &&
    patch --force --dry-run --reverse --silent -d "$source_dir" -p1 < "${patches[1]}" >/dev/null; then
    patches=()
fi
if [[ ${2:-lu} == gpu-static || ${2:-lu} == gpu-static-solve ]]; then
    patches+=("$script_dir/chameleon-static-gemm.patch")
fi
if [[ ${2:-lu} == gpu-static-solve ]]; then
    patches+=("$script_dir/chameleon-static-solve.patch")
    # The solve patch extends the GEMM owner function, so checking the older
    # GEMM patch in reverse is no longer sufficient on a subsequent build.
    if patch --force --dry-run --reverse --silent -d "$source_dir" -p1 \
        < "$script_dir/chameleon-static-solve.patch" >/dev/null; then
        patches=()
    fi
fi
for patch_file in "${patches[@]}"; do
    if patch --dry-run --forward --silent -d "$source_dir" -p1 < "$patch_file" >/dev/null; then
        patch --forward --silent -d "$source_dir" -p1 < "$patch_file"
    else
        patch --force --dry-run --reverse --silent -d "$source_dir" -p1 < "$patch_file"
    fi
done

cuda_root="$CUDA_HOME"
mkl_lib="${dare_mkl_lib:-$MKLROOT/lib}"
if [[ -n ${dare_math:-} ]]; then
    cuda_root="$root/cuda"
    export CC="$(command -v gcc)" CXX="$(command -v g++)" FC="$(command -v gfortran)"
fi

cmake -S "$source_dir" -B "$build" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$root/install;$MKLROOT" \
    -DCMAKE_LIBRARY_PATH="$mkl_lib" -DBLA_VENDOR=Intel10_64lp_seq \
    -DCBLAS_INCDIR="$MKLROOT/include" -DLAPACKE_INCDIR="$MKLROOT/include" \
    -DBUILD_SHARED_LIBS=ON -DCHAMELEON_USE_CUDA=ON -DCHAMELEON_USE_MPI=ON \
    -DCHAMELEON_ENABLE_EXAMPLE=OFF -DCHAMELEON_ENABLE_TESTING=OFF \
    -DCHAMELEON_DARE_CUSOLVER_LU=ON \
    "${cuda_options[@]}" \
    -DCUDA_TOOLKIT_ROOT_DIR="$cuda_root" -DCUDA_NVCC_EXECUTABLE="$CUDA_HOME/bin/nvcc" \
    -DCUDA_CUDA_LIBRARY="$CUDA_HOME/lib64/stubs/libcuda.so" \
    -DMPI_C_COMPILER="$(command -v mpicc)" -DMPI_CXX_COMPILER="$(command -v mpicxx)" \
    > "$build/configure.log" 2>&1
cmake --build "$build" --target chameleon_starpu --parallel "${DARE_BUILD_JOBS:-6}" \
    > "$build/build.log" 2>&1
printf '%s\n' "$build/runtime/starpu"
