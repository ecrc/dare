#!/usr/bin/env bash
# Build an isolated ARM copy of the same FP64 solver source used on Perlmutter.
set -eo pipefail
root=/work/hdd/bhwj/yhong4/dare-dai-28210
source "$root/env.sh"
set -u
suite="$root/ao-size-sweep"
work=$(mktemp -d /tmp/dare-ao.XXXXXX)
printf '%s\n' "$work" > "$suite/build-work.txt"
cp -a "$suite/src" "$work/dare-src"
sed -i 's/set(BLA_VENDOR Intel10_64lp_seq)/set(BLA_VENDOR OpenBLAS)/g' "$work/dare-src/CMakeLists.txt"
export CC="$(command -v gcc)" CXX="$(command -v g++)"
cmake -S "$work/dare-src" -B "$work/dare-build" -G Ninja \
    -DCHAMELEON_BACKEND=ON -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" -DCMAKE_CUDA_ARCHITECTURES=90 \
    -DCMAKE_PREFIX_PATH="$root/install;$OPENBLAS_HOME" -DCMAKE_LIBRARY_PATH="$OPENBLAS_HOME/lib"
cmake --build "$work/dare-build" -j4
mkdir -p "$suite/bin"
cp "$work/dare-build/ddarechameleon" "$work/dare-build/libfitsio.so" "$suite/bin/"
base=/tmp/dare-build-28210.tGviJC
cp -a "$base/src/chameleon-1.1.0" "$work/chameleon-src"
patch -d "$work/chameleon-src" -p1 < "$suite/chameleon-static-gemm.patch"
cmake -S "$work/chameleon-src" -B "$work/chameleon-build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$root/install" \
    -DCMAKE_PREFIX_PATH="$root/install;$OPENBLAS_HOME" -DCMAKE_LIBRARY_PATH="$OPENBLAS_HOME/lib" \
    -DBLA_VENDOR=OpenBLAS -DCBLAS_INCDIR="$OPENBLAS_HOME/include" -DLAPACKE_INCDIR="$OPENBLAS_HOME/include" \
    -DBUILD_SHARED_LIBS=ON -DCHAMELEON_USE_CUDA=ON -DCHAMELEON_USE_MPI=ON \
    -DCHAMELEON_ENABLE_EXAMPLE=OFF -DCHAMELEON_ENABLE_TESTING=OFF \
    -DCHAMELEON_PREC_C=ON -DCHAMELEON_PREC_Z=ON -DCMAKE_C_FLAGS=-Wno-error=incompatible-pointer-types \
    -DCHAMELEON_DARE_CUSOLVER_LU=ON -DCHAMELEON_DARE_GPU_NORM=ON \
    -DCUDA_TOOLKIT_ROOT_DIR="$base/cuda" -DCUDA_NVCC_EXECUTABLE="$CUDA_HOME/bin/nvcc" \
    -DCUDA_CUDA_LIBRARY="$CUDA_HOME/lib64/stubs/libcuda.so" \
    -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" -DCMAKE_CUDA_ARCHITECTURES=90 \
    -DCMAKE_CUDA_HOST_COMPILER="$CXX" \
    -DMPI_C_COMPILER="$(command -v mpicc)" -DMPI_CXX_COMPILER="$(command -v mpicxx)"
cmake --build "$work/chameleon-build" --target chameleon_starpu -j4
cp -a "$work/chameleon-build/runtime/starpu/"libchameleon_starpu.so* "$suite/bin/"
sha256sum "$suite/bin/ddarechameleon" "$suite/bin/libchameleon_starpu.so" > "$suite/build-sha256.txt"
