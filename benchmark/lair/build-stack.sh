#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

dare_src=${1:?Pass the DARE source path}
cd "$dare_root"
mkdir -p build/starpu-1.3.11 logs src
for archive in starpu-1.3.11 chameleon-1.1.0; do
    if [[ ! -d src/$archive ]]; then
        tar xf "downloads/$archive.tar.gz" -C src
    fi
done

if ! pkg-config --exact-version=1.3.11 starpu-1.3; then
    (
        cd build/starpu-1.3.11
        export NVCCFLAGS="-arch=sm_$dare_cuda_arch"
        export LDFLAGS="-L$CUDA_HOME/lib64/stubs"
        export CFLAGS="-O2 -Wno-deprecated-declarations"
        "$dare_root/src/starpu-1.3.11/configure" --prefix="$dare_root/install" \
            --with-cuda-dir="$CUDA_HOME" --with-cuda-lib-dir="$CUDA_HOME/lib64" \
            --disable-opencl --disable-build-tests --disable-build-examples --disable-fortran \
            > "$dare_root/logs/starpu-1.3.11-configure.log" 2>&1
        make -j8 install > "$dare_root/logs/starpu-1.3.11-build.log" 2>&1
    )
fi
grep -q '^#define STARPU_USE_CUDA 1' install/include/starpu/1.3/starpu_config.h
pkg-config --modversion starpu-1.3 starpumpi-1.3

if patch --dry-run --forward --silent -d src/chameleon-1.1.0 -p1 < chameleon-cuda12.patch; then
    patch --forward --silent -d src/chameleon-1.1.0 -p1 < chameleon-cuda12.patch
else
    patch --dry-run --reverse --silent -d src/chameleon-1.1.0 -p1 < chameleon-cuda12.patch
fi

cmake -S src/chameleon-1.1.0 -B build/chameleon \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$dare_root/install" \
    -DCMAKE_PREFIX_PATH="$dare_root/install;$MKLROOT" -DCMAKE_LIBRARY_PATH="$dare_mkl_lib" \
    -DBLA_VENDOR=Intel10_64lp_seq -DCBLAS_INCDIR="$MKLROOT/include" -DLAPACKE_INCDIR="$MKLROOT/include" \
    -DBUILD_SHARED_LIBS=ON -DCHAMELEON_USE_CUDA=ON -DCHAMELEON_USE_MPI=ON \
    -DCHAMELEON_ENABLE_EXAMPLE=OFF -DCHAMELEON_ENABLE_TESTING=OFF \
    -DCUDA_TOOLKIT_ROOT_DIR="$CUDA_HOME" -DCUDA_CUDA_LIBRARY="$CUDA_HOME/lib64/stubs/libcuda.so" \
    > logs/chameleon-configure.log 2>&1
cmake --build build/chameleon --parallel 8 > logs/chameleon-build.log 2>&1
cmake --install build/chameleon > logs/chameleon-install.log 2>&1

cmake -S "$dare_src" -B build/dare \
    -DCMAKE_BUILD_TYPE=Release -DCHAMELEON_BACKEND=ON \
    -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" -DCMAKE_CUDA_ARCHITECTURES="$dare_cuda_arch" \
    -DCMAKE_PREFIX_PATH="$dare_root/install;$MKLROOT" -DCMAKE_LIBRARY_PATH="$dare_mkl_lib" \
    > logs/dare-configure.log 2>&1
cmake --build build/dare --parallel 8 > logs/dare-build.log 2>&1
