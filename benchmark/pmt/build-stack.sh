#!/usr/bin/env bash
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
set -u
export CC="$(command -v gcc)" CXX="$(command -v g++)" FC="$(command -v gfortran)"
cd "$dare_root"
mkdir -p build/{cfitsio,starpu,chameleon,dare} logs src cuda/{include,lib64}
for archive in cfitsio-4.0.0 starpu-1.3.11 chameleon-1.1.0; do
    if [[ ! -d src/$archive ]]; then
        tar xf "downloads/$archive.tar.gz" -C src
    fi
done

apply_once() {
    local target=$1 patch_file=$2
    if patch --dry-run --forward --silent -d "$target" -p1 < "$patch_file" >/dev/null; then
        patch --forward --silent -d "$target" -p1 < "$patch_file"
    else
        patch --dry-run --reverse --silent -d "$target" -p1 < "$patch_file"
    fi
}

apply_once src/chameleon-1.1.0 chameleon-cuda12.patch
apply_once src/dare dare-report-sequence-status.patch

# The NVIDIA HPC SDK keeps the toolkit and math libraries in separate directories.
for dir in "$CUDA_HOME" "$dare_math"; do
    for part in include lib64; do
        for entry in "$dir/$part/"*; do
            link="$dare_root/cuda/$part/${entry##*/}"
            [[ -e "$link" || -L "$link" ]] || ln -s "$entry" "$link"
        done
    done
done

if ! pkg-config --exact-version=4.0.0 cfitsio; then
    cmake --fresh -S src/cfitsio-4.0.0 -B build/cfitsio -G Ninja \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$dare_root/install" \
        -DUSE_CURL=OFF -DUSE_PTHREADS=ON \
        -DCFITSIO_MAJOR=4 -DCFITSIO_MINOR=0 -DCFITSIO_MICRO=0 -DPKG_CONFIG_LIBS=-lcfitsio > logs/cfitsio-configure.log 2>&1
    cmake --build build/cfitsio --target install --parallel 6 > logs/cfitsio-build.log 2>&1
fi

if ! pkg-config --exact-version=1.3.11 starpu-1.3; then
    (
        cd build/starpu
        export NVCCFLAGS="-arch=sm_80"
        export CPPFLAGS="-I$dare_root/compat/include"
        export LDFLAGS="-L$CUDA_HOME/lib64/stubs"
        export CFLAGS="-O2 -Wno-deprecated-declarations"
        "$dare_root/src/starpu-1.3.11/configure" --prefix="$dare_root/install" \
            --with-cuda-dir="$dare_root/cuda" --with-cuda-lib-dir="$dare_root/cuda/lib64" \
            --with-mpicc="$(command -v mpicc)" --with-mpicxx="$(command -v mpicxx)" \
            --disable-opencl --disable-build-tests --disable-build-examples --disable-fortran \
            > "$dare_root/logs/starpu-configure.log" 2>&1
        make -j6 install > "$dare_root/logs/starpu-build.log" 2>&1
    )
fi
grep -q '^#define STARPU_USE_CUDA 1' install/include/starpu/1.3/starpu_config.h
pkg-config --modversion starpu-1.3 starpumpi-1.3 cfitsio

cmake -S src/chameleon-1.1.0 -B build/chameleon -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$dare_root/install" \
    -DCMAKE_PREFIX_PATH="$dare_root/install;$MKLROOT" -DCMAKE_LIBRARY_PATH="$MKLROOT/lib" \
    -DBLA_VENDOR=Intel10_64lp_seq -DCBLAS_INCDIR="$MKLROOT/include" -DLAPACKE_INCDIR="$MKLROOT/include" \
    -DBUILD_SHARED_LIBS=ON -DCHAMELEON_USE_CUDA=ON -DCHAMELEON_USE_MPI=ON \
    -DCHAMELEON_ENABLE_EXAMPLE=OFF -DCHAMELEON_ENABLE_TESTING=OFF \
    -DCUDA_TOOLKIT_ROOT_DIR="$dare_root/cuda" -DCUDA_NVCC_EXECUTABLE="$CUDA_HOME/bin/nvcc" \
    -DCUDA_CUDA_LIBRARY="$CUDA_HOME/lib64/stubs/libcuda.so" \
    -DMPI_C_COMPILER="$(command -v mpicc)" -DMPI_CXX_COMPILER="$(command -v mpicxx)" \
    > logs/chameleon-configure.log 2>&1
cmake --build build/chameleon --target install --parallel 6 > logs/chameleon-build.log 2>&1

cmake --fresh -S src/dare -B build/dare -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DCHAMELEON_BACKEND=ON \
    -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" -DCMAKE_CUDA_ARCHITECTURES=80 \
    -DCMAKE_PREFIX_PATH="$dare_root/install;$MKLROOT" -DCMAKE_LIBRARY_PATH="$MKLROOT/lib" \
    > logs/dare-configure.log 2>&1
cmake --build build/dare --parallel 6 > logs/dare-build.log 2>&1
