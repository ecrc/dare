#!/usr/bin/env bash
# Build an isolated diagnostic executable beside the original benchmark binary.
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
set -u
profile=${1:-dare-profile}
case $profile in
    dare-profile|dare-profile-swap|dare-profile-mpi) ;;
    *) printf 'Unknown diagnostic build %s\n' "$profile" >&2; exit 2 ;;
esac
cmake_args=()
if [[ -n ${dare_math:-} ]]; then
    export CC="$(command -v gcc)" CXX="$(command -v g++)" FC="$(command -v gfortran)"
    cmake_args=(-G Ninja)
fi
if [[ ! -d "$dare_root/src/$profile" ]]; then
    cp -a "$dare_root/src/dare" "$dare_root/src/$profile"
fi
profile_patch="$dare_root/dare-profile-copy.patch"
if patch --dry-run --forward --silent -d "$dare_root/src/$profile" -p1 < "$profile_patch" >/dev/null; then
    patch --forward --silent -d "$dare_root/src/$profile" -p1 < "$profile_patch"
else
    patch --dry-run --reverse --silent -d "$dare_root/src/$profile" -p1 < "$profile_patch"
fi
cmake -S "$dare_root/src/$profile" -B "$dare_root/build/$profile" "${cmake_args[@]}" \
    -DCMAKE_BUILD_TYPE=Release -DCHAMELEON_BACKEND=ON \
    -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" -DCMAKE_CUDA_ARCHITECTURES="${dare_cuda_arch:-80}" \
    -DCMAKE_PREFIX_PATH="$dare_root/install;$MKLROOT" -DCMAKE_LIBRARY_PATH="${dare_mkl_lib:-$MKLROOT/lib}" \
    > "$dare_root/logs/$profile-configure.log" 2>&1
cmake --build "$dare_root/build/$profile" --parallel 4 \
    > "$dare_root/logs/$profile-build.log" 2>&1
