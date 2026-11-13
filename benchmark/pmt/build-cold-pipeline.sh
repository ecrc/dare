#!/usr/bin/env bash
set -euo pipefail
src=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
output=${1:?output executable}
arch=${2:-sm_80}
nvcc_bin=${NVCC:-nvcc}
flags=()
if [[ -n ${MVM_ROWS_PER_BLOCK:-} ]]; then
    flags+=(-DMVM_ROWS_PER_BLOCK="$MVM_ROWS_PER_BLOCK")
fi
if [[ -n ${CUBLAS_LIB_DIR:-} ]]; then
    flags+=(-L"$CUBLAS_LIB_DIR" -Xlinker -rpath -Xlinker "$CUBLAS_LIB_DIR")
fi
if [[ -n ${CUDA_HOST_CXX:-} ]]; then
    flags+=(-ccbin "$CUDA_HOST_CXX")
fi
"$nvcc_bin" -O3 -std=c++17 -arch="$arch" "$src/benchmark-cold-pipeline.cu" \
    "${flags[@]}" -lcublas -o "$output"
sha256sum "$src/benchmark-cold-pipeline.cu" "$output"
