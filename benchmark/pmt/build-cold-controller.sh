#!/usr/bin/env bash
set -euo pipefail
src=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
output=${1:?output executable}
arch=${2:-sm_80}
rows=${3:-8}
flags=()
if [[ -n ${CUBLAS_LIB_DIR:-} ]]; then
    flags+=(-L"$CUBLAS_LIB_DIR" -Xlinker -rpath -Xlinker "$CUBLAS_LIB_DIR")
fi
if [[ -n ${CUDA_HOST_CXX:-} ]]; then
    flags+=(-ccbin "$CUDA_HOST_CXX")
fi
"${NVCC:-nvcc}" -O3 -std=c++17 -arch="$arch" -DMVM_ROWS_PER_BLOCK="$rows" \
    "$src/benchmark-cold-controller.cu" "${flags[@]}" -lcublas -o "$output"
sha256sum "$src/benchmark-cold-controller.cu" "$src/benchmark-cold-pipeline.cu" "$output"
