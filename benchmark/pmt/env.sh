# Source after the Perlmutter module environment has been initialized.
dare_root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
module load cudatoolkit/12.4 gcc-native/12
export MKLROOT=/opt/intel/oneapi/mkl/2025.3
dare_math="${CUDA_HOME%/cuda/*}/math_libs/12.4"
export PKG_CONFIG_PATH="$dare_root/install/lib/pkgconfig:$dare_root/install/lib64/pkgconfig:$MKLROOT/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
export LD_LIBRARY_PATH="$dare_root/install/lib:$dare_root/install/lib64:$MKLROOT/lib:$CUDA_HOME/lib64:$dare_math/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
