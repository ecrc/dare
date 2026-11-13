dare_root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
case "$dare_root" in
    /data/project/yuxilab/*)
        export CUDA_HOME=/data/project/yuxilab/cuda/cuda-12.8
        export MKLROOT=/data/project/yuxilab/intel/oneapi/mkl/2026.1
        dare_cuda_arch=89
        dare_mkl_lib="$MKLROOT/lib"
        ;;
    *)
        export CUDA_HOME=/media/volume/workspace/tools/cuda-12.4.1
        export MKLROOT="$dare_root/mkl"
        dare_cuda_arch=80
        dare_mkl_lib="$MKLROOT/lib/intel64"
        ;;
esac
export PATH="$dare_root/install/bin:$CUDA_HOME/bin:$PATH"
export PKG_CONFIG_PATH="$dare_root/install/lib/pkgconfig:$dare_mkl_lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
export LD_LIBRARY_PATH="$dare_root/install/lib:$dare_mkl_lib:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
