#!/bin/bash

# script to install dare with starpu backend on ecrc nvidia gpu server.
curdir=$(pwd)
# starpu
module load gcc/10.2.0
module load cmake/3.19.2
module load cuda/11.4
module load openmpi/4.1.0-gcc-10.2.0
module load hwloc/2.4.0-gcc-10.2.0
module load python-3.9.9-gcc-7.5.0-bp37qr2
module load mkl/2020.0.166

python3 -m venv $HOME/.venv/starpu-dare-dev
source $HOME/.venv/starpu-dare-dev/bin/activate
pip install conan

export workdir=$HOME/tmpbuild
rm -rf $workdir
mkdir -p $workdir

cd $workdir && rm -rf starpu-1.3.9.tar.gz starpu-1.3.9
wget https://files.inria.fr/starpu/starpu-1.3.9/starpu-1.3.9.tar.gz
tar -xzf starpu-1.3.9.tar.gz
cd starpu-1.3.9
mkdir build && cd build 
../configure --prefix=$(pwd)/install --with-cuda-dir=$CUDA_HOME
make install -j
export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:$workdir/starpu-1.3.9/build/install/lib/pkgconfig
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$workdir/starpu-1.3.9/build/install/lib

# chameleon
cd $workdir && rm -rf chameleon-1.1.0.tar.gz chameleon-1.1.0
wget https://gitlab.inria.fr/solverstack/chameleon/uploads/b299d6037d7636c6be16108c89bc2aab/chameleon-1.1.0.tar.gz
tar xvf chameleon-1.1.0.tar.gz
cd chameleon-1.1.0
mkdir build && cd build
cmake .. -DCHAMELEON_USE_CUDA=ON -DCHAMELEON_USE_MPI=ON -DBUILD_SHARED_LIBS=ON \
-DCMAKE_INSTALL_PREFIX=$(pwd)/install
make -j install
export CHAMELEON_TESTING=$workdir/chameleon-1.1.0/testing
export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:$workdir/chameleon-1.1.0/build/install/lib/pkgconfig
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$workdir/chameleon-1.1.0/build/install/lib

cd $workdir
rm -rf dare-dev
git clone git@github.com:ecrc/dare-dev.git
cd dare-dev
git checkout yuxi/dev # currrently the code is at my branch
rm -rf build && mkdir build && cd build
conan install -b missing ..
export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:$(pwd)
cmake -DCMAKE_CUDA_COMPILER=$(which nvcc) -DCHAMELEON_BACKEND=ON -DDPLASMA_BACKEND=OFF ..
make -j

# test random dataset
CHAMELEON_USE_CUDA=1 STARPU_SILENT=1 OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 STARPU_CUDA_PIPELINE=4 STARPU_NWORKER_PER_CUDA=4 \
STARPU_SCHED=dmda STARPU_CALIBRATE=1 CUDA_VISIBLE_DEVICES=0 \
./ddarechameleon --threads=5 --n=7090 --ninstr=18694 --nb=720 --ib=180 --gpus=1

# return
cd $curdir
