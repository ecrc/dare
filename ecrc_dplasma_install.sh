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
cd $workdir
rm -rf dplasma
git clone --recursive https://github.com/ICLDisco/dplasma.git
cd dplasma
mkdir build && cd build
../configure --prefix=$(pwd)/install \
--with-cuda=$CUDA_HOME -DCMAKE_CUDA_COMPILER=$(which nvcc)
make -j
make install -j
export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:$workdir/dplasma/build/install/lib/pkgconfig
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$workdir/dplasma/build/install/lib

cd $workdir
rm -rf dare-dev
git clone git@github.com:ecrc/dare-dev.git
cd dare-dev
mkdir build
cd build
conan install .. -b missing
export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:$(pwd)
cmake -DCMAKE_INSTALL_PREFIX=$(pwd)/install \
-DDPLASMA_BACKEND=ON -DCMAKE_CUDA_COMPILER=$(which nvcc) ..

make -j

./ddareparsec -c 20 -g 1

# return
cd $curdir
