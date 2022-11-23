pipeline {
    agent { label 'Vulture' }
    triggers {
        pollSCM('H/10 * * * *')
    }
    options {
        disableConcurrentBuilds()
        buildDiscarder(logRotator(numToKeepStr: '50'))
        timestamps()
    }
    stages {
        stage ('build-dare-chameleon') {
            steps {
            sh '''#!/bin/bash -el
                    module purge
                    module load gcc/10.2.0
                    module load cmake/3.19.2
                    module load cuda/11.4
                    module load openmpi/4.1.0-gcc-10.2.0
                    module load hwloc/2.4.0-gcc-10.2.0
                    module load python-3.9.9-gcc-7.5.0-bp37qr2
                    module load mkl/2020.0.166
                    module list
                    daredir=$(pwd)
                   # starpu
                    cd $HOME && rm -rf starpu-1.3.9.tar.gz starpu-1.3.9 7090
                    wget https://files.inria.fr/starpu/starpu-1.3.9/starpu-1.3.9.tar.gz
                    tar -xzf starpu-1.3.9.tar.gz
                    cd starpu-1.3.9
                    export STARPU_ROOT=$(pwd)
                    mkdir build && cd build 
                    echo "CUDA HOME is "
                    echo $CUDA_HOME
                    # make sure your cuda is in right place. starpu won't fail if cuda is not found.
                    ../configure --prefix=$(pwd)/install --with-cuda-dir=$CUDA_HOME
                    make install -j
                    export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:$STARPU_ROOT/build/install/lib/pkgconfig
                    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$STARPU_ROOT/build/install/lib
                    # chameleon
                    cd $HOME && rm -rf chameleon-1.1.0.tar.gz chameleon-1.1.0
                    wget https://gitlab.inria.fr/solverstack/chameleon/uploads/b299d6037d7636c6be16108c89bc2aab/chameleon-1.1.0.tar.gz
                    tar xvf chameleon-1.1.0.tar.gz
                    cd chameleon-1.1.0
                    export CHAMELEON_ROOT=$(pwd)
                    mkdir build && cd build
                    cmake .. -DCHAMELEON_USE_CUDA=ON -DCHAMELEON_USE_MPI=ON -DBUILD_SHARED_LIBS=ON \
                    -DCMAKE_INSTALL_PREFIX=$(pwd)/install
                    make -j install
                    export CHAMELEON_TESTING=$CHAMELEON_ROOT/testing
                    export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:$CHAMELEON_ROOT/build/install/lib/pkgconfig
                    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CHAMELEON_ROOT/build/install/lib
                    rm -rf ~/.venv/dareenv
                    python -m venv ~/.venv/dareenv
                    source ~/.venv/dareenv/bin/activate
                    pip install conan
                    # dare installation
                    cd $daredir
                    rm -rf build-chameleon && mkdir build-chameleon && cd build-chameleon
                    conan install -b missing ..
                    export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:$(pwd)
                    cmake -DCMAKE_CUDA_COMPILER=$(which nvcc) -DCHAMELEON_BACKEND=ON -DDPLASMA_BACKEND=OFF ..
                    make -j
                    '''
            }
        }
        stage ('dare-test-chameleon') {
            steps {
            sh '''#!/bin/bash -el                    
                    module purge
                    module load gcc/10.2.0
                    module load cmake/3.19.2
                    module load cuda/11.4
                    module load openmpi/4.1.0-gcc-10.2.0
                    module load hwloc/2.4.0-gcc-10.2.0
                    module load python-3.9.9-gcc-7.5.0-bp37qr2
                    module load mkl/2020.0.166
                    module list
                    echo $(pwd)
                    ls
                    echo "look into home"
                    ls $HOME
                    export STARPU_ROOT=$HOME/starpu-1.3.9
                    export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:$STARPU_ROOT/build/install/lib/pkgconfig
                    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$STARPU_ROOT/build/install/lib
                    export CHAMELEON_ROOT=$HOME/chameleon-1.1.0
                    export CHAMELEON_TESTING=$CHAMELEON_ROOT/testing
                    export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:$CHAMELEON_ROOT/build/install/lib/pkgconfig
                    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CHAMELEON_ROOT/build/install/lib
                    cd build-chameleon
                    CHAMELEON_USE_CUDA=1 STARPU_SILENT=1 OMP_NUM_THREADS=1 \
                    MKL_NUM_THREADS=1 STARPU_CUDA_PIPELINE=4 STARPU_NWORKER_PER_CUDA=4 \
                    STARPU_SCHED=dmda STARPU_CALIBRATE=1 CUDA_VISIBLE_DEVICES=0 \
                    ./ddarechameleon --threads=5 --n=7090 --ninstr=18694 --nb=720 --ib=180 --gpus=1
                    '''
            }
        }
                stage ('build-dare-dplasma') {
            steps {
            sh '''#!/bin/bash -el
                    module purge
                    module load gcc/10.2.0
                    module load cmake/3.19.2
                    module load cuda/11.4
                    module load openmpi/4.1.0-gcc-10.2.0
                    module load hwloc/2.4.0-gcc-10.2.0
                    module load python-3.9.9-gcc-7.5.0-bp37qr2
                    module load mkl/2020.0.166
                    module list
                    daredir=$(pwd)
                   # starpu
                    cd $HOME
                    rm -rf dplasma
                    git clone --recursive https://github.com/ICLDisco/dplasma.git
                    cd dplasma
                    mkdir build && cd build
                    ../configure --prefix=$(pwd)/install \
                    --with-cuda=$CUDA_HOME -DCMAKE_CUDA_COMPILER=$(which nvcc)
                    make install -j
                    export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:$HOME/dplasma/build/install/lib/pkgconfig
                    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/dplasma/build/install/lib
                    rm -rf ~/.venv/dareenv
                    python -m venv ~/.venv/dareenv
                    source ~/.venv/dareenv/bin/activate
                    pip install conan
                    # dare installation
                    cd $daredir
                    rm -rf build-dplasma && mkdir build-dplasma && cd build-dplasma
                    conan install -b missing ..
                    export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:$(pwd)
                    cmake -DCMAKE_CUDA_COMPILER=$(which nvcc) -DCHAMELEON_BACKEND=OFF -DDPLASMA_BACKEND=ON ..
                    make -j
                    '''
            }
        }
        stage ('dare-test-dplasma') {
            steps {
            sh '''#!/bin/bash -el
                    module purge
                    module load gcc/10.2.0
                    module load cmake/3.19.2
                    module load cuda/11.4
                    module load openmpi/4.1.0-gcc-10.2.0
                    module load hwloc/2.4.0-gcc-10.2.0
                    module load python-3.9.9-gcc-7.5.0-bp37qr2
                    module load mkl/2020.0.166
                    module list
                    echo $(pwd)
                    ls
                    echo "look into home"
                    ls $HOME
                    cd build-dplasma
                    export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:$HOME/dplasma/build/install/lib/pkgconfig
                    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/dplasma/build/install/lib
                    export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:$(pwd)
                    ./ddareparsec -c 20 -g 1 > log.txt
                    '''
            }
        }

    }
}

