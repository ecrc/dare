# DARE Solver

# 1. Introduction
High Performance Discrete time Algebraic Riccati Equation [(DARE)](https://en.wikipedia.org/wiki/Algebraic_Riccati_equation) Solver Library. The DARE solver is used in Adaptive Optics system of extreme large ground-based Telescope such as MAVIS. It has significant better application peformance than other traditional methods. However the computation cost is also much higher. According to our benchmark of MAVIS system, the application has best accuracy when the matrices dimension is more than 20k x 20k. Our High Performance DARE solver exploit advanced linear algebra runtime system such as [Chameleon](https://solverstack.gitlabpages.inria.fr/chameleon/) or [DPLASMA](https://github.com/icldisco/dplasma) to deploy the algorithm onto multiple GPUs/CPUs at a shared memory system.

# 2. Dependencies
  * CMake (>=3.19)
  * CUDA (>=11.0)
  * Intel MKL (>=2018)
  * OpenMPI (>=4.0)
  * Chameleon or DPLASMA runtime system


# 3. Installation

We provide two [install scripts](https://zenodo.org/record/7309032) for user to download dependencies and compile
the dare solver library. 


# 4. Dataset
Input files available at:
https://kaust-my.sharepoint.com/:f:/g/personal/ltaiefh_kaust_edu_sa/EgaqEKCdWURBgDq3x2CNsXQBteQs4x0VIdiiHJcY8AVyVA?e=EzETNF


# 5. Running
The usage is provided by `./ddare --help`:
```Usage:
./ddare [options]

Options are:
  --help           Show this help

  --n=X            #samples x #layers. states    (default: 500)
  --ninstr=X       Instrument dimension. measurements (default: 500)
  --nb=X           tile size.            (default: 128)

  --datapath=X     path to the data folder
                   this folder must contain the following files:
                       At.fits
                       BinvRt.fits
                       Btinitial.fits
                       Rt.fits
                       Qt.fits
                   the parameters '--n' and '--ninstr' will be overwritten by the
                   corresponding dimensions of the files

  --threads=X      Number of CPU workers (default: _SC_NPROCESSORS_ONLN)
  --gpus=X         Number of GPU workers (default: 0)
  --sync           Synchronize the execution of all calls (default: async)
  --nooptalgo      Leverage matrix structure in the algorithm (default: optalgo)
  --profile        Profile the execution of all calls (default: no profile)
  --check          Check numerical correctness (default: no check)
  ```

The StarPU runtime also needs to be set up.
Example:
```
STARPU_SILENT=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 STARPU_CUDA_PIPELINE=4 STARPU_NWORKER_PER_CUDA=4 STARPU_SCHED=prio STARPU_CALIBRATE=1 ./ddare --threads=31 --n=7090 --ninstr=19078 --nb=320 --ib=80 --gpus=1
```
where `n` and `ninstr` are the dimensions of the problem. In this case, the matrices will be generated randomly
Or using input matrices (as fits files):
```
STARPU_SILENT=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 STARPU_CUDA_PIPELINE=2 STARPU_NWORKER_PER_CUDA=12 STARPU_SCHED=$j STARPU_CALIBRATE=1 ./ddare --threads=4 --nb=720 --ib=180 --gpus=1 --datapath=<path/to/data>
```
Question
========

For more information and questions please send email to yuxi.hong@kaust.edu.sa and hatem.ltaief@kaust.edu.sa.

Handout
========
![alt text](https://github.com/ecrc/dare/blob/main/misc/dare-handout-final.pdf)
