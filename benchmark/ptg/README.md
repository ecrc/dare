# PaRSEC PTG doubling benchmark

`ddareptg` runs the FP64 unpivoted-LU doubling recurrence on synthetic
tiled matrices in a single process. The diagonal addition in
`S = I + G H` is a DARE-specific JDF. Each round composes it with
DPLASMA task graphs for GEMM, LU, triangular solves, copies, and the
convergence norm. A taskpool completes before the next one starts.

Build against a DPLASMA installation with FP64 enabled, its matching
PaRSEC installation (including `parsec-ptgpp`), and a CBLAS/LAPACKE provider.
The prototype was tested with DPLASMA `dc82dfac` and PaRSEC `d5212212`.
CUDA runs require both libraries built with CUDA support.

```sh
cmake -S benchmark/ptg -B build-ptg -DCMAKE_PREFIX_PATH=/path/to/install
cmake --build build-ptg
export MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
PARSEC_MCA_device_cuda_enabled=0 ./build-ptg/ddareptg
PARSEC_MCA_device_cuda_enabled=4 PARSEC_MCA_device_show_statistics=1 \
  ./build-ptg/ddareptg -n 257 -b 64 -c 8
PARSEC_MCA_device_cuda_enabled=4 PARSEC_MCA_device_show_statistics=1 \
  ./build-ptg/ddareptg -n 28210 -b 720 -i 2 -c 8 --no-verify
```

The small runs compare each round against a dense LAPACK reference.
The large run reports doubling time per round, including graph submission
and the convergence check. Matrix initialization and reference work are
outside that timer. FITS input, gain recovery, and AO validation belong
to the full solver.
