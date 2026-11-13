#!/usr/bin/env python3
"""Measure Chameleon's sequential FP64 diagonal LU kernel inside an allocation."""
import ctypes
import os
import statistics
import sys
import time

import numpy as np

if not os.environ.get("SLURM_JOB_ID"):
    raise SystemExit("Run this diagnostic in a Slurm compute allocation.")
library = ctypes.CDLL(sys.argv[1])
lu = library.CORE_dgetrf_nopiv
lu.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
               ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
lu.restype = ctypes.c_int
random = np.random.RandomState(7)
print("tile,inner,median_seconds", flush=True)
for tile in (720, 960, 1440):
    matrix = np.asfortranarray(random.normal(size=(tile, tile)))
    matrix[np.diag_indices(tile)] += tile
    for inner in (16, 32, 64, 96, 180):
        durations = []
        for repeat in range(6):
            work = matrix.copy(order="F")
            info = ctypes.c_int()
            start = time.perf_counter()
            status = lu(tile, tile, inner, work.ctypes.data, tile, ctypes.byref(info))
            elapsed = time.perf_counter() - start
            if status or info.value:
                raise RuntimeError(f"LU failed: status={status}, info={info.value}")
            if repeat:
                durations.append(elapsed)
        print(f"{tile},{inner},{statistics.median(durations):.9f}", flush=True)
