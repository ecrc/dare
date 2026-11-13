#!/usr/bin/env python3
"""Evaluate the DARE equation on the FP64 P and K exported by the solver."""

import argparse
import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("prefixes", nargs="+", type=Path)
    args = parser.parse_args()

    spec = spec_from_file_location("compare_fits", Path(__file__).with_name("compare-fits.py"))
    fits = module_from_spec(spec)
    spec.loader.exec_module(fits)

    # Chameleon reads and writes the FITS pixel stream as a column-major matrix.
    a = np.asarray(fits.read_matrix(args.dataset / "At.fits").T, dtype=np.float64)
    b = np.asarray(fits.read_matrix(args.dataset / "Btinitial.fits").T, dtype=np.float64)
    q = np.asarray(fits.read_matrix(args.dataset / "Qt.fits").T, dtype=np.float64)
    for prefix in args.prefixes:
        p = np.asarray(fits.read_matrix(prefix.with_name(prefix.name + "-P.fits")).T,
                       dtype=np.float64)
        k = np.asarray(fits.read_matrix(prefix.with_name(prefix.name + "-K.fits")).T,
                       dtype=np.float64)
        residual = a.T @ (p @ (a - b @ k))
        residual += q
        residual -= p
        relative = np.linalg.norm(residual) / np.linalg.norm(p)
        if not np.isfinite(relative):
            raise ValueError(f"{prefix}: nonfinite Riccati residual")
        print(json.dumps(dict(prefix=str(prefix), relative_riccati_residual=relative)), flush=True)


if __name__ == "__main__":
    main()
