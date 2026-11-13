"""Memory-bounded projected FP64 DARE residual; this is not an AO quality test."""
import argparse
import json
from pathlib import Path

import numpy as np
from astropy.io import fits


def multiply(matrix, vectors, transpose=False, rows=256):
    if transpose:
        out = np.zeros((matrix.shape[1], vectors.shape[1]), dtype=np.float64)
        for lo in range(0, matrix.shape[0], rows):
            hi = min(lo+rows, matrix.shape[0])
            out += np.asarray(matrix[lo:hi], dtype=np.float64).T @ vectors[lo:hi]
    else:
        out = np.empty((matrix.shape[0], vectors.shape[1]), dtype=np.float64)
        for lo in range(0, matrix.shape[0], rows):
            hi = min(lo+rows, matrix.shape[0])
            out[lo:hi] = np.asarray(matrix[lo:hi], dtype=np.float64) @ vectors
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--solution', type=Path, required=True, help='Directory containing result-P/K.fits')
    p.add_argument('--rtol', type=float, default=1e-10)
    args = p.parse_args()
    a, c, q = (fits.getdata(args.inputs/name, memmap=True)
               for name in ('At.fits', 'Btinitial.fits', 'Qt.fits'))
    cov, gain = (fits.getdata(args.solution/name, memmap=True)
                 for name in ('result-P.fits', 'result-K.fits'))
    n = a.shape[0]
    v = np.random.RandomState(28210).choice([-1., 1.], (n, 16))
    # Stored FITS arrays are A and L of the estimation convention; the solver's
    # LQR convention uses A.T, C.T, L.T. P is symmetric (transpose kept explicit).
    av = multiply(a, v, True)
    kv = multiply(gain, v, True)
    inner = av - multiply(c, kv, True)
    pv = multiply(cov, v, True)
    residual = multiply(a, multiply(cov, inner, True)) + multiply(q, v, True) - pv
    value = float(np.linalg.norm(residual)/np.linalg.norm(pv))
    result = dict(states=n, probes=16, relative_projected_residual=value,
                  tolerance=args.rtol, passed=bool(np.isfinite(value) and value <= args.rtol))
    (args.solution/'residual.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)
    raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()
