#!/usr/bin/env python3
"""Compare a packed DARE FP64 matrix against a primary FITS image tile by tile."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "python"))
from dare_tiles import HEADER, MAGIC, tile_regions

spec = importlib.util.spec_from_file_location("fits_reference", HERE / "compare-fits.py")
fits_reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fits_reference)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fits", type=Path)
    parser.add_argument("packed", type=Path)
    parser.add_argument("--rtol", type=float, default=1e-10)
    args = parser.parse_args()
    reference = fits_reference.read_matrix(args.fits).T
    with args.packed.open("rb") as stream:
        magic, rows, cols, block = HEADER.unpack(stream.read(HEADER.size))
    if magic != MAGIC or (rows, cols) != reference.shape or block < 1:
        raise ValueError("incompatible packed matrix header")
    if args.packed.stat().st_size != HEADER.size + 8 * rows * cols:
        raise ValueError("incomplete packed matrix")
    data = np.memmap(str(args.packed), dtype="<f8", mode="r", offset=HEADER.size)

    ref_sq = diff_sq = max_diff = 0.0
    offset = 0
    for row_tiles, col_tiles in tile_regions(rows, cols, block):
        if not row_tiles or not col_tiles:
            continue
        row_start = row_tiles.start * block
        row_stop = min(rows, row_tiles.stop * block)
        height = min(block, row_stop - row_start)
        for j in col_tiles:
            col_start = j * block
            col_stop = min(cols, col_start + block)
            width = col_stop - col_start
            panel = np.ascontiguousarray(reference[row_start:row_stop, col_start:col_stop].T,
                                         dtype=np.float64)
            a = np.ascontiguousarray(panel.reshape(width, len(row_tiles), height).transpose(1, 0, 2))
            a = a.ravel()
            end = offset + a.size
            b = data[offset:end]
            if not (np.isfinite(a).all() and np.isfinite(b).all()):
                raise ValueError("nonfinite matrix element")
            delta = b - a
            ref_sq += np.einsum("i,i->", a, a)
            diff_sq += np.einsum("i,i->", delta, delta)
            max_diff = max(max_diff, float(np.max(np.abs(delta))))
            offset = end

    relative = float(np.sqrt(diff_sq / ref_sq)) if ref_sq else (0.0 if diff_sq == 0 else float("inf"))
    passed = relative <= args.rtol
    print(json.dumps(dict(reference=str(args.fits), candidate=str(args.packed), shape=[rows, cols],
                          relative_frobenius_error=relative, max_absolute_error=max_diff,
                          tolerance=args.rtol, passed=passed)))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
