#!/usr/bin/env python3
"""Compare all FP64 words in two identically shaped packed tile matrices."""
import argparse
import json
from pathlib import Path
import struct

import numpy as np


def compare(reference, candidate, tolerance):
    with reference.open("rb") as a, candidate.open("rb") as b:
        header = a.read(32)
        if b.read(32) != header or len(header) != 32:
            raise ValueError("Packed headers differ or are incomplete")
        magic, m, n, nb = struct.unpack("<8sQQQ", header)
        if magic != b"DARETIL1" or min(m, n, nb) < 1:
            raise ValueError("Invalid packed header")
        size = 32 + m * n * 8
        if reference.stat().st_size != size or candidate.stat().st_size != size:
            raise ValueError("Incomplete packed payload")
        square_reference = square_error = max_error = 0.0
        remaining = m * n
        while remaining:
            count = min(remaining, 4 * 1024 * 1024)
            x = np.fromfile(a, dtype="<f8", count=count)
            y = np.fromfile(b, dtype="<f8", count=count)
            if x.size != count or y.size != count:
                raise ValueError("Short packed read")
            if not np.isfinite(x).all() or not np.isfinite(y).all():
                raise ValueError("Nonfinite matrix entries")
            difference = y - x
            square_reference += float(x @ x)
            square_error += float(difference @ difference)
            max_error = max(max_error, float(np.abs(difference).max()))
            remaining -= count
        relative = np.sqrt(square_error) / max(np.sqrt(square_reference), np.finfo(float).tiny)
        return dict(reference=str(reference), candidate=str(candidate), rows=m, columns=n,
                    relative_frobenius=float(relative), max_abs=max_error,
                    tolerance=tolerance, passed=bool(relative <= tolerance))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--rtol", type=float, default=1e-10)
    args = parser.parse_args()
    if not np.isfinite(args.rtol) or args.rtol < 0:
        parser.error("--rtol must be finite and nonnegative")
    result = compare(args.reference, args.candidate, args.rtol)
    print(json.dumps(result, allow_nan=False))
    raise SystemExit(0 if result["passed"] else 1)
