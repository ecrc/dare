#!/usr/bin/env python3
"""Compare uncompressed primary-HDU FP32 or FP64 FITS matrices in chunks."""

import argparse
import json
from pathlib import Path

import numpy as np


def read_matrix(path):
    header = {}
    with path.open("rb") as stream:
        finished = False
        while not finished:
            block = stream.read(2880)
            if len(block) != 2880:
                raise ValueError(f"{path}: truncated FITS header")
            for start in range(0, 2880, 80):
                card = block[start:start + 80].decode("ascii")
                key = card[:8].strip()
                if key == "END":
                    finished = True
                    break
                if card[8:10] == "= ":
                    header[key] = card[10:].split("/")[0].strip()
        offset = stream.tell()
    dtype = {-32: ">f4", -64: ">f8"}.get(int(header["BITPIX"]))
    if dtype is None or int(header["NAXIS"]) != 2:
        raise ValueError(f"{path}: expected a two-dimensional FP32 or FP64 FITS image")
    if float(header.get("BSCALE", "1")) != 1 or float(header.get("BZERO", "0")) != 0:
        raise ValueError(f"{path}: scaled FITS images are not supported")
    shape = (int(header["NAXIS2"]), int(header["NAXIS1"]))
    return np.memmap(str(path), dtype=dtype, mode="r", offset=offset, shape=shape)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--rtol", type=float, default=1e-10)
    args = parser.parse_args()
    reference, candidate = read_matrix(args.reference), read_matrix(args.candidate)
    if reference.shape != candidate.shape:
        raise ValueError("matrix shapes differ")
    ref_sq = diff_sq = maximum = 0.0
    for start in range(0, len(reference), 128):
        a = np.asarray(reference[start:start + 128], dtype=np.float64)
        b = np.asarray(candidate[start:start + 128], dtype=np.float64)
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            raise ValueError("nonfinite matrix element")
        difference = b - a
        ref_sq += np.einsum("ij,ij->", a, a)
        diff_sq += np.einsum("ij,ij->", difference, difference)
        maximum = max(maximum, float(np.max(np.abs(difference))))
    relative = float(np.sqrt(diff_sq / ref_sq)) if ref_sq else (0.0 if diff_sq == 0 else float("inf"))
    print(json.dumps(dict(reference=str(args.reference), candidate=str(args.candidate),
                          shape=reference.shape, relative_frobenius_error=relative,
                          reference_frobenius_norm=float(np.sqrt(ref_sq)),
                          max_absolute_error=maximum, tolerance=args.rtol,
                          passed=relative <= args.rtol)))
    raise SystemExit(0 if relative <= args.rtol else 1)


if __name__ == "__main__":
    main()
