#!/usr/bin/env python3
"""Convert one archived DARE input generation into packed FP64 tiles."""

import argparse
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "python"))
from dare_tiles import HEADER, MAGIC, tile_regions, write_tiles

spec = importlib.util.spec_from_file_location("fits_reference", HERE / "compare-fits.py")
fits_reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fits_reference)


def verify(path, matrix, block):
    rows, columns = matrix.shape
    with path.open("rb") as stream:
        if stream.read(HEADER.size) != HEADER.pack(MAGIC, rows, columns, block):
            raise ValueError(f"{path}: inconsistent header")
    source = matrix.view(np.dtype("u8").newbyteorder(matrix.dtype.byteorder))
    words = np.memmap(str(path), dtype="<u8", mode="r", offset=HEADER.size)
    offset = 0
    for row_tiles, col_tiles in tile_regions(rows, columns, block):
        for j in col_tiles:
            for i in row_tiles:
                tile = source[i * block:(i + 1) * block, j * block:(j + 1) * block].T
                end = offset + tile.size
                if not np.array_equal(words[offset:end].reshape(tile.shape), tile):
                    raise ValueError(f"{path}: different FP64 words at tile ({i}, {j})")
                offset = end
    if offset != words.size:
        raise ValueError(f"{path}: unexpected trailing data")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--states", required=True, type=int)
    parser.add_argument("--measurements", required=True, type=int)
    parser.add_argument("--block", type=int, default=720)
    parser.add_argument("--resume-from", type=Path)
    args = parser.parse_args()
    n, m = args.states, args.measurements
    if args.output.exists():
        parser.error("output generation already exists")
    if min(n, m, args.block) < 1:
        parser.error("dimensions must be positive")
    matrices = (("At", n, n), ("BinvRt", n, m), ("Btinitial", n, m),
                ("Qt", n, n), ("Rt", m, m))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=str(args.output.parent)) as workspace:
        stage = Path(workspace) / "generation"
        stage.mkdir()
        for name, rows, columns in matrices:
            stem = f"{name}{n}"
            options = [f"{stem}.fits", f"{name}.fits"]
            if name in {"At", "BinvRt", "Btinitial"}:
                options.insert(0, f"{stem}_transpose.fits")
            source = next((args.inputs / p for p in options if (args.inputs / p).is_file()), None)
            if source is None:
                raise FileNotFoundError(f"Missing input {stem}")
            matrix = fits_reference.read_matrix(source).T
            if matrix.shape != (rows, columns):
                raise ValueError(f"{source}: expected shape {(rows, columns)}, got {matrix.shape}")
            dest = stage / f"{stem}.tiles"
            start = time.monotonic()
            previous = args.resume_from / dest.name if args.resume_from else None
            if previous is not None and previous.is_file():
                verify(previous, matrix, args.block)
                os.link(previous, dest)
            else:
                write_tiles(dest, matrix, args.block)
                verify(dest, matrix, args.block)
            print(f"{source.name} {rows}x{columns} {dest.stat().st_size} bytes "
                  f"bitwise verified in {time.monotonic() - start:.3f} s", flush=True)
        os.rename(stage, args.output)


if __name__ == "__main__":
    main()
