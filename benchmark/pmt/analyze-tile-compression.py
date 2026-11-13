#!/usr/bin/env python3
"""Measure smoothness and lossless compression of FITS or sampled MPI tiles."""

import argparse
import csv
import ctypes as C
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import re
import sys
from time import perf_counter

import numpy as np


class Codecs:
    def __init__(self):
        zstd = C.CDLL("libzstd.so.1")
        lz4 = C.CDLL("liblz4.so.1")
        self.zstd_bound = zstd.ZSTD_compressBound
        self.zstd_bound.argtypes, self.zstd_bound.restype = [C.c_size_t], C.c_size_t
        self.zstd_encode = zstd.ZSTD_compress
        self.zstd_encode.argtypes = [C.c_void_p, C.c_size_t, C.c_char_p, C.c_size_t, C.c_int]
        self.zstd_encode.restype = C.c_size_t
        self.zstd_decode = zstd.ZSTD_decompress
        self.zstd_decode.argtypes = self.zstd_encode.argtypes[:-1]
        self.zstd_decode.restype = C.c_size_t
        self.lz4_bound = lz4.LZ4_compressBound
        self.lz4_bound.argtypes, self.lz4_bound.restype = [C.c_int], C.c_int
        self.lz4_encode = lz4.LZ4_compress_default
        self.lz4_decode = lz4.LZ4_decompress_safe
        for function in (self.lz4_encode, self.lz4_decode):
            function.argtypes = [C.c_char_p, C.c_void_p, C.c_int, C.c_int]
            function.restype = C.c_int

    def encode(self, raw, name):
        if name == "lz4":
            out = C.create_string_buffer(self.lz4_bound(len(raw)))
            size = self.lz4_encode(raw, out, len(raw), len(out))
        else:
            out = C.create_string_buffer(self.zstd_bound(len(raw)))
            size = self.zstd_encode(out, len(out), raw, len(raw), 1)
        if not 0 < size <= len(out):
            raise ValueError(f"{name} compression failed")
        return out[:size]

    def decode(self, compressed, size, name):
        out = C.create_string_buffer(size)
        if name == "lz4":
            decoded = self.lz4_decode(compressed, out, len(compressed), size)
        else:
            decoded = self.zstd_decode(out, size, compressed, len(compressed))
        if decoded != size:
            raise ValueError(f"{name} decompression failed")
        return out.raw


def transform(raw, rows, method, reverse=False):
    if method == "raw":
        return raw
    if reverse:
        raw = np.frombuffer(raw, dtype=np.uint8).reshape(8, -1).T.tobytes()
    if method == "xor-shuffle":
        columns = np.frombuffer(raw, dtype="<u8").reshape(-1, rows)
        if reverse:
            return np.bitwise_xor.accumulate(columns, axis=1).tobytes()
        delta = columns.copy()
        delta[:, 1:] ^= columns[:, :-1]
        raw = delta.tobytes()
    return raw if reverse else np.frombuffer(raw, dtype=np.uint8).reshape(-1, 8).T.tobytes()


def metrics(name, kind, tile, codecs):
    a = np.asarray(tile, dtype="<f8", order="F")
    if not np.isfinite(a).all():
        raise ValueError(f"{name}: nonfinite tile")
    rms = float(np.sqrt(np.mean(a * a)))
    neighbor = [
        float(np.sqrt(np.mean(np.diff(a, axis=axis) ** 2))) / rms
        if rms and a.shape[axis] > 1 else 0.0
        for axis in (0, 1)
    ]
    fp32_error = float(np.sqrt(np.mean((a - a.astype(np.float32)) ** 2))) / rms if rms else 0.0
    raw = a.tobytes(order="F")
    for codec, method in (("lz4", "raw"), ("lz4", "shuffle"), ("zstd", "raw"),
                          ("zstd", "shuffle"), ("zstd", "xor-shuffle")):
        start = perf_counter()
        compressed = codecs.encode(transform(raw, a.shape[0], method), codec)
        encode_ms = (perf_counter() - start) * 1000
        start = perf_counter()
        decoded = transform(codecs.decode(compressed, len(raw), codec), a.shape[0], method, reverse=True)
        decode_ms = (perf_counter() - start) * 1000
        if decoded != raw:
            raise ValueError(f"{name}: failed bit-exact round trip for {codec}/{method}")
        yield (name, kind, *a.shape, np.count_nonzero(a) / a.size, *neighbor, fp32_error,
               codec, method, len(raw), len(compressed), encode_ms, decode_ms)


def fits_tiles(paths, tile_size):
    spec = spec_from_file_location("compare_fits", Path(__file__).with_name("compare-fits.py"))
    fits = module_from_spec(spec)
    spec.loader.exec_module(fits)
    rng = np.random.default_rng(42)
    for path in paths:
        a = fits.read_matrix(path).T
        row_tiles, col_tiles = ((length + tile_size - 1) // tile_size for length in a.shape)
        diagonal = list(dict.fromkeys(np.linspace(0, min(row_tiles, col_tiles) - 1, 12, dtype=int)))
        off_diagonal = [(r, c) for r in range(row_tiles) for c in range(col_tiles) if r != c]
        chosen = [(i, i) for i in diagonal]
        chosen += [off_diagonal[i] for i in rng.choice(len(off_diagonal), min(12, len(off_diagonal)), replace=False)]
        for row, col in chosen:
            yield (f"{path.name}:{row},{col}", "fits-diagonal" if row == col else "fits-off-diagonal",
                   a[row * tile_size:(row + 1) * tile_size, col * tile_size:(col + 1) * tile_size])


def mpi_tiles(directory):
    files = sorted(directory.glob("*.f64"))
    if not files:
        raise ValueError(f"{directory}: no MPI tile samples")
    for path in files:
        rows, cols = map(int, re.search(r"-(\d+)x(\d+)\.f64$", path.name).groups())
        yield path.name, "mpi", np.fromfile(path, dtype="<f8").reshape((rows, cols), order="F")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--fits", nargs="+", type=Path)
    source.add_argument("--mpi-tiles", type=Path)
    parser.add_argument("--tile-size", type=int, default=720)
    args = parser.parse_args()
    tiles = fits_tiles(args.fits, args.tile_size) if args.fits else mpi_tiles(args.mpi_tiles)
    codecs = Codecs()
    writer = csv.writer(sys.stdout)
    writer.writerow(("tile", "kind", "rows", "cols", "nonzero_fraction", "row_neighbor_rms_ratio",
                     "col_neighbor_rms_ratio", "fp32_relative_rms_error", "codec", "transform",
                     "bytes", "compressed_bytes", "encode_ms", "decode_ms"))
    for name, kind, tile in tiles:
        writer.writerows(metrics(name, kind, tile, codecs))


if __name__ == "__main__":
    main()
