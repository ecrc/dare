"""Write FP64 matrices in the packed Chameleon tile layout used by DARE."""

import os
from pathlib import Path
import struct
import tempfile

import numpy as np


HEADER = struct.Struct("<8sQQQ")
MAGIC = b"DARETIL1"


def tile_regions(rows, columns, block):
    """Yield each packed region as (row tiles, column tiles)."""
    full_rows, edge_rows = divmod(rows, block)
    full_cols, edge_cols = divmod(columns, block)
    top = range(full_rows)
    bottom = range(full_rows, full_rows + bool(edge_rows))
    left = range(full_cols)
    right = range(full_cols, full_cols + bool(edge_cols))
    return (top, left), (bottom, left), (top, right), (bottom, right)


def write_tile_blocks(path, rows, columns, tile_at, block=720):
    """Write tiles obtained from tile_at(i, j) with zero-based tile coordinates."""
    if min(rows, columns, block) < 1:
        raise ValueError("matrix and tile dimensions must be positive")
    path = Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".",
                                     delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(HEADER.pack(MAGIC, rows, columns, block))
            for row_tiles, col_tiles in tile_regions(rows, columns, block):
                for j in col_tiles:
                    for i in row_tiles:
                        tile = np.asarray(tile_at(i, j))
                        shape = (min(block, rows - i * block), min(block, columns - j * block))
                        if tile.shape != shape or tile.dtype.kind != "f" or tile.dtype.itemsize != 8:
                            raise ValueError(f"tile ({i}, {j}) must be FP64 with shape {shape}")
                        words = tile.view(np.dtype("u8").newbyteorder(tile.dtype.byteorder))
                        if np.any(words):
                            stream.write(np.ascontiguousarray(words.T, dtype="<u8").tobytes())
                        else:
                            stream.seek(words.nbytes, os.SEEK_CUR)
            stream.truncate()
            stream.close()
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()


def write_tiles(path, matrix, block=720):
    """Write a logical (rows, columns) NumPy FP64 array without changing its bits."""
    matrix = np.asarray(matrix)
    if matrix.ndim != 2:
        raise ValueError("expected a two-dimensional FP64 matrix")

    def tile_at(i, j):
        return matrix[i * block:(i + 1) * block, j * block:(j + 1) * block]

    write_tile_blocks(path, *matrix.shape, tile_at, block)
