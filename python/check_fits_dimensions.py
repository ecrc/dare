#!/usr/bin/env python3
"""Print primary-image FITS dimensions without loading the image or using NumPy."""

import argparse
from pathlib import Path
import sys


def read_dimensions(path):
    """Read the primary HDU header and return FITS axes and BITPIX."""
    values = {}
    with path.open("rb") as stream:
        while True:
            block = stream.read(2880)
            if len(block) != 2880:
                raise ValueError("incomplete FITS header: END card was not found")
            for offset in range(0, 2880, 80):
                card = block[offset:offset + 80].decode("ascii")
                key = card[:8].strip()
                if not values and offset == 0 and key != "SIMPLE":
                    raise ValueError("not a FITS primary HDU")
                if key == "END":
                    if values.get("SIMPLE") != "T":
                        raise ValueError("SIMPLE is not true")
                    axis_count = int(values["NAXIS"])
                    if not 0 <= axis_count <= 999:
                        raise ValueError("invalid NAXIS")
                    axes = tuple(int(values[f"NAXIS{i}"]) for i in range(1, axis_count + 1))
                    if any(axis < 0 for axis in axes):
                        raise ValueError("negative axis length")
                    return axes, int(values["BITPIX"])
                if card[8:10] == "= ":
                    values[key] = card[10:].split("/", 1)[0].strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", type=Path, help="FITS files; defaults to *.fits here")
    args = parser.parse_args()
    files = args.files or sorted(Path.cwd().glob("*.fits"))
    if not files:
        parser.error("no FITS files found; provide their paths")
    failed = False
    for path in files:
        try:
            axes, bitpix = read_dimensions(path)
            print(f"{path}:")
            print(f"  FITS axes (NAXIS1 first): {axes}")
            print(f"  NumPy shape (reversed axes): {axes[::-1]}")
            print(f"  BITPIX: {bitpix}")
        except (OSError, ValueError, KeyError) as error:
            print(f"{path}: {error}", file=sys.stderr)
            failed = True
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
