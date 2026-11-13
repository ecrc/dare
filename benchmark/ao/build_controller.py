"""Assemble ATA's A/L/G/K from an FP64 DARE gain (single-rate archive)."""
import argparse
import json
from pathlib import Path
import time

import cupy as cp
import numpy as np
from astropy.io import fits


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--gain', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--rtc-dtype', choices=('float32', 'float64'), default='float32')
    args = p.parse_args()
    state = fits.getdata(args.inputs / 'At.fits', memmap=True)
    obs = fits.getdata(args.inputs / 'Btinitial.fits', memmap=True)
    gain = fits.getdata(args.gain, memmap=True)
    n, m = state.shape[0], obs.shape[0]
    if state.shape != (n, n) or obs.shape != (m, n) or gain.shape != (n, m):
        raise ValueError('Incorrect dimensions or gain transpose')
    if gain.dtype.itemsize != 8:
        raise ValueError('This workflow requires an FP64 DARE gain')
    with np.load(args.model / 'layout.npz') as f:
        layers, slopes, tt = (f[k] for k in ('states', 'slopes', 'filter_tt'))
    interaction = np.load(args.model / 'interaction.npy', mmap_mode='r')
    projector = np.load(args.model / 'command-projector.npy', mmap_mode='r')
    nact = projector.shape[0]
    if layers[-1] != n or slopes[-1] != m or interaction.shape != (m, nact):
        raise ValueError('Model layout mismatch')
    args.out.mkdir(parents=True, exist_ok=True)
    arrays = {k: np.lib.format.open_memmap(args.out / (k+'.npy.part'),
                                         mode='w+', dtype=args.rtc_dtype, shape=shape)
              for k, shape in dict(A=(n,n), L=(n,m), G=(n,nact), K=(nact,n)).items()}
    start = time.monotonic()
    for lo, hi in zip(layers[:-1], layers[1:]):
        rows = slice(lo, hi)
        raw = cp.asarray(np.asarray(gain[rows], dtype=np.float64))
        if not bool(cp.isfinite(raw).all()):
            raise ValueError('Nonfinite gain')
        for col in range(0, n, 1024):
            cols = slice(col, min(col+1024, n))
            block = -raw @ cp.asarray(np.asarray(obs[:, cols], dtype=np.float64))
            block += cp.asarray(np.asarray(state[rows, cols], dtype=np.float64))
            block -= block.mean(axis=0, keepdims=True)
            arrays['A'][rows, cols] = cp.asnumpy(block)
        # G = Pf L_raw C2. Apply Pf to gain before G and C_filter.
        raw -= raw.mean(axis=0, keepdims=True)
        for col in range(0, nact, 512):
            cols = slice(col, min(col+512, nact))
            block = raw @ cp.asarray(np.asarray(interaction[:, cols], dtype=np.float64))
            arrays['G'][rows, cols] = cp.asnumpy(block)
        for i, enabled in enumerate(tt):
            if enabled:
                left, right = slopes[i:i+2]
                mid = (left+right)//2
                for cols in (slice(left, mid), slice(mid, right)):
                    raw[:, cols] -= raw[:, cols].mean(axis=1, keepdims=True)
        arrays['L'][rows] = cp.asnumpy(raw)
        arrays['K'][:, rows] = projector[:, rows]
        del raw, block
        cp.get_default_memory_pool().free_all_blocks()
        print('Assembled through state', int(hi), 'seconds', time.monotonic()-start, flush=True)
    for name, arr in arrays.items():
        arr.flush()
        (args.out / (name+'.npy.part')).replace(args.out / (name+'.npy'))
    manifest = dict(states=n, measurements=m, actuators=nact, dare_dtype='float64',
                    assembly_dtype='float64', rtc_dtype=args.rtc_dtype,
                    gain=str(args.gain.resolve()), model=str(args.model.resolve()),
                    assembly_seconds=time.monotonic()-start,
                    model_check=json.loads((args.model/'model-check.json').read_text()))
    (args.out / 'controller.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    main()
