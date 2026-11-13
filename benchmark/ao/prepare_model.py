"""Reconstruct an archived R=1 controller's optical maps with Jesse's ATA.

The archive predates DARE_opt: preserve its calibrated geometry and 5x5
projection directions explicitly. Model mismatch is reported, never hidden.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

import numpy as np
from astropy.io import fits
from scipy import stats


def meta_coords(D, h, rad, res, mask="circle"):
    # Same ordering/mask as ATA.getMetaCoords without its dense selection matrix.
    extent = D / 2 + h * np.tan(rad)
    x, y = np.meshgrid(np.linspace(-extent, extent, res),
                       np.linspace(-extent, extent, res))
    x, y = x.T.ravel(), y.T.ravel()
    if isinstance(mask, str):
        if mask != "circle":
            raise ValueError(mask)
        a, b = np.meshgrid(np.linspace(-1, 1, res), np.linspace(-1, 1, res))
        valid = (np.sqrt(a*a + b*b) <= 1).T.ravel()
    else:
        ix = np.around((x + D/2) * (mask.shape[0]-1) / D).astype(np.int64)
        iy = np.around((y + D/2) * (mask.shape[0]-1) / D).astype(np.int64)
        valid = mask[ix, iy] == 1
    return [y[valid], x[valid]]


def filtered(matrix, enabled):
    if enabled:
        for part in np.split(matrix, 2, axis=0):
            part -= part.mean(axis=0, keepdims=True)
    return matrix


def relative(a, b):
    return float(np.linalg.norm(a-b) / np.linalg.norm(b))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ata', type=Path, required=True)
    p.add_argument('--params', type=Path, required=True)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--states', type=int, required=True)
    p.add_argument('--resolution', type=int, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    for name in ('ata', 'params', 'inputs', 'out'):
        setattr(args, name, getattr(args, name).resolve())
    args.out.mkdir(parents=True, exist_ok=True)
    # argparsing reads params_<simname>.pkl from cwd; do not modify the checkout.
    work = args.out / 'ata-work'
    work.mkdir(exist_ok=True)
    link = work / 'params_mavis_archive.pkl'
    if not link.exists():
        link.symlink_to(args.params)
    os.chdir(work)
    sys.path.insert(0, str(args.ata))
    sys.argv = ['step_3.py', 'mavis_archive', '--metares', str(args.resolution),
                '--sigwngs', '0.1', '--sigwlgs', '20', '--projreg', '1e-7',
                '--numtargets', '25', '--fovrad', '15']
    code = runpy.run_path(str(args.ata / 'step_3.py'))
    ao = code['ao']
    assert ao.R == 1
    # Archived generator uses wind speed as stored; DARE_opt divides it by
    # cos(zenith). Undo that additional scaling for these existing inputs.
    ao.vy = np.asarray(ao.vy) * np.cos(ao.zenithangle)
    ao.vx = np.asarray(ao.vx) * np.cos(ao.zenithangle)
    dx, dy = np.meshgrid(np.linspace(-15, 15, 5), np.linspace(-15, 15, 5))
    ao.dirs = np.array([dx.ravel()*4.848e-6, dy.ravel()*4.848e-6,
                       np.ones(dx.size)]).T
    state = fits.getdata(args.inputs / 'At.fits', memmap=True)
    observation = fits.getdata(args.inputs / 'Btinitial.fits', memmap=True)
    dm = [[ao.K_y[i], ao.K_x[i]] for i in range(ao.N)]
    pitch = [float(stats.mode(d[d > 0], keepdims=True).mode[0])
             for _, x in dm for d in [np.abs(x - np.roll(x, 1))]]
    meta = [meta_coords(ao.D, h, ao.metafov, n)
            for h, n in zip(ao.h, ao.n_meta)]
    proj = [meta_coords(ao.D, 0, 0, ao.proj_res, ao.pupil_mask)]
    states = np.cumsum([0] + [len(x[0]) for x in meta])
    slopes = np.cumsum([0] + [x.shape[0] for x in ao.C_smls])
    if state.shape != (args.states, args.states) or states[-1] != args.states:
        raise ValueError('State size/grid/archive mismatch')
    if observation.shape != (slopes[-1], states[-1]):
        raise ValueError('Observation size/orientation mismatch')
    gy, gx = dm[0]
    gauss, cov = code['gaussMap'], code['covMap']
    interp = gauss(gy, gx, gy, gx, pitch[0], ao.infl[0]).get()
    blocks, checks = [], []
    # Test both first and last layer for EVERY WFS, including weak NGS blocks.
    audit = {}
    for j in (0, len(meta)-1):
        y, x = meta[j]
        ci = cov(y, x, y, x, ao.r0[j], ao.L0[j]).get()
        shifted = cov(y + ao.ittime*ao.vy[j], x + ao.ittime*ao.vx[j],
                      y, x, ao.r0[j], ao.L0[j]).get()
        trans = (1-ao.boil[0]) * np.linalg.solve(ci, shifted.T).T
        audit[j] = (ci, trans)
    for i, geometry in enumerate(ao.C_smls):
        c = np.linalg.solve(interp.T, geometry.T).T
        parts = []
        for j in range(ao.N):
            factor = 1-ao.h_dm[j]/ao.h_gs[i]
            parts.append(c @ gauss(gy*factor+ao.h_dm[j]*ao.yc[i],
                                   gx*factor+ao.h_dm[j]*ao.xc[i],
                                   dm[j][0], dm[j][1], pitch[j], ao.infl[j]).get())
        blocks.append(filtered(np.concatenate(parts, axis=1), ao.filter_TT[i]))
        slope_transform = None
        for j, (ci, trans) in audit.items():
            y, x = meta[j]
            factor, h = 1-ao.h[j]/ao.h_gs[i], ao.h[j]
            wave = cov(gy*factor+h*ao.yc[i], gx*factor+h*ao.xc[i],
                       y, x, ao.r0[j], ao.L0[j]).get()
            observed = filtered(c @ np.linalg.solve(ci, wave.T).T,
                                ao.filter_TT[i]) @ trans
            s, t = slice(states[j], states[j+1]), slice(slopes[i], slopes[i+1])
            row = dict(wfs=i, layer=j,
                transition_relative_error=relative(trans, state[s, s]),
                observation_relative_error=relative(observed, observation[t, s]),
                observation_reference_norm=float(np.linalg.norm(observation[t, s])),
                observation_difference_norm=float(np.linalg.norm(observed-observation[t, s])))
            if not ao.filter_TT[i]:
                if slope_transform is None:
                    slope_transform = np.linalg.lstsq(observed.T, observation[t, s].T, rcond=None)[0].T
                row['diagnostic_slope_transform'] = slope_transform.tolist()
                row['transformed_observation_error'] = relative(slope_transform @ observed, observation[t, s])
                # Diagnostic only: do NOT alter the archived model or slopes.
            checks.append(row)
        print('Prepared WFS', i, checks[-2:], flush=True)
    # Preserve full precision for G = Pf L C2; RTC casts are explicit later.
    np.save(args.out / 'interaction.npy', np.concatenate(blocks, axis=0))
    del blocks
    np.savez(args.out / 'layout.npz', states=states, slopes=slopes,
             filter_tt=ao.filter_TT)
    matched = all(c['transition_relative_error'] < 1e-5 and
                  c['observation_relative_error'] < 2e-3 for c in checks)
    manifest = dict(states=int(states[-1]), measurements=int(slopes[-1]),
                    resolution=args.resolution, sampled_model_match=matched,
                    status='sampled-match' if matched else 'reconstructed-model-mismatch',
                    checks=checks, projection='5x5 square, +/-15 arcsec',
                    ata_step3_sha256=hashlib.sha256((args.ata/'step_3.py').read_bytes()).hexdigest(),
                    params_sha256=hashlib.sha256(args.params.read_bytes()).hexdigest())
    (args.out / 'model-check.json').write_text(json.dumps(manifest, indent=2)+'\n')
    projector = code['kMCAO'](ao.dirs, proj, meta, dm, ao.h, ao.h_dm,
                              ao.r0, ao.L0, pitch, ao.infl, ao.proj_reg)
    for lo, hi in zip(states[:-1], states[1:]):
        projector[:, lo:hi] = projector[:, lo:hi] @ np.asarray(state[lo:hi, lo:hi], dtype=np.float64)
    projector *= -1/(1-ao.boil[0])
    np.save(args.out / 'command-projector.npy', projector)
    print('MODEL_READY', manifest['status'], flush=True)


if __name__ == '__main__':
    main()
