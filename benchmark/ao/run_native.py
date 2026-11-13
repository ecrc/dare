"""Measure AO science metrics using Jesse's unmodified nextLQG frame ordering.

The GPU option changes only dense MVM execution. This is an optical simulation,
not a real-time latency benchmark. An archive model mismatch must be opted into.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import cupy as cp


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ata', type=Path, required=True)
    p.add_argument('--params', type=Path, required=True)
    p.add_argument('--controller', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--frames', type=int, default=1000)
    p.add_argument('--burn-in', type=int, default=200)
    p.add_argument('--seed', type=int, default=1100)
    p.add_argument('--backend', choices=('cpu', 'gpu'), default='gpu')
    p.add_argument('--open-loop', action='store_true')
    p.add_argument('--allow-model-mismatch', action='store_true')
    p.add_argument('--rtc-delay', type=float, default=0., help='ATA already includes delay explicitly; use 0')
    p.add_argument('--max-command', type=float, default=100., help='Abort threshold in command units, not a DM stroke specification')
    args = p.parse_args()
    if not 0 <= args.burn_in < args.frames:
        raise ValueError('Require 0 <= burn-in < frames')
    manifest = json.loads((args.controller/'controller.json').read_text())
    if not manifest['model_check']['sampled_model_match'] and not args.allow_model_mismatch:
        raise ValueError('Model mismatch: use --allow-model-mismatch for diagnostic runs only')
    sys.path.insert(0, str(args.ata.resolve()))
    from ataSupervisor import ATASupervisor
    from shesha.config import ParamConfig
    import shesha.init.atmos_init as atmos_module
    if args.backend == 'gpu':
        import cupy as cp

    class NativeGPU(ATASupervisor):
        def doLQGControl(self, nControl, **kwargs):
            if self.x is None:
                self.dx = cp.zeros(self.A.shape[0], dtype=self.A.dtype)
                self.du = cp.zeros(self.K.shape[0], dtype=self.K.dtype)
                self.dum1 = self.du.copy()
            self.dx = self.A @ self.dx + self.L @ cp.asarray(self.s) - self.G @ self.dum1
            self.dum1 = self.du
            self.du = self.K @ self.dx
            # Mirror upstream state variables for its monitoring code.
            self.x = cp.asnumpy(self.dx)
            self.um2 = self.um1
            self.um1 = self.u
            self.u = cp.asnumpy(self.du)
            self.rtc.set_command(0, self.u.astype(np.float32))

    class OpenLoop(ATASupervisor):
        def doLQGControl(self, nControl, **kwargs):
            self.x = np.zeros(self.A.shape[0], dtype=np.float32)
            self.u = np.zeros(self.K.shape[0], dtype=np.float32)
            self.um1 = self.u
            self.rtc.set_command(0, self.u)

    config = ParamConfig(str(args.params.resolve()))
    config.p_controllers[0].set_type('generic')
    config.p_controllers[0].set_delay(args.rtc_delay)
    config.p_loop.set_devices([0])
    config.p_atmos.set_seeds(np.arange(config.p_atmos.nscreens, dtype=np.int64)+args.seed)
    cls = OpenLoop if args.open_loop else NativeGPU if args.backend == 'gpu' else ATASupervisor
    sup = cls(config)
    sup.reset()
    sup.iter = 0
    matrices = {k: np.load(args.controller / (k+'.npy'), mmap_mode='r')
                for k in ('A', 'L', 'G', 'K')}
    n, m, nact = manifest['states'], manifest['measurements'], manifest['actuators']
    if len(sup.rtc.get_slopes(0)) != m or len(sup.rtc.get_command(0)) != nact:
        raise ValueError('COMPASS geometry does not match controller')
    if args.backend == 'gpu' and not args.open_loop:
        class ConstantMatrix:
            # Upstream copies its inputs. These private device copies are never
            # mutated, so avoid a second full set occupying the GPU during loop.
            def __init__(self, host):
                self.device = cp.asarray(host)
            def copy(self):
                return self.device
        matrices = {k: ConstantMatrix(v) for k, v in matrices.items()}
    # loopLQG checks this argument's presence, but uses it only for compute_pols.
    # Filtering is already included in L and G; no huge identity is allocated.
    matrices['tt_filter'] = np.empty((0, 0))
    targets = len(config.p_targets)
    wavelength_nm = np.array([t.Lambda for t in config.p_targets])*1000
    native_next = sup.nextLQG
    command_max = 0.
    clipped_frames = 0
    def measured_next(**kwargs):
        nonlocal command_max, clipped_frames
        if sup.iter < 3:
            print('Starting native frame', sup.iter, flush=True)
        if sup.iter == args.burn_in:
            for j in range(targets):
                sup.target.reset_strehl(j)
        native_next(**kwargs)
        if not np.isfinite(sup.u).all() or not np.isfinite(sup.x).all():
            raise ValueError('Nonfinite controller state/command')
        if np.max(np.abs(sup.u)) > args.max_command:
            raise ValueError(f'Unstable command at frame {sup.iter}: {np.max(np.abs(sup.u))}')
        if sup.iter > args.burn_in:
            command_max = max(command_max, float(np.max(np.abs(sup.u))))
            clipped_frames += int(not np.allclose(sup.rtc.get_command(0), sup.u,
                                                  rtol=1e-6, atol=1e-7))
    sup.nextLQG = measured_next
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    try:
        series = sup.loopLQG(args.frames, monitoring_freq=100, compute_pols=False,
                             abortThresh=0., **matrices)
    except (ValueError, RuntimeError) as error:
        failure = dict(status='failed', frame=sup.iter, reason=str(error),
                       states=n, seed_base=args.seed, rtc_delay_frames=args.rtc_delay)
        (args.out/'failure.json').write_text(json.dumps(failure, indent=2)+'\n')
        raise
    if series.shape != (args.frames, targets, 4) or not np.isfinite(series).all():
        raise ValueError('Invalid metric series')
    settled = series[args.burn_in:]
    wfe = np.sqrt(np.mean(settled[:,:,2] * (wavelength_nm/(2*np.pi))**2, axis=0))
    np.save(args.out/'strehl.npy', series)
    result = dict(states=n, measurements=m, actuators=nact, targets=targets,
                  status='diagnostic-model-mismatch' if not manifest['model_check']['sampled_model_match'] else 'ok',
                  frames=args.frames, burn_in=args.burn_in, seed_base=args.seed,
                  wfs_noise_seed_base=1234,
                  simulated_seconds=args.frames*float(config.p_loop.ittime),
                  backend=args.backend, dare_dtype=manifest['dare_dtype'],
                  rtc_delay_frames=args.rtc_delay, max_command_threshold=args.max_command,
                  rtc_dtype=manifest['rtc_dtype'], open_loop=args.open_loop,
                  rtc='Jesse ATA nextLQG; original two-frame slope/command ordering',
                  model_status=manifest['model_check']['status'],
                  field_mean_le_strehl=float(settled[-1,:,1].mean()),
                  on_axis_le_strehl=float(settled[-1,0,1]),
                  field_mean_se_strehl=float(settled[:,:,0].mean()),
                  field_wfe_rms_nm=float(np.sqrt(np.mean(wfe*wfe))),
                  per_target_le_strehl=settled[-1,:,1].tolist(),
                  per_target_wfe_rms_nm=wfe.tolist(), max_command_abs=command_max,
                  clipped_frames=clipped_frames, wall_seconds=time.monotonic()-start,
                  supervisor_sha256=hashlib.sha256((args.ata/'ataSupervisor.py').read_bytes()).hexdigest(),
                  atmos_init_sha256=hashlib.sha256(Path(atmos_module.__file__).read_bytes()).hexdigest(),
                  config_sha256=hashlib.sha256(args.params.read_bytes()).hexdigest())
    (args.out/'metrics.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
