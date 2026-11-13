"""Compare one CPU/GPU ATA state update against an FP64 reference."""
import argparse
import json
from pathlib import Path
import cupy as cp
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('controller', type=Path)
    p.add_argument('output', type=Path)
    args = p.parse_args()
    a,l,g,k = (np.load(args.controller/(name+'.npy'),mmap_mode='r') for name in 'ALGK')
    rng=np.random.RandomState(311)
    x,s,u=(rng.normal(size=n).astype(np.float32) for n in (a.shape[1],l.shape[1],g.shape[1]))
    reference = np.asarray(a,dtype=np.float64)@x + np.asarray(l,dtype=np.float64)@s - np.asarray(g,dtype=np.float64)@u
    reference_command = np.asarray(k,dtype=np.float64)@reference
    cpu=a@x+l@s-g@u
    cpu_command=k@cpu
    gpu=cp.asarray(a)@cp.asarray(x)+cp.asarray(l)@cp.asarray(s)-cp.asarray(g)@cp.asarray(u)
    gpu_command=cp.asnumpy(cp.asarray(k)@gpu)
    gpu=cp.asnumpy(gpu)
    def rel(x,y): return float(np.linalg.norm(x-y)/np.linalg.norm(y))
    result=dict(cpu_state_vs_fp64=rel(cpu,reference),gpu_state_vs_fp64=rel(gpu,reference),
                cpu_command_vs_fp64=rel(cpu_command,reference_command),
                gpu_command_vs_fp64=rel(gpu_command,reference_command),
                gpu_vs_cpu_state=rel(gpu,cpu),gpu_vs_cpu_command=rel(gpu_command,cpu_command))
    result['passed']=all(np.isfinite(v) and v<1e-5 for v in result.values())
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
    raise SystemExit(0 if result['passed'] else 1)


if __name__=='__main__': main()
