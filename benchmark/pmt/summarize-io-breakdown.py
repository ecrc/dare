#!/usr/bin/env python3
"""Account for process wall time using non-overlapping main checkpoints."""
import argparse,csv,json,re
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('results',type=Path);a=p.parse_args()
rows=[]
for f in sorted(a.results.glob('run-*.log')):
 text=f.read_text()
 phases={name:float(value) for name,value in re.findall(r'WALL_PHASE (\w+) ([\d.]+)$',text,re.M)}
 if len(phases)!=10: raise ValueError(f'{f}: incomplete phases {phases}')
 wall=float(re.search(r'^Wall seconds: ([\d.]+)$',text,re.M)[1])
 main=float(re.search(r'^WALL_MAIN ([\d.]+)$',text,re.M)[1])
 if abs(sum(phases.values())-main)>0.001: raise ValueError('Phase sum mismatch')
 if wall-main < -0.011: raise ValueError('Main longer than process')
 row={'case':f.stem,**phases,'outside_main':wall-sum(phases.values()),'wall':wall,'main':main,
      'reported_solver':float(re.search(r'^Time:\s+([\d.]+)',text,re.M)[1])}
 row['io']=row['input']+row['output']
 row['other']=wall-row['solver']-row['io']
 rows.append(row)
if not rows: raise ValueError('No complete runs')
result={'job':a.results.name,'states':28210,'hardware':'4 A100 SXM4 80GB','precision':'FP64','tile':720,'readers_per_file':2,'writers_per_file':4,
        'notes':['Five concurrent input files; P/K output concurrent','No cache drop; first and repeated processes, not guaranteed cold disk','Output includes StarPU host acquisition/D2H and pwrite/close; no fsync','Wall is executable process time, excluding Slurm queue/step startup, input prepacking and numerical validation','Only clock checkpoints added; no extra GPU synchronization'],
        'runs':rows,'mean':{key:sum(x[key] for x in rows)/len(rows) for key in rows[0] if key!='case'}}
(a.results/'timings.json').write_text(json.dumps(result,indent=2)+'\n')
with (a.results/'timings.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
print(json.dumps(result,indent=2))
