#!/usr/bin/env python3
"""Plot measured full-process phases, with historical FITS comparison labeled."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('results',type=Path);a=p.parse_args()
d=json.loads((a.results/'timings.json').read_text())
rows=[dict(case='FITS (historical)',input=20.476834,solver=124.951,output=31.68434,other=16.457826,wall=193.57),*d['runs']]
fig,ax=plt.subplots(figsize=(9,3.6),layout='constrained')
left=[0]*len(rows)
for key,label,color in [('input','Input','#5b8db8'),('solver','FP64 solve','#de9450'),('output','Output','#67a78c'),('other','Startup / cleanup / other','#a6a6ad')]:
 values=[r[key] for r in rows]
 ax.barh(range(len(rows)),values,left=left,label=label,color=color,height=.54)
 for i,v in enumerate(values):
  if v>=7: ax.text(left[i]+v/2,i,f'{v:.1f}',ha='center',va='center',fontsize=9)
 left=[x+y for x,y in zip(left,values)]
for i,r in enumerate(rows): ax.text(r['wall']+2,i,f"{r['wall']:.2f} s",va='center',fontsize=10)
ax.set_yticks(range(len(rows)),[r['case'].replace('run-','Tile run ') for r in rows]);ax.invert_yaxis()
ax.set_xlabel('Executable wall time (seconds)');ax.set_xlim(0,220)
ax.set_title('28,210 states · FP64 LU · 4 × A100 80 GB',loc='left')
ax.legend(loc='upper center',bbox_to_anchor=(.5,-.2),ncol=4,frameon=False,fontsize=9)
ax.spines[['top','right']].set_visible(False)
fig.text(.02,-.06,'Tile job '+d['job']+'; FITS reference job 61125702 (different run/cache).\nIncludes host acquisition/transfers in I/O. No fsync; excludes queue, prepacking and controller assembly.',fontsize=8,color='#555555')
for ext in ['png','pdf']:fig.savefig(a.results/('time-breakdown.'+ext),dpi=180,bbox_inches='tight')
