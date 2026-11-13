"""Export paired-seed science diagnostics; no inference of absolute MAVIS performance."""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    args=p.parse_args()
    rows=list(csv.DictReader((args.root/'ao-by-seed.csv').open()))
    sites=sorted(set(r['site'] for r in rows)-{'open-loop'})
    seeds=(1100,2300,3700)
    sizes=(7090,12570,28210)
    fig,axes=plt.subplots(1,3,figsize=(13,4.6))
    keys=('se_strehl','le_strehl','wfe_rms_nm')
    titles=('Short-exposure Strehl (%)','Long-exposure Strehl (%)','Residual wavefront RMS (nm)')
    colors=plt.get_cmap('tab10').colors
    for i,site in enumerate(sites):
        for j,seed in enumerate(seeds):
            found={int(r['states']):r for r in rows if r['site']==site and int(r['seed'])==seed}
            if not all(n in found for n in sizes): continue
            for ax,key,title in zip(axes,keys,titles):
                scale=1 if key=='wfe_rms_nm' else 100
                values=[float(found[n][key])*scale for n in sizes]
                ax.plot(np.arange(3),values,marker='o',color=colors[j],
                        linestyle='-' if i==0 else '--',label=f'{site}; seed {seed}')
                ax.set_title(title,fontsize=10)
                ax.set_xticks(range(3),['7,090','12,570','28,210'])
                ax.set_xlabel('DARE states')
                ax.grid(alpha=.22)
    fig.suptitle('FP64 DARE / FP32 ATA RTC — reconstructed-model diagnostics',fontsize=13)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=3,fontsize=9,bbox_to_anchor=(.5,.055))
    fig.text(.5,.015,'1,000 frames, 200 burn-in; NGS calibration mismatch remains. Not an absolute MAVIS validation.',
             ha='center',fontsize=9,color='#9a3510')
    fig.tight_layout(rect=(0,.19,1,.93))
    fig.savefig(args.root/'ao-comparison.png',dpi=180)
    fig.savefig(args.root/'ao-comparison.pdf')


if __name__=='__main__': main()
