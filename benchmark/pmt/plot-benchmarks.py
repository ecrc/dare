#!/usr/bin/env python3
"""Render publication figures from measured DARE benchmarks using UltraPlot."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use('Agg')
import ultraplot as uplt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter

CUDA_PEAK = 9.7  # TFLOP/s per A100 SXM, published theoretical FP64 peak
TENSOR_PEAK = 19.5  # TFLOP/s per A100 SXM, dense FP64 Tensor Core peak
COLORS = {7090: '#D55E00', 12570: '#0072B2', 19610: '#009E73', 28210: '#8B5AA3'}
MARKERS = {7090: 'o', 12570: 's', 19610: '^', 28210: 'D'}
GRAY = '#9BA4AD'
SOURCES = {
    'A100_80GB_peak': 'https://www.nvidia.com/content/dam/en-zz/Solutions/data-center/a100/pdf/a100-80gb-datasheet-update-nvidia-us-1521051-r2-web.pdf',
    'A100_40GB_peak': 'https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet.pdf',
    'UltraPlot': 'https://ultraplot.readthedocs.io/en/latest/basics.html',
}


def configure():
    uplt.rc.update({
        'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
        'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 9,
        'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
        'legend.fontsize': 7.5, 'axes.linewidth': 0.65,
        'lines.linewidth': 1.35, 'lines.markersize': 4.5,
        'figure.facecolor': 'white', 'savefig.facecolor': 'white',
        'mathtext.fontset': 'stix', 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'svg.fonttype': 'none', 'grid.alpha': 0.35,
        'grid.linewidth': 0.45, 'grid.color': '#ABB3BA',
    })


def panels(n=3, top='0.62in', height=3.7):
    fig, axs = uplt.subplots(ncols=n, figwidth=7.4, figheight=height, share=False,
                            left='0.58in', right='0.13in', bottom='0.88in',
                            top=top, wspace='0.62in', tight=False)
    for ax in axs:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis='y', which='major', visible=True)
        ax.tick_params(direction='out', length=3, width=0.65)
    return fig, axs


def footnote(fig, lines):
    fig.text(0.075, 0.035, '\n'.join(lines), fontsize=6.8,
             ha='left', va='bottom', linespacing=1.5, color='#444444')


def legend(fig, handles, labels, ncol=2):
    # Native Matplotlib legend on an UltraPlot Figure, with fixed journal margins.
    Figure.legend(fig, handles, labels, loc='upper center',
                  bbox_to_anchor=(0.53, 0.995), ncol=ncol, frameon=False,
                  columnspacing=1.6, handlelength=2.3, handletextpad=0.6)


def sample_stats(values):
    values = np.array(values, dtype=float)
    assert np.all(np.isfinite(values)) and np.all(values > 0)
    center = float(np.median(values))
    return center, np.array([[center-values.min()], [values.max()-center]])


def errorpoint(ax, x, values, color, marker='o', **kwargs):
    center, error = sample_stats(values)
    ax.errorbar([x], [center], yerr=error, color=color, marker=marker,
                linestyle='none', capsize=2, elinewidth=0.8, zorder=5, **kwargs)
    return center


def load_data(workspace):
    path = workspace/'log/pmt/pmt-benchmark.csv'
    grouped = {}
    table = []
    for row in csv.DictReader(path.open()):
        n, p = int(row['States']), int(row['GPUs'])
        times = [float(row[f'Solver run {i} seconds']) for i in (1, 2, 3)]
        t = float(row['Median solver seconds'])
        assert abs(median(times)-t) < 1e-8
        grouped.setdefault(n, {})[p] = dict(times=times, t=t,
                    rate=float(row['Median GFLOP/s'])/1000, gpu=row['GPU'])
    for n, records in sorted(grouped.items()):
        assert sorted(records) == [1, 2, 4]
        assert len({d['gpu'] for d in records.values()}) == 1
        t1 = records[1]['t']
        for p, d in sorted(records.items()):
            d['speedup'] = t1/d['t']
            d['efficiency'] = 100*d['speedup']/p
            table.append(dict(states=n, gpus=p, gpu=d['gpu'], median_seconds=d['t'],
                min_seconds=min(d['times']), max_seconds=max(d['times']),
                speedup=d['speedup'], parallel_efficiency_percent=d['efficiency'],
                achieved_tflops=d['rate'], cuda_peak_tflops=CUDA_PEAK*p,
                fp64_tensor_peak_tflops=TENSOR_PEAK*p,
                percent_fp64_tensor_peak=100*d['rate']/(TENSOR_PEAK*p)))
    opt_path=workspace/'log/pmt/profiling/performance-comparison.json'
    tuning=json.loads(opt_path.read_text())
    for case in tuning:
        for key, field in [('baseline_samples','baseline_median_seconds'),
                           ('optimized_samples','optimized_median_seconds')]:
            assert abs(median(s['seconds'] for s in case[key])-case['summary'][field]) < 1e-8
    return grouped, tuning, table, [path, opt_path]


def strong_scaling(data):
    fig, axs = panels()
    handles, labels = [], []
    for n, records in sorted(data.items()):
        p=np.array(sorted(records))
        times=np.array([records[g]['t'] for g in p])
        t1=records[1]['t']
        color=COLORS[n]
        style='--' if n == 7090 else '-'
        h=axs[0].plot(p, times, color=color, marker=MARKERS[n], linestyle=style)[0]
        handles.append(h)
        labels.append(f'$n={n:,}$  ({40 if n == 7090 else 80} GB)')
        axs[1].plot(p, t1/times, color=color, marker=MARKERS[n], linestyle=style)
        axs[2].plot(p, 100*t1/(p*times), color=color, marker=MARKERS[n], linestyle=style)
        for g in p:
            errorpoint(axs[0], g, records[g]['times'], color, MARKERS[n])
            # Fixed measured median T1 is the normalizer; these are ranges, not CIs.
            if g != 1:
                errorpoint(axs[1], g, [t1/t for t in records[g]['times']], color, MARKERS[n])
                errorpoint(axs[2], g, [100*t1/(g*t) for t in records[g]['times']], color, MARKERS[n])
    for ax in axs:
        ax.set_xlim(0.85,4.15)
        ax.set_xticks([1,2,4])
        ax.set_xlabel('Number of GPUs, $p$')
    axs[0].set_yscale('log')
    axs[0].set_ylim(6,650)
    axs[0].set_yticks([10,30,100,300])
    axs[0].yaxis.set_major_formatter(ScalarFormatter())
    axs[0].set_ylabel('Solver time (s)')
    axs[0].set_title('(a) Time to solution', loc='left')
    axs[1].plot([1,4],[1,4],color='#606060',linestyle=':',linewidth=1.1,zorder=1)
    axs[1].text(2.35,3.45,'Ideal: $S_p=p$',fontsize=7,color='#555555')
    axs[1].set_ylim(0.85,4.25)
    axs[1].set_yticks([1,2,3,4])
    axs[1].set_ylabel('Speedup, $S_p=T_1/T_p$')
    axs[1].set_title('(b) Strong scaling',loc='left')
    axs[2].axhline(100,color='#606060',linestyle=':',linewidth=1.1)
    axs[2].set_ylim(25,106)
    axs[2].set_yticks([25,50,75,100])
    axs[2].set_ylabel('Parallel efficiency (%)')
    axs[2].set_title('(c) $100 S_p/p$',loc='left')
    legend(fig,handles,labels)
    footnote(fig,['Original solver; FP64; 18,694 measurements; median of 3 runs. Whiskers: observed min–max.',
                  'Speedup ranges hold median $T_1$ fixed. Different problem sizes are separate strong-scaling series.'])
    return fig


def optimization(tuning):
    fig, axs=panels(top='0.4in',height=3.55)
    for i, case in enumerate(tuning):
        n=case['summary']['states']; color=COLORS[n]
        ax=axs[i]
        b=[s['seconds'] for s in case['baseline_samples']]
        o=[s['seconds'] for s in case['optimized_samples']]
        bm, om=median(b),median(o)
        upper=bm*1.33
        ax.bar([0,1],[bm,om],color=[GRAY,color],width=0.58,edgecolor='white',linewidth=0.6,zorder=3)
        for x, values in enumerate([b,o]):
            errorpoint(ax,x,values,'#222222',marker='_',markersize=5)
            jitter=np.linspace(-0.11,0.11,len(values))
            ax.scatter(x+jitter,values,s=12,facecolor='white',edgecolor='#222222',linewidth=0.6,zorder=6)
            ax.text(x,median(values)+upper*0.045,f'{median(values):.2f}',ha='center',fontsize=8)
        ax.set_xlim(-0.6,1.6)
        ax.set_ylim(0,upper)
        ax.set_xticks([0,1],['Baseline','Optimized'])
        ax.set_ylabel('Solver time (s)')
        ax.set_title(f'({chr(97+i)}) $n={n:,}$',loc='left')
        ax.text(0.5,0.94,f'{100*(1-om/bm):.1f}% less time',transform=ax.transAxes,
                ha='center',va='top',color=color,fontsize=8)
        ratio=bm/om
        axs[2].bar([i],[ratio],color=color,width=0.55,edgecolor='white',zorder=3)
        axs[2].text(i,ratio+0.05,f'{ratio:.2f}'+r'$\times$',ha='center',fontsize=8)
    axs[2].axhline(1,color='#555555',linestyle=':',linewidth=1)
    axs[2].set_ylim(0,1.85)
    axs[2].set_xlim(-0.6,1.6)
    axs[2].set_yticks([0,0.5,1,1.5])
    axs[2].set_xticks([0,1],['12,570','28,210'])
    axs[2].set_xlabel('Number of states, $n$')
    axs[2].set_ylabel(r'$T_{\mathrm{base}}/T_{\mathrm{opt}}$')
    axs[2].set_title('(c) Four-GPU speedup',loc='left')
    footnote(fig,['4 × A100 SXM4 80 GB; FP64. Dots: individual runs; bars: medians; whiskers: observed min–max.',
                  '12,570 states: 2 runs/configuration, same node. 28,210: 3 runs/configuration, different nodes.'])
    return fig


def throughput(data,tuning):
    fig,axs=panels(n=2,top='0.67in',height=3.9)
    handles,labels=[],[]
    for n,records in sorted(data.items()):
        p=np.array(sorted(records)); color=COLORS[n]
        h=axs[0].plot(p,[records[g]['rate'] for g in p],color=color,
                     marker=MARKERS[n],linestyle='--' if n==7090 else '-')[0]
        handles.append(h); labels.append(f'$n={n:,}$ ({40 if n==7090 else 80} GB)')
    for case in tuning:
        n=case['summary']['states']
        errorpoint(axs[0],4,[s['tflops'] for s in case['optimized_samples']],COLORS[n],
                   marker='*',markersize=10,markeredgecolor='white',markeredgewidth=0.5)
    p=np.array([1,2,4])
    tensor_line=axs[0].plot(p,TENSOR_PEAK*p,color='#333333',linestyle=':',linewidth=1.3)[0]
    cuda_line=axs[0].plot(p,CUDA_PEAK*p,color='#777777',linestyle='--',linewidth=1)[0]
    optimized_marker=Line2D([],[],color='#444444',marker='*',linestyle='none',markersize=8)
    axs[0].legend(handles=[tensor_line,cuda_line,optimized_marker],
                  labels=['FP64 Tensor Core peak','Standard FP64 peak','Optimized (4 GPUs)'],
                  loc='upper left',frameon=False,fontsize=7,handlelength=2.0,ncols=1)
    axs[0].set_xticks([1,2,4])
    axs[0].set_xlim(0.85,4.2)
    axs[0].set_ylim(0,86)
    axs[0].set_yticks([0,20,40,60,80])
    axs[0].set_xlabel('Number of GPUs, $p$')
    axs[0].set_ylabel('Solver throughput (TFLOP/s)')
    axs[0].set_title('(a) Throughput and hardware peaks',loc='left')
    for i,case in enumerate(tuning):
        n=case['summary']['states']
        for dx,key,color in [(-0.18,'baseline_samples',GRAY),(0.18,'optimized_samples',COLORS[n])]:
            values=[100*s['tflops']/(4*TENSOR_PEAK) for s in case[key]]
            val=median(values)
            axs[1].bar([i+dx],[val],width=0.31,color=color,edgecolor='white',zorder=3)
            errorpoint(axs[1],i+dx,values,'#222222',marker='_',markersize=3)
            axs[1].text(i+dx,val+3,f'{val:.1f}%',ha='center',fontsize=7.5)
    axs[1].axhline(100,color='#333333',linestyle=':',linewidth=1.3)
    axs[1].text(0.97,0.97,'78.0 TFLOP/s: FP64 Tensor Core peak',
                transform=axs[1].transAxes,ha='right',fontsize=6.8)
    # The conventional CUDA-core peak is a reference, not a ceiling for tensor work.
    axs[1].set_ylim(0,107)
    axs[1].set_yticks([0,25,50,75,100])
    axs[1].set_xlim(-0.6,1.6)
    axs[1].set_xticks([0,1],['12,570','28,210'])
    axs[1].set_xlabel('Number of states, $n$')
    axs[1].set_ylabel('Throughput / FP64 Tensor peak (%)')
    axs[1].set_title('(b) Four-GPU tuning runs',loc='left')
    axs[1].text(0.04,0.80,'Gray: baseline\nColor: optimized',transform=axs[1].transAxes,
                fontsize=7,va='top',linespacing=1.5)
    legend(fig,handles,labels)
    footnote(fig,['Theoretical per-GPU peaks: 9.7 TFLOP/s (standard FP64), 19.5 TFLOP/s (FP64 Tensor Cores); NVIDIA.',
                  'Throughput uses solver-reported FLOP counts; peak fractions are not GPU utilization measurements.'])
    return fig


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    script=Path(__file__).resolve()
    default=script.parents[3] if script.parent.parent.name == 'benchmark' else Path.cwd()
    parser.add_argument('--workspace',type=Path,default=default)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    out=args.output or args.workspace/'plots/pmt'
    out.mkdir(parents=True,exist_ok=True)
    configure()
    data,tuning,table,paths=load_data(args.workspace)
    with (out/'strong-scaling-data.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    shutil.copy2(paths[1],out/'optimization-data.json')
    shutil.copy2(args.workspace/'log/pmt/profiling/performance-comparison.csv',out/'optimization-data.csv')
    figs=[('01-strong-scaling',strong_scaling(data)),
          ('02-time-and-speedup',optimization(tuning)),
          ('03-throughput-vs-peak',throughput(data,tuning))]
    with PdfPages(out/'dare-a100-benchmark-figures.pdf',metadata={
            'Title':'DARE FP64 performance on NVIDIA A100 GPUs',
            'Author':'Yuxi Hong', 'Subject':'Strong scaling, optimization, and theoretical peak comparison',
            'Creator':f'UltraPlot {uplt.__version__}'}) as pdf:
        for name,fig in figs:
            for suffix in ['pdf','svg','png']:
                fig.savefig(out/f'{name}.{suffix}',dpi=600)
            fig.savefig(out/f'{name}-preview.png',dpi=150)
            pdf.savefig(fig)
            print(name,fig.get_size_inches())
    manifest=dict(renderer='UltraPlot',ultraplot=uplt.__version__,matplotlib=matplotlib.__version__,
        numpy=np.__version__,peaks_tflops_per_gpu=dict(fp64=CUDA_PEAK,fp64_tensor=TENSOR_PEAK),
        peak_sources=SOURCES,inputs=[dict(path=str(p.relative_to(args.workspace)),
                       sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths],
        uncertainty='Observed min–max ranges, not confidence intervals. No optimized scaling curve is inferred.',
        hardware='7,090-state original scaling: A100 SXM4 40 GB; all other cases: A100 SXM4 80 GB.')
    (out/'provenance.json').write_text(json.dumps(manifest,indent=2)+'\n')
    shutil.copy2(Path(__file__).resolve(),out/'plot-benchmarks.py')

if __name__=='__main__':
    main()
