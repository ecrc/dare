"""Summarize solver time separately from paired ATA science diagnostics."""
import argparse
import csv
import json
from pathlib import Path
import re
import statistics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    args = p.parse_args()
    summary, by_seed, timings = [], [], []
    for site in ('perlmutter', 'dai'):
        for case in sorted((args.root/site).glob('*/*')):
            if not case.name.isdecimal() or not (case/'solver.log').exists():
                continue
            n, job = int(case.name), case.parent.name
            text = (case/'solver.log').read_text()
            def number(pattern):
                match = re.search(pattern, text, re.M)
                return float(match[1]) if match else None
            timing = dict(site=site, job=job, states=n,
                solver_seconds=number(r'^Time:\s+(\S+)'),
                wall_seconds=number(r'^Wall seconds:\s+(\S+)'),
                fits_read_seconds=number(r'^Parallel FITS read seconds:\s+(\S+)'),
                input_layout_seconds=number(r'^Parallel Lap2Desc seconds:\s+(\S+)'),
                output_collect_seconds=sum(map(float,re.findall(r'Desc2Lap seconds: (\S+)',text))),
                fits_write_seconds=sum(map(float,re.findall(r'FITS write seconds: (\S+)',text))))
            timings.append(timing)
            metrics = [json.loads(path.read_text()) for path in sorted(case.glob('ao-*/metrics.json'))]
            if not metrics:
                continue
            for d in metrics:
                if d['frames'] != 1000 or d['burn_in'] != 200 or d['rtc_delay_frames'] != 0:
                    raise ValueError('Inconsistent simulation protocol in '+str(case))
                if d['max_command_abs'] > 100 or d['model_status'] not in ('sampled-match','reconstructed-model-mismatch'):
                    raise ValueError('Invalid AO result in '+str(case))
                by_seed.append(dict(site=site, job=job, states=n, seed=d['seed_base'],
                    le_strehl=d['field_mean_le_strehl'], se_strehl=d['field_mean_se_strehl'],
                    wfe_rms_nm=d['field_wfe_rms_nm'], max_command=d['max_command_abs'],
                    model_status=d['model_status']))
            seeds = sorted(d['seed_base'] for d in metrics)
            row = dict(site=site, job=job, states=n, seeds=seeds,
                       complete=seeds == [1100,2300,3700],
                       model_status=metrics[0]['model_status'])
            for key in ('field_mean_le_strehl','field_mean_se_strehl','field_wfe_rms_nm'):
                values = [d[key] for d in metrics]
                row[key+'_mean'] = statistics.mean(values)
                row[key+'_sample_sd'] = statistics.stdev(values) if len(values)>1 else None
            summary.append(row)
    open_metrics=[json.loads(p.read_text()) for p in sorted((args.root/'open-loop').glob('ao-*/metrics.json'))]
    if open_metrics:
        seeds=sorted(d['seed_base'] for d in open_metrics)
        row=dict(site='open-loop',job=None,states=0,seeds=seeds,complete=seeds==[1100,2300,3700],model_status='reconstructed optical configuration')
        for key in ('field_mean_le_strehl','field_mean_se_strehl','field_wfe_rms_nm'):
            values=[d[key] for d in open_metrics]
            row[key+'_mean']=statistics.mean(values)
            row[key+'_sample_sd']=statistics.stdev(values) if len(values)>1 else None
        summary.append(row)
        for d in open_metrics:
            by_seed.append(dict(site='open-loop',job=None,states=0,seed=d['seed_base'],
                le_strehl=d['field_mean_le_strehl'],se_strehl=d['field_mean_se_strehl'],
                wfe_rms_nm=d['field_wfe_rms_nm'],max_command=0,model_status='reconstructed optical configuration'))
    for name, rows in (('solver-timings',timings),('ao-by-seed',by_seed)):
        if rows:
            with (args.root/(name+'.csv')).open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
    (args.root/'ao-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(dict(timings=timings,ao=summary),indent=2))


if __name__ == '__main__':
    main()
