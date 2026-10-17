#!/usr/bin/python3
"""Export compact scene summaries and validation-only integration memory curves."""
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]/'results/mentor_benchmark'

if __name__=='__main__':
    home=ROOT/'home_increment_memory_v1';cross=ROOT/'cross_scene_increment_memory_v1'
    groups=defaultdict(list)
    for row in csv.DictReader((home/'results.csv').open()):
        groups[row['scene'],row['method'],row['seed']].append(row)
    summary=[]
    for (scene,method,seed),rows in sorted(groups.items()):
        summary.append(dict(scene=scene,method=method,seed=seed,sequences=len(rows),
            raw_macro_ape_m=float(np.mean([float(r['raw_ape_m']) for r in rows])),
            macro_ape_m=float(np.mean([float(r['ape_m']) for r in rows])),
            macro_rpe_1s_m=float(np.mean([float(r['rpe_1s_m']) for r in rows])),
            tau_s=rows[0]['tau_s'],alpha=rows[0]['alpha']))
    with (home/'scene_summary.csv').open('w',newline='') as out:
        writer=csv.DictWriter(out,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    for j,(root,name) in enumerate([(home,'Home validation'),(cross,'Cafe validation')]):
        rows=list(csv.DictReader((root/'validation_curves.csv').open()))
        for i,model in enumerate(sorted({r['model'] for r in rows})):
            values=[r for r in rows if r['model']==model]
            labels=[r['tau_s'] or 'unlimited' for r in values]
            for k,key in enumerate(['validation_gated_ape_m','forced_alpha1_ungated_validation_ape_m']):
                axes[k,j].plot(np.arange(len(values)),[float(r[key]) for r in values],marker='o',markersize=3,
                               label=model.rsplit('_seed',1)[1])
                axes[k,j].set_xticks(np.arange(len(values)));axes[k,j].set_xticklabels(labels)
        for k in range(2):
            axes[k,j].set(xlabel='memory time constant (s)',ylabel='validation macro APE (m)',
                          title=name+(' / frozen gate + alpha' if k==0 else ' / alpha=1, ungated diagnostic'))
            axes[k,j].grid(alpha=.2);axes[k,j].legend(title='seed',fontsize=8)
    for ext in ['png','pdf']:fig.savefig(home/f'validation_memory_curves.{ext}',dpi=180)
    plt.close(fig)
    print('Saved scene_summary.csv and validation_memory_curves.png/pdf')
