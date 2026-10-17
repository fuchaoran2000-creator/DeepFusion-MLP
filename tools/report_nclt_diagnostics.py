#!/usr/bin/python3
"""Score controlled offline node poses on one common 10 Hz reference grid."""
import csv
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mentor_benchmark import Sequence,load_pose,interpolate,local,metrics
from tools.extract_pbstream_diagnostics import extract
from tools.run_nclt_diagnostics import VARIANTS

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'results/mentor_benchmark/nclt_diagnostic_v1'


def write_csv(name,rows):
    with (OUTPUT/name).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def run():
    arrays={};resources={};counts={}
    for name,_,_,_ in VARIANTS:
        directory=OUTPUT/f'nclt-2013-01-10_{name}'
        if not (directory/'cartographer_noimu_resources.json').exists():raise ValueError(f'Incomplete diagnostic: {name}')
        counts[name]=extract(directory/'noimu.pbstream')
        for kind in ['local','global']:
            arrays[name,kind]=load_pose(directory/f'noimu_{kind}_nodes.tum')
        resources[name]=json.loads((directory/'cartographer_noimu_resources.json').read_text())
        expected=json.loads((directory/'condition.json').read_text())['configuration_sha256']
        if resources[name]['configuration_sha256']!=expected:raise AssertionError('Config snapshot mismatch')
    prepared=ROOT/'data/prepared/nclt-2013-01-10'
    gt,gp=load_pose(prepared/'groundtruth.tum');ot,op=load_pose(prepared/'odom.tum')
    start=max(gt[0],ot[0],*[v[0][0] for v in arrays.values()])
    end=min(gt[-1],ot[-1],*[v[0][-1] for v in arrays.values()])
    t=start+np.arange(int(np.floor((end-start)*10))+1)/10.
    right=np.clip(np.searchsorted(gt,t),0,len(gt)-1);left=np.maximum(right-1,0)
    valid=(gt[right]-gt[left])<=.25
    truth=local(interpolate(t,gt,gp));odom=local(interpolate(t,ot,op))
    seq=Sequence('nclt-2013-01-10','nclt',t,odom,odom,truth,valid,'offline_node_diagnostic',
                 dict(alignment='individual common-time origin SE2; identical grid and reference',grid_hz=10,
                      input_grid_uses_gt=True,diagnostic_only=True,
                      interpolation='motion-filtered node poses; not live published poses'))
    scored=[];series={};failure=[]
    for name,kind in [*arrays.keys(),('provided_ekf_odom','input')]:
        pose=odom if kind=='input' else local(interpolate(t,*arrays[name,kind]))
        m=metrics(seq,pose);error=np.linalg.norm(pose[:,:2]-truth[:,:2],axis=1)
        yaw=np.abs(np.degrees((pose[:,2]-truth[:,2]+np.pi)%(2*np.pi)-np.pi))
        series[name,kind]=(pose,error,yaw)
        resource={} if kind=='input' else resources[name]
        count={} if kind=='input' else counts[name]
        scored.append(dict(condition=name,trajectory=kind,ape_m=m['ape_m']['rmse'],
            rpe_1s_m=m['rpe']['1s']['translation_m']['rmse'],yaw_ape_deg=m['yaw_ape_deg']['rmse'],
            endpoint_error_m=float(error[-1]),samples=int(valid.sum()),
            nodes=count.get('local_nodes'),inter_submap_constraints=count.get('inter_submap_constraints'),
            wall_s=resource.get('wall_s'),peak_rss_mib=resource.get('peak_rss_mib')))
        for threshold in [1,5,10,50,100]:
            index=np.flatnonzero(valid&(error>threshold))
            failure.append(dict(condition=name,trajectory=kind,threshold_m=threshold,
                first_exceedance_s=float(t[index[0]]-t[0]) if len(index) else None,
                fraction_over_threshold=float((error[valid]>threshold).mean())))
    write_csv('common_grid_metrics.csv',scored);write_csv('first_exceedance.csv',failure)
    np.savez_compressed(OUTPUT/'common_grid.npz',t=t,valid=valid,gt=truth,odom=odom,
        **{f'{name}_{kind}':v[0] for (name,kind),v in series.items()})
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    colors=plt.cm.tab10.colors
    for i,(name,_,_,_) in enumerate(VARIANTS):
        for j,kind in enumerate(['local','global']):
            pose,error,yaw=series[name,kind]
            axes[0,j].plot((t-t[0])[valid],error[valid],label=name,color=colors[i],linewidth=1)
            axes[1,j].plot(pose[:,0],pose[:,1],label=name,color=colors[i],linewidth=.9)
    for j,kind in enumerate(['local','global']):
        axes[0,j].plot((t-t[0])[valid],series['provided_ekf_odom','input'][1][valid],color='black',label='provided EKF odom')
        axes[0,j].set(xlabel='elapsed time (s)',ylabel='origin-aligned position error (m)',title=f'{kind} node poses')
        axes[0,j].grid(alpha=.2)
        axes[1,j].plot(truth[:,0],truth[:,1],color='black',label='SLAM-derived reference',linewidth=1.5)
        axes[1,j].set(xlabel='local x (m)',ylabel='local y (m)');axes[1,j].axis('equal');axes[1,j].grid(alpha=.2)
    axes[0,0].legend(fontsize=7);axes[1,1].legend(fontsize=7)
    for ext in ['png','pdf']:fig.savefig(OUTPUT/f'factor_contrasts.{ext}',dpi=180)
    plt.close(fig)
    manifest={'independent_sessions':1,'diagnostic_conditions':len(VARIANTS),'common_samples':int(valid.sum()),
              'common_start_unix_s':float(t[0]),'common_end_unix_s':float(t[-1]),
              'physical_duration_s':float(t[-1]-t[0]),'GT_used_for_conversion_or_SLAM':False,
              'GT_used_for_diagnostic_scoring_grid':True,'test_used_to_select_contrasts':False,
              'source_checksums':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [
                  ROOT/'tools/run_nclt_diagnostics.py',ROOT/'tools/replay_cartographer.py',
                  ROOT/'tools/extract_pbstream_diagnostics.py',ROOT/'tools/report_nclt_diagnostics.py']},
              'resource_caveat':'offline complete pipeline including final optimization; other analysis jobs overlapped; not isolated real-time measurement',
              'reference_caveat':'dense GT interpolated with supplied EKF odometry; short RPE not independent'}
    (OUTPUT/'provenance.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(scored,indent=2),flush=True)


if __name__=='__main__':run()
