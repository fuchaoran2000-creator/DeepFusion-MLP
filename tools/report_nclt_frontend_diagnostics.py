#!/usr/bin/python3
"""Score adaptive frontend contrasts on the already frozen stage-1 grid."""
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

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'results/mentor_benchmark/nclt_frontend_diagnostic_v2'
ORIGINAL=ROOT/'results/mentor_benchmark/nclt_diagnostic_v1'

if __name__=='__main__':
    with np.load(ORIGINAL/'common_grid.npz') as data:
        t,valid,truth,odom=[data[k].copy() for k in ['t','valid','gt','odom']]
        reference_conditions={k:data[k].copy() for k in ['original10_loop_local','original10_loop_global',
                                                        'original10_noloop_local','original10_noloop_global']}
    seq=Sequence('nclt','nclt',t,odom,odom,truth,valid,'offline_node_diagnostic',{})
    results=[];poses={};node_checks=[]
    for name in ['ceres10_loop','ceres10_noloop']:
        directory=OUTPUT/f'nclt-2013-01-10_{name}'
        stats=extract(directory/'noimu.pbstream')
        resource=json.loads((directory/'cartographer_noimu_resources.json').read_text())
        condition=json.loads((directory/'condition.json').read_text())
        assert resource['configuration_sha256']==condition['configuration_sha256']
        exported=np.loadtxt(directory/'cartographer_noimu_final.tum')
        upstream=np.loadtxt(directory/'noimu_global_nodes.tum')
        # q and -q encode the same rotation; the ROS exporter canonicalizes
        # through a rotation matrix while protobuf retains the solver sign.
        upstream[:,4:8] *= np.where(np.sum(upstream[:,4:8]*exported[:,4:8],axis=1)<0,-1.,1.)[:,None]
        np.testing.assert_allclose(exported,upstream,atol=2e-6,rtol=0)
        node_checks.append(dict(condition=name,upstream_export_max_error=float(np.abs(exported-upstream).max())))
        for kind in ['local','global']:
            pt,pp=load_pose(directory/f'noimu_{kind}_nodes.tum')
            assert pt[0]<=t[0] and pt[-1]>=t[-1]
            pose=local(interpolate(t,pt,pp));poses[name+'_'+kind]=pose
            m=metrics(seq,pose)
            results.append(dict(condition=name,trajectory=kind,ape_m=m['ape_m']['rmse'],
                rpe_1s_m=m['rpe']['1s']['translation_m']['rmse'],yaw_ape_deg=m['yaw_ape_deg']['rmse'],
                samples=int(valid.sum()),local_nodes=stats['local_nodes'],
                inter_submap_constraints=stats['inter_submap_constraints'],
                wall_s=resource['wall_s'],peak_rss_mib=resource['peak_rss_mib'],adaptive_diagnostic=True))
    with (OUTPUT/'common_grid_metrics.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)
    np.savez_compressed(OUTPUT/'common_grid.npz',t=t,valid=valid,gt=truth,odom=odom,**poses)
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for j,kind in enumerate(['local','global']):
        for name in ['original10_loop','original10_noloop','ceres10_loop','ceres10_noloop']:
            key=name+'_'+kind;p=poses.get(key,reference_conditions.get(key))
            error=np.linalg.norm(p[:,:2]-truth[:,:2],axis=1)
            axes[j].plot((t-t[0])[valid],error[valid],label=name,linewidth=1.2)
        axes[j].set(xlabel='elapsed time (s)',ylabel='position error (m)',title=kind+' node poses')
        axes[j].grid(alpha=.2);axes[j].legend(fontsize=8)
    for extension in ['png','pdf']:fig.savefig(OUTPUT/f'frontend_contrasts.{extension}',dpi=180)
    plt.close(fig)
    (OUTPUT/'verification.json').write_text(json.dumps(dict(upstream_export_checks=node_checks,
        scoring_grid_sha256=hashlib.sha256((ORIGINAL/'common_grid.npz').read_bytes()).hexdigest(),
        resource_caveat='offline complete pipeline; no isolated live or embedded timing',
        independent_sequences=1,conditions=2,reference_used_for_parameter_fitting=False),indent=2)+'\n')
    print(json.dumps(results,indent=2),flush=True)
