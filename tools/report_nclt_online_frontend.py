#!/usr/bin/python3
"""Keep online frontend improvement separate from frozen residual correction."""
import csv
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from mentor_benchmark import Sequence,load_sequence,load_pose,interpolate,local,metrics

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'results/mentor_benchmark/nclt_frontend_diagnostic_v2'


def write_csv(name,rows):
    with (OUTPUT/name).open('w',newline='') as out:
        writer=csv.DictWriter(out,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def run():
    old_manifest=json.loads((ROOT/'config/benchmark_cross_scene_causal_v1.json').read_text())
    new_manifest=json.loads((ROOT/'config/benchmark_cross_scene_frontend_v2.json').read_text())
    old_row=next(r for r in old_manifest['sequences'] if r['sequence_id'].startswith('nclt-'))
    new_row=next(r for r in new_manifest['sequences'] if r['sequence_id'].startswith('nclt-'))
    rows=[old_row,new_row];independent=[];arrays=[]
    for name,row in zip(['original','ceres_only'],rows):
        seq=load_sequence(row);m=metrics(seq,seq.cart)
        resource_path=Path(row['cart']).with_name('cartographer_noimu_online_resources_'+
            ('unified_inputs' if name=='original' else 'frontend_v2')+'.json')
        resource=json.loads(resource_path.read_text());assert resource['shutdown_success']
        independent.append(dict(configuration=name,ape_m=m['ape_m']['rmse'],
            rpe_1s_m=m['rpe']['1s']['translation_m']['rmse'],yaw_ape_deg=m['yaw_ape_deg']['rmse'],
            samples=m['ape_m']['count'],input_available_fraction=float(seq.available.mean()),
            replay_rate=resource['replay_rate'],wall_s=resource['wall_s'],
            slam_peak_rss_mib=resource['peak_rss_mib'],raw_pose_count=resource['pose_count']))
        with np.load(row['sampled_stream']) as data:
            arrays.append({k:data[k].copy() for k in ['t','cart','odom','available']})
    start=max(a['t'][0] for a in arrays);end=min(a['t'][-1] for a in arrays)
    t=start+np.arange(int((end-start)*10)+1)/10.
    gt,gp=load_pose(new_row['gt']);truth=local(interpolate(t,gt,gp))
    right=np.clip(np.searchsorted(gt,t),0,len(gt)-1);left=np.maximum(right-1,0)
    valid=(t>=gt[0])&(t<=gt[-1])&((gt[right]-gt[left])<=.25)
    matched=[]
    for a in arrays:
        i=np.searchsorted(a['t'],t,side='right')-1;i=i.clip(0,len(a['t'])-1)
        valid &= a['available'][i] & (t-a['t'][i]<=.15)
        matched.append(a['cart'][i])
    anchor=int(np.flatnonzero(valid)[0])
    # This grid is scoring only, never feeds training or the live runtime.
    from causal_stream import local as point_local
    truth=np.array([point_local(p,truth[anchor]) for p in truth])
    shared=[];series={}
    for name,pose in zip(['original','ceres_only'],matched):
        pose=np.array([point_local(p,pose[anchor]) for p in pose])
        seq=Sequence('nclt','nclt',t,pose,pose,truth,valid,'online_common_time',{})
        m=metrics(seq,pose);series[name]=pose
        shared.append(dict(configuration=name,ape_m=m['ape_m']['rmse'],
            rpe_1s_m=m['rpe']['1s']['translation_m']['rmse'],yaw_ape_deg=m['yaw_ape_deg']['rmse'],
            common_valid_samples=int(valid.sum()),common_duration_s=float(t[-1]-t[0]),
            alignment='each raw trajectory origin SE2 at one common scoring time',
            pose_sampling='past-only captured grid hold, <=150ms; reference interpolation only for scoring'))
    write_csv('online_individual_metrics.csv',independent);write_csv('online_common_time_metrics.csv',shared)
    np.savez_compressed(OUTPUT/'online_common_time.npz',t=t,gt=truth,valid=valid,**series)
    new_seq=load_sequence(new_row)
    nodes=np.loadtxt(ROOT/'data/nclt/cov_2013-01-10.csv',delimiter=',')[:,0]/1e6
    nodes=nodes[(nodes>=new_seq.t[0])&(nodes<=new_seq.t[-1])]
    indices=np.searchsorted(new_seq.t,nodes).clip(0,len(new_seq.t)-1);prior=np.maximum(indices-1,0)
    indices=np.where(abs(new_seq.t[prior]-nodes)<abs(new_seq.t[indices]-nodes),prior,indices)
    indices=np.unique(indices[(abs(new_seq.t[indices]-nodes)<=.051)&new_seq.valid[indices]])
    sparse=[]
    for name,pose in [('cartographer',new_seq.cart),('provided_ekf_odom',new_seq.odom)]:
        error=np.linalg.norm(pose[indices,:2]-new_seq.gt[indices,:2],axis=1)
        sparse.append(dict(method=name,graph_node_samples=len(indices),ape_m=float(np.sqrt(np.mean(error**2))),
            tolerance_s=.051,reference_kind='SLAM graph reference; not independent mocap'))
    write_csv('online_graph_node_ape.csv',sparse)
    resources=json.loads((Path(new_row['cart']).parent/'cartographer_noimu_metrics_frontend_v2.json').read_text())
    gauges={}
    for sample in resources['samples']:
        for k,v in sample['gauges'].items():gauges.setdefault(k,[]).append(v)
    summary={k:dict(count=len(v),min=float(min(v)),p05=float(np.quantile(v,.05)),median=float(np.median(v)),p95=float(np.quantile(v,.95)),max=float(max(v))) for k,v in gauges.items()}
    for family in resources['before_shutdown']['families']:
        if 'search' in family['name'] or 'constraints' in family['name']:
            summary[family['name']]=family
    (OUTPUT/'online_metric_summary.json').write_text(json.dumps(dict(summary=summary,
        caution='local trajectory latency = inter-update interval; real-time ratios are duration/compute gauges; queue length includes batch results, not pending-backlog count; found constraints are not loop accuracy'),indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for name,pose in series.items():
        axes[0].plot((t-t[0])[valid],np.linalg.norm(pose[valid,:2]-truth[valid,:2],axis=1),label=name)
        axes[1].plot(pose[:,0],pose[:,1],label=name,linewidth=1)
    axes[0].set(xlabel='elapsed time (s)',ylabel='position error (m)');axes[0].legend();axes[0].grid(alpha=.2)
    axes[1].plot(truth[:,0],truth[:,1],color='black',label='SLAM-derived reference');axes[1].axis('equal')
    axes[1].set(xlabel='local x (m)',ylabel='local y (m)');axes[1].legend();axes[1].grid(alpha=.2)
    for ext in ['png','pdf']:fig.savefig(OUTPUT/f'online_frontend_comparison.{ext}',dpi=180)
    plt.close(fig)
    (OUTPUT/'online_provenance.json').write_text(json.dumps(dict(
        independent_sessions=1,new_online_captures=1,rate=1.,adaptive_diagnostic=True,
        learned_model_in_slam=False,correction_feedback_to_slam=False,
        frozen_evaluation_manifest='config/benchmark_cross_scene_frontend_v2.json',
        artifacts_sha256={str(Path(r['sampled_stream']).relative_to(ROOT)):hashlib.sha256(Path(r['sampled_stream']).read_bytes()).hexdigest() for r in rows}),indent=2)+'\n')
    print(json.dumps(dict(individual=independent,common_time=shared,graph_nodes=sparse),indent=2),flush=True)


if __name__=='__main__':run()
