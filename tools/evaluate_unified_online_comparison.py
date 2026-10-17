#!/usr/bin/python3
"""Score current paired correction and SLAM alternatives on common input times."""
import csv
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from mentor_benchmark import Sequence,load_pose,interpolate,metrics
from tools.summarize_mentor_benchmark import anchor_alignment,shared_anchor_alignment,at_times

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/paper_revision'


def run():
    results=[];sources=[]
    for scene,label,names in [('home','v2',['home1-3','home1-4','home1-5']),
                              ('corridor','crossscene',['corridor1-3','corridor1-4','corridor1-5'])]:
        for name in names:
            directory=ROOT/'data/prepared'/name
            paths={a:directory/f'{a}_online_{label}.tum' for a in ['cartographer_noimu','cartographer_imu','2dliw']}
            if scene=='home':
                paths['cartographer_noimu']=directory/'cartographer_noimu_online_unified20.tum'
                paths['mlp20_confidence']=directory/'cartographer_noimu_corrected_online_unified20.tum'
            paths['wheel_odometry']=directory/'odom.tum'
            ot,op=load_pose(paths['wheel_odometry']);t=np.arange(ot[0],ot[-1],.1)
            gt,gp=load_pose(directory/'groundtruth.tum');truth=interpolate(t,gt,gp)
            ri=np.clip(np.searchsorted(gt,t),0,len(gt)-1);li=np.maximum(ri-1,0)
            ref=(t>=gt[0])&(t<=gt[-1])&((gt[ri]-gt[li])<=.25)
            values={method:at_times(path,t) for method,path in paths.items()}
            if scene=='home':
                with np.load(directory/'cartographer_noimu_sampled_stream_unified20.npz') as d:
                    index=np.clip(np.searchsorted(d['t'],t,side='right')-1,0,len(d['t'])-1)
                    # Reject stale base poses, but retain correct raw fallback
                    # during missing odometry: an output remains available.
                    age=t-d['cart_source_t'][index]
                    pose,mask=values['mlp20_confidence'];mask&=(age>=0)&(age<=.15)
                    values['mlp20_confidence']=(pose,mask)
            common=ref.copy()
            for pose,mask in values.values():common&=mask&np.isfinite(pose).all(1)
            assert common.sum()>=20
            anchor=int(np.flatnonzero(common)[0]);seq=Sequence(name,scene,t,truth,truth,truth,common,'online',{})
            for method,(pose,mask) in values.items():
                aligned=anchor_alignment(pose,truth,anchor);m=metrics(seq,aligned)
                row=dict(scene=scene,sequence=name,method=method,ape_rmse_m=m['ape_m']['rmse'],
                    rpe_1s_m=m['rpe']['1s']['translation_m']['rmse'],
                    rpe_1s_deg=m['rpe']['1s']['rotation_deg']['rmse'],
                    rpe_5s_m=m['rpe']['5s']['translation_m']['rmse'],
                    common_coverage=float(common.mean()),common_frames=int(common.sum()),
                    full_input_coverage=float(mask.mean()),reference_coverage=float(ref.mean()),
                    paired_map_ape_m=None,replay_wall_s=None,slam_peak_rss_mib=None,
                    input_budget='wheel odometry',callback_p99_ms=None,
                    alignment='Each estimate anchored at the same first common valid input time; planar SE2, no full-trajectory fit or scale.',
                    grid='Wheel-odometry input span at 10Hz; grid independent of reference.')
                if method!='wheel_odometry':
                    parent='cartographer_noimu' if method=='mlp20_confidence' else method
                    selected='unified20' if scene=='home' and parent=='cartographer_noimu' else label
                    path=directory/f'{parent}_online_resources_{selected}.json';res=json.loads(path.read_text())
                    row.update(replay_wall_s=res['wall_s'],slam_peak_rss_mib=res['peak_rss_mib'],
                               input_budget='LiDAR + wheel odometry'+(' + IMU' if parent!='cartographer_noimu' else ''))
                    if method=='mlp20_confidence':
                        same_map=shared_anchor_alignment(pose,values['cartographer_noimu'][0],truth,anchor)
                        row['paired_map_ape_m']=metrics(seq,same_map)['ape_m']['rmse']
                        row['callback_p99_ms']=res['correction']['p99_ms']
                results.append(row)
            sources.extend(dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths.values())
    with (OUT/'online_unified_common_time.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
    (OUT/'online_unified_protocol.json').write_text(json.dumps(dict(
        scope='Current validation-selected 20D actual paired Home correction plus tight/IMU SLAM alternatives; Corridor SLAM-only panel.',
        source_slam_runs='Independent ROS replays; performance depends on configuration and sensor budget.',
        models_reselected=False,training_or_configuration_changes=False,
        timing_scope='One-times paced replay wall time is not pure computation time; correction callback and SLAM child RSS kept separate.',
        sources=sources),indent=2)+'\n')
    print('Unified current online comparison',len(results),'rows')


if __name__=='__main__':run()
