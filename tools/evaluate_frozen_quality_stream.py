#!/usr/bin/python3
"""Evaluate a frozen quality model on one actual-arrival ROS capture."""
import argparse
import copy
import csv
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from causal_stream import wrap,local
from trajectory_quality import QualityResidualStream,extract_quality,PROTOCOL
from residual_models import Regressor
from mentor_benchmark import load_sequence,load_pose,metrics,interpolate


def reference_path(directory):
    path=directory/'groundtruth.tum'
    return path if path.exists() else directory/'groundtruth.tum.gz'


def run(args):
    torch.set_num_threads(1)
    if args.plan:
        plan=json.loads(args.plan.read_text())
        assert hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()==plan['checkpoint_sha256']
        configuration=args.directory/f'cartographer_noimu_configuration_{args.label}.lua'
        assert hashlib.sha256(configuration.read_bytes()).hexdigest()==plan['configuration_sha256']
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    assert state['stream_protocol']==PROTOCOL
    norm={k:v.numpy() for k,v in state['norm'].items()}
    model=Regressor(44,'mlp',True);model.load_state_dict(state['model_state_dict']);model.eval()
    def forward(z):
        with torch.inference_mode():return model(torch.from_numpy(z[None])).numpy()[0]
    capture=args.directory/f'cartographer_noimu_sampled_stream_{args.label}.npz'
    with np.load(capture,allow_pickle=False) as d:data={k:d[k].copy() for k in d.files}
    assert str(data['feature_protocol'])==PROTOCOL
    assert str(data['quality_arrival_scope'])=='actual received scan callbacks'
    t,cart,odom,q,available=[data[k] for k in ['t','cart','odom','quality','available']]
    x,ready=extract_quality(t,cart,odom,available,q,state['feature_config'])
    np.testing.assert_array_equal(x,data['features']);np.testing.assert_array_equal(ready,data['ready'])
    present=q[:,11]==1
    assert np.all(data['scan_source_t'][present]<=t[present])
    if args.same_strength:
        state=dict(state,alpha=state['gate_parameters']['gate_alpha'])
    outputs={};flags={};reasons={};effective_alpha_by_gate={}
    for name in ['none','rule','uncertainty']:
        gate=None if name=='none' else dict(state['gate_parameters'])
        if name=='rule':gate['sigma_threshold']=1e30
        engine=QualityResidualStream(state['feature_config'],norm,forward,state['alpha'],gate,state.get('integration_tau_s'))
        # ResidualStream uses gate_alpha when a gate is present; reporting the
        # checkpoint's ungated alpha for every condition mislabels the control.
        effective_alpha_by_gate[name]=float(engine.engine.alpha)
        poses=[];fallback=[];why=[]
        for tick,c,o,quality,a in zip(t,cart,odom,q,available):
            pose,status=engine.step(tick,c,o,quality,bool(a))
            poses.append(pose);fallback.append(status['fallback']);why.append(status['reason'])
        poses=np.array(poses);fallback=np.array(fallback)
        np.testing.assert_array_equal(poses[fallback],cart[fallback])
        outputs[name]=poses;flags[name]=fallback;reasons[name]=why
    difference=outputs['uncertainty']-data['output'];difference[:,2]=wrap(difference[:,2])
    np.testing.assert_allclose(difference,0,atol=3e-5,rtol=0)
    np.testing.assert_array_equal(flags['uncertainty'],data['fallback'])
    pt,pp=load_pose(args.directory/f'cartographer_noimu_corrected_online_{args.label}.tum')
    np.testing.assert_allclose(pt,t,atol=1e-8,rtol=0)
    pd=pp-data['output'];pd[:,2]=wrap(pd[:,2]);np.testing.assert_allclose(pd,0,atol=1e-8,rtol=0)
    inference=dict(feature_error_max=float(np.abs(x-data['features']).max()),
                   torch_replay_pose_error_max=float(np.abs(difference).max()),
                   publication_error_max=float(np.abs(pd).max()),exact_raw_fallback=True,
                   effective_alpha_by_gate=effective_alpha_by_gate,
                   same_alpha_and_tau_for_all_gates=len(set(effective_alpha_by_gate.values()))==1,
                   reference_access_after_inference=True)
    # Only now read reference poses. All choices remain frozen.
    row=dict(sequence_id=args.directory.name,scene_id='nclt' if args.directory.name.startswith('nclt-') else 'home',
             cart=str(args.directory/f'cartographer_noimu_online_{args.label}.tum'),
             odom=str(args.directory/'odom.tum'),gt=str(reference_path(args.directory)),
             trajectory_kind='online',sampled_stream=str(capture),stream_protocol='causal_grid_v1')
    seq=load_sequence(row);seq.ready=ready
    origin=cart[int(np.flatnonzero(available)[0])]
    local_outputs={name:np.array([local(p,origin) for p in pose]) for name,pose in outputs.items()}
    local_outputs['cartographer']=seq.cart;local_outputs['provided_odometry']=seq.odom
    rawerror=np.linalg.norm(seq.cart[:,:2]-seq.gt[:,:2],axis=1)
    raw=metrics(seq,seq.cart);results=[];full_metrics={};events=[]
    gt_t,gt_p=load_pose(reference_path(args.directory))
    ri=np.searchsorted(gt_t,t);left=np.clip(ri-1,0,len(gt_t)-1);right=np.clip(ri,0,len(gt_t)-1)
    all_ref=(t>=gt_t[0])&(t<=gt_t[-1])&((gt_t[right]-gt_t[left])<=.25)
    for name,pose in local_outputs.items():
        m=metrics(seq,pose);full_metrics[name]=m
        error=np.linalg.norm(pose[:,:2]-seq.gt[:,:2],axis=1)
        results.append(dict(sequence=seq.name,method=name,alpha=effective_alpha_by_gate[name] if name in outputs else None,
            tau_s=state.get('integration_tau_s') if name in outputs else None,
            ape_m=m['ape_m']['rmse'],all_reference_ape_m=float(np.sqrt(np.mean(error[all_ref]**2))),
            rpe_1s_m=m['rpe']['1s']['translation_m']['rmse'],rpe_1s_deg=m['rpe']['1s']['rotation_deg']['rmse'],
            rpe_5s_m=m['rpe']['5s']['translation_m']['rmse'],
            harmful_fraction_over_5cm=float(((error-rawerror)[seq.valid]>.05).mean()) if name in outputs else None,
            fallback_fraction=float(flags[name].mean()) if name in outputs else None,
            input_available_fraction=float(available.mean()),reference_valid_fraction=float(seq.valid.mean()),
            all_reference_fraction=float(all_ref.mean()),grid_frames=len(t),output_fraction=1.0))
    fault_path=args.directory/'fault_plan.json'
    if fault_path.exists():
        faults=json.loads(fault_path.read_text());affected=np.zeros(len(t),bool)
        for event in faults['events']:
            start=faults['source_start_t']+event['start_offset_s'];end=faults['source_start_t']+event['end_offset_s']
            interval=(t>=start)&(t<end);affected|=interval
            for name in outputs:
                rejected=np.flatnonzero(interval&flags[name]);recovery=np.flatnonzero((t>=end)&~flags[name])
                events.append(dict(event=event['kind'],gate=name,grid_samples=int(interval.sum()),
                    interval_fallback_fraction=float(flags[name][interval].mean()) if interval.any() else None,
                    first_fallback_offset_s=float(t[rejected[0]]-start) if len(rejected) else None,
                    first_accept_after_event_s=float(t[recovery[0]]-end) if len(recovery) else None,
                    input_available_fraction=float(available[interval].mean()) if interval.any() else None,
                    measurement_scope='Source-time grid offsets at 10Hz, not wall-clock fault detection or calibrated false positives.'))
        inference['outside_fault_fallback_fraction']={n:float(f[~affected].mean()) for n,f in flags.items()}
    node_results=[]
    if seq.name.startswith('nclt-'):
        cov=Path('data/nclt')/('cov_'+seq.name[5:]+'.csv')
        if cov.exists():
            nt=np.loadtxt(cov,delimiter=',',usecols=0)/1e6
            nr=np.searchsorted(gt_t,nt);nr=np.clip(nr,0,len(gt_t)-1);nl=np.maximum(nr-1,0)
            ni=np.where(np.abs(gt_t[nl]-nt)<np.abs(gt_t[nr]-nt),nl,nr)
            bracket=np.searchsorted(t,nt);bracket=np.clip(bracket,1,len(t)-1)
            keep=(nt>=t[0])&(nt<=t[-1])&(np.abs(gt_t[ni]-nt)<=.0001)&seq.valid[bracket]&seq.valid[bracket-1]
            nt=nt[keep];truth=gt_p[ni[keep]].copy()
            anchor=int(np.flatnonzero(seq.valid)[0]);a=interpolate(np.array([t[anchor]]),gt_t,gt_p)[0]
            delta=seq.cart[anchor,2]-a[2];c,s=np.cos(delta),np.sin(delta)
            truth[:,:2]=(truth[:,:2]-a[:2])@np.array([[c,s],[-s,c]])+seq.cart[anchor,:2]
            truth[:,2]=wrap(truth[:,2]-a[2]+seq.cart[anchor,2])
            for name,pose in local_outputs.items():
                estimate=interpolate(nt,t,pose)
                node_results.append(dict(method=name,node_count=len(nt),graph_node_ape_m=
                    float(np.sqrt(np.mean(np.sum((estimate[:,:2]-truth[:,:2])**2,axis=1)))) if len(nt) else None,
                    scope='Reference samples matching covariance graph-node timestamps within 100us; linear estimate interpolation on valid grid brackets; SLAM-derived reference.'))
    report=dict(sequence=seq.name,checks=inference,results=results,metrics=full_metrics,events=events,
                graph_nodes=node_results,checkpoint=str(args.checkpoint),
                checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                capture_sha256=hashlib.sha256(capture.read_bytes()).hexdigest(),
                reference_in_inference=False,retuned_after_evaluation=False,
                correction_scope='Output-only, no feedback to SLAM; gate none retains mandatory input/history guards.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    for suffix,values in [('results',results),('events',events),('graph_nodes',node_results)]:
        if values:
            with args.output.with_name(args.output.stem+'_'+suffix+'.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
    print(json.dumps(dict(sequence=seq.name,checks=inference,results=results),indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--label',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--plan',type=Path)
    p.add_argument('--same-strength',action='store_true',help='Use checkpoint gate_alpha for all three policies, including none')
    run(p.parse_args())
