#!/usr/bin/python3
"""Verify saved actual-arrival quality, features and publication before GT scoring."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from causal_stream import wrap,local
from trajectory_quality import QualityResidualStream,QualityPortable,extract_quality,PROTOCOL
from residual_models import Regressor
from mentor_benchmark import load_sequence,load_pose,corrected,metrics


def run(args):
    torch.set_num_threads(1)
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    assert state['stream_protocol']==PROTOCOL
    norm={k:v.numpy() for k,v in state['norm'].items()}
    model=Regressor(44,'mlp',True);model.load_state_dict(state['model_state_dict']);model.eval()
    def forward(z):
        with torch.inference_mode():return model(torch.from_numpy(z[None])).numpy()[0]
    with np.load(args.capture,allow_pickle=False) as d:data={k:d[k].copy() for k in d.files}
    assert str(data['feature_protocol'])==PROTOCOL and str(data['quality_arrival_scope'])=='actual received scan callbacks'
    t,cart,odom,q,available=[data[k] for k in ['t','cart','odom','quality','available']]
    scan_present=q[:,11]==1
    assert (data['scan_source_t'][scan_present]<=t[scan_present]).all()
    assert (q[scan_present,2]>=0).all() and (q[scan_present,2]<=.15).all()
    np.testing.assert_allclose(q[:,0],t-data['cart_source_t'],atol=1e-7)
    finite=np.isfinite(data['odom_source_t'])
    np.testing.assert_allclose(q[finite,1],t[finite]-data['odom_source_t'][finite],atol=1e-7)
    features,ready=extract_quality(t,cart,odom,available,q,state['feature_config'])
    np.testing.assert_array_equal(ready,data['ready']);np.testing.assert_array_equal(features,data['features'])
    engine=QualityResidualStream(state['feature_config'],norm,forward,state['alpha'],state['gate_parameters'],state.get('integration_tau_s'))
    out=[];fallback=[]
    portable=QualityPortable(args.portable) if args.portable else None
    portable_error=[]
    for tick,c,o,quality,a in zip(t,cart,odom,q,available):
        value,status=engine.step(tick,c,o,quality,bool(a));out.append(value);fallback.append(status['fallback'])
        if portable:
            value2,status2=portable.step(tick,c,o,quality,bool(a));v=value2-value;v[2]=wrap(v[2]);portable_error.append(v)
            assert status2['fallback']==status['fallback']
    out=np.array(out);fallback=np.array(fallback)
    delta=out-data['output'];delta[:,2]=wrap(delta[:,2]);np.testing.assert_allclose(delta,0,atol=3e-5,rtol=0)
    np.testing.assert_array_equal(fallback,data['fallback']);np.testing.assert_array_equal(out[fallback],cart[fallback])
    publication=args.capture.with_name(args.capture.name.replace('_sampled_stream','_corrected_online')).with_suffix('.tum')
    pt,pp=load_pose(publication);np.testing.assert_allclose(pt,t,atol=1e-8,rtol=0)
    pd=pp-data['output'];pd[:,2]=wrap(pd[:,2]);np.testing.assert_allclose(pd,0,atol=1e-8,rtol=0)
    checks=dict(actual_arrival_feature_error_max=float(np.abs(features-data['features']).max()),
        torch_replay_pose_error_max=float(np.abs(delta).max()),published_pose_error_max=float(np.abs(pd).max()),
        exact_raw_fallback=True,frame_count=len(t),ready_fraction=float(ready.mean()),
        scan_present_fraction=float(scan_present.mean()),fallback_fraction=float(fallback.mean()))
    if portable:
        np.testing.assert_allclose(portable_error,0,atol=3e-5,rtol=0)
        checks['numpy_replay_pose_error_max']=float(np.abs(portable_error).max())
    # Only after runtime verification do reference files enter scoring.
    manifest=json.loads((args.checkpoint.parent.parent/'manifest.json').read_text())
    row=next(r.copy() for r in manifest['sequences'] if r['sequence_id']==args.sequence)
    row['sampled_stream']=str(args.capture.resolve());seq=load_sequence(row)
    seq.ready=ready
    z=((features[1:]-norm['xm'])/norm['xs']).astype(np.float32)
    with torch.inference_mode():pred=model(torch.from_numpy(z)).numpy()
    mean=np.clip(pred[:,:3]*norm['ys']+norm['ym'],-norm['clip'],norm['clip']);mean[:,2]=wrap(mean[:,2])
    gate=state['gate_parameters']
    mask=ready[1:]&(np.abs(z).max(1)<=gate['ood_threshold'])&(np.exp(.5*np.clip(pred[:,3:],-8,6)).mean(1)<=gate['sigma_threshold'])
    batch=corrected(seq,mean,gate['gate_alpha'],'increment',mask,state.get('integration_tau_s'))
    first=int(np.flatnonzero(available)[0]);origin=cart[first]
    published_local=np.array([local(p,origin) for p in data['output']])
    d=published_local-batch;d[:,2]=wrap(d[:,2]);np.testing.assert_allclose(d,0,atol=3e-5,rtol=0)
    checks['batch_vs_publication_error_max']=float(np.abs(d).max())
    raw=metrics(seq,seq.cart);corr=metrics(seq,published_local)
    rawerror=np.linalg.norm(seq.cart[:,:2]-seq.gt[:,:2],axis=1)
    error=np.linalg.norm(published_local[:,:2]-seq.gt[:,:2],axis=1)
    report=dict(sequence=args.sequence,checks=checks,raw_metrics=raw,corrected_metrics=corr,
        harmful_fraction_over_5cm=float(((error-rawerror)[seq.valid]>.05).mean()),
        source_checkpoint=str(args.checkpoint.resolve()),checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        capture_sha256=hashlib.sha256(args.capture.read_bytes()).hexdigest(),
        scope='same-run paired raw/corrected ROS output; development verification; no feedback to SLAM',
        reference_input=False)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(checks,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--capture',type=Path,required=True)
    p.add_argument('--sequence',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--portable',type=Path)
    run(p.parse_args())
