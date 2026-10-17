#!/usr/bin/python3
"""Export and verify validation-selected global and quality+path candidates."""
import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from causal_stream import wrap
from residual_models import Regressor
from trajectory_quality import QualityPortable,PROTOCOL
from tools.train_trajectory_quality import load


def run(args):
    torch.set_num_threads(1)
    selection=json.loads((args.study/'deployment_selection.json').read_text())
    names=[selection['selected']['model']]
    group=[r for r in selection['candidates'] if r['feature_config']=='quality44' and r['trajectory_loss']]
    quality=min(group,key=lambda r:(r['validation_ape_m'],r['model']))['model']
    if quality not in names:names.append(quality)
    manifest=json.loads((args.study/'manifest.json').read_text());rows={r['sequence_id']:r for r in manifest['sequences']}
    sequences=[load(rows[n],args.quality_root) for n in manifest['test']]
    summary=[]
    for name in names:
        checkpoint=args.study/name/'model.pt'
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert state['stream_protocol']==PROTOCOL and not state['training_selection']['test_used']
        directory=args.study/'portable'/name;directory.mkdir(parents=True,exist_ok=True)
        arrays={k:v.numpy() for k,v in state['norm'].items()}
        for i,layer in enumerate([0,2,4,6]):
            for target,source in [('w','weight'),('b','bias')]:arrays[f'{target}{i}']=state['model_state_dict'][f'net.{layer}.{source}'].numpy()
        np.savez_compressed(directory/'portable_weights.npz',**arrays)
        meta=dict(stream_protocol=PROTOCOL,grid_hz=state['grid_hz'],feature_config=state['feature_config'],
            alpha=state['alpha'],gate_parameters=state['gate_parameters'],integration_tau_s=state.get('integration_tau_s'),
            runtime_dependencies=['numpy'],source_checkpoint=str(checkpoint.resolve()),
            source_checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            selection='global validation best' if name==names[0] else 'quality+trajectory group validation best',
            evaluation_status='experimental candidate, public development sequences only',
            input_contract='10 Hz causal sampled poses and twelve source-causal or actual received quality values; no reference inputs',
            quality_names=__import__('trajectory_quality').QUALITY_NAMES)
        (directory/'portable.json').write_text(json.dumps(meta,indent=2)+'\n')
        for filename in ['trajectory_quality.py','causal_stream.py']:
            shutil.copyfile(Path(__file__).resolve().parents[1]/filename,directory/filename)
        portable=QualityPortable(directory)
        model=Regressor(44,'mlp',True);model.load_state_dict(state['model_state_dict']);model.eval()
        random=np.random.default_rng(91).normal(size=(100,44)).astype(np.float32)
        with torch.inference_mode():expected=model(torch.from_numpy(random)).numpy()
        actual=np.array([portable.forward_normalized(z) for z in random])
        np.testing.assert_allclose(actual,expected,atol=2e-5,rtol=2e-5)
        checks=[]
        from trajectory_quality import QualityResidualStream
        def forward(z):
            with torch.inference_mode():return model(torch.from_numpy(z[None])).numpy()[0]
        for d in sequences:
            s=d['seq'];portable.reset();engine=QualityResidualStream(state['feature_config'],portable.norm,
                forward,state['alpha'],state['gate_parameters'],state.get('integration_tau_s'))
            delta=[];timings=[]
            for t,c,o,q,a in zip(s.t,s.cart,s.odom,d['quality'],s.available):
                start=time.perf_counter_ns();p,ps=portable.step(t,c,o,q,bool(a));timings.append((time.perf_counter_ns()-start)/1e6)
                e,es=engine.step(t,c,o,q,bool(a));v=p-e;v[2]=wrap(v[2]);delta.append(v)
                assert ps['fallback']==es['fallback']
            delta=np.array(delta);np.testing.assert_allclose(delta,0,atol=3e-5,rtol=0)
            checks.append(dict(sequence=s.name,frames=len(s.t),max_pose_error=float(np.abs(delta).max()),
                numpy_step_p50_ms=float(np.quantile(timings,.5)),numpy_step_p99_ms=float(np.quantile(timings,.99)),
                timing_scope='host batch-one quality feature/model/gate/integration; excludes scan descriptor, ROS and I/O'))
        meta['verification']=dict(random_cases=100,random_max_error=float(np.abs(actual-expected).max()),captured_sequences=checks)
        meta['artifact_weights_bytes']=(directory/'portable_weights.npz').stat().st_size
        (directory/'portable.json').write_text(json.dumps(meta,indent=2)+'\n')
        summary.append(dict(model=name,package=str(directory),verification=meta['verification'],selection=meta['selection']))
        print('Exported',directory,flush=True)
    (args.study/'portable_verification.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('study',type=Path);p.add_argument('--quality-root',type=Path,required=True)
    run(p.parse_args())
