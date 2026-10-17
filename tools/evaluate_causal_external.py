#!/usr/bin/python3
"""Evaluate a predeclared external stream with frozen weights/validation settings."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from mentor_benchmark import (load_sequence,validate_manifest,metrics,predict,corrected,save_tum,
    covariance_intersection,recursive_filter,fixed_lag,full_factor_graph,ridge_fit,gp_fit)
from residual_models import Regressor


def run(args):
    torch.set_num_threads(1)
    original=json.loads((args.root/'manifest.json').read_text())
    manifest=json.loads(args.manifest.read_text());validate_manifest(manifest)
    assert original['train']==manifest['train'] and original['validation']==manifest['validation']
    declared=set(original.get('external_test',[]))
    if not set(args.sequences).issubset(declared):raise ValueError('External sequences were not declared before training')
    rows={r['sequence_id']:r for r in manifest['sequences']}
    train=[load_sequence(rows[n]) for n in original['train']]
    external=[load_sequence(rows[n]) for n in args.sequences]
    previous=json.loads((args.root/'results.json').read_text());results=[]
    args.output.mkdir(parents=True,exist_ok=True)
    def record(method,seq,pose,seed=None,extra=None):
        stats=metrics(seq,pose);stats.update(method=method,sequence=seq.name,scene=seq.scene,seed=seed,
            trajectory_kind=seq.kind,split='external_test',fit_or_tuning_used_external=False)
        if extra:stats.update(extra)
        directory=args.output/f'{method}_seed{seed}';directory.mkdir(exist_ok=True)
        save_tum(directory/f'{seq.name}.tum',seq,pose);results.append(stats)
    for seq in external:
        record('cartographer',seq,seq.cart);record('wheel_odometry',seq,seq.odom)
    for name,method in [('covariance_intersection',covariance_intersection),('ekf_conservative_noise',recursive_filter),
            ('fixed_lag_factor_graph',fixed_lag),('full_trajectory_factor_graph',full_factor_graph)]:
        parameter=next(r['parameter_selected_on_validation'] for r in previous if r['method']==name)
        for seq in external:record(name,seq,method(seq,parameter),extra={'parameter_selected_on_validation':parameter})
    for name,fit in [('ridge_pose_residual',ridge_fit),('gp_pose_residual_adaptation',gp_fit)]:
        predictor=fit(train,'full20');alpha=next(r['alpha'] for r in previous if r['method']==name)
        for seq in external:record(name,seq,corrected(seq,predictor(seq),alpha),extra={'alpha':alpha})
    for path in sorted(args.root.glob('*/model.pt')):
        state=torch.load(path,map_location='cpu',weights_only=False)
        norm={k:v.numpy() for k,v in state['norm'].items()}
        model=Regressor(len(norm['xm']),'mlp',state['uncertainty']);model.load_state_dict(state['model_state_dict']);model.eval()
        name=path.parent.name.rsplit('_seed',1)[0]
        for seq in external:
            mean,sigma,z=predict(model,norm,seq,state['config'])
            record(name,seq,corrected(seq,mean,state['alpha'],state['target'],integration_tau_s=state.get('integration_tau_s')),state['seed'],{'alpha':state['alpha']})
            if state['uncertainty']:
                gate=state['gate_parameters']
                mask=(sigma/norm['ys']).mean(1)<=gate['sigma_threshold']
                mask &= np.abs(z).max(1)<=gate['ood_threshold']
                mask &= seq.ready[1:]
                record(name+'_hard_fallback',seq,corrected(seq,mean,gate['gate_alpha'],state['target'],mask,state.get('integration_tau_s')),state['seed'],
                       dict(gate,fallback_fraction=1. if gate['gate_alpha']==0 else float((~mask).mean())))
    (args.output/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    (args.output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(f'Saved {len(results)} external rows; no model/threshold tuning on external data')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--sequences',nargs='+',required=True);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
