#!/usr/bin/python3
"""Measure each postprocessor in a fresh process; verify its frozen saved output."""
import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def worker(case):
    import numpy as np
    import torch
    from mentor_benchmark import (load_sequence,load_pose,corrected,predict,
        covariance_intersection,recursive_filter,fixed_lag,full_factor_graph,ridge_fit,gp_fit)
    from residual_models import Regressor
    torch.set_num_threads(1)
    directory=ROOT/'results/mentor_benchmark'/case['run']
    manifest=json.loads((directory/'manifest.json').read_text())
    seq=load_sequence(next(r for r in manifest['sequences'] if r['sequence_id']==case['sequence']))
    row=next(r for r in json.loads((directory/'results.json').read_text()) if r['method']==case['method'] and r['sequence']==case['sequence'] and r['seed'] in (None,11))
    methods={'covariance_intersection':covariance_intersection,'ekf_conservative_noise':recursive_filter,
             'fixed_lag_factor_graph':fixed_lag,'full_trajectory_factor_graph':full_factor_graph}
    name=case['method']
    if name=='cartographer':operation=lambda:seq.cart.copy()
    elif name in methods:operation=lambda:methods[name](seq,row['parameter_selected_on_validation'])
    elif name in ['ridge_pose_residual','gp_pose_residual_adaptation']:
        train=[load_sequence(r) for r in manifest['sequences'] if r['sequence_id'] in manifest['train']]
        predictor=(ridge_fit if name.startswith('ridge') else gp_fit)(train,'full20')
        operation=lambda:corrected(seq,predictor(seq),row['alpha'])
    else:
        model_name=name.removesuffix('_hard_fallback') if hasattr(str,'removesuffix') else name.replace('_hard_fallback','')
        state=torch.load(directory/f'{model_name}_seed11/model.pt',map_location='cpu',weights_only=False)
        norm={k:v.numpy() for k,v in state['norm'].items()}
        model=Regressor(len(norm['xm']),state['kind'],state['uncertainty']);model.load_state_dict(state['model_state_dict']);model.eval()
        def operation():
            mean,sigma,x=predict(model,norm,seq,state['config'])
            alpha=state['alpha'];gate=None
            if name.endswith('_hard_fallback'):
                parameters=state['gate_parameters'];alpha=parameters['gate_alpha']
                gate=((sigma/norm['ys']).mean(1)<=parameters['sigma_threshold'])&(np.abs(x).max(1)<=parameters['ood_threshold'])
            return corrected(seq,mean,alpha,state['target'],gate,state.get('integration_tau_s'))
    actual=operation()
    _,expected=load_pose(directory/f"{name}_seed{row['seed']}"/(seq.name+'.tum'))
    np.testing.assert_allclose(actual,expected,atol=2e-6,rtol=1e-6)
    times=[]
    for _ in range(3):
        started=time.perf_counter();operation();times.append(time.perf_counter()-started)
    return {**case,'seed':row['seed'],'frames':len(seq.t),'warmup':1,'repeats':3,
            'mean_trajectory_wall_s':float(np.mean(times)),
            'mean_ms_per_frame':float(np.mean(times)/len(seq.t)*1000),
            'fresh_process_peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            'memory_scope':'whole fresh worker including common Torch/Scipy framework, data loading and any baseline fitting; not incremental model memory',
            'time_scope':'batch postprocessing of loaded input trajectory; excludes SLAM, input loading and training',
            'output_verified_against_saved_trajectory':True}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker')
    parser.add_argument('--run',help='One explicit benchmark run with its own frozen outputs')
    parser.add_argument('--methods',nargs='+')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.worker:
        print(json.dumps(worker(json.loads(args.worker))));return
    methods=['cartographer','covariance_intersection','ekf_conservative_noise','fixed_lag_factor_graph',
             'full_trajectory_factor_graph','ridge_pose_residual','gp_pose_residual_adaptation',
             'mlp_full20_pose','mlp_relative32_increment','gru_relative32_increment','tcn_relative32_increment',
             'mlp_relative32_increment_uncertainty_hard_fallback']
    if args.methods:methods=args.methods
    cases=[{'run':run,'sequence':seq,'method':method} for run,seq in [
        ('home_online_inputs','home1-3'),('cross_scene_and_nclt','corridor1-3'),
        ('cross_scene_and_nclt','nclt-2013-01-10')] for method in methods]
    if args.run:
        manifest=json.loads((ROOT/'results/mentor_benchmark'/args.run/'manifest.json').read_text())
        cases=[dict(run=args.run,sequence=seq,method=method) for seq in manifest['test'] for method in methods]
    path=args.output or ROOT/'results/mentor_benchmark/postprocessing_fresh_process_resources.json'
    path.parent.mkdir(parents=True,exist_ok=True)
    results=json.loads(path.read_text()) if path.exists() else []
    for case in cases:
        if any(all(row[k]==v for k,v in case.items()) for row in results):continue
        completed=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',json.dumps(case)],
                                 cwd=ROOT,check=True,capture_output=True,text=True,
                                 env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
        row=json.loads(completed.stdout);results.append(row)
        path.write_text(json.dumps(results,indent=2)+'\n')
        print(f"Measured {case['sequence']}/{case['method']}: {row['fresh_process_peak_rss_mib']:.1f} MiB",flush=True)
if __name__=='__main__':main()
