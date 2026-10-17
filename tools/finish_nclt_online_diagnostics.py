#!/usr/bin/python3
"""Finish the predeclared frozen-model evaluation after the live capture closes."""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
R='results/mentor_benchmark'


def run(args):
    capture=ROOT/R/'nclt_frontend_diagnostic_v2/nclt-2013-01-10_ceres10_loop/cartographer_noimu_online_resources_frontend_v2.json'
    if args.wait_live:
        start=time.monotonic()
        while not capture.exists():
            if time.monotonic()-start>1800:raise TimeoutError('Live capture did not finish within 30 minutes')
            time.sleep(5)
    if not capture.exists():raise ValueError('Complete live resource artifact is required')
    assert json.loads(capture.read_text())['shutdown_success']
    env=dict(os.environ,MPLCONFIGDIR='/tmp/nclt_mpl')
    def call(tool,*arguments):
        print('Running',tool,flush=True)
        subprocess.run(['/usr/bin/python3','-u',tool,*arguments],cwd=ROOT,env=env,check=True)
    manifest='config/benchmark_cross_scene_frontend_v2.json'
    call('tools/verify_causal_stream.py','--manifest',manifest,'--split','test','--output',
         f'{R}/nclt_frontend_diagnostic_v2/input_equivalence.json')
    call('tools/report_nclt_online_frontend.py')
    call('tools/evaluate_causal_external.py','--root',f'{R}/cross_scene_causal_v1','--manifest',manifest,
         '--sequences','nclt-2013-01-10','--output',f'{R}/cross_scene_causal_frontend_external_v2')
    home=[f'{R}/home_causal_v1/mlp_full20_pose_uncertainty_seed11/model.pt',
          f'{R}/home_causal_v1/mlp_relative32_increment_uncertainty_seed22/model.pt']
    cross=[f'{R}/cross_scene_causal_v1/mlp_{feature}_{target}_uncertainty_seed{seed}/model.pt'
           for feature,target in [('full20','pose'),('relative32','increment')] for seed in [11,22,33]]
    for checkpoints,name in [(home,'home_frozen_transfer_nclt_frontend_v2'),(cross,'cross_scene_gate_stress_nclt_frontend_v2')]:
        call('tools/benchmark_gate_stress.py','--manifest',manifest,'--sequences','nclt-2013-01-10',
             '--checkpoints',*checkpoints,'--output',f'{R}/{name}')
    call('tools/write_nclt_diagnostic_report.py')
    print('NCLT live frontend and frozen-model diagnostic work complete',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--wait-live',action='store_true');run(p.parse_args())
