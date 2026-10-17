#!/usr/bin/env python3
"""Run fixed numerical checks, retraining smoke, or clean frozen transfer."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]

def run(command,cwd=ROOT):
    return subprocess.run(command,cwd=cwd,check=True,text=True,capture_output=True,
      env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-smoke',action='store_true',help='Two training epochs on recorded Home inputs, not a paper-score claim')
    p.add_argument('--transfer',action='store_true',help='Re-evaluate clean C20 transfer using all archived target inputs')
    p.add_argument('--output',type=Path,default=ROOT/'results/public_reproduction')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    report={'scope':'Frozen numerical checks; no source SLAM rerun.','passed':False}
    source=ROOT/'reproduction/C20'
    hashes=json.loads((source/'artifact_sha256.json').read_text())
    for name,h in hashes.items():assert hashlib.sha256((source/name).read_bytes()).hexdigest()==h,name
    with tempfile.TemporaryDirectory(prefix='c20_replay_') as temp:
        replay=Path(temp)/'C20';shutil.copytree(source,replay)
        run([sys.executable,str(replay/'replay_c20.py')],cwd=replay)
        report['table4']=json.loads((replay/'replay_verification.json').read_text())
    report['table7']=json.loads(run([sys.executable,str(ROOT/'reproduction/transfer/recompute_table7.py')]).stdout)
    assert report['table4']['passed'] and report['table7']['passed']
    if args.train_smoke:
        command=[sys.executable,'mentor_benchmark.py','--manifest','config/recorded/benchmark_home_causal_v1.json',
          '--output',str(args.output/'training_smoke'),'--configs','full20','--seeds','11','--epochs','2','--no-extra-models','--with-uncertainty']
        result=run(command);(args.output/'training_smoke.log').write_text(result.stdout+result.stderr)
        report['training_smoke']={'passed':True,'epochs':2,'scope':'Execution and data portability only; not a retrained reproduction of paper accuracy.'}
    if args.transfer:
        protocol=json.loads((ROOT/'config/causal_gate_stress_v1.json').read_text());protocol['corruptions']={'clean':{}}
        config=args.output/'clean_protocol.json';config.write_text(json.dumps(protocol,indent=2)+'\n')
        checkpoint=ROOT/'artifacts/checkpoints/home_causal_v1/mlp_full20_pose_uncertainty_seed11/model.pt'
        sequences=['office1-'+str(i) for i in range(1,8)]+['cafe1-1','cafe1-2','corridor1-3','corridor1-4','corridor1-5','nclt-2013-01-10']
        for name,manifest,selected in [('initial','benchmark_cross_scene_causal_v1.json',sequences),
                                       ('adapted','benchmark_cross_scene_frontend_v2.json',['nclt-2013-01-10'])]:
            command=[sys.executable,'tools/benchmark_gate_stress.py','--manifest','config/recorded/'+manifest,
               '--split','all','--sequences',*selected,'--protocol',str(config),'--checkpoints',str(checkpoint),'--output',str(args.output/('transfer_'+name))]
            result=run(command);(args.output/f'transfer_{name}.log').write_text(result.stdout+result.stderr)
        report['transfer_inference']={'passed':True,'scope':'New inference on recorded inputs with frozen C20, not a new ROS execution.'}
    report['passed']=True
    (args.output/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
