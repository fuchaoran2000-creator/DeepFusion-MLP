#!/usr/bin/python3
"""Recover input-only gate thresholds for frozen pooled selections, without NCLT."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from residual_models import Regressor
from tools.train_trajectory_quality import load,prediction,dump
from tools.train_scene_rotation import balanced_quantile


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('study',type=Path);p.add_argument('--quality-root',type=Path,required=True);args=p.parse_args()
    torch.set_num_threads(1)
    root=args.study/'test_corridor_val_cafe';manifest=json.loads((root/'manifest.json').read_text())
    data=[load(r,args.quality_root) for r in manifest['sequences'] if r['scene_id'] in ['cafe','corridor']]
    selection=json.loads((args.study/'pooled_selection.json').read_text());output={}
    for name in sorted({r['model'] for r in selection['candidates']}):
        checkpoint=root/name/'model.pt';state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        model=Regressor(44,'mlp',True);model.load_state_dict(state['model_state_dict']);model.eval()
        norm={k:v.numpy() for k,v in state['norm'].items()};ps=[prediction(model,norm,d,'quality44') for d in data]
        sg=[];zg=[]
        for scene in ['cafe','corridor']:
            ss=[];zz=[]
            for d,pred in zip(data,ps):
                if d['seq'].scene==scene:
                    ready=d['seq'].ready[1:];ss.append(pred[1][ready].mean(1));zz.append(np.abs(pred[2][ready]).max(1))
            sg.append(ss);zg.append(zz)
        pooled=dict(sigma_threshold=balanced_quantile(sg,.9),ood_threshold=balanced_quantile(zg,.99),
            fit_split='cafe_corridor_validation',weights='equal scenes, equal sequences within scene, equal ready frames within sequence')
        for c in selection['candidates']:
            if c['model']!=name:continue
            gate=state['gate_parameters'] if c['protocol'].endswith('_training_gate') else pooled
            output[c['protocol']+'/'+name]=dict(choice={k:c[k] for k in ['alpha','tau_s','validation_score']},
                gate=dict(gate,gate_alpha=c['alpha']),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    path=args.study/'pooled_calibration.json'
    if path.exists():assert json.loads(path.read_text())==output,'Persisted gate thresholds differ'
    dump(path,output);print('Stored',len(output),'frozen gate/alpha/tau records; NCLT not loaded',flush=True)
