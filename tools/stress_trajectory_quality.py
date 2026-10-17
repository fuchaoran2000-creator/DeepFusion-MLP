#!/usr/bin/python3
"""Frozen candidates: missing scans and no-return descriptors at equal alpha."""
import argparse
import copy
import csv
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from trajectory_quality import extract_quality
from residual_models import Regressor
from mentor_benchmark import corrected,metrics
from tools.train_trajectory_quality import load,prediction,masks,score_ape


def run(args):
    torch.set_num_threads(1)
    exports=json.loads((args.study/'portable_verification.json').read_text())
    manifest=json.loads((args.study/'manifest.json').read_text());rows={r['sequence_id']:r for r in manifest['sequences']}
    sequences=[load(rows[n],args.quality_root) for n in manifest['test']]
    results=[];reset_checks=[]
    for export in exports:
        name=export['model'];state=torch.load(args.study/name/'model.pt',map_location='cpu',weights_only=False)
        norm={k:v.numpy() for k,v in state['norm'].items()};model=Regressor(44,'mlp',True)
        model.load_state_dict(state['model_state_dict']);model.eval();gate=state['gate_parameters']
        for source in sequences:
            for case in ['clean','scan_missing_2s','scan_no_returns_2s']:
                d=copy.deepcopy(source);s=d['seq'];q=d['quality'];start=s.t[len(s.t)//2];interval=(s.t>=start)&(s.t<start+2.)
                if case=='scan_missing_2s':q[interval,11]=0.;q[interval,2]=1.15
                elif case=='scan_no_returns_2s':q[interval,3:11]=0.
                x,ready=extract_quality(s.t,s.cart,s.odom,s.available,q,state['feature_config'])
                d['x'][state['feature_config']]=x[1:];s.ready=ready
                p=prediction(model,norm,d,state['feature_config']);ms=masks(d,p,gate)
                for method,mask in ms.items():
                    pose=corrected(s,p[0],gate['gate_alpha'],'increment',mask,state.get('integration_tau_s'))
                    active=np.r_[False,mask&(gate['gate_alpha']!=0)]
                    np.testing.assert_array_equal(pose[~active],s.cart[~active])
                    if case=='scan_missing_2s':
                        np.testing.assert_array_equal(pose[interval],s.cart[interval])
                        after=np.flatnonzero(s.t>=start+2.)
                        if len(after):np.testing.assert_array_equal(pose[after[0]],s.cart[after[0]])
                        reset_checks.append(dict(model=name,sequence=s.name,method=method,exact_raw_during_outage_and_first_recovery=True))
                    rawerror=np.linalg.norm(s.cart[:,:2]-s.gt[:,:2],axis=1)
                    error=np.linalg.norm(pose[:,:2]-s.gt[:,:2],axis=1)
                    m=metrics(s,pose)
                    results.append(dict(model=name,sequence=s.name,scene=s.scene,case=case,gate=method,
                        alpha=gate['gate_alpha'],tau_s=state.get('integration_tau_s'),raw_ape_m=score_ape(s,s.cart),
                        ape_m=m['ape_m']['rmse'],rpe_1s_m=m['rpe']['1s']['translation_m']['rmse'],
                        harmful_fraction=float(((error-rawerror)[s.valid]>.05).mean()),fallback_fraction=float((~active).mean()),
                        interval_fallback_fraction=float((~active)[interval].mean()),
                        gt_mask_changed=False,selection_uses_test=False))
    with (args.study/'quality_stress.csv').open('w',newline='') as out:
        w=csv.DictWriter(out,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
    (args.study/'quality_stress_protocol.json').write_text(json.dumps(dict(cases=len(results),
        scope='descriptor/input availability perturbations only; Cartographer raw trajectory and reference mask fixed; no sensor/frontend replay perturbation',
        equal_alpha_and_tau_across_gates=True,reset_checks=reset_checks),indent=2)+'\n')
    print(args.study,len(results),'quality perturbation cases',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('study',type=Path);p.add_argument('--quality-root',type=Path,required=True)
    run(p.parse_args())
