#!/usr/bin/python3
"""Record validation-only alpha curves/calibration; do not alter checkpoints."""
import argparse
import csv
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from mentor_benchmark import load_sequence,predict,targets,corrected,metrics
from residual_models import Regressor


def main(root):
    torch.set_num_threads(1)
    manifest=json.loads((root/'manifest.json').read_text())
    sequences=[load_sequence(r) for r in manifest['sequences'] if r['sequence_id'] in manifest['validation']]
    rows=[];curves=[]
    for path in sorted(root.glob('*/model.pt')):
        state=torch.load(path,map_location='cpu',weights_only=False)
        norm={k:v.numpy() for k,v in state['norm'].items()}
        model=Regressor(len(norm['xm']),'mlp',state['uncertainty']);model.load_state_dict(state['model_state_dict']);model.eval()
        predictions=[predict(model,norm,s,state['config']) for s in sequences]
        errors=[];zero_errors=[];sigma_error=[];coverage=[];gate_parameters=state['gate_parameters']
        gates={'none':[s.ready[1:].copy() for s in sequences]}
        if gate_parameters:
            gates['rule']=[s.ready[1:] & (np.abs(p[2]).max(1)<=gate_parameters['ood_threshold']) for s,p in zip(sequences,predictions)]
            gates['uncertainty']=[m & ((p[1]/norm['ys']).mean(1)<=gate_parameters['sigma_threshold']) for m,p in zip(gates['rule'],predictions)]
        for s,(mean,sigma,z) in zip(sequences,predictions):
            labels,valid=targets(s,state['target']);error=(labels-mean)/norm['ys']
            errors.append(np.sum(error[valid]**2,axis=1));zero_errors.append(np.sum((labels[valid]/norm['ys'])**2,axis=1))
            if sigma is not None:
                sigma_error.append(np.column_stack([(sigma[valid]/norm['ys']).mean(1),np.linalg.norm(error[valid],axis=1)]))
                standardized=(labels[valid]-mean[valid])/np.maximum(sigma[valid]*gate_parameters['variance_calibration_scale'],1e-8)
                coverage.append(np.abs(standardized)<=1.95996)
        for mode,masks in gates.items():
            for alpha in np.linspace(0,1,21):
                ape=np.mean([metrics(s,corrected(s,p[0],float(alpha),state['target'],mask,state.get('integration_tau_s')))['ape_m']['rmse'] for s,p,mask in zip(sequences,predictions,masks)])
                curves.append({'model':path.parent.name,'gate_mode':mode,'alpha':float(alpha),'validation_macro_ape_m':float(ape)})
        corr=None
        if sigma_error:
            values=np.concatenate(sigma_error)
            if values[:,0].std()>1e-10 and values[:,1].std()>1e-10:corr=float(np.corrcoef(values.T)[0,1])
        rows.append({'model':path.parent.name,'config':state['config'],'seed':state['seed'],'uncertainty':state['uncertainty'],
            'validation_sequences':len(sequences),'validation_raw_ape_m':float(np.mean([metrics(s,s.cart)['ape_m']['rmse'] for s in sequences])),
            'selected_alpha':state['alpha'],'selected_gate_alpha':None if not gate_parameters else gate_parameters['gate_alpha'],
            'residual_rmse_relative_to_zero_correction':float(np.sqrt(np.mean(np.concatenate(errors))/np.mean(np.concatenate(zero_errors)))),
            'sigma_error_correlation':corr,'calibrated_95pct_coverage':float(np.concatenate(coverage).mean()) if coverage else None,
            'scope':'validation diagnostic only; checkpoints and deployed settings unchanged'})
    for name,data in [('validation_residual_diagnostics.csv',rows),('validation_alpha_curves.csv',curves)]:
        with (root/name).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    print(json.dumps(rows,ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);main(p.parse_args().root)
