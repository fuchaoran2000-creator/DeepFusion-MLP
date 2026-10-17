#!/usr/bin/python3
"""Frozen-model memory experiment: select tau on validation, then evaluate.

Alpha, learned weights, normalizers and uncertainty thresholds remain frozen.
The force-alpha=1 curves are validation diagnostics only, never selection.
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from mentor_benchmark import load_sequence,predict,targets,corrected,metrics,save_tum
from residual_models import Regressor

TAUS=[None,.5,1.,2.,5.,10.,30.]


def write_csv(path,rows):
    with path.open('w',newline='') as out:
        writer=csv.DictWriter(out,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def run(args):
    torch.set_num_threads(1);args.output.mkdir(parents=True,exist_ok=True)
    frozen=json.loads((args.root/'manifest.json').read_text())
    all_rows=json.loads(args.manifest.read_text())['sequences']
    rows={r['sequence_id']:r for r in all_rows}
    if set(args.evaluate) & set(frozen['train']+frozen['validation']):
        raise ValueError('Evaluation cannot overlap training/validation')
    # Save the candidate grid and split before loading evaluation references.
    plan=dict(tau_candidates_s=TAUS,selection='minimum validation macro APE; exact ties favor original unlimited integration',
              frozen=['weights','alpha','normalization','clip','gate thresholds'],
              train=frozen['train'],validation=frozen['validation'],evaluate=args.evaluate,
              evaluation_kind='previously inspected public development sequences',
              sources_root=str(args.root.resolve()))
    (args.output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    validation=[load_sequence(rows[n]) for n in frozen['validation']]
    selection=[];curves=[];diagnostics=[];states=[]
    for checkpoint in sorted(args.root.glob('mlp_relative32_increment_uncertainty_seed*/model.pt')):
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        norm={k:v.numpy() for k,v in state['norm'].items()}
        model=Regressor(32,'mlp',True);model.load_state_dict(state['model_state_dict']);model.eval()
        gate=state['gate_parameters'];alpha=gate['gate_alpha']
        vp=[predict(model,norm,s,'relative32') for s in validation]
        def mask(seq,p):
            return seq.ready[1:] & (np.abs(p[2]).max(1)<=gate['ood_threshold']) & ((p[1]/norm['ys']).mean(1)<=gate['sigma_threshold'])
        masks=[mask(s,p) for s,p in zip(validation,vp)]
        scores=[]
        for tau in TAUS:
            scores.append(float(np.mean([metrics(s,corrected(s,p[0],alpha,'increment',m,tau))['ape_m']['rmse']
                                         for s,p,m in zip(validation,vp,masks)])))
            forced=float(np.mean([metrics(s,corrected(s,p[0],1.,'increment',s.ready[1:],tau))['ape_m']['rmse'] for s,p in zip(validation,vp)]))
            curves.append(dict(model=checkpoint.parent.name,tau_s=tau,validation_gated_ape_m=scores[-1],
                               forced_alpha1_ungated_validation_ape_m=forced,forced_curve_used_for_selection=False))
        index=int(np.argmin(scores));tau=TAUS[index]
        entry=dict(model=checkpoint.parent.name,seed=state['seed'],checkpoint=str(checkpoint.resolve()),
                   checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),alpha=alpha,
                   selected_tau_s=tau,validation_ape_m=scores[index],original_validation_ape_m=scores[0],
                   selection_used_test=False)
        selection.append(entry);states.append((state,model,norm,tau,checkpoint))
        for seq,p in zip(validation,vp):
            y,valid=targets(seq,'increment');e=p[0]-y
            rmse=np.sqrt(np.mean((e[valid]/norm['ys'])**2));zero=np.sqrt(np.mean((y[valid]/norm['ys'])**2))
            diagnostics.append(dict(model=checkpoint.parent.name,sequence=seq.name,split='validation',
                normalized_residual_rmse_ratio_to_zero=float(rmse/zero),
                residual_error_mean_body_x_m=float(e[valid,0].mean()),residual_error_mean_body_y_m=float(e[valid,1].mean()),
                residual_error_mean_yaw_deg=float(np.degrees(e[valid,2].mean())),
                label_exceeds_training_clip_fraction=float(np.any(np.abs(y[valid])>norm['clip'],axis=1).mean()),
                samples=int(valid.sum())))
    if len(selection)!=3:raise ValueError('Expected all three frozen seeds')
    # Critical ordering: selections are written before any evaluation GT is read.
    (args.output/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
    write_csv(args.output/'validation_curves.csv',curves);write_csv(args.output/'validation_bias.csv',diagnostics)
    evaluation=[load_sequence(rows[n]) for n in args.evaluate];results=[];equivalence=[]
    from causal_stream import ResidualStream,local
    for state,model,norm,tau,checkpoint in states:
        gate=state['gate_parameters'];alpha=gate['gate_alpha']
        for seq in evaluation:
            p=predict(model,norm,seq,'relative32')
            mask=seq.ready[1:] & (np.abs(p[2]).max(1)<=gate['ood_threshold']) & ((p[1]/norm['ys']).mean(1)<=gate['sigma_threshold'])
            estimates={k:corrected(seq,p[0],alpha,'increment',mask,v) for k,v in [('original',None),('selected_memory',tau)]}
            for name,estimate in estimates.items():
                stats=metrics(seq,estimate)
                results.append(dict(model=checkpoint.parent.name,seed=state['seed'],sequence=seq.name,scene=seq.scene,
                    method=name,tau_s=tau if name=='selected_memory' else None,alpha=alpha,
                    raw_ape_m=metrics(seq,seq.cart)['ape_m']['rmse'],ape_m=stats['ape_m']['rmse'],
                    rpe_1s_m=stats['rpe']['1s']['translation_m']['rmse'],rpe_1s_deg=stats['rpe']['1s']['rotation_deg']['rmse'],
                    max_offset_m=float(np.linalg.norm(estimate[:,:2]-seq.cart[:,:2],axis=1).max()),
                    endpoint_offset_m=float(np.linalg.norm(estimate[-1,:2]-seq.cart[-1,:2])),
                    fallback_fraction=1. if alpha==0 else float((~mask).mean()),
                    evaluation_used_for_selection=False))
                directory=args.output/checkpoint.parent.name/name;directory.mkdir(parents=True,exist_ok=True)
                save_tum(directory/f'{seq.name}.tum',seq,estimate)
            def forward(z):
                with torch.inference_mode():return model(torch.from_numpy(z[None])).numpy()[0]
            engine=ResidualStream('relative32','increment',norm,forward,state['alpha'],gate,tau)
            outputs=[]
            for t,c,o,a in zip(seq.t,seq.cart,seq.odom,seq.available):
                out,_=engine.step(t,c,o,a);outputs.append(local(out,np.zeros(3)))
            error=float(np.abs(np.array(outputs)-estimates['selected_memory']).max())
            # Batched BLAS and single-frame BLAS may differ slightly; yaw wrapping
            # near +-pi is compared by shortest offset.
            delta=np.array(outputs)-estimates['selected_memory'];delta[:,2]=(delta[:,2]+np.pi)%(2*np.pi)-np.pi
            error=float(np.abs(delta).max())
            if error>3e-5:raise AssertionError(f'Batch/stream mismatch {seq.name}: {error}')
            equivalence.append(dict(model=checkpoint.parent.name,sequence=seq.name,max_abs_error=error))
    write_csv(args.output/'results.csv',results)
    (args.output/'equivalence.json').write_text(json.dumps(equivalence,indent=2)+'\n')
    for filename in ['causal_stream.py','mentor_benchmark.py','tools/benchmark_increment_memory.py']:
        (args.output/('source_'+Path(filename).name)).write_bytes((Path(__file__).resolve().parents[1]/filename).read_bytes())
    print(json.dumps(selection,indent=2),flush=True)
    print(f'{len(results)} result rows; {len(equivalence)} actual-stream equivalence checks passed',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--evaluate',nargs='+',required=True);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
