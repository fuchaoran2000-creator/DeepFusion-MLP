#!/usr/bin/python3
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from gate_stress import evaluate
from mentor_benchmark import load_sequence,save_tum,validate_manifest
from residual_models import Regressor


def run(args):
    torch.set_num_threads(1)
    manifest=json.loads(args.manifest.read_text());validate_manifest(manifest)
    protocol=json.loads(args.protocol.read_text())
    args.output.mkdir(parents=True,exist_ok=True)
    for source in ['gate_stress.py','causal_stream.py','mentor_benchmark.py']:
        (args.output/('source_'+source)).write_bytes(Path(__file__).resolve().parents[1].joinpath(source).read_bytes())
    (args.output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    (args.output/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n')
    selected=[r for r in manifest['sequences'] if args.split=='all' or r['sequence_id'] in manifest[args.split]]
    if args.sequences:selected=[r for r in selected if r['sequence_id'] in args.sequences]
    datasets=[]
    for row in selected:
        with np.load(row['sampled_stream'],allow_pickle=False) as data:
            arrays={k:data[k].copy() for k in data.files}
        datasets.append((load_sequence(row),arrays))
    results=[];checkpoints=[]
    for checkpoint in args.checkpoints:
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if state.get('stream_protocol')!='causal_grid_v1' or not state['uncertainty']:
            raise ValueError('Expected uncertainty checkpoint matching causal_grid_v1')
        model=Regressor(len(state['norm']['xm']),'mlp',True)
        model.load_state_dict(state['model_state_dict']);model.eval()
        checkpoints.append({'path':str(checkpoint.resolve()),'sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                            'alpha':state['gate_parameters']['gate_alpha'],'seed':state['seed']})
        for seq,arrays in datasets:
            root=args.output/checkpoint.parent.name/seq.name;root.mkdir(parents=True,exist_ok=True)
            def save(anomaly,mode,work,estimate,active):
                save_tum(root/f'{anomaly}_{mode}.tum',work,estimate)
                np.savez_compressed(root/f'{anomaly}_{mode}_status.npz',t=work.t,accepted=active,
                                    available=work.available,scoring_valid=seq.valid)
            reports=evaluate(seq,arrays,state,model,protocol,save)
            for r in reports:r['checkpoint']=str(checkpoint.resolve())
            results.extend(reports)
            (args.output/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
            print(f'{checkpoint.parent.name}/{seq.name}: {len(reports)} paired gate cases',flush=True)
    flat=[]
    for r in results:
        flat.append({k:r[k] for k in ['sequence','scene','config','seed','anomaly','gate_mode','alpha',
            'alpha_disabled','scored_samples','raw_clean_ape_m','raw_corrupted_ape_m',
            'harmful_fraction_vs_corrupted_raw','harmful_fraction_vs_clean_raw','region_harmful_fraction_vs_corrupted_raw',
            'harmful_fraction_among_accepted','fallback_fraction_all','fallback_fraction_region',
            'clean_rejected_fraction_ready','newly_rejected_fraction_region','rejection_of_beneficial_none_fraction',
            'new_rejection_latency_s','recovery_s','recovery_censored','clean_recovery_s','additional_recovery_delay_s',
            'max_pose_step_m','max_correction_offset_step_m','sigma_additional_rejection_fraction']})
        flat[-1].update(ape_rmse_m=r['ape_m']['rmse'],rpe_1s_m=r['rpe']['1s']['translation_m']['rmse'])
    with (args.output/'benchmark.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    (args.output/'checkpoints.json').write_text(json.dumps(checkpoints,indent=2)+'\n')
    print(f'Saved {len(results)} gate/sequence/anomaly cases; not independent sequences',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--protocol',type=Path,default=Path('config/causal_gate_stress_v1.json'))
    parser.add_argument('--checkpoints',nargs='+',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--split',choices=['all','train','validation','test'],default='test')
    parser.add_argument('--sequences',nargs='+')
    run(parser.parse_args())
