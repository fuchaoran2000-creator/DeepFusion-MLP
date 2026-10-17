#!/usr/bin/python3
"""Choose each uncertainty configuration by validation score, never test score."""
import argparse
import json
from pathlib import Path
import torch

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path)
    args=parser.parse_args();decisions={}
    for config,target in [('full20','pose'),('relative32','increment')]:
        candidates=[]
        for path in sorted(args.root.glob(f'mlp_{config}_{target}_uncertainty_seed*/model.pt')):
            state=torch.load(path,map_location='cpu',weights_only=False)
            assert state['stream_protocol']=='causal_grid_v1'
            gate=state['gate_parameters']
            candidates.append({'checkpoint':str(path.resolve()),'seed':state['seed'],
                'validation_ape_m':gate['gate_validation_ape_m'],'gate_alpha':gate['gate_alpha']})
        if len(candidates)!=3:raise ValueError('Expected all three trained seeds')
        selected=min(candidates,key=lambda c:(c['validation_ape_m'],c['seed']))
        decisions[config]={'criterion':'minimum validation macro APE after hard gate; ties resolved by seed',
            'test_used_for_selection':False,'candidates':candidates,'selected':selected}
    path=args.root/'deployment_selection.json'
    path.write_text(json.dumps(decisions,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(decisions,indent=2))
