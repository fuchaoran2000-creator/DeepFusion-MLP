#!/usr/bin/python3
"""Export the originally validation-selected seed with its new validation tau."""
import argparse
import hashlib
import json
from pathlib import Path
import torch

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('study',type=Path);args=p.parse_args()
    plan=json.loads((args.study/'plan.json').read_text())
    root=Path(plan['sources_root'])
    original=json.loads((root/'deployment_selection.json').read_text())['relative32']['selected']
    selected=next(r for r in json.loads((args.study/'selection.json').read_text()) if r['seed']==original['seed'])
    path=Path(selected['checkpoint'])
    assert path.resolve()==Path(original['checkpoint']).resolve()
    assert hashlib.sha256(path.read_bytes()).hexdigest()==selected['checkpoint_sha256']
    state=torch.load(path,map_location='cpu',weights_only=False)
    tau=selected['selected_tau_s']
    if tau is None:raise ValueError('Validation did not select finite memory; preserve original checkpoint')
    state['integration_tau_s']=tau
    state['integration_memory_selection']=dict(selected,experimental_candidate=True,actual_ros_replay_verified=False)
    output=args.study/f'candidate_seed{original["seed"]}_memory{tau:g}.pt'
    torch.save(state,output)
    print(output)
