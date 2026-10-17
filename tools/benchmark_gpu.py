#!/usr/bin/python3
"""Measure frozen models; save GPU incompatibility as a measured failure."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from residual_models import Regressor


def benchmark_latency(model, x, device):
    model.to(device).eval()
    tensor=torch.from_numpy(x).to(device)
    if model.kind in ('gru','tcn'):tensor=tensor[:,None,:].expand(-1,5,-1)
    with torch.inference_mode():
        for _ in range(30):model(tensor)
        timings=[]
        for _ in range(500):
            if device=='cuda':torch.cuda.synchronize()
            start=time.perf_counter_ns()
            model(tensor)
            if device=='cuda':torch.cuda.synchronize()
            timings.append((time.perf_counter_ns()-start)/1e6)
    return {'device':device,'batch_size':1,'repeats':500,'warmup':30,
            'mean_ms':float(np.mean(timings)),'p50_ms':float(np.quantile(timings,.5)),
            'p95_ms':float(np.quantile(timings,.95)),'p99_ms':float(np.quantile(timings,.99)),
            'measurement':'model forward only; CUDA synchronized; excludes feature and transfer costs'}

parser=argparse.ArgumentParser()
parser.add_argument('--root',type=Path,required=True)
parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
parser.add_argument('--checkpoints',nargs='+',type=Path,help='Explicit frozen models instead of historical default models')
parser.add_argument('--output',type=Path,help='Separate result path; never overwrite historical measurements')
args=parser.parse_args()
torch.set_num_threads(1)
result={'torch':torch.__version__,'cuda_runtime':torch.version.cuda,
        'available':torch.cuda.is_available(),'models':[]}
try:
    result['device']=args.device
    if args.device=='cuda':
        result['gpu_name']=torch.cuda.get_device_name(0)
        result['capability']=list(torch.cuda.get_device_capability(0))
    paths=args.checkpoints or [args.root/name/'model.pt' for name in
        ['mlp_full20_pose_seed11','mlp_relative32_increment_seed11',
         'mlp_relative32_increment_uncertainty_seed11','gru_relative32_increment_seed11']]
    for path in paths:
        name=path.parent.name
        checkpoint=torch.load(path,map_location='cpu',weights_only=False)
        model=Regressor(len(checkpoint['norm']['xm']),checkpoint['kind'],checkpoint['uncertainty'])
        model.load_state_dict(checkpoint['model_state_dict'])
        if args.device=='cuda':torch.cuda.reset_peak_memory_stats()
        x=np.zeros((1,len(checkpoint['norm']['xm'])),dtype=np.float32)
        stats=benchmark_latency(model,x,device=args.device)
        stats.update(model=name)
        import hashlib
        stats.update(checkpoint=str(path.resolve()),checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        if args.device=='cuda':
            stats.update(peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
                         peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2)
        result['models'].append(stats)
except Exception as exc:
    result['failure']=f'{type(exc).__name__}: {exc}'
    result['status']='failed; no GPU real-time performance claim'
else:
    result['status']='model-forward microbenchmark completed; excludes transfer and feature processing'
existing=args.output or args.root/('gpu_latency.json' if args.device=='cuda' else 'cpu_matched_runtime_latency.json')
existing.parent.mkdir(parents=True,exist_ok=True)
if existing.exists():
    previous=json.loads(existing.read_text())
    version=previous['torch'].replace('+','_')
    (args.root/f'gpu_latency_{version}.json').write_text(json.dumps(previous,indent=2)+'\n')
existing.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2),flush=True)
