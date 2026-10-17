#!/usr/bin/python3
import argparse
import json
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from portable_residual import PortableResidual

parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path)
parser.add_argument('--manifest',type=Path);args=parser.parse_args()
times=[];fallbacks=0;branches={'accepted':[],'fallback':[]};sequence_results=[]
def measure(model,stamp,cp,op,warmup=False,available=True):
    global fallbacks
    start=time.perf_counter_ns();pose,status=model.step(stamp,cp,op,available)
    elapsed=(time.perf_counter_ns()-start)/1e6
    if not warmup:
        times.append(elapsed);fallbacks+=status['fallback']
        branches['fallback' if status['fallback'] else 'accepted'].append(elapsed)
    return pose
if args.manifest:
    manifest=json.loads(args.manifest.read_text())
    for row in manifest['sequences']:
        if row['sequence_id'] not in manifest['test']:continue
        model=PortableResidual(args.directory)
        if model.metadata.get('stream_protocol') != row.get('stream_protocol'):
            raise ValueError('Portable model and input stream protocols differ')
        def read(path):
            values=np.loadtxt(path,ndmin=2);qx,qy,qz,qw=values[:,4:8].T
            yaw=np.arctan2(2*(qw*qz+qx*qy),1-2*(qy*qy+qz*qz))
            return values[:,0],np.column_stack([values[:,1:3],yaw])
        if row.get('stream_protocol')=='causal_grid_v1':
            with np.load(row['sampled_stream'],allow_pickle=False) as data:
                t,cart,odom,available=[data[k].copy() for k in ['t','cart','odom','available']]
        else:
            ct,cp=read(row['cart']);ot,op=read(row['odom'])
            t=np.arange(max(ct[0],ot[0]),min(ct[-1],ot[-1]),.1)
            ci=np.searchsorted(ct,t,side='right')-1;oi=np.searchsorted(ot,t,side='right')-1
            available=(t-ct[ci]<=.15)&(t-ot[oi]<=.15)
            cart,odom=cp[ci],op[oi]
        poses=[]
        for index,stamp in enumerate(t):
            op_current=odom[index] if available[index] else np.full(3,np.nan)
            poses.append(measure(model,stamp,cart[index],op_current,warmup=index<30,available=bool(available[index])))
        poses=np.asarray(poses)
        values=np.column_stack([t,poses[:,:2],np.zeros(len(t)),np.zeros((len(t),2)),
                                np.sin(poses[:,2]/2),np.cos(poses[:,2]/2)])
        np.savetxt(args.directory/(row['sequence_id']+'_portable.tum'),values,fmt='%.9f')
        sequence_results.append({'sequence':row['sequence_id'],'poses':len(t),'input_available_fraction':float(available.mean())})
else:
    model=PortableResidual(args.directory)
    for i in range(1030):
        measure(model,i*.1,np.array([i*.03,0.,i*.002]),np.array([i*.031,0.,i*.0021]),warmup=i<30)
result={'device':'host CPU','runtime':'NumPy-only','repeats':1000,'warmup':30,
        'mean_ms':float(np.mean(times)),'p95_ms':float(np.quantile(times,.95)),
        'p99_ms':float(np.quantile(times,.99)),'fallback_fraction':fallbacks/len(times),
        'process_peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        'weights_bytes':(args.directory/'portable_weights.npz').stat().st_size,
        'timing_scope':'causal features, model, confidence and trajectory integration; input lookup excluded',
        'input':'recorded online poses on causal 10Hz grid' if args.manifest else 'synthetic stream',
        'sequence_results':sequence_results,
        'branch_latency':{k:{'count':len(v),'p99_ms':float(np.quantile(v,.99)) if v else None} for k,v in branches.items()},
        'edge_device_tested':False}
result['repeats']=len(times)
(args.directory/('portable_recorded_latency.json' if args.manifest else 'portable_latency.json')).write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2),flush=True)
