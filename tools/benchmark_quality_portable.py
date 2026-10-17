#!/usr/bin/python3
"""Measure each NumPy candidate in a separate CPU process, without Torch/ROS."""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


def child(package,quality_root):
    for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='1'
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    import time
    import numpy as np
    import psutil
    from trajectory_quality import QualityPortable
    process=psutil.Process();before=process.memory_info().rss
    portable=QualityPortable(package);after=process.memory_info().rss
    manifest=json.loads((package.parent.parent/'manifest.json').read_text())
    name=manifest['test'][0];row=next(r for r in manifest['sequences'] if r['sequence_id']==name)
    with np.load(row['sampled_stream']) as d:data={k:d[k].copy() for k in ['t','cart','odom','available']}
    with np.load(quality_root/(name+'.npz')) as d:
        np.testing.assert_array_equal(data['t'],d['t']);quality=d['quality'].copy()
    arrays=[data['t'],data['cart'],data['odom'],quality,data['available']]
    for values in list(zip(*arrays))[:20]:portable.step(*values)
    portable.reset();timings=[];peak=process.memory_info().rss;fallbacks=0
    for values in zip(*arrays):
        started=time.perf_counter_ns();_,status=portable.step(*values)
        timings.append((time.perf_counter_ns()-started)/1e6);fallbacks+=int(status['fallback'])
        peak=max(peak,process.memory_info().rss)
    assert 'torch' not in sys.modules and 'rospy' not in sys.modules
    return dict(model=package.name,sequence=name,frames=len(timings),
        mean_ms=float(np.mean(timings)),p50_ms=float(np.quantile(timings,.5)),
        p95_ms=float(np.quantile(timings,.95)),p99_ms=float(np.quantile(timings,.99)),max_ms=float(np.max(timings)),
        process_rss_before_model_mib=before/1024**2,process_rss_after_model_mib=after/1024**2,
        process_peak_rss_mib=peak/1024**2,weights_bytes=(package/'portable_weights.npz').stat().st_size,
        fallback_fraction=fallbacks/len(timings),torch_imported=False,ros_imported=False,
        timing_scope='single-thread BLAS, batch-one feature/model/gate/integration; excludes scan descriptor/ROS/I/O and input loading',
        memory_scope='whole Python/NumPy process including one captured sequence in RAM; RSS delta is coarse, not tensor allocation',
        cpu_only=True,edge_device_tested=False)


def run(args):
    rows=[]
    for study in args.studies:
        packages=json.loads((study/'portable_verification.json').read_text())
        for r in packages:
            cmd=[sys.executable,str(Path(__file__).resolve()),'--child-package',str(Path(r['package']).resolve()),
                '--quality-root',str(args.quality_root.resolve())]
            result=subprocess.run(cmd,text=True,capture_output=True,check=True)
            row=json.loads(result.stdout);row['study']=study.name;rows.append(row)
            print(row['study'],row['model'],round(row['p99_ms'],4),'ms',flush=True)
    cpu=next((r.split(':',1)[1].strip() for r in Path('/proc/cpuinfo').read_text().splitlines() if r.startswith('model name')),platform.processor())
    result=dict(hardware=dict(cpu=cpu,system=platform.platform(),python=platform.python_version()),results=rows,
        source_sha256={p:hashlib.sha256((Path(__file__).resolve().parents[1]/p).read_bytes()).hexdigest()
                       for p in ['trajectory_quality.py','causal_stream.py','tools/benchmark_quality_portable.py']})
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--studies',nargs='+',type=Path);p.add_argument('--quality-root',type=Path,required=True)
    p.add_argument('--output',type=Path);p.add_argument('--child-package',type=Path);args=p.parse_args()
    if args.child_package:print(json.dumps(child(args.child_package,args.quality_root)))
    else:run(args)
