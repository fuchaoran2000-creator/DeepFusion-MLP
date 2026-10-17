#!/usr/bin/python3
"""Import exported sources in isolated Python, outside the repository cwd."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

SCRIPT='''
import sys,json
from pathlib import Path
import numpy as np
package,capture,quality_file=sys.argv[1:]
sys.path.insert(0,package)
import trajectory_quality,causal_stream
assert Path(trajectory_quality.__file__).resolve().parent==Path(package).resolve()
assert Path(causal_stream.__file__).resolve().parent==Path(package).resolve()
runtime=trajectory_quality.QualityPortable(package)
with np.load(capture) as d:rows={k:d[k].copy() for k in ['t','cart','odom','available']}
with np.load(quality_file) as d:
    np.testing.assert_array_equal(d['t'],rows['t']);quality=d['quality'].copy()
fallback=0
for t,c,o,q,a in zip(rows['t'],rows['cart'],rows['odom'],quality,rows['available']):
    output,status=runtime.step(t,c,o,q,bool(a))
    assert np.isfinite(output).all()
    if status['fallback']:np.testing.assert_array_equal(output,c);fallback+=1
assert 'torch' not in sys.modules and 'rospy' not in sys.modules
print(json.dumps(dict(model=Path(package).name,frames=len(rows['t']),fallbacks=fallback,isolated_import=True,
    imports_use_exported_source=True,torch_imported=False,rospy_imported=False)))
'''


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--studies',nargs='+',type=Path,required=True)
    p.add_argument('--quality-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    results=[]
    for study in args.studies:
        manifest=json.loads((study/'manifest.json').read_text());name=manifest['test'][0]
        row=next(r for r in manifest['sequences'] if r['sequence_id']==name)
        for export in json.loads((study/'portable_verification.json').read_text()):
            package=Path(export['package']).resolve()
            result=subprocess.run([sys.executable,'-I','-c',SCRIPT,str(package),row['sampled_stream'],
                str((args.quality_root/(name+'.npz')).resolve())],cwd='/tmp',capture_output=True,text=True,check=True)
            results.append(json.loads(result.stdout));print(export['model'],'isolated package OK',flush=True)
    args.output.write_text(json.dumps(results,indent=2)+'\n')
