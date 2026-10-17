"""Replay the archived C20 inputs; references are read only after inference."""
import csv
import json
from pathlib import Path
import numpy as np
from portable_residual import PortableResidual
from causal_stream import local, wrap

HERE=Path(__file__).resolve().parent

def score(t, truth, estimate, valid):
    ape=float(np.sqrt(np.mean(np.sum((truth[valid,:2]-estimate[valid,:2])**2,axis=1))))
    prefix=np.r_[0,np.cumsum(~valid)]
    i=np.arange(len(t));j=np.searchsorted(t,t+1.)
    use=(t+1.)<=t[-1]+1e-7;i,j=i[use],j[use]
    j=np.minimum(j,len(t)-1);previous=np.maximum(j-1,0)
    j=np.where(np.abs(t[previous]-t[i]-1)<np.abs(t[j]-t[i]-1),previous,j)
    use=(np.abs(t[j]-t[i]-1)<=.11)&valid[i]&valid[j]&((prefix[j+1]-prefix[i])==0)
    i,j=i[use],j[use]
    def relative(p):
        d=p[j,:2]-p[i,:2];c,s=np.cos(p[i,2]),np.sin(p[i,2])
        return np.column_stack([c*d[:,0]+s*d[:,1],-s*d[:,0]+c*d[:,1]])
    rpe=float(np.sqrt(np.mean(np.sum((relative(truth)-relative(estimate))**2,axis=1))))
    return ape,rpe

def main():
    expected={r['sequence']:r for r in csv.DictReader((HERE/'expected_table4.csv').open()) if r['model']=='unified20'}
    reports=[]
    for name in ['home1-3','home1-4','home1-5']:
        with np.load(HERE/'captures'/f'{name}.npz',allow_pickle=False) as z:
            d={k:z[k].copy() for k in z.files}
        model=PortableResidual(HERE/'model');poses=[];fallback=[];features=[];ready=[]
        for tick,c,o,a in zip(d['t'],d['cart'],d['odom'],d['available']):
            p,status=model.step(tick,c,o,bool(a));poses.append(p);fallback.append(status['fallback'])
            features.append(status['feature']);ready.append(status['ready'])
        poses=np.array(poses);fallback=np.array(fallback)
        error=poses-d['output'];error[:,2]=wrap(error[:,2])
        np.testing.assert_array_equal(fallback,d['fallback'])
        np.testing.assert_array_equal(ready,d['ready'])
        np.testing.assert_array_equal(np.array(features),d['features'])
        np.testing.assert_allclose(error,0,atol=3e-5,rtol=0)
        np.testing.assert_array_equal(poses[fallback],d['cart'][fallback])
        # No reference data have been opened before this point.
        with np.load(HERE/'scoring'/f'{name}.npz',allow_pickle=False) as z:
            truth,valid=z['aligned_reference'].copy(),z['valid'].copy()
        anchor=d['cart'][np.flatnonzero(d['available'])[0]]
        corrected=np.array([local(p,anchor) for p in poses])
        raw=np.array([local(p,anchor) for p in d['cart']])
        raw_ape,raw_rpe=score(d['t'],truth,raw,valid)
        ape,rpe=score(d['t'],truth,corrected,valid)
        row=dict(sequence=name,frames=len(poses),raw_ape_m=raw_ape,corrected_ape_m=ape,
            raw_rpe_1s_m=raw_rpe,corrected_rpe_1s_m=rpe,
            fallback_fraction=float(fallback.mean()),valid_fraction=float(valid.mean()),
            pose_component_max_error=float(np.abs(error).max()))
        for key,old in [('raw_ape_m','raw_ape_m'),('corrected_ape_m','shared_map_corrected_ape_m'),
            ('raw_rpe_1s_m','raw_rpe_1s_m'),('corrected_rpe_1s_m','corrected_rpe_1s_m'),
            ('fallback_fraction','fallback_fraction')]:
            assert abs(row[key]-float(expected[name][old]))<3e-5,(key,row[key],expected[name][old])
        reports.append(row)
    result=dict(passed=True,reference_in_inference=False,sequences=reports,
        macro_ape_raw_m=float(np.mean([r['raw_ape_m'] for r in reports])),
        macro_ape_corrected_m=float(np.mean([r['corrected_ape_m'] for r in reports])),
        scope='Frozen-recorded-input replay; no new ROS run, training, threshold fitting or test-based selection.')
    (HERE/'replay_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
