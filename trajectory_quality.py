"""NumPy-only quality proxies and runtime for trajectory-trained residual models.

Geometry descriptors are observable proxies, not calibrated scan-match scores.
No reference poses or ground truth enter this module.
"""
from collections import deque
import numpy as np
from causal_stream import FeatureStream, ResidualStream

PROTOCOL = 'causal_quality_v1'
QUALITY_NAMES = ['cart_age_s','odom_age_s','scan_age_s','near_hit_fraction',
    'far_hit_fraction','sensor_hit_fraction','near_median_scaled','near_iqr_scaled',
    'near_angular_coverage','near_centroid_scaled','near_covariance_isotropy','scan_present']


def scan_descriptor(ranges, angle_min, angle_increment, sensor_min, sensor_max, cart_max=10.):
    r=np.asarray(ranges,dtype=float);angles=angle_min+np.arange(len(r))*angle_increment
    valid=np.isfinite(r)&(r>=sensor_min)&(r<=sensor_max)
    near=valid&(r>=.12)&(r<=cart_max);far=valid&(r>cart_max)
    values=np.zeros(8,dtype=np.float32)
    values[:3]=[near.mean(),far.mean(),valid.mean()] if len(r) else [0,0,0]
    if near.sum()>=3:
        rn=r[near];a=angles[near]
        q=np.quantile(rn,[.25,.5,.75]);values[3:5]=[q[1]/cart_max,(q[2]-q[0])/cart_max]
        bins=np.floor(np.arange(len(r))*12/max(len(r),1)).astype(int)
        values[5]=len(np.unique(bins[near]))/12.
        points=np.column_stack([rn*np.cos(a),rn*np.sin(a)])
        centered=points-points.mean(0);eig=np.linalg.eigvalsh(centered.T@centered/len(points))
        values[6]=np.linalg.norm(points.mean(0))/cart_max
        values[7]=max(float(eig[0]),0)/max(float(eig.sum()),1e-12)
    last_index=int(np.flatnonzero(valid)[-1]) if valid.any() else 0
    return values,last_index


class ScanQualityBuffer:
    def __init__(self,max_age=.15):
        self.rows=deque(maxlen=4096);self.max_age=max_age

    def push(self,stamp,descriptor):
        self.rows.append((float(stamp),np.asarray(descriptor,dtype=np.float32).copy()))

    def receive(self,msg):
        d,last=scan_descriptor(msg.ranges,msg.angle_min,msg.angle_increment,msg.range_min,msg.range_max)
        # Mirrors Cartographer's last valid beam timestamp conversion. This
        # preserves the existing bag's timing assumptions, rather than fitting GT.
        self.push(msg.header.stamp.to_sec()+last*msg.time_increment,d)

    def at(self,tick,cart_stamp,odom_stamp):
        usable=[r for r in self.rows if r[0]<=tick]
        row=max(usable,key=lambda r:r[0]) if usable else None
        q=np.zeros(12,dtype=np.float32)
        q[:2]=[tick-cart_stamp,tick-odom_stamp]
        q[2]=self.max_age+1 if row is None else tick-row[0]
        if row is not None and 0<=q[2]<=self.max_age and np.isfinite(q[:2]).all():
            q[3:11]=row[1];q[11]=1.
        return q,np.nan if row is None else row[0]


class QualityFeatureStream:
    """Same reset/initialization guard in padded and quality ablations."""
    def __init__(self,config):
        if config not in ('padded44','quality44'):raise ValueError('Unknown quality feature configuration')
        self.base=FeatureStream('relative32');self.config=config;self.quality=None

    def __getattr__(self,name):return getattr(self.base,name)

    def step(self,stamp,cart,odom,available=True):
        q=self.quality
        good=q is not None and np.shape(q)==(12,) and np.isfinite(q).all() and q[11]==1
        # Origins still come from the original first available pose pair.
        cp,op,x,ready=self.base.step(stamp,cart,odom,available)
        if not good:
            self.base.history.clear();return cp,op,np.zeros(44,np.float32),False
        tail=np.asarray(q,np.float32) if self.config=='quality44' else np.zeros(12,np.float32)
        return cp,op,np.r_[x,tail].astype(np.float32),bool(ready)


def extract_quality(t,cart,odom,available,quality,config):
    stream=QualityFeatureStream(config);rows=[]
    if len(quality)!=len(t):raise ValueError('Quality array length differs from captured grid')
    for s,c,o,a,q in zip(t,cart,odom,available,quality):
        stream.quality=q;rows.append(stream.step(s,c,o,a))
    return np.array([r[2] for r in rows]),np.array([r[3] for r in rows])


class QualityResidualStream:
    def __init__(self,config,norm,forward,alpha,gate=None,tau_s=None):
        self.engine=ResidualStream('relative32','increment',norm,forward,alpha,gate,tau_s)
        self.engine.features=QualityFeatureStream(config)

    def step(self,stamp,cart,odom,quality,available=True):
        self.engine.features.quality=quality
        return self.engine.step(stamp,cart,odom,available)


class QualityPortable:
    """Portable batch-one runtime; requires only NumPy and sampled inputs."""
    def __init__(self,directory):
        import json
        from pathlib import Path
        directory=Path(directory)
        self.meta=json.loads((directory/'portable.json').read_text())
        if self.meta['stream_protocol']!=PROTOCOL:raise ValueError('Wrong portable protocol')
        with np.load(directory/'portable_weights.npz') as d:self.arrays={k:d[k].copy() for k in d.files}
        self.norm={k:self.arrays[k] for k in ['xm','xs','ym','ys','clip']}
        self.reset()

    def forward_normalized(self,z):
        h=np.asarray(z,dtype=np.float32)
        for i in range(4):
            h=self.arrays[f'w{i}']@h+self.arrays[f'b{i}']
            if i<3:h=np.maximum(h,0)
        return h

    def reset(self):
        m=self.meta
        self.engine=QualityResidualStream(m['feature_config'],self.norm,self.forward_normalized,
            m['alpha'],m['gate_parameters'],m.get('integration_tau_s'))

    def step(self,stamp,cart,odom,quality,available=True):
        return self.engine.step(stamp,cart,odom,quality,available)
