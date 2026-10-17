"""NumPy-only causal relative-motion residual correction with exact hard fallback.

Accept poses of the same physical robot reference point, in each estimator's
consistent frame. No ground truth, ROS or PyTorch is needed for inference.
"""
from collections import deque
import json
from pathlib import Path

import numpy as np


def wrap(value):return (value+np.pi)%(2*np.pi)-np.pi


def body_increments(poses):
    d=np.diff(poses[:,:2],axis=0);c=np.cos(poses[:-1,2]);s=np.sin(poses[:-1,2])
    return np.column_stack([c*d[:,0]+s*d[:,1],-s*d[:,0]+c*d[:,1],wrap(np.diff(poses[:,2]))])


def motion_feature(t,cart,odom):
    ci,oi=body_increments(cart),body_increments(odom)
    diff=ci-oi;diff[:,2]=wrap(diff[:,2])
    current=np.column_stack([oi,ci,diff])[-5:]
    dt=t[-1]-t[-2]
    return np.r_[current[-1],current.mean(0),current.std(0),dt,
                 np.linalg.norm(oi[-1,:2])/dt,np.linalg.norm(ci[-1,:2])/dt,
                 oi[-1,2]/dt,ci[-1,2]/dt].astype(np.float32)


class PortableResidual:
    def __init__(self,directory):
        directory=Path(directory)
        self.metadata=json.loads((directory/'portable.json').read_text())
        with np.load(directory/'portable_weights.npz',allow_pickle=False) as data:
            self.weights={k:data[k].copy() for k in data.files}
        self.history=deque(maxlen=6);self.previous=None
        self.shared_engine=None
        if self.metadata.get('stream_protocol') == 'causal_grid_v1':
            from causal_stream import ResidualStream
            self.shared_engine=ResidualStream(self.metadata['config'],self.metadata['target'],self.weights,
                lambda z:self._forward_normalized(z),self.metadata['alpha'],self.metadata['gate_parameters'],
                self.metadata.get('integration_tau_s'))

    def _forward_normalized(self,x):
        x=x.astype(np.float32)
        for index in range(4):
            x=self.weights[f'w{index}']@x+self.weights[f'b{index}']
            if index<3:x=np.maximum(x,0)
        return x

    def forward(self,feature):
        x=((feature-self.weights['xm'])/self.weights['xs']).astype(np.float32)
        normalized=x.copy()
        for index in range(4):
            x=self.weights[f'w{index}']@x+self.weights[f'b{index}']
            if index<3:x=np.maximum(x,0)
        return x,normalized

    def step(self,stamp,cart,odom,available=True):
        if getattr(self,'shared_engine',None) is not None:
            return self.shared_engine.step(stamp,cart,odom,available)
        cart=np.asarray(cart,dtype=float);odom=np.asarray(odom,dtype=float)
        if not np.isfinite(cart).all():
            raise ValueError('Original Cartographer pose invalid; no valid fallback exists')
        if not np.isfinite(odom).all() or (self.history and stamp<=self.history[-1][0]):
            self.history.clear();self.previous=cart.copy()
            return cart.copy(),{'fallback':True,'reason':'invalid input/timestamp'}
        self.history.append((stamp,cart.copy(),odom.copy()))
        if len(self.history)<2:
            self.previous=cart.copy()
            return cart.copy(),{'fallback':True,'reason':'initialization'}
        history=list(self.history)
        t=np.array([v[0] for v in history]);cp=np.array([v[1] for v in history]);op=np.array([v[2] for v in history])
        feature=motion_feature(t,cp,op)
        output,x=self.forward(feature)
        gate=self.metadata['gate_parameters']
        alpha=self.metadata['alpha'] if gate is None else gate['gate_alpha']
        accepted=True
        sigma=None
        if gate:
            sigma=float(np.exp(.5*np.clip(output[3:],-8,6)).mean())
            accepted=sigma<=gate['sigma_threshold'] and np.abs(x).max()<=gate['ood_threshold']
        if not accepted or alpha==0:
            self.previous=cart.copy()
            return cart.copy(),{'fallback':True,'reason':'confidence' if not accepted else 'alpha_zero',
                                'normalized_sigma':sigma,'max_z':float(np.abs(x).max())}
        residual=np.clip(output[:3]*self.weights['ys']+self.weights['ym'],
                         -self.weights['clip'],self.weights['clip'])
        inc=body_increments(cp)[-1]+alpha*residual
        c,s=np.cos(self.previous[2]),np.sin(self.previous[2])
        self.previous=np.array([self.previous[0]+c*inc[0]-s*inc[1],
                                self.previous[1]+s*inc[0]+c*inc[1],wrap(self.previous[2]+inc[2])])
        return self.previous.copy(),{'fallback':False,'normalized_sigma':sigma,
                                    'max_z':float(np.abs(x).max())}
