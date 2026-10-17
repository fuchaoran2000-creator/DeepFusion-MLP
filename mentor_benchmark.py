#!/usr/bin/python3
"""Reproducible grouped-sequence benchmark; never tune on held-out test data.

Prepared Cartographer final trajectories are offline inputs, explicitly labelled.
Online trajectories can be supplied through the manifest with trajectory_kind=online.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import resource
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import psutil
import torch
from scipy.linalg import cho_factor, cho_solve, solve_banded
from torch import nn

from deepfusion_mlp_train_relative_v2 import read_tum, wrap_angle, rolling_statistics
from residual_models import Regressor

GROUPS = {'P_O': slice(0, 4), 'P_C': slice(4, 8), 'D_O': slice(8, 12),
          'D_C': slice(12, 16), 'E': slice(16, 20)}
CONFIGS = ['full20', 'no_P_C', 'no_P_O', 'no_D_C', 'no_D_O', 'no_increments',
           'no_E', 'poses8', 'motion_difference12', 'relative6', 'relative9',
           'relative14', 'relative32', 'no_window_mean', 'no_window_std',
           'no_motion_state']


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def load_pose(path):
    tr = read_tum(Path(path))
    return tr.t, np.column_stack([tr.xyz[:, :2], tr.yaw])


def local(pose):
    result = pose.copy()
    c, s = np.cos(pose[0, 2]), np.sin(pose[0, 2])
    result[:, :2] = (pose[:, :2] - pose[0, :2]) @ np.array([[c, -s], [s, c]])
    result[:, 2] = wrap_angle(pose[:, 2] - pose[0, 2])
    return result


def increments(pose):
    delta = np.diff(pose[:, :2], axis=0)
    c, s = np.cos(pose[:-1, 2]), np.sin(pose[:-1, 2])
    return np.column_stack([c*delta[:, 0]+s*delta[:, 1],
                            -s*delta[:, 0]+c*delta[:, 1],
                            wrap_angle(np.diff(pose[:, 2]))])


def add_increment(pose, inc):
    c, s = np.cos(pose[2]), np.sin(pose[2])
    return np.array([pose[0]+c*inc[0]-s*inc[1], pose[1]+s*inc[0]+c*inc[1],
                     wrap_angle(pose[2]+inc[2])])


def interpolate(t, source_t, pose):
    return np.column_stack([np.interp(t, source_t, pose[:, 0]),
                            np.interp(t, source_t, pose[:, 1]),
                            wrap_angle(np.interp(t, source_t, np.unwrap(pose[:, 2])))])


@dataclass
class Sequence:
    name: str
    scene: str
    t: np.ndarray
    cart: np.ndarray
    odom: np.ndarray
    gt: np.ndarray
    valid: np.ndarray
    kind: str
    coverage: dict
    available: np.ndarray = None
    ready: np.ndarray = None
    stream_protocol: str = None


def load_sequence(row, hz=10.0):
    if row.get('stream_protocol') == 'causal_grid_v1':
        return load_causal_sequence(row, hz)
    ct, cp = load_pose(row['cart'])
    ot, op = load_pose(row['odom'])
    gt, gp = load_pose(row['gt'])
    t = np.arange(max(ct[0], ot[0]), min(ct[-1], ot[-1]), 1.0/hz)
    if len(t) < 3:
        raise ValueError(f"Insufficient overlapping inputs: {row['sequence_id']}")
    # Input timeline is independent of reference availability.
    oi = np.searchsorted(ot, t, side='right')-1
    available = (oi >= 0) & ((t-ot[np.maximum(oi, 0)]) <= 0.15)
    oi = np.clip(oi, 0, len(ot)-1)
    if row.get('trajectory_kind') == 'online':
        ci = np.searchsorted(ct, t, side='right')-1
        available &= (ci >= 0) & ((t-ct[np.maximum(ci, 0)]) <= 0.15)
        cart = cp[np.clip(ci, 0, len(ct)-1)].copy()
    else:
        cart = interpolate(t, ct, cp)
    # Preserve missing samples as unavailable; never reconnect disjoint windows.
    odom = op[oi].copy()
    truth = interpolate(t, gt, gp)
    right = np.searchsorted(gt, t, side='left')
    left = np.clip(right-1, 0, len(gt)-1)
    right = np.clip(right, 0, len(gt)-1)
    valid = available & (t >= gt[0]) & (t <= gt[-1]) & ((gt[right]-gt[left]) <= 0.25)
    if valid.sum() < 20:
        raise ValueError(f"Insufficient reference coverage: {row['sequence_id']}")
    first = int(np.flatnonzero(available)[0])
    # Keep the input stream and its origin independent of reference timestamps.
    t, cart, odom, truth, valid = [v[first:] for v in (t, cart, odom, truth, valid)]
    available = available[first:]
    cart, odom = local(cart), local(odom)
    anchor = int(np.flatnonzero(valid)[0])
    # Align reference labels to the first valid input pose only for supervision/scoring.
    c, s = np.cos(cart[anchor,2]-truth[anchor,2]), np.sin(cart[anchor,2]-truth[anchor,2])
    truth[:,:2] = (truth[:,:2]-truth[anchor,:2]) @ np.array([[c,s],[-s,c]]) + cart[anchor,:2]
    truth[:,2] = wrap_angle(truth[:,2]-truth[anchor,2]+cart[anchor,2])
    return Sequence(row['sequence_id'], row['scene_id'], t, cart, odom,
                    truth, valid, row.get('trajectory_kind', 'offline_final_optimized'),
                    {'input_available_fraction': float(available.mean()),
                     'reference_valid_fraction': float(valid.mean()), 'grid_hz': hz,
                     'input_grid_uses_gt': False, 'alignment': 'common-time origin SE2',
                     'cart_interpolation': row.get('trajectory_kind') != 'online'})


def load_causal_sequence(row, hz):
    from causal_stream import extract, PROTOCOL
    with np.load(row['sampled_stream'], allow_pickle=False) as data:
        if str(data['protocol']) != PROTOCOL or abs(float(data['hz'])-hz) > 1e-9:
            raise ValueError('Captured stream protocol / grid frequency mismatch')
        t, cp, op, available = [data[k].copy() for k in ['t', 'cart', 'odom', 'available']]
    # Captured pairs represent arrival availability, never reconstruct it from GT.
    cart, odom, _, ready, _ = extract(t, cp, op, available, 'full20')
    # Finite causal held values let conventional baselines run; availability is
    # retained separately and shared learned history never consumes these gaps.
    for i in range(len(odom)):
        if not np.isfinite(odom[i]).all():
            odom[i] = odom[i-1] if i else np.zeros(3)
    gt, gp = load_pose(row['gt'])
    truth = interpolate(t, gt, gp)
    right = np.clip(np.searchsorted(gt, t, side='left'), 0, len(gt)-1)
    left = np.maximum(right-1, 0)
    valid = available & (t >= gt[0]) & (t <= gt[-1]) & ((gt[right]-gt[left]) <= .25)
    if valid.sum() < 20:
        raise ValueError('Insufficient reference coverage')
    anchor = int(np.flatnonzero(valid)[0])
    delta = cart[anchor, 2]-truth[anchor, 2]
    c, s = np.cos(delta), np.sin(delta)
    truth[:, :2] = (truth[:, :2]-truth[anchor, :2]) @ np.array([[c, s], [-s, c]]) + cart[anchor, :2]
    truth[:, 2] = wrap_angle(truth[:, 2]-truth[anchor, 2]+cart[anchor, 2])
    return Sequence(row['sequence_id'], row['scene_id'], t, cart, odom, truth,
                    valid, 'online', {'input_available_fraction': float(available.mean()),
                    'reference_valid_fraction': float(valid.mean()), 'grid_hz': hz,
                    'input_grid_uses_gt': False, 'alignment': 'common-time origin SE2',
                    'cart_interpolation': False, 'stream_protocol': PROTOCOL,
                    'ready_fraction': float(ready.mean()), 'availability_source': 'captured ROS arrivals'},
                    available, ready, PROTOCOL)


def validate_manifest(manifest):
    sets = [set(manifest[key]) for key in ['train', 'validation', 'test']]
    if not all(sets) or any(sets[i] & sets[j] for i in range(3) for j in range(i)):
        raise ValueError('Train, validation and test must be non-empty, disjoint complete sequences')
    rows = manifest['sequences']
    ids = [r['sequence_id'] for r in rows]
    if len(set(ids)) != len(ids) or not set.union(*sets).issubset(ids):
        raise ValueError('Duplicate or missing sequence IDs')
    paths = [str(Path(r['cart']).resolve()) for r in rows]
    if len(set(paths)) != len(paths):
        raise ValueError('The same Cartographer trajectory cannot appear under multiple IDs')
    stream_paths = [str(Path(r.get('sampled_stream',r['cart'])).resolve()) for r in rows]
    if len(set(stream_paths)) != len(stream_paths):
        raise ValueError('The same captured input stream cannot appear under multiple IDs')
    protocols = {r.get('stream_protocol') for r in rows}
    if len(protocols) > 1:
        raise ValueError('Do not mix legacy and versioned input contracts in one benchmark')
    field = manifest.get('isolation_field')
    if field:
        groups = [{r[field] for r in rows if r['sequence_id'] in group} for group in sets]
        if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
            raise ValueError(f'{field} leakage across splits')


def features(seq, config, window=5):
    if seq.stream_protocol == 'causal_grid_v1':
        from causal_stream import extract
        # Sequence poses are already in input-origin coordinates; origin zero is
        # preserved by the shared extractor's first available sample.
        return extract(seq.t, seq.cart, seq.odom, seq.available, config)[2][1:]
    cart, odom = seq.cart, seq.odom
    dc, do = np.diff(cart, axis=0), np.diff(odom, axis=0)
    dc[:, 2], do[:, 2] = wrap_angle(dc[:, 2]), wrap_angle(do[:, 2])
    def encode(p):
        return np.column_stack([p[:, :2], np.sin(p[:, 2]), np.cos(p[:, 2])])
    difference = cart[1:]-odom[1:]
    difference[:, 2] = wrap_angle(difference[:, 2])
    full = np.column_stack([encode(odom[1:]), encode(cart[1:]), encode(do), encode(dc), encode(difference)])
    if config in ['full20', *['no_'+k for k in GROUPS], 'no_increments', 'poses8', 'motion_difference12']:
        keep = list(GROUPS)
        if config.startswith('no_') and config[3:] in GROUPS:
            keep.remove(config[3:])
        elif config == 'no_increments':
            keep = ['P_O', 'P_C', 'E']
        elif config == 'poses8':
            keep = ['P_O', 'P_C']
        elif config == 'motion_difference12':
            keep = ['D_O', 'D_C', 'E']
        return np.concatenate([full[:, GROUPS[key]] for key in keep], axis=1).astype(np.float32)
    ci, oi = increments(cart), increments(odom)
    diff = ci-oi
    diff[:, 2] = wrap_angle(diff[:, 2])
    current = np.column_stack([oi, ci, diff])
    mean, std = rolling_statistics(current, window)
    dt = np.diff(seq.t)
    state = np.column_stack([dt, np.linalg.norm(oi[:, :2], axis=1)/dt,
                             np.linalg.norm(ci[:, :2], axis=1)/dt, oi[:, 2]/dt, ci[:, 2]/dt])
    options = {'relative6': current[:, :6], 'relative9': current,
               'relative14': np.column_stack([current, state]),
               'relative32': np.column_stack([current, mean, std, state]),
               'no_window_mean': np.column_stack([current, std, state]),
               'no_window_std': np.column_stack([current, mean, state]),
               'no_motion_state': np.column_stack([current, mean, std])}
    return options[config].astype(np.float32)


def targets(seq, target):
    y = seq.gt[1:]-seq.cart[1:] if target == 'pose' else increments(seq.gt)-increments(seq.cart)
    y[:, 2] = wrap_angle(y[:, 2])
    valid = seq.valid[1:] & seq.valid[:-1]
    if seq.ready is not None:
        valid &= seq.ready[1:]
    return y.astype(np.float32), valid


def corrected(seq, prediction, alpha=1.0, target='pose', gate=None, integration_tau_s=None):
    if integration_tau_s is not None and (target != 'increment' or not np.isfinite(integration_tau_s) or integration_tau_s <= 0):
        raise ValueError('Integration memory requires an increment target and a finite positive time constant')
    weights = np.ones(len(prediction)) if gate is None else np.asarray(gate)
    if seq.ready is not None:
        weights = weights * seq.ready[1:]
    result = seq.cart.copy()
    if target == 'pose':
        result[1:] += (alpha*weights)[:, None]*prediction
        result[:, 2] = wrap_angle(result[:, 2])
    else:
        ci = increments(seq.cart)
        for i in range(len(prediction)):
            if weights[i] == 0 or alpha == 0:
                # A real fallback removes previous integrated offsets, not only the next increment.
                result[i+1] = seq.cart[i+1]
            else:
                result[i+1] = add_increment(result[i], ci[i]+alpha*weights[i]*prediction[i])
                if integration_tau_s is not None:
                    from causal_stream import retain_offset
                    result[i+1] = retain_offset(result[i+1], seq.cart[i+1], seq.t[i+1]-seq.t[i], integration_tau_s)
    return result


def summary(values):
    values = np.asarray(values)
    if not len(values):
        return {'count': 0, 'rmse': None, 'mean': None, 'median': None, 'p95': None, 'max': None}
    return {'count': len(values), 'rmse': float(np.sqrt(np.mean(values**2))),
            'mean': float(values.mean()), 'median': float(np.median(values)),
            'p95': float(np.quantile(values, .95)), 'max': float(values.max())}


def metrics(seq, estimate):
    valid = seq.valid & np.isfinite(estimate).all(axis=1)
    ape = np.linalg.norm(estimate[:, :2]-seq.gt[:, :2], axis=1)
    yaw = np.abs(np.degrees(wrap_angle(estimate[:, 2]-seq.gt[:, 2])))
    report = {'ape_m': summary(ape[valid]), 'yaw_ape_deg': summary(yaw[valid]),
              'coverage': dict(seq.coverage, output_fraction=float(np.isfinite(estimate).all(axis=1).mean())),
              'rpe': {}}
    length = np.r_[0, np.cumsum(np.linalg.norm(np.diff(seq.gt[:, :2], axis=0), axis=1))]
    invalid_prefix = np.r_[0, np.cumsum(~valid)]
    for delta, unit, axis in [(1,'s',seq.t), (5,'s',seq.t), (1,'m',length), (5,'m',length)]:
        i = np.arange(len(seq.t))
        j = np.searchsorted(axis, axis+delta)
        use = (axis+delta) <= axis[-1]+1e-7
        i, j = i[use], j[use]
        j=np.minimum(j,len(axis)-1)
        previous=np.maximum(j-1,0)
        j=np.where(np.abs(axis[previous]-axis[i]-delta)<np.abs(axis[j]-axis[i]-delta),previous,j)
        use = (np.abs(axis[j]-axis[i]-delta) <= max(.1*delta, .11 if unit=='s' else .1))
        use &= valid[i] & valid[j] & ((invalid_prefix[j+1]-invalid_prefix[i]) == 0)
        i, j = i[use], j[use]
        def relative(p):
            d = p[j, :2]-p[i, :2]
            c, s = np.cos(p[i, 2]), np.sin(p[i, 2])
            return np.column_stack([c*d[:,0]+s*d[:,1], -s*d[:,0]+c*d[:,1],
                                    wrap_angle(p[j,2]-p[i,2])])
        a, b = relative(seq.gt), relative(estimate)
        report['rpe'][f'{delta}{unit}'] = {
            'translation_m': summary(np.linalg.norm(a[:,:2]-b[:,:2], axis=1)),
            'rotation_deg': summary(np.abs(np.degrees(wrap_angle(a[:,2]-b[:,2]))))}
    return report


def causal_windows(x, window=5):
    return np.stack([x[np.maximum(np.arange(i-window+1, i+1), 0)] for i in range(len(x))])


def loss(output, y, uncertainty):
    if uncertainty:
        mean, logvar = output[:, :3], output[:, 3:].clamp(-8, 6)
        return (.5*((y-mean)**2*torch.exp(-logvar)+logvar)).mean()
    return ((output-y)**2).mean()


def train_model(train, validation, config, target, seed, epochs, kind='mlp', uncertainty=False, feature_builder=None):
    build_features = features if feature_builder is None else feature_builder
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    datasets = []
    for seq in train:
        x = build_features(seq, config)
        y, valid = targets(seq, target)
        datasets.append((x, y, valid))
    all_x = np.concatenate([x[v] for x,y,v in datasets])
    all_y = np.concatenate([y[v] for x,y,v in datasets])
    xm = all_x.mean(0)
    raw_std=all_x.std(0)
    # Constant channels (notably dt on a fixed grid) must not amplify arrival jitter.
    xs=np.where(raw_std<1e-5,1.,raw_std)
    ym, ys = all_y.mean(0), np.maximum(all_y.std(0), 1e-6)
    clip = np.maximum(np.quantile(np.abs(all_y), .995, axis=0)*1.5, 3*ys)
    def prepare(sequences):
        chunks = []
        for seq in sequences:
            x = ((build_features(seq,config)-xm)/xs).astype(np.float32)
            y, valid = targets(seq,target)
            if kind in ('gru','tcn'):
                x = causal_windows(x)
                valid &= np.convolve((~seq.valid[:-1]).astype(int), np.ones(5,dtype=int), 'full')[:len(valid)] == 0
            chunks.append((x[valid], ((y-ym)/ys)[valid]))
        return [torch.from_numpy(np.concatenate([v[k] for v in chunks]).astype(np.float32)) for k in (0,1)]
    x, y = prepare(train)
    vx, vy = prepare(validation)
    model = Regressor(all_x.shape[1], kind, uncertainty)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=1e-5)
    best, best_epoch, stale = math.inf, 0, 0
    best_state = None
    start = time.perf_counter()
    history = []
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(x))
        losses = []
        for i in range(0,len(x),256):
            idx = order[i:i+256]
            optimizer.zero_grad()
            batch_loss = loss(model(x[idx]), y[idx], uncertainty)
            if not torch.isfinite(batch_loss):
                raise ValueError('Non-finite training loss')
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            losses.append(float(batch_loss.detach()))
        model.eval()
        with torch.inference_mode():
            val = float(loss(model(vx),vy,uncertainty))
        history.append({'epoch':epoch+1,'train_loss':float(np.mean(losses)),'validation_loss':val})
        if val < best-1e-6:
            best, best_epoch, stale = val, epoch+1, 0
            best_state = {k:v.detach().clone() for k,v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= 25:
            break
    model.load_state_dict(best_state)
    model.eval()
    return model, {'xm':xm,'xs':xs,'ym':ym,'ys':ys,'clip':clip}, {
        'wall_s':time.perf_counter()-start, 'best_epoch':best_epoch,'best_validation_loss':best,
        'parameter_count':sum(p.numel() for p in model.parameters()), 'train_samples':len(x),
        'validation_samples':len(vx),'history':history}


def predict(model, norm, seq, config, feature_builder=None):
    build_features = features if feature_builder is None else feature_builder
    x = ((build_features(seq, config)-norm['xm'])/norm['xs']).astype(np.float32)
    inputs = causal_windows(x) if model.kind in ('gru','tcn') else x
    with torch.inference_mode():
        output = model(torch.from_numpy(inputs)).numpy()
    mean = np.clip(output[:,:3]*norm['ys']+norm['ym'], -norm['clip'], norm['clip'])
    mean[:,2] = wrap_angle(mean[:,2])
    sigma = np.exp(.5*np.clip(output[:,3:],-8,6))*norm['ys'] if model.uncertainty else None
    return mean, sigma, x


def choose_alpha(sequences, predictions, target, gates=None):
    scores = []
    for alpha in np.linspace(0,1,21):
        scores.append(np.mean([metrics(s, corrected(s,p,float(alpha),target,
                            None if gates is None else gates[i]))['ape_m']['rmse']
                               for i,(s,p) in enumerate(zip(sequences,predictions))]))
    index = int(np.argmin(scores))
    return float(np.linspace(0,1,21)[index]), float(scores[index])


def benchmark_latency(model, x, repeats=500, device='cpu'):
    start = time.perf_counter()
    model = model.to(device)
    tensor = torch.from_numpy(x[:1]).to(device)
    if model.kind in ('gru','tcn') and tensor.ndim == 2:
        tensor = tensor[:,None,:].expand(-1,5,-1)
    timings = []
    peak = psutil.Process().memory_info().rss
    with torch.inference_mode():
        for _ in range(30):
            model(tensor)
        for _ in range(repeats):
            if device == 'cuda':
                torch.cuda.synchronize()
            tick = time.perf_counter_ns()
            model(tensor)
            if device == 'cuda':
                torch.cuda.synchronize()
            timings.append((time.perf_counter_ns()-tick)/1e6)
            peak = max(peak, psutil.Process().memory_info().rss)
    model.cpu()
    return {'batch_size':1,'device':device,'repeats':repeats,'warmup':30,
            'mean_ms':float(np.mean(timings)),'p50_ms':float(np.quantile(timings,.5)),
            'p95_ms':float(np.quantile(timings,.95)),'p99_ms':float(np.quantile(timings,.99)),
            'process_rss_mib':peak/1024**2,'measurement':'model forward only; not complete SLAM latency',
            'measurement_wall_s':time.perf_counter()-start}


def ridge_fit(train, config):
    x, y = [], []
    for seq in train:
        target, valid = targets(seq,'pose')
        x.append(features(seq,config)[valid]); y.append(target[valid])
    x, y = np.concatenate(x), np.concatenate(y)
    mean, std = x.mean(0), np.maximum(x.std(0),1e-6)
    design = np.column_stack([(x-mean)/std, np.ones(len(x))])
    coef = np.linalg.solve(design.T@design+.1*np.eye(design.shape[1]),design.T@y)
    return lambda seq: np.column_stack([(features(seq,config)-mean)/std,
                                        np.ones(len(seq.t)-1)])@coef


def gp_fit(train, config):
    # Bounded-size GP adaptation: shared inputs/labels, not a reproduction of sensor descriptors.
    x, y = [], []
    for seq in train:
        target, valid = targets(seq,'pose')
        x.append(features(seq,config)[valid]); y.append(target[valid])
    x, y = np.concatenate(x).astype(float), np.concatenate(y).astype(float)
    xm, xs, ym = x.mean(0), np.maximum(x.std(0),1e-6), y.mean(0)
    x=(x-xm)/xs
    idx=np.linspace(0,len(x)-1,min(256,len(x)),dtype=int)
    basis=x[idx]
    def kernel(a,b):
        distance=np.maximum((a*a).sum(1)[:,None]+(b*b).sum(1)-2*a@b.T,0)
        return np.exp(-distance/(2*x.shape[1]))
    weights=cho_solve(cho_factor(kernel(basis,basis)+.05*np.eye(len(basis))),y[idx]-ym)
    return lambda seq: kernel((features(seq,config)-xm)/xs,basis)@weights+ym


def covariance_intersection(seq, weight):
    # Conservative fusion of correlated estimates; identical isotropic covariance proxy.
    out=seq.cart.copy()
    out[:,:2]=weight*seq.cart[:,:2]+(1-weight)*seq.odom[:,:2]
    out[:,2]=wrap_angle(seq.cart[:,2]+(1-weight)*wrap_angle(seq.odom[:,2]-seq.cart[:,2]))
    return out


def recursive_filter(seq, noise_ratio):
    out=seq.cart.copy()
    oi=increments(seq.odom)
    covariance=np.diag([.01,.01,np.deg2rad(5)**2])
    # Inflate Cartographer observation noise because it already contains wheel information.
    measurement=np.diag([.1**2,.1**2,np.deg2rad(5)**2])*25
    process=measurement*noise_ratio
    for i in range(len(oi)):
        prior=add_increment(out[i],oi[i])
        yaw=out[i,2]; dx,dy=oi[i,:2]
        jacobian=np.eye(3)
        jacobian[0,2]=-np.sin(yaw)*dx-np.cos(yaw)*dy
        jacobian[1,2]=np.cos(yaw)*dx-np.sin(yaw)*dy
        covariance=jacobian@covariance@jacobian.T+process
        gain=np.linalg.solve((covariance+measurement).T,covariance.T).T
        residual=seq.cart[i+1]-prior
        residual[2]=wrap_angle(residual[2])
        out[i+1]=prior+gain@residual
        out[i+1,2]=wrap_angle(out[i+1,2])
        identity=np.eye(3)-gain
        covariance=identity@covariance@identity.T+gain@measurement@gain.T
    return out


def fixed_lag(seq, weight, window=10):
    # Causal linearized pose-graph smoothing with odometry increments and Cartographer priors.
    # Priors and odometry are correlated; validated conservative weight is an approximation.
    out=seq.cart.copy()
    cart=seq.cart.copy(); odom=seq.odom.copy()
    cart[:,2]=np.unwrap(cart[:,2]); odom[:,2]=np.unwrap(odom[:,2])
    for end in range(1,len(cart)):
        start=max(0,end-window+1)
        n=end-start+1
        a=np.eye(n)
        rhs=cart[start:end+1].copy()
        for j in range(n-1):
            delta=odom[start+j+1]-odom[start+j]
            a[j,j]+=weight; a[j+1,j+1]+=weight
            a[j,j+1]-=weight; a[j+1,j]-=weight
            rhs[j]-=weight*delta; rhs[j+1]+=weight*delta
        out[end]=np.linalg.solve(a,rhs)[-1]
    out[:,2]=wrap_angle(out[:,2])
    return out


def full_factor_graph(seq,weight):
    cart=seq.cart.copy();odom=seq.odom.copy()
    cart[:,2]=np.unwrap(cart[:,2]);odom[:,2]=np.unwrap(odom[:,2])
    n=len(cart);band=np.zeros((3,n));band[1]=1+2*weight
    band[1,0]=band[1,-1]=1+weight
    band[0,1:]=-weight;band[2,:-1]=-weight
    rhs=cart.copy();delta=np.diff(odom,axis=0)
    rhs[:-1]-=weight*delta;rhs[1:]+=weight*delta
    band[1,0]+=1e6;rhs[0]+=1e6*cart[0]
    result=solve_banded((1,1),band,rhs)
    result[:,2]=wrap_angle(result[:,2])
    return result


def save_tum(path, seq, pose):
    values=np.column_stack([seq.t,pose[:,:2],np.zeros(len(pose)),np.zeros((len(pose),2)),
                            np.sin(pose[:,2]/2),np.cos(pose[:,2]/2)])
    np.savetxt(path,values,fmt='%.9f')


def run(args):
    torch.set_num_threads(1)
    manifest=json.loads(args.manifest.read_text())
    validate_manifest(manifest)
    args.output.mkdir(parents=True,exist_ok=True)
    dump(args.output/'manifest.json',manifest)
    for filename in ['mentor_benchmark.py','residual_models.py','deepfusion_mlp_train_relative_v2.py','causal_stream.py']:
        (args.output/('source_'+filename)).write_bytes(Path(__file__).with_name(filename).read_bytes())
    dump(args.output/'environment.json',{'python':platform.python_version(),'platform':platform.platform(),
        'processor':platform.processor(),'torch':torch.__version__,'numpy':np.__version__,
        'cpu_threads':1,'cuda_available_in_process':torch.cuda.is_available(),
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    sequences={r['sequence_id']:load_sequence(r,args.hz) for r in manifest['sequences']}
    train=[sequences[n] for n in manifest['train']]
    val=[sequences[n] for n in manifest['validation']]
    test=[sequences[n] for n in manifest['test']]
    if any(s.stream_protocol for s in sequences.values()) and (args.configs is None or
            any(c not in ('full20','relative32') for c in args.configs) or not args.no_extra_models):
        raise ValueError('Versioned stream currently supports explicit full20/relative32 MLP configurations with --no-extra-models')
    results=[]
    def record(name,seq,pose,seed=None,extra=None):
        stats=metrics(seq,pose)
        stats.update({'method':name,'sequence':seq.name,'scene':seq.scene,'seed':seed,
                      'trajectory_kind':seq.kind,'split':'test'})
        if extra: stats.update(extra)
        results.append(stats)
        path=args.output/f'{name}_seed{seed}'
        path.mkdir(exist_ok=True)
        save_tum(path/f'{seq.name}.tum',seq,pose)
        dump(args.output/'results.json',results)
    for seq in test:
        record('cartographer',seq,seq.cart)
        record('wheel_odometry',seq,seq.odom)
        save_tum(args.output/f'{seq.name}_gt.tum',seq,seq.gt)
    baselines=[('covariance_intersection',covariance_intersection,np.linspace(0,1,11)),
               ('ekf_conservative_noise',recursive_filter,np.linspace(.1,1,10)),
               ('fixed_lag_factor_graph',fixed_lag,[.01,.1,1,10]),
               ('full_trajectory_factor_graph',full_factor_graph,[.01,.1,1,10])]
    for name,method,grid in baselines:
        best=min(grid,key=lambda w:np.mean([metrics(s,method(s,w))['ape_m']['rmse'] for s in val]))
        for seq in test:
            tick=time.perf_counter(); pose=method(seq,best); wall=time.perf_counter()-tick
            record(name,seq,pose,extra={'parameter_selected_on_validation':float(best),
                                       'postprocessing_wall_s':wall,'mean_ms_per_frame':wall/len(seq.t)*1000,
                                       'covariance_assumption':'conservative proxy; not independent measurements',
                                       'postprocessing_causal':name!='full_trajectory_factor_graph'})
    for name,fit in [('ridge_pose_residual',ridge_fit),('gp_pose_residual_adaptation',gp_fit)]:
        tick=time.perf_counter(); predictor=fit(train,'full20'); training=time.perf_counter()-tick
        alpha,score=choose_alpha(val,[predictor(s) for s in val],'pose')
        for seq in test:
            tick=time.perf_counter(); prediction=predictor(seq); wall=time.perf_counter()-tick
            record(name,seq,corrected(seq,prediction,alpha),extra={'alpha':alpha,'train_wall_s':training,
                'validation_ape_m':score,'postprocessing_wall_s':wall,'implementation':'shared-feature adaptation'})
    configs=args.configs or CONFIGS
    settings=[(config,'increment' if config.startswith('relative') or config.startswith('no_window') or config=='no_motion_state' else 'pose','mlp',False) for config in configs]
    if args.uncertainty_only:
        settings=[(config,target,kind,True) for config,target,kind,_ in settings]
    elif args.with_uncertainty:
        settings += [(config,target,kind,True) for config,target,kind,_ in settings.copy()]
    elif not args.no_extra_models:
        settings += [('full20','increment','mlp',False),('relative32','pose','mlp',False),
                     ('relative32','increment','gru',False),('relative32','increment','tcn',False),
                     ('relative32','increment','mlp',True)]
    for config,target,kind,uncertainty in settings:
        name=f'{kind}_{config}_{target}'+('_uncertainty' if uncertainty else '')
        for seed in args.seeds:
            print(f'Train {name} seed={seed}',flush=True)
            model,norm,training=train_model(train,val,config,target,seed,args.epochs,kind,uncertainty)
            val_outputs=[predict(model,norm,s,config) for s in val]
            alpha,score=choose_alpha(val,[v[0] for v in val_outputs],target)
            gate_parameters=None
            if uncertainty:
                errors=[]; variances=[]; ood=[]
                for seq,(mean,sigma,x) in zip(val,val_outputs):
                    truth,valid=targets(seq,target)
                    errors.append(((truth[valid]-mean[valid])/np.maximum(sigma[valid],1e-8))**2)
                    variances.append((sigma/norm['ys']).mean(1))
                    ood.append(np.abs(x).max(1))
                calibration=float(np.sqrt(np.mean(np.concatenate(errors))))
                selection_masks=[np.ones(len(s.t)-1,dtype=bool) if s.ready is None else s.ready[1:] for s in val]
                sigma_threshold=float(np.quantile(np.concatenate([a[m] for a,m in zip(variances,selection_masks)]),.9))
                ood_threshold=float(np.quantile(np.concatenate([a[m] for a,m in zip(ood,selection_masks)]),.99))
                gates=[(s<=sigma_threshold)&(o<=ood_threshold) for s,o in zip(variances,ood)]
                gate_alpha,gate_score=choose_alpha(val,[v[0] for v in val_outputs],target,gates)
                gate_parameters={'sigma_threshold':sigma_threshold,'ood_threshold':ood_threshold,
                                 'variance_calibration_scale':calibration,'gate_alpha':gate_alpha,
                                 'gate_validation_ape_m':gate_score,'fit_split':'validation'}
            checkpoint={'model_state_dict':model.state_dict(),'config':config,'target':target,'kind':kind,
                        'uncertainty':uncertainty,'norm':{k:torch.from_numpy(np.asarray(v)) for k,v in norm.items()},
                        'alpha':alpha,'gate_parameters':gate_parameters,'seed':seed,'manifest':manifest}
            if train[0].stream_protocol:
                checkpoint.update(stream_protocol=train[0].stream_protocol, grid_hz=args.hz)
            directory=args.output/f'{name}_seed{seed}'; directory.mkdir(exist_ok=True)
            torch.save(checkpoint,directory/'model.pt')
            dump(directory/'training.json',training)
            latency=benchmark_latency(model,val_outputs[0][2])
            dump(directory/'cpu_latency.json',latency)
            for seq in test:
                tick=time.perf_counter(); mean,sigma,x=predict(model,norm,seq,config)
                pose=corrected(seq,mean,alpha,target); wall=time.perf_counter()-tick
                extra={'alpha':alpha,'validation_ape_m':score,'parameter_count':training['parameter_count'],
                       'postprocessing_wall_s':wall,'cpu_forward_latency':latency,
                       'process_peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                       'rss_scope':'cumulative benchmark process, not isolated model memory'}
                record(name,seq,pose,seed,extra)
                if uncertainty:
                    normalized_sigma=(sigma/norm['ys']).mean(1)
                    gate=(normalized_sigma<=gate_parameters['sigma_threshold'])&(np.abs(x).max(1)<=gate_parameters['ood_threshold'])
                    if seq.ready is not None:
                        gate &= seq.ready[1:]
                    safe=corrected(seq,mean,gate_parameters['gate_alpha'],target,gate)
                    labels,valid=targets(seq,target)
                    calibrated=sigma*gate_parameters['variance_calibration_scale']
                    standardized=(labels-mean)/np.maximum(calibrated,1e-8)
                    normalized_sigma_calibrated=np.maximum(calibrated/norm['ys'],1e-8)
                    extra=dict(extra,**gate_parameters,
                        confidence_rejected_fraction=float((~gate).mean()),
                        fallback_fraction=float((~gate).mean()) if gate_parameters['gate_alpha']!=0 else 1.0,
                        marginal_95pct_coverage=float((np.abs(standardized[valid])<=1.96).mean()),
                        normalized_gaussian_nll=float((.5*(standardized[valid]**2+
                            2*np.log(normalized_sigma_calibrated[valid])+np.log(2*np.pi))).mean()),
                        nominal_interval_coverage={str(p):float((np.abs(standardized[valid])<=z).mean())
                            for p,z in [(.5,.67449),(.68,.99446),(.9,1.64485),(.95,1.95996)]},
                        sigma_error_correlation=float(np.corrcoef(normalized_sigma[valid],
                            np.linalg.norm((labels[valid]-mean[valid])/norm['ys'],axis=1))[0,1])
                            if normalized_sigma[valid].std()>1e-10 else None)
                    record(name+'_hard_fallback',seq,safe,seed,extra)
                    # Controlled input corruption only: clean reference and Cartographer stay fixed.
                    for anomaly in ['wheel_scale','wheel_jump','feature_outlier']:
                        altered=Sequence(**vars(seq)); altered.odom=seq.odom.copy()
                        region=slice(len(seq.t)//3,2*len(seq.t)//3)
                        if anomaly=='wheel_scale': altered.odom[region,:2]*=1.3
                        elif anomaly=='wheel_jump': altered.odom[region,0]+=2
                        ax=features(altered,config)
                        normalized=(ax-norm['xm'])/norm['xs']
                        if anomaly=='feature_outlier': normalized[region]*=10
                        with torch.inference_mode(): output=model(torch.from_numpy(normalized.astype(np.float32))).numpy()
                        ap=np.clip(output[:,:3]*norm['ys']+norm['ym'],-norm['clip'],norm['clip'])
                        asigma=np.exp(.5*np.clip(output[:,3:],-8,6))
                        agate=(asigma.mean(1)<=gate_parameters['sigma_threshold'])&(np.abs(normalized).max(1)<=gate_parameters['ood_threshold'])
                        if seq.ready is not None:
                            agate &= seq.ready[1:]
                        ungated=corrected(seq,ap,alpha,target)
                        gated=corrected(seq,ap,gate_parameters['gate_alpha'],target,agate)
                        base_error=np.linalg.norm(seq.cart[:,:2]-seq.gt[:,:2],axis=1)
                        ungated_error=np.linalg.norm(ungated[:,:2]-seq.gt[:,:2],axis=1)
                        gated_error=np.linalg.norm(gated[:,:2]-seq.gt[:,:2],axis=1)
                        extra_a=dict(extra,anomaly=anomaly,anomaly_kind='controlled postprocessor-input corruption; SLAM not rerun',
                                     anomaly_region=[region.start,region.stop],
                                     confidence_rejected_fraction=float((~agate).mean()),
                                     fallback_fraction=float((~agate).mean()) if gate_parameters['gate_alpha']!=0 else 1.0,
                                     harmful_frame_fraction=float(((gated_error-base_error)>.05)[seq.valid].mean()),
                                     ungated_harmful_frame_fraction=float(((ungated_error-base_error)>.05)[seq.valid].mean()))
                        record(name+'_anomaly_'+anomaly,seq,gated,seed,extra_a)
    flat=[]
    for row in results:
        flat.append({'method':row['method'],'sequence':row['sequence'],'seed':row['seed'],
                     'trajectory_kind':row['trajectory_kind'],'ape_rmse_m':row['ape_m']['rmse'],
                     'ape_p95_m':row['ape_m']['p95'],'yaw_ape_rmse_deg':row['yaw_ape_deg']['rmse'],
                     'rpe_1s_rmse_m':row['rpe']['1s']['translation_m']['rmse'],
                     'rpe_1s_rmse_deg':row['rpe']['1s']['rotation_deg']['rmse'],
                     'alpha':row.get('alpha'),'postprocessing_wall_s':row.get('postprocessing_wall_s'),
                     'coverage':row['coverage']['reference_valid_fraction']})
    with (args.output/'benchmark.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    print(f'Saved {len(results)} measured rows to {args.output}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--configs',nargs='+',choices=CONFIGS)
    parser.add_argument('--seeds',nargs='+',type=int,default=[11,22,33])
    parser.add_argument('--epochs',type=int,default=150)
    parser.add_argument('--hz',type=float,default=10)
    parser.add_argument('--no-extra-models',action='store_true')
    parser.add_argument('--uncertainty-only',action='store_true',help='train only variance-head versions of selected configurations')
    parser.add_argument('--with-uncertainty',action='store_true',help='also train variance-head versions of explicitly selected configurations')
    run(parser.parse_args())
