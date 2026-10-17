"""Versioned, NumPy-only input contract shared by training and ROS inference.

Ground truth must never enter this module. Grid samples use only received poses
with source timestamp <= grid timestamp, age <= 150 ms. Missing inputs reset
motion history and integrated correction, but never change coordinate origins.
"""
from collections import deque
import numpy as np

PROTOCOL = 'causal_grid_v1'
DIMENSIONS = {'full20': 20, 'relative32': 32}


def wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def local(p, origin):
    c, s = np.cos(origin[2]), np.sin(origin[2])
    d = p[:2] - origin[:2]
    return np.array([c*d[0]+s*d[1], -s*d[0]+c*d[1], wrap(p[2]-origin[2])])


def world(p, origin):
    c, s = np.cos(origin[2]), np.sin(origin[2])
    return np.array([origin[0]+c*p[0]-s*p[1], origin[1]+s*p[0]+c*p[1], wrap(p[2]+origin[2])])


def increments(p):
    d = np.diff(p[:, :2], axis=0)
    c, s = np.cos(p[:-1, 2]), np.sin(p[:-1, 2])
    return np.column_stack([c*d[:, 0]+s*d[:, 1], -s*d[:, 0]+c*d[:, 1], wrap(np.diff(p[:, 2]))])


def add_increment(p, d):
    c, s = np.cos(p[2]), np.sin(p[2])
    return np.array([p[0]+c*d[0]-s*d[1], p[1]+s*d[0]+c*d[1], wrap(p[2]+d[2])])


def feature(t, cart, odom, config):
    if config == 'full20':
        def encode(p):
            return np.r_[p[:2], np.sin(p[2]), np.cos(p[2])]
        dc, do, diff = cart[-1]-cart[-2], odom[-1]-odom[-2], cart[-1]-odom[-1]
        dc[2], do[2], diff[2] = wrap(dc[2]), wrap(do[2]), wrap(diff[2])
        return np.concatenate([encode(odom[-1]), encode(cart[-1]), encode(do),
                               encode(dc), encode(diff)]).astype(np.float32)
    if config != 'relative32':
        raise ValueError('Shared protocol supports full20 and relative32 MLPs')
    ci, oi = increments(cart), increments(odom)
    diff = ci-oi
    diff[:, 2] = wrap(diff[:, 2])
    current = np.column_stack([oi, ci, diff])[-5:]
    dt = t[-1]-t[-2]
    return np.r_[current[-1], current.mean(0), current.std(0), dt,
                 np.linalg.norm(oi[-1, :2])/dt, np.linalg.norm(ci[-1, :2])/dt,
                 oi[-1, 2]/dt, ci[-1, 2]/dt].astype(np.float32)


class FeatureStream:
    def __init__(self, config):
        self.config = config
        self.history = deque(maxlen=6)
        self.origin_cart = self.origin_odom = None
        self.last_stamp = None

    def step(self, stamp, cart, odom, available=True):
        cart, odom = np.asarray(cart, dtype=float), np.asarray(odom, dtype=float)
        monotonic = self.last_stamp is None or stamp > self.last_stamp
        self.last_stamp = stamp
        available = bool(available and monotonic and np.isfinite(stamp)
                         and np.isfinite(cart).all() and np.isfinite(odom).all())
        if self.origin_cart is None and available:
            self.origin_cart, self.origin_odom = cart.copy(), odom.copy()
        cp = cart.copy() if self.origin_cart is None else local(cart, self.origin_cart)
        op = odom.copy() if self.origin_odom is None else local(odom, self.origin_odom)
        if not available:
            self.history.clear()
            return cp, op, np.zeros(DIMENSIONS[self.config], np.float32), False
        self.history.append((stamp, cp, op))
        if len(self.history) < 2:
            return cp, op, np.zeros(DIMENSIONS[self.config], np.float32), False
        rows = list(self.history)
        x = feature(np.array([r[0] for r in rows]), np.array([r[1] for r in rows]),
                    np.array([r[2] for r in rows]), self.config)
        return cp, op, x, True


def extract(t, cart, odom, available, config):
    stream = FeatureStream(config)
    rows = [stream.step(s, c, o, a) for s, c, o, a in zip(t, cart, odom, available)]
    return (np.array([r[0] for r in rows]), np.array([r[1] for r in rows]),
            np.array([r[2] for r in rows]), np.array([r[3] for r in rows]), stream)


class GridSampler:
    """Grid anchored at first Cartographer publication; bounded arrival buffers.

    Future odometry is retained without hiding older usable messages. No waiting
    for later arrival is performed: the caller records the exact chosen pairs.
    """
    def __init__(self, hz=10., max_age=.15):
        self.hz, self.max_age = float(hz), max_age
        self.cart, self.odom = deque(maxlen=4096), deque(maxlen=4096)
        self.anchor = None
        self.index = 0
        self.last_cart_stamp = None

    def push_odom(self, stamp, pose):
        if np.isfinite(stamp):
            self.odom.append((float(stamp), np.asarray(pose).copy()))

    @staticmethod
    def latest(buffer, stamp):
        usable = [r for r in buffer if r[0] <= stamp]
        return max(usable, key=lambda r: r[0]) if usable else None

    def push_cart(self, stamp, pose):
        if not np.isfinite(stamp) or (self.last_cart_stamp is not None and stamp <= self.last_cart_stamp):
            return []
        self.last_cart_stamp = stamp
        self.cart.append((float(stamp), np.asarray(pose).copy()))
        if self.anchor is None:
            self.anchor = float(stamp)
        rows = []
        while self.anchor+self.index/self.hz <= stamp:
            tick = self.anchor+self.index/self.hz
            c, o = self.latest(self.cart, tick), self.latest(self.odom, tick)
            available = (c is not None and o is not None and
                         0 <= tick-c[0] <= self.max_age and 0 <= tick-o[0] <= self.max_age
                         and np.isfinite(c[1]).all() and np.isfinite(o[1]).all())
            rows.append((tick, c[1].copy(), np.full(3, np.nan) if o is None else o[1].copy(),
                         bool(available), c[0], np.nan if o is None else o[0]))
            self.index += 1
        return rows


class ResidualStream:
    """Same normalization, gate, reset and integration for all runtime adapters."""
    def __init__(self, config, target, norm, forward, alpha, gate=None):
        self.features = FeatureStream(config)
        self.target, self.norm, self.forward = target, norm, forward
        self.gate = gate
        self.alpha = alpha if gate is None else gate['gate_alpha']
        self.previous = None

    def step(self, stamp, cart, odom, available=True):
        if not np.isfinite(cart).all():
            raise ValueError('Original Cartographer pose invalid; no valid fallback exists')
        cp, op, x, ready = self.features.step(stamp, cart, odom, available)
        status = {'ready': ready, 'feature': x, 'fallback': True, 'reason': 'input_or_initialization'}
        if not ready:
            self.previous = cp.copy()
            return np.asarray(cart).copy(), status
        z = ((x-self.norm['xm'])/self.norm['xs']).astype(np.float32)
        output = self.forward(z)
        accepted = np.isfinite(output).all()
        if self.gate:
            sigma = float(np.exp(.5*np.clip(output[3:], -8, 6)).mean())
            accepted &= sigma <= self.gate['sigma_threshold'] and np.abs(z).max() <= self.gate['ood_threshold']
            status['sigma'] = sigma
        status.update(max_z=float(np.abs(z).max()), reason='confidence' if not accepted else 'alpha_zero')
        if not accepted or self.alpha == 0:
            self.previous = cp.copy()
            return np.asarray(cart).copy(), status
        residual = np.clip(output[:3]*self.norm['ys']+self.norm['ym'], -self.norm['clip'], self.norm['clip'])
        residual[2] = wrap(residual[2])
        if self.target == 'pose':
            self.previous = cp+self.alpha*residual
            self.previous[2] = wrap(self.previous[2])
        else:
            hist = list(self.features.history)
            ci = increments(np.array([r[1] for r in hist]))[-1]
            self.previous = add_increment(self.previous, ci+self.alpha*residual)
        status.update(fallback=False, reason='accepted')
        return world(self.previous, self.features.origin_cart), status
