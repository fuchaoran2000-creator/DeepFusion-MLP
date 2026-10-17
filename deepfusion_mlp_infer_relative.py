#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Apply a home1-1-trained relative-motion DeepFusion-MLP checkpoint to an
unseen sequence such as home1-2, without retraining.

The checkpoint must be produced by deepfusion_mlp_train_relative_v2.py
(checkpoint_version == 3).

Groundtruth is used only for:
1. choosing common evaluation timestamps;
2. computing evaluation metrics.
It is never included in the model input features.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
from torch import nn


class Trajectory:
    def __init__(self, t: np.ndarray, xyz: np.ndarray, yaw: np.ndarray):
        self.t = np.asarray(t, dtype=np.float64)
        self.xyz = np.asarray(xyz, dtype=np.float64)
        self.yaw = np.asarray(yaw, dtype=np.float64)


def wrap_angle(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def quaternion_to_yaw(qx, qy, qz, qw):
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return np.arctan2(siny_cosp, cosy_cosp)


def yaw_to_quaternion(yaw: np.ndarray) -> np.ndarray:
    q = np.zeros((len(yaw), 4), dtype=np.float64)
    q[:, 2] = np.sin(yaw / 2.0)
    q[:, 3] = np.cos(yaw / 2.0)
    return q


def read_tum(path: Path) -> Trajectory:
    if not path.exists():
        raise FileNotFoundError(f"TUM file not found: {path}")

    rows = []
    seen = set()

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 8:
                raise ValueError(
                    f"{path}: line {line_no} has {len(parts)} columns; expected 8."
                )

            values = [float(v) for v in parts[:8]]
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{path}: non-finite value at line {line_no}.")

            timestamp = values[0]
            if timestamp in seen:
                continue

            seen.add(timestamp)
            rows.append(values)

    if len(rows) < 3:
        raise ValueError(f"{path}: fewer than three valid poses.")

    data = np.asarray(rows, dtype=np.float64)
    data = data[np.argsort(data[:, 0])]

    return Trajectory(
        t=data[:, 0],
        xyz=data[:, 1:4],
        yaw=quaternion_to_yaw(
            data[:, 4], data[:, 5], data[:, 6], data[:, 7]
        ),
    )


def nearest_indices(query_t: np.ndarray, source_t: np.ndarray):
    right = np.searchsorted(source_t, query_t, side="left")
    right = np.clip(right, 0, len(source_t) - 1)
    left = np.clip(right - 1, 0, len(source_t) - 1)

    left_error = np.abs(source_t[left] - query_t)
    right_error = np.abs(source_t[right] - query_t)

    choose_right = right_error < left_error
    indices = np.where(choose_right, right, left)
    errors = np.minimum(left_error, right_error)
    return indices, errors


def synchronize_to_gt(
    gt: Trajectory,
    cart: Trajectory,
    odom: Trajectory,
    max_dt: float,
) -> Tuple[Trajectory, Trajectory, Trajectory]:
    cart_idx, cart_dt = nearest_indices(gt.t, cart.t)
    odom_idx, odom_dt = nearest_indices(gt.t, odom.t)

    keep = (cart_dt <= max_dt) & (odom_dt <= max_dt)
    kept = int(keep.sum())

    if kept < 20:
        raise RuntimeError(
            f"Only {kept} synchronized poses remain. "
            "Inspect timestamp ranges or increase --max_dt."
        )

    return (
        Trajectory(gt.t[keep], gt.xyz[keep].copy(), gt.yaw[keep].copy()),
        Trajectory(
            gt.t[keep],
            cart.xyz[cart_idx[keep]].copy(),
            cart.yaw[cart_idx[keep]].copy(),
        ),
        Trajectory(
            gt.t[keep],
            odom.xyz[odom_idx[keep]].copy(),
            odom.yaw[odom_idx[keep]].copy(),
        ),
    )


def to_local_frame(traj: Trajectory) -> Trajectory:
    """Express a trajectory relative to its own first pose."""
    xy0 = traj.xyz[0, :2].copy()
    yaw0 = float(traj.yaw[0])

    c = math.cos(-yaw0)
    s = math.sin(-yaw0)
    rotation = np.array([[c, -s], [s, c]], dtype=np.float64)

    xyz = traj.xyz.copy()
    xyz[:, :2] = (rotation @ (traj.xyz[:, :2] - xy0).T).T
    xyz[:, 2] = traj.xyz[:, 2] - traj.xyz[0, 2]
    yaw = wrap_angle(traj.yaw - yaw0)

    return Trajectory(traj.t.copy(), xyz, yaw)


def body_frame_increments(traj: Trajectory) -> Tuple[np.ndarray, np.ndarray]:
    global_dxy = traj.xyz[1:, :2] - traj.xyz[:-1, :2]
    previous_yaw = traj.yaw[:-1]

    c = np.cos(previous_yaw)
    s = np.sin(previous_yaw)

    dx_body = c * global_dxy[:, 0] + s * global_dxy[:, 1]
    dy_body = -s * global_dxy[:, 0] + c * global_dxy[:, 1]
    d_yaw = wrap_angle(traj.yaw[1:] - traj.yaw[:-1])

    dt = traj.t[1:] - traj.t[:-1]
    if np.any(dt <= 0):
        raise ValueError("Synchronized timestamps must be strictly increasing.")

    increments = np.column_stack([dx_body, dy_body, d_yaw])
    return increments.astype(np.float64), dt.astype(np.float64)


def rolling_statistics(values: np.ndarray, window_size: int):
    n, d = values.shape
    means = np.zeros((n, d), dtype=np.float64)
    stds = np.zeros((n, d), dtype=np.float64)

    for i in range(n):
        start = max(0, i - window_size + 1)
        window = values[start:i + 1]
        means[i] = window.mean(axis=0)
        stds[i] = window.std(axis=0)

    return means, stds


def make_features(
    cart: Trajectory,
    odom: Trajectory,
    window_size: int,
):
    cart_inc, cart_dt = body_frame_increments(cart)
    odom_inc, odom_dt = body_frame_increments(odom)

    if not np.allclose(cart_dt, odom_dt, atol=1e-9):
        raise RuntimeError("Cartographer and odometry produced inconsistent dt.")

    dt = cart_dt
    inc_diff = cart_inc - odom_inc
    inc_diff[:, 2] = wrap_angle(inc_diff[:, 2])

    current = np.column_stack([odom_inc, cart_inc, inc_diff])
    rolling_mean, rolling_std = rolling_statistics(current, window_size)

    odom_speed = np.linalg.norm(odom_inc[:, :2], axis=1) / dt
    cart_speed = np.linalg.norm(cart_inc[:, :2], axis=1) / dt
    odom_yaw_rate = odom_inc[:, 2] / dt
    cart_yaw_rate = cart_inc[:, 2] / dt

    features = np.column_stack([
        current,
        rolling_mean,
        rolling_std,
        dt,
        odom_speed,
        cart_speed,
        odom_yaw_rate,
        cart_yaw_rate,
    ]).astype(np.float32)

    return features, cart_inc, odom_inc


def integrate_body_increments(
    increments: np.ndarray,
    timestamps: np.ndarray,
) -> Trajectory:
    n = len(increments) + 1
    if len(timestamps) != n:
        raise ValueError("timestamps length must equal increments length + 1.")

    xyz = np.zeros((n, 3), dtype=np.float64)
    yaw = np.zeros(n, dtype=np.float64)

    for i, (dx_body, dy_body, d_yaw) in enumerate(increments, start=1):
        previous_yaw = yaw[i - 1]
        c = math.cos(previous_yaw)
        s = math.sin(previous_yaw)

        xyz[i, 0] = xyz[i - 1, 0] + c * dx_body - s * dy_body
        xyz[i, 1] = xyz[i - 1, 1] + s * dx_body + c * dy_body
        yaw[i] = wrap_angle(previous_yaw + d_yaw)

    return Trajectory(timestamps.copy(), xyz, yaw)


class MLPResidual(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_sizes: Sequence[int],
        output_dim: int = 3,
    ):
        super().__init__()

        layers: List[nn.Module] = []
        previous = input_dim

        for hidden in hidden_sizes:
            layers.append(nn.Linear(previous, int(hidden)))
            layers.append(nn.ReLU())
            previous = int(hidden)

        layers.append(nn.Linear(previous, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def xy_errors(reference: Trajectory, estimate: Trajectory) -> np.ndarray:
    if len(reference.t) != len(estimate.t):
        raise ValueError("Reference and estimate must have equal lengths.")
    delta = estimate.xyz[:, :2] - reference.xyz[:, :2]
    return np.linalg.norm(delta, axis=1)


def xy_rmse(reference: Trajectory, estimate: Trajectory) -> float:
    error = xy_errors(reference, estimate)
    return float(np.sqrt(np.mean(error ** 2)))


def xy_stats(reference: Trajectory, estimate: Trajectory):
    error = xy_errors(reference, estimate)
    return {
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mean": float(np.mean(error)),
        "median": float(np.median(error)),
        "std": float(np.std(error)),
        "max": float(np.max(error)),
    }


def write_tum(path: Path, traj: Trajectory) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    q = yaw_to_quaternion(traj.yaw)

    with path.open("w", encoding="utf-8") as f:
        for i in range(len(traj.t)):
            f.write(
                f"{traj.t[i]:.9f} "
                f"{traj.xyz[i,0]:.9f} {traj.xyz[i,1]:.9f} "
                f"{traj.xyz[i,2]:.9f} "
                f"{q[i,0]:.9f} {q[i,1]:.9f} "
                f"{q[i,2]:.9f} {q[i,3]:.9f}\n"
            )


def run(args):
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if checkpoint.get("checkpoint_version") != 3:
        raise ValueError(
            "This inference script requires checkpoint_version == 3."
        )

    if checkpoint.get("feature_mode") != "relative_motion_window_v1":
        raise ValueError(
            "Checkpoint feature_mode is not relative_motion_window_v1."
        )

    input_dim = int(checkpoint["input_dim"])
    hidden_sizes = [int(v) for v in checkpoint["hidden_sizes"]]
    output_dim = int(checkpoint.get("output_dim", 3))
    window_size = int(checkpoint["window_size"])
    blend_alpha = float(checkpoint["blend_alpha"])
    max_dt = float(args.max_dt) if args.max_dt is not None else float(
        checkpoint["max_dt"]
    )

    feature_mean = checkpoint["feature_mean"].cpu().numpy().astype(np.float32)
    feature_std = checkpoint["feature_std"].cpu().numpy().astype(np.float32)
    target_mean = checkpoint["target_mean"].cpu().numpy().astype(np.float32)
    target_std = checkpoint["target_std"].cpu().numpy().astype(np.float32)
    target_clip = checkpoint["target_clip"].cpu().numpy().astype(np.float32)

    model = MLPResidual(input_dim, hidden_sizes, output_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    gt_raw = read_tum(Path(args.gt).expanduser())
    cart_raw = read_tum(Path(args.cart).expanduser())
    odom_raw = read_tum(Path(args.odom).expanduser())

    gt, cart, odom = synchronize_to_gt(
        gt_raw, cart_raw, odom_raw, max_dt
    )

    features, cart_inc, _ = make_features(
        cart, odom, window_size
    )

    if features.shape[1] != input_dim:
        raise RuntimeError(
            f"Feature dimension mismatch: checkpoint expects {input_dim}, "
            f"but current sequence produced {features.shape[1]}."
        )

    normalized_features = (
        (features - feature_mean) / feature_std
    ).astype(np.float32)

    sample_max_z = np.max(np.abs(normalized_features), axis=1)

    print("\nOut-of-distribution diagnostics:")
    print(
        "Samples with any |z| > 3:",
        np.mean(sample_max_z > 3.0)
    )
    print(
        "Samples with any |z| > 5:",
        np.mean(sample_max_z > 5.0)
    )
    print(
        "Samples with any |z| > 10:",
        np.mean(sample_max_z > 10.0)
    )

    with torch.no_grad():
        x_tensor = torch.from_numpy(normalized_features).to(device)
        normalized_prediction = model(x_tensor).cpu().numpy()

    predicted_residual = normalized_prediction * target_std + target_mean
    predicted_residual = np.clip(
        predicted_residual, -target_clip, target_clip
    )
    predicted_residual[:, 2] = wrap_angle(predicted_residual[:, 2])

    np.set_printoptions(precision=8, suppress=True)

    residual_mean = predicted_residual.mean(axis=0)
    residual_std = predicted_residual.std(axis=0)
    residual_abs_mean = np.abs(predicted_residual).mean(axis=0)
    residual_sum = predicted_residual.sum(axis=0)

    print("\nPredicted residual diagnostics [dx, dy, dyaw]:")
    print("Mean residual:          ", residual_mean)
    print("Std residual:           ", residual_std)
    print("Mean absolute residual: ", residual_abs_mean)
    print("Cumulative residual:    ", residual_sum)
    print(
        "Alpha-weighted sum:     ",
        blend_alpha * residual_sum
    )

    print(
        "95% absolute quantile:  ",
        np.quantile(np.abs(predicted_residual), 0.95, axis=0)
    )
    print(
        "99% absolute quantile:  ",
        np.quantile(np.abs(predicted_residual), 0.99, axis=0)
    )

    print(
        "Mean translation correction per step (mm):",
        residual_mean[:2] * 1000
    )
    print(
        "Mean yaw correction per step (degree):",
        np.degrees(residual_mean[2])
    )    

    corrected_inc = cart_inc + blend_alpha * predicted_residual
    corrected_inc[:, 2] = wrap_angle(corrected_inc[:, 2])

    # Evaluation in a common first-pose-relative convention.
    gt_local = to_local_frame(gt)
    cart_local = to_local_frame(cart)
    odom_local = to_local_frame(odom)
    fused_local = integrate_body_increments(corrected_inc, gt.t)

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = args.sequence
    write_tum(out_dir / f"{prefix}_gt_relative.tum", gt_local)
    write_tum(out_dir / f"{prefix}_odom_relative.tum", odom_local)
    write_tum(
        out_dir / f"{prefix}_cartographer_relative.tum",
        cart_local,
    )
    write_tum(
        out_dir / f"{prefix}_deepfusion_relative.tum",
        fused_local,
    )

    stats = {
        "sequence": prefix,
        "checkpoint": str(checkpoint_path),
        "checkpoint_source_sequence": checkpoint["source_sequence"],
        "checkpoint_model_name": checkpoint["model_name"],
        "feature_mode": checkpoint["feature_mode"],
        "hidden_sizes": hidden_sizes,
        "window_size": window_size,
        "blend_alpha_from_home1_1_validation": blend_alpha,
        "max_dt": max_dt,
        "synchronized_poses": len(gt.t),
        "groundtruth_used_as_model_input": False,
        "groundtruth_usage": (
            "timestamp synchronization and evaluation only"
        ),
        "xy_error_m": {
            "odometry": xy_stats(gt_local, odom_local),
            "cartographer": xy_stats(gt_local, cart_local),
            "deepfusion_mlp": xy_stats(gt_local, fused_local),
        },
        "feature_distribution_diagnostics": {
            "mean_absolute_z_score": float(
                np.mean(np.abs(normalized_features))
            ),
            "fraction_abs_z_above_3": float(
                np.mean(np.abs(normalized_features) > 3.0)
            ),
            "maximum_absolute_z_score": float(
                np.max(np.abs(normalized_features))
            ),
        },
    }

    with (out_dir / f"{prefix}_inference_metrics.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    cart_rmse = stats["xy_error_m"]["cartographer"]["rmse"]
    deep_rmse = stats["xy_error_m"]["deepfusion_mlp"]["rmse"]
    odom_rmse = stats["xy_error_m"]["odometry"]["rmse"]

    improvement_vs_cart = (
        (cart_rmse - deep_rmse) / cart_rmse * 100.0
        if cart_rmse > 0 else float("nan")
    )

    print()
    print(f"Device: {device}")
    print(f"Sequence: {prefix}")
    print(
        f"Checkpoint trained on: "
        f"{checkpoint['source_sequence']}"
    )
    print(f"Retraining performed: No")
    print(f"Feature mode: {checkpoint['feature_mode']}")
    print(f"Hidden sizes: {hidden_sizes}")
    print(f"Window size: {window_size}")
    print(f"Fixed blend alpha: {blend_alpha:.2f}")
    print(f"Synchronized poses: {len(gt.t)}")
    print("XY RMSE (first-pose-relative evaluation):")
    print(f"  Odometry:       {odom_rmse:.6f} m")
    print(f"  Cartographer:   {cart_rmse:.6f} m")
    print(f"  DeepFusion-MLP: {deep_rmse:.6f} m")
    print(
        f"DeepFusion change vs Cartographer: "
        f"{improvement_vs_cart:+.2f}% "
        "(positive means lower error)"
    )
    print("Feature distribution diagnostics:")
    print(
        "  Mean |z|: "
        f"{stats['feature_distribution_diagnostics']['mean_absolute_z_score']:.3f}"
    )
    print(
        "  Fraction |z| > 3: "
        f"{stats['feature_distribution_diagnostics']['fraction_abs_z_above_3']:.3%}"
    )
    print(
        "  Max |z|: "
        f"{stats['feature_distribution_diagnostics']['maximum_absolute_z_score']:.3f}"
    )
    print(f"Saved outputs to: {out_dir}")


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Apply a saved relative-motion DeepFusion-MLP checkpoint "
            "to an unseen trajectory sequence without retraining."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gt", required=True)
    parser.add_argument("--cart", required=True)
    parser.add_argument("--odom", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument(
        "--max_dt",
        type=float,
        default=None,
        help=(
            "Optional synchronization threshold override. "
            "Default: value saved in checkpoint."
        ),
    )
    parser.add_argument("--cpu", action="store_true")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
