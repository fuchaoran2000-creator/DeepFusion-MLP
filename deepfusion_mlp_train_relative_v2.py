#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepFusion-MLP v2: relative-motion residual learning for cross-sequence testing.

Why this version:
- Absolute x/y pose features extrapolate poorly across time and sequences.
- This version learns corrections to frame-to-frame body-frame motion.
- It saves all preprocessing statistics needed for home1-2 ... home1-5 inference.
- A residual blend factor is selected using validation data only.

Groundtruth is used only for supervised targets and evaluation; it is never an
input feature at inference time.
"""

from __future__ import annotations

import argparse
import gzip
import csv
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

EPS = 1e-8


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

    with (gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open("r", encoding="utf-8")) as f:
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

    if int(keep.sum()) < 20:
        raise RuntimeError(
            f"Only {int(keep.sum())} synchronized poses remain. "
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


def slice_trajectory(traj: Trajectory, start: int, stop: int) -> Trajectory:
    return Trajectory(
        traj.t[start:stop].copy(),
        traj.xyz[start:stop].copy(),
        traj.yaw[start:stop].copy(),
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
    """
    Relative motion from pose i to i+1, expressed in the body frame at pose i.

    Returns:
        increments: [N-1, 3] = dx_body, dy_body, d_yaw
        dt:         [N-1]
    """
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
    """Causal rolling mean/std, including the current sample."""
    n, d = values.shape
    means = np.zeros((n, d), dtype=np.float64)
    stds = np.zeros((n, d), dtype=np.float64)

    for i in range(n):
        start = max(0, i - window_size + 1)
        window = values[start:i + 1]
        means[i] = window.mean(axis=0)
        stds[i] = window.std(axis=0)

    return means, stds


def make_motion_dataset(
    gt: Trajectory,
    cart: Trajectory,
    odom: Trajectory,
    window_size: int,
):
    """
    Coordinate-invariant MLP inputs and residual-motion targets.

    Input contains only relative motion information:
      current odom increment (3)
      current cart increment (3)
      current cart-odom increment difference (3)
      causal rolling mean of the above values (9)
      causal rolling std of the above values (9)
      dt, odom speed, cart speed, odom yaw rate, cart yaw rate (5)

    Total input dimension: 32.

    Target:
      groundtruth body-frame increment - Cartographer body-frame increment (3)
    """
    gt_inc, gt_dt = body_frame_increments(gt)
    cart_inc, cart_dt = body_frame_increments(cart)
    odom_inc, odom_dt = body_frame_increments(odom)

    if not (
        np.allclose(gt_dt, cart_dt, atol=1e-9)
        and np.allclose(gt_dt, odom_dt, atol=1e-9)
    ):
        raise RuntimeError("Synchronized trajectories produced inconsistent dt.")

    dt = gt_dt
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

    targets = gt_inc - cart_inc
    targets[:, 2] = wrap_angle(targets[:, 2])
    targets = targets.astype(np.float32)

    return features, targets, gt_inc, cart_inc, odom_inc, dt


def fit_standardizer(values: np.ndarray):
    mean = values.mean(axis=0).astype(np.float32)
    std = values.std(axis=0).astype(np.float32)
    std = np.where(std < EPS, 1.0, std).astype(np.float32)
    return mean, std


def apply_standardizer(values: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return ((values - mean) / std).astype(np.float32)


def integrate_body_increments(
    increments: np.ndarray,
    timestamps: np.ndarray,
) -> Trajectory:
    """Integrate body-frame increments from the origin with zero initial yaw."""
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


def xy_rmse(reference: Trajectory, estimate: Trajectory) -> float:
    if len(reference.t) != len(estimate.t):
        raise ValueError("Reference and estimate must have equal lengths.")
    error = estimate.xyz[:, :2] - reference.xyz[:, :2]
    return float(np.sqrt(np.mean(np.sum(error * error, axis=1))))


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


class MLPResidual(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: Sequence[int], output_dim: int = 3):
        super().__init__()

        if not hidden_sizes:
            raise ValueError("hidden_sizes cannot be empty.")

        layers: List[nn.Module] = []
        previous = input_dim

        for hidden in hidden_sizes:
            if hidden <= 0:
                raise ValueError("Hidden layer sizes must be positive.")
            layers.append(nn.Linear(previous, hidden))
            layers.append(nn.ReLU())
            previous = hidden

        layers.append(nn.Linear(previous, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def parse_hidden_sizes(value: str):
    try:
        sizes = tuple(int(v.strip()) for v in value.split(",") if v.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Use comma-separated integers, for example 64,64,32"
        ) from exc

    if not sizes or any(v <= 0 for v in sizes):
        raise argparse.ArgumentTypeError("All hidden sizes must be positive.")

    return sizes


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def predict_residuals(
    model: nn.Module,
    x_normalized: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    target_clip: np.ndarray,
    device: torch.device,
):
    model.eval()

    with torch.no_grad():
        x_tensor = torch.from_numpy(x_normalized).to(device)
        prediction_normalized = model(x_tensor).cpu().numpy()

    prediction = prediction_normalized * y_std + y_mean
    prediction = np.clip(prediction, -target_clip, target_clip)
    prediction[:, 2] = wrap_angle(prediction[:, 2])
    return prediction


def select_blend_factor(
    predicted_residual: np.ndarray,
    cart_increments: np.ndarray,
    reference_local: Trajectory,
    timestamps: np.ndarray,
):
    """Select residual strength using validation data only."""
    best_alpha = 0.0
    best_rmse = float("inf")

    for alpha in np.linspace(0.0, 1.0, 21):
        corrected = cart_increments + alpha * predicted_residual
        corrected[:, 2] = wrap_angle(corrected[:, 2])
        estimate = integrate_body_increments(corrected, timestamps)
        score = xy_rmse(reference_local, estimate)

        if score < best_rmse:
            best_rmse = score
            best_alpha = float(alpha)

    return best_alpha, best_rmse


def train(args):
    set_seed(args.seed)

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_raw = read_tum(Path(args.gt).expanduser())
    cart_raw = read_tum(Path(args.cart).expanduser())
    odom_raw = read_tum(Path(args.odom).expanduser())

    gt, cart, odom = synchronize_to_gt(
        gt_raw, cart_raw, odom_raw, args.max_dt
    )

    x_all, y_all, _, cart_inc_all, odom_inc_all, _ = make_motion_dataset(
        gt, cart, odom, args.window_size
    )

    n_steps = len(x_all)
    train_end = int(n_steps * args.train_ratio)
    val_end = train_end + int(n_steps * args.val_ratio)

    if train_end < 20 or val_end <= train_end or val_end >= n_steps:
        raise ValueError(
            "Invalid train/validation/test split. "
            "Check --train_ratio and --val_ratio."
        )

    train_slice = slice(0, train_end)
    val_slice = slice(train_end, val_end)
    test_slice = slice(val_end, n_steps)

    x_mean, x_std = fit_standardizer(x_all[train_slice])
    y_mean, y_std = fit_standardizer(y_all[train_slice])

    x_normalized = apply_standardizer(x_all, x_mean, x_std)
    y_normalized = apply_standardizer(y_all, y_mean, y_std)

    # Avoid uncontrolled corrections outside the home1-1 training distribution.
    train_abs_q = np.quantile(
        np.abs(y_all[train_slice]), 0.995, axis=0
    ).astype(np.float32)
    target_clip = np.maximum(train_abs_q * 1.5, y_std * 3.0).astype(np.float32)

    x_train = torch.from_numpy(x_normalized[train_slice])
    y_train = torch.from_numpy(y_normalized[train_slice])
    x_val = torch.from_numpy(x_normalized[val_slice])
    y_val = torch.from_numpy(y_normalized[val_slice])

    loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )

    model = MLPResidual(
        input_dim=x_all.shape[1],
        hidden_sizes=args.hidden_sizes,
        output_dim=3,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state: Dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0

        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            prediction = model(xb)
            loss = criterion(prediction, yb)

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}.")

            loss.backward()
            optimizer.step()

            batch_size = xb.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_count += batch_size

        train_loss = total_loss / max(total_count, 1)

        model.eval()
        with torch.no_grad():
            val_prediction = model(x_val.to(device))
            val_loss = float(
                criterion(val_prediction, y_val.to(device)).item()
            )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })

        if epoch == 1 or epoch % 50 == 0:
            print(
                f"Epoch {epoch:04d} | "
                f"train_loss={train_loss:.6f} | "
                f"val_loss={val_loss:.6f}"
            )

        if val_loss < best_val_loss - args.min_delta:
            best_val_loss = val_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1

        if stale_epochs >= args.patience:
            print(
                f"Early stopping at epoch {epoch}. "
                f"Best val_loss={best_val_loss:.6f}"
            )
            break

    if best_state is None:
        raise RuntimeError("No valid model checkpoint was produced.")

    model.load_state_dict(best_state)

    val_prediction = predict_residuals(
        model,
        x_normalized[val_slice],
        y_mean,
        y_std,
        target_clip,
        device,
    )

    # Validation pose indices are [train_end, val_end].
    val_gt_local = to_local_frame(
        slice_trajectory(gt, train_end, val_end + 1)
    )
    val_cart_inc = cart_inc_all[val_slice]
    val_timestamps = gt.t[train_end:val_end + 1]

    blend_alpha, val_blended_rmse = select_blend_factor(
        val_prediction,
        val_cart_inc,
        val_gt_local,
        val_timestamps,
    )

    test_prediction = predict_residuals(
        model,
        x_normalized[test_slice],
        y_mean,
        y_std,
        target_clip,
        device,
    )

    test_start = val_end
    test_stop = n_steps + 1

    test_gt_local = to_local_frame(
        slice_trajectory(gt, test_start, test_stop)
    )
    test_cart_local = to_local_frame(
        slice_trajectory(cart, test_start, test_stop)
    )
    test_odom_local = to_local_frame(
        slice_trajectory(odom, test_start, test_stop)
    )

    test_cart_inc = cart_inc_all[test_slice]
    corrected_test_inc = test_cart_inc + blend_alpha * test_prediction
    corrected_test_inc[:, 2] = wrap_angle(corrected_test_inc[:, 2])

    test_fused = integrate_body_increments(
        corrected_test_inc,
        gt.t[test_start:test_stop],
    )

    odom_rmse = xy_rmse(test_gt_local, test_odom_local)
    cart_rmse = xy_rmse(test_gt_local, test_cart_local)
    fused_rmse = xy_rmse(test_gt_local, test_fused)

    checkpoint = {
        "checkpoint_version": 3,
        "model_name": "DeepFusion-MLP-relative-motion",
        "model_state_dict": best_state,
        "input_dim": int(x_all.shape[1]),
        "output_dim": 3,
        "hidden_sizes": list(args.hidden_sizes),
        "feature_mean": torch.from_numpy(x_mean),
        "feature_std": torch.from_numpy(x_std),
        "target_mean": torch.from_numpy(y_mean),
        "target_std": torch.from_numpy(y_std),
        "target_clip": torch.from_numpy(target_clip),
        "feature_mode": "relative_motion_window_v1",
        "window_size": int(args.window_size),
        "blend_alpha": float(blend_alpha),
        "max_dt": float(args.max_dt),
        "train_ratio": float(args.train_ratio),
        "val_ratio": float(args.val_ratio),
        "seed": int(args.seed),
        "best_val_loss": float(best_val_loss),
        "validation_blended_xy_rmse_m": float(val_blended_rmse),
        "parameter_count": int(count_parameters(model)),
        "source_sequence": args.source_sequence,
    }

    checkpoint_path = out_dir / "deepfusion_mlp_relative_best.pt"
    torch.save(checkpoint, checkpoint_path)

    with (out_dir / "training_history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=["epoch", "train_loss", "val_loss"]
        )
        writer.writeheader()
        writer.writerows(history)

    metadata = {
        "synchronized_poses": len(gt.t),
        "motion_samples": n_steps,
        "train_motion_samples": train_end,
        "val_motion_samples": val_end - train_end,
        "test_motion_samples": n_steps - val_end,
        "device": str(device),
        "hidden_sizes": list(args.hidden_sizes),
        "input_dim": int(x_all.shape[1]),
        "parameter_count": count_parameters(model),
        "best_val_loss": best_val_loss,
        "blend_alpha_selected_on_validation": blend_alpha,
        "validation_blended_xy_rmse_m": val_blended_rmse,
        "test_xy_rmse_m": {
            "odometry": odom_rmse,
            "cartographer": cart_rmse,
            "deepfusion_mlp": fused_rmse,
        },
        "checkpoint": str(checkpoint_path),
        "preprocessing": (
            "Relative body-frame motion only; no absolute x/y pose features. "
            "Input and target normalization fitted on home1-1 training data only."
        ),
    }

    with (out_dir / "training_metadata.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    write_tum(
        out_dir / f"{args.source_sequence}_gt_test_relative.tum",
        test_gt_local,
    )
    write_tum(
        out_dir / f"{args.source_sequence}_odom_test_relative.tum",
        test_odom_local,
    )
    write_tum(
        out_dir / f"{args.source_sequence}_cartographer_test_relative.tum",
        test_cart_local,
    )
    write_tum(
        out_dir / f"{args.source_sequence}_deepfusion_test_relative.tum",
        test_fused,
    )

    print()
    print(f"Device: {device}")
    print(f"Feature mode: relative_motion_window_v1")
    print(f"Input dimension: {x_all.shape[1]}")
    print(f"Hidden sizes: {list(args.hidden_sizes)}")
    print(f"Trainable parameters: {count_parameters(model)}")
    print(f"Synchronized poses: {len(gt.t)}")
    print(
        f"Train / Val / Test motion samples: "
        f"{train_end} / {val_end - train_end} / {n_steps - val_end}"
    )
    print(f"Validation-selected residual alpha: {blend_alpha:.2f}")
    print(f"Validation blended XY RMSE: {val_blended_rmse:.6f} m")
    print("TEST XY RMSE (relative-motion evaluation):")
    print(f"  Odometry:       {odom_rmse:.6f} m")
    print(f"  Cartographer:   {cart_rmse:.6f} m")
    print(f"  DeepFusion-MLP: {fused_rmse:.6f} m")
    print(f"Saved checkpoint: {checkpoint_path}")
    print(f"Saved outputs to: {out_dir}")


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Train a cross-sequence DeepFusion-MLP using relative-motion residuals."
        )
    )

    parser.add_argument("--gt", required=True)
    parser.add_argument("--cart", required=True)
    parser.add_argument("--odom", required=True)
    parser.add_argument("--out_dir", required=True)

    parser.add_argument("--source_sequence", default="home1-1")
    parser.add_argument(
        "--hidden_sizes",
        type=parse_hidden_sizes,
        default=(64, 64, 32),
    )
    parser.add_argument("--window_size", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--min_delta", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_dt", type=float, default=0.05)
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.10)
    parser.add_argument("--cpu", action="store_true")

    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
