#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a reusable DeepFusion-MLP checkpoint on home1-1.

This version is prepared for cross-sequence testing on home1-2 ... home1-5:
1. Saves model weights, architecture, feature mean/std and metadata to .pt.
2. Uses local first-pose coordinates to reduce dependence on map origin.
3. Keeps groundtruth out of model inputs.
4. Supports configurable hidden layers for later architecture experiments.
"""

from __future__ import annotations

import argparse
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
        self.t = t
        self.xyz = xyz
        self.yaw = yaw


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

    if len(rows) < 2:
        raise ValueError(f"{path}: fewer than two valid poses.")

    data = np.asarray(rows, dtype=np.float64)
    data = data[np.argsort(data[:, 0])]
    return Trajectory(
        t=data[:, 0],
        xyz=data[:, 1:4],
        yaw=quaternion_to_yaw(data[:, 4], data[:, 5], data[:, 6], data[:, 7]),
    )


def nearest_indices(query_t: np.ndarray, source_t: np.ndarray):
    idx_right = np.searchsorted(source_t, query_t, side="left")
    idx_right = np.clip(idx_right, 0, len(source_t) - 1)
    idx_left = np.clip(idx_right - 1, 0, len(source_t) - 1)

    err_left = np.abs(source_t[idx_left] - query_t)
    err_right = np.abs(source_t[idx_right] - query_t)
    choose_right = err_right < err_left
    indices = np.where(choose_right, idx_right, idx_left)
    time_error = np.minimum(err_left, err_right)
    return indices, time_error


def synchronize_to_gt(gt: Trajectory, cart: Trajectory, odom: Trajectory, max_dt: float):
    cart_idx, cart_dt = nearest_indices(gt.t, cart.t)
    odom_idx, odom_dt = nearest_indices(gt.t, odom.t)
    keep = (cart_dt <= max_dt) & (odom_dt <= max_dt)

    if int(keep.sum()) < 10:
        raise RuntimeError(
            f"Only {int(keep.sum())} synchronized samples remain. "
            "Inspect timestamp ranges or increase --max_dt."
        )

    return (
        Trajectory(gt.t[keep], gt.xyz[keep].copy(), gt.yaw[keep].copy()),
        Trajectory(gt.t[keep], cart.xyz[cart_idx[keep]].copy(), cart.yaw[cart_idx[keep]].copy()),
        Trajectory(gt.t[keep], odom.xyz[odom_idx[keep]].copy(), odom.yaw[odom_idx[keep]].copy()),
    )


def to_local_frame(traj: Trajectory) -> Trajectory:
    """Express a trajectory relative to its own first pose."""
    xy0 = traj.xyz[0, :2].copy()
    yaw0 = float(traj.yaw[0])
    c = math.cos(-yaw0)
    s = math.sin(-yaw0)
    rotation = np.array([[c, -s], [s, c]], dtype=np.float64)

    local_xyz = traj.xyz.copy()
    local_xyz[:, :2] = (rotation @ (traj.xyz[:, :2] - xy0).T).T
    local_xyz[:, 2] = traj.xyz[:, 2] - traj.xyz[0, 2]
    local_yaw = wrap_angle(traj.yaw - yaw0)
    return Trajectory(traj.t.copy(), local_xyz, local_yaw)


def trajectory_delta(traj: Trajectory):
    dxy = np.zeros((len(traj.t), 2), dtype=np.float64)
    dyaw = np.zeros(len(traj.t), dtype=np.float64)
    dxy[1:] = traj.xyz[1:, :2] - traj.xyz[:-1, :2]
    dyaw[1:] = wrap_angle(traj.yaw[1:] - traj.yaw[:-1])
    return dxy, dyaw


def make_features(cart: Trajectory, odom: Trajectory) -> np.ndarray:
    odom_dxy, odom_dyaw = trajectory_delta(odom)
    cart_dxy, cart_dyaw = trajectory_delta(cart)
    yaw_diff = wrap_angle(cart.yaw - odom.yaw)

    return np.column_stack([
        odom.xyz[:, 0], odom.xyz[:, 1], np.sin(odom.yaw), np.cos(odom.yaw),
        cart.xyz[:, 0], cart.xyz[:, 1], np.sin(cart.yaw), np.cos(cart.yaw),
        odom_dxy[:, 0], odom_dxy[:, 1], np.sin(odom_dyaw), np.cos(odom_dyaw),
        cart_dxy[:, 0], cart_dxy[:, 1], np.sin(cart_dyaw), np.cos(cart_dyaw),
        cart.xyz[:, 0] - odom.xyz[:, 0],
        cart.xyz[:, 1] - odom.xyz[:, 1],
        np.sin(yaw_diff), np.cos(yaw_diff),
    ]).astype(np.float32)


def make_targets(gt: Trajectory, cart: Trajectory) -> np.ndarray:
    return np.column_stack([
        gt.xyz[:, 0] - cart.xyz[:, 0],
        gt.xyz[:, 1] - cart.xyz[:, 1],
        wrap_angle(gt.yaw - cart.yaw),
    ]).astype(np.float32)


class MLPResidual(nn.Module):
    def __init__(self, in_dim: int, hidden_sizes: Sequence[int], out_dim: int = 3):
        super().__init__()
        if not hidden_sizes:
            raise ValueError("hidden_sizes cannot be empty.")
        layers: List[nn.Module] = []
        previous = in_dim
        for hidden in hidden_sizes:
            if hidden <= 0:
                raise ValueError("Hidden layer sizes must be positive.")
            layers.append(nn.Linear(previous, hidden))
            layers.append(nn.ReLU())
            previous = hidden
        layers.append(nn.Linear(previous, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parse_hidden_sizes(value: str) -> Tuple[int, ...]:
    try:
        sizes = tuple(int(v.strip()) for v in value.split(",") if v.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Use comma-separated integers, for example 64,64,32"
        ) from exc
    if not sizes or any(v <= 0 for v in sizes):
        raise argparse.ArgumentTypeError("All hidden sizes must be positive.")
    return sizes


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def standardize_fit(x_train: np.ndarray):
    mean = x_train.mean(axis=0).astype(np.float32)
    std = x_train.std(axis=0).astype(np.float32)
    std = np.where(std < EPS, 1.0, std).astype(np.float32)
    return mean, std


def standardize_apply(x: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return ((x - mean) / std).astype(np.float32)


def xy_rmse(reference: Trajectory, estimate_xy: np.ndarray) -> float:
    error = estimate_xy - reference.xyz[:, :2]
    return float(np.sqrt(np.mean(np.sum(error * error, axis=1))))


def write_tum(path: Path, t: np.ndarray, xyz: np.ndarray, yaw: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    quat = yaw_to_quaternion(yaw)
    with path.open("w", encoding="utf-8") as f:
        for i in range(len(t)):
            f.write(
                f"{t[i]:.9f} "
                f"{xyz[i,0]:.9f} {xyz[i,1]:.9f} {xyz[i,2]:.9f} "
                f"{quat[i,0]:.9f} {quat[i,1]:.9f} "
                f"{quat[i,2]:.9f} {quat[i,3]:.9f}\n"
            )


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_raw = read_tum(Path(args.gt).expanduser())
    cart_raw = read_tum(Path(args.cart).expanduser())
    odom_raw = read_tum(Path(args.odom).expanduser())
    gt_sync, cart_sync, odom_sync = synchronize_to_gt(
        gt_raw, cart_raw, odom_raw, args.max_dt
    )

    # Cross-sequence-ready coordinates: no best-fit test alignment.
    gt = to_local_frame(gt_sync)
    cart = to_local_frame(cart_sync)
    odom = to_local_frame(odom_sync)

    x_all = make_features(cart, odom)
    y_all = make_targets(gt, cart)

    n = len(x_all)
    train_end = int(n * args.train_ratio)
    val_end = train_end + int(n * args.val_ratio)
    if train_end < 10 or val_end <= train_end or val_end >= n:
        raise ValueError("Invalid train/validation/test split.")

    train_slice = slice(0, train_end)
    val_slice = slice(train_end, val_end)
    test_slice = slice(val_end, n)

    feature_mean, feature_std = standardize_fit(x_all[train_slice])
    x_all_norm = standardize_apply(x_all, feature_mean, feature_std)

    x_train = torch.from_numpy(x_all_norm[train_slice])
    y_train = torch.from_numpy(y_all[train_slice])
    x_val = torch.from_numpy(x_all_norm[val_slice])
    y_val = torch.from_numpy(y_all[val_slice])
    x_test = torch.from_numpy(x_all_norm[test_slice])

    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    model = MLPResidual(
        in_dim=x_all.shape[1],
        hidden_sizes=args.hidden_sizes,
        out_dim=3,
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

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(xb)
            loss = criterion(prediction, yb)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}.")
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * xb.shape[0]
            total_count += xb.shape[0]

        train_loss = total_loss / max(total_count, 1)
        model.eval()
        with torch.no_grad():
            val_loss = float(
                criterion(model(x_val.to(device)), y_val.to(device)).item()
            )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })

        if epoch == 1 or epoch % 50 == 0:
            print(
                f"Epoch {epoch:04d} | train_loss={train_loss:.6f} | "
                f"val_loss={val_loss:.6f}"
            )

        if val_loss < best_val_loss - args.min_delta:
            best_val_loss = val_loss
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
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
        raise RuntimeError("No valid model state was produced.")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        predicted_residual = model(x_test.to(device)).cpu().numpy()

    test_gt = Trajectory(gt.t[test_slice], gt.xyz[test_slice].copy(), gt.yaw[test_slice].copy())
    test_cart = Trajectory(cart.t[test_slice], cart.xyz[test_slice].copy(), cart.yaw[test_slice].copy())
    test_odom = Trajectory(odom.t[test_slice], odom.xyz[test_slice].copy(), odom.yaw[test_slice].copy())

    fused_xyz = test_cart.xyz.copy()
    fused_xyz[:, 0] += predicted_residual[:, 0]
    fused_xyz[:, 1] += predicted_residual[:, 1]
    fused_yaw = wrap_angle(test_cart.yaw + predicted_residual[:, 2])

    odom_rmse = xy_rmse(test_gt, test_odom.xyz[:, :2])
    cart_rmse = xy_rmse(test_gt, test_cart.xyz[:, :2])
    fusion_rmse = xy_rmse(test_gt, fused_xyz[:, :2])

    checkpoint = {
        "checkpoint_version": 2,
        "model_name": "DeepFusion-MLP",
        "model_state_dict": best_state,
        "input_dim": int(x_all.shape[1]),
        "output_dim": 3,
        "hidden_sizes": list(args.hidden_sizes),
        "feature_mean": torch.from_numpy(feature_mean),
        "feature_std": torch.from_numpy(feature_std),
        "feature_mode": "local_first_pose",
        "max_dt": float(args.max_dt),
        "source_sequence": args.source_sequence,
        "seed": int(args.seed),
        "best_val_loss": float(best_val_loss),
        "parameter_count": int(parameter_count(model)),
    }
    checkpoint_path = out_dir / "deepfusion_mlp_best.pt"
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
        "synchronized_samples": n,
        "train_samples": train_end,
        "val_samples": val_end - train_end,
        "test_samples": n - val_end,
        "device": str(device),
        "hidden_sizes": list(args.hidden_sizes),
        "parameter_count": parameter_count(model),
        "best_val_loss": best_val_loss,
        "test_xy_rmse_m": {
            "odometry": odom_rmse,
            "cartographer": cart_rmse,
            "deepfusion_mlp": fusion_rmse,
        },
        "checkpoint": str(checkpoint_path),
        "preprocessing": (
            "Each synchronized trajectory is transformed to its own first-pose "
            "local coordinate frame. Feature normalization uses home1-1 training only."
        ),
    }
    with (out_dir / "training_metadata.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    write_tum(
        out_dir / f"{args.source_sequence}_gt_test_local.tum",
        test_gt.t, test_gt.xyz, test_gt.yaw,
    )
    write_tum(
        out_dir / f"{args.source_sequence}_odom_test_local.tum",
        test_odom.t, test_odom.xyz, test_odom.yaw,
    )
    write_tum(
        out_dir / f"{args.source_sequence}_cartographer_test_local.tum",
        test_cart.t, test_cart.xyz, test_cart.yaw,
    )
    write_tum(
        out_dir / f"{args.source_sequence}_deepfusion_test_local.tum",
        test_gt.t, fused_xyz, fused_yaw,
    )

    print()
    print(f"Device: {device}")
    print(f"Hidden sizes: {list(args.hidden_sizes)}")
    print(f"Trainable parameters: {parameter_count(model)}")
    print(f"Synchronized samples: {n}")
    print(
        f"Train / Val / Test: "
        f"{train_end} / {val_end-train_end} / {n-val_end}"
    )
    print("TEST XY RMSE (local first-pose coordinates):")
    print(f"  Odometry:       {odom_rmse:.6f} m")
    print(f"  Cartographer:   {cart_rmse:.6f} m")
    print(f"  DeepFusion-MLP: {fusion_rmse:.6f} m")
    print(f"Saved checkpoint: {checkpoint_path}")
    print(f"Saved outputs to: {out_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and save a reusable DeepFusion-MLP checkpoint."
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
        help="Example: 64,64,32",
    )
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
