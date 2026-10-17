#!/usr/bin/env python3
import os
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def quat_to_yaw(qx, qy, qz, qw):
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def yaw_to_quat(yaw):
    return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)


def read_tum(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split()
            if len(p) != 8:
                continue
            t = float(p[0])
            x, y, z = float(p[1]), float(p[2]), float(p[3])
            qx, qy, qz, qw = map(float, p[4:8])
            yaw = quat_to_yaw(qx, qy, qz, qw)
            data.append([t, x, y, z, yaw])
    arr = np.array(data, dtype=np.float64)
    arr = arr[np.argsort(arr[:, 0])]
    return arr


def nearest_sync(ref, src, max_dt):
    """
    For each timestamp in ref, find nearest timestamp in src.
    Return indices in src and valid mask.
    """
    src_t = src[:, 0]
    idx = np.searchsorted(src_t, ref[:, 0])
    idx0 = np.clip(idx - 1, 0, len(src_t) - 1)
    idx1 = np.clip(idx, 0, len(src_t) - 1)

    dt0 = np.abs(src_t[idx0] - ref[:, 0])
    dt1 = np.abs(src_t[idx1] - ref[:, 0])
    use1 = dt1 < dt0
    best = np.where(use1, idx1, idx0)
    best_dt = np.minimum(dt0, dt1)

    valid = best_dt <= max_dt
    return best, valid


def fit_se2_alignment(src_xy, tgt_xy):
    """
    Fit rigid 2D transform:
        tgt ≈ R @ src + t
    Use training data only.
    """
    src_mean = src_xy.mean(axis=0)
    tgt_mean = tgt_xy.mean(axis=0)

    X = src_xy - src_mean
    Y = tgt_xy - tgt_mean

    H = X.T @ Y
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = tgt_mean - R @ src_mean
    theta = math.atan2(R[1, 0], R[0, 0])
    return R, t, theta


def apply_se2_alignment(traj, R, t, theta):
    out = traj.copy()
    xy = traj[:, 1:3]
    out[:, 1:3] = (R @ xy.T).T + t
    out[:, 4] = wrap_angle(traj[:, 4] + theta)
    return out


def make_features(gt, cart, odom):
    """
    Feature design:
    - current odom pose
    - current cartographer pose
    - odom delta
    - cartographer delta
    - cartographer minus odom
    Label:
    - gt - cartographer residual
    """
    n = len(gt)

    odom_dx = np.zeros(n)
    odom_dy = np.zeros(n)
    odom_dyaw = np.zeros(n)

    cart_dx = np.zeros(n)
    cart_dy = np.zeros(n)
    cart_dyaw = np.zeros(n)

    odom_dx[1:] = np.diff(odom[:, 1])
    odom_dy[1:] = np.diff(odom[:, 2])
    odom_dyaw[1:] = wrap_angle(np.diff(odom[:, 4]))

    cart_dx[1:] = np.diff(cart[:, 1])
    cart_dy[1:] = np.diff(cart[:, 2])
    cart_dyaw[1:] = wrap_angle(np.diff(cart[:, 4]))

    yaw_diff = wrap_angle(cart[:, 4] - odom[:, 4])

    X = np.column_stack([
        odom[:, 1], odom[:, 2], np.sin(odom[:, 4]), np.cos(odom[:, 4]),
        cart[:, 1], cart[:, 2], np.sin(cart[:, 4]), np.cos(cart[:, 4]),

        odom_dx, odom_dy, np.sin(odom_dyaw), np.cos(odom_dyaw),
        cart_dx, cart_dy, np.sin(cart_dyaw), np.cos(cart_dyaw),

        cart[:, 1] - odom[:, 1],
        cart[:, 2] - odom[:, 2],
        np.sin(yaw_diff),
        np.cos(yaw_diff),
    ])

    y = np.column_stack([
        gt[:, 1] - cart[:, 1],
        gt[:, 2] - cart[:, 2],
        wrap_angle(gt[:, 4] - cart[:, 4]),
    ])

    return X, y


class MLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )

    def forward(self, x):
        return self.net(x)


def write_tum(path, traj):
    with open(path, "w") as f:
        for row in traj:
            ts, x, y, z, yaw = row
            qx, qy, qz, qw = yaw_to_quat(yaw)
            f.write(
                f"{ts:.9f} {x:.9f} {y:.9f} {z:.9f} "
                f"{qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}\n"
            )


def rmse_xy(pred, gt):
    e = pred[:, 1:3] - gt[:, 1:3]
    return float(np.sqrt(np.mean(np.sum(e * e, axis=1))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", required=True)
    parser.add_argument("--cart", required=True)
    parser.add_argument("--odom", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--max_dt", type=float, default=0.05)
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    gt_raw = read_tum(args.gt)
    cart_raw = read_tum(args.cart)
    odom_raw = read_tum(args.odom)

    cart_idx, valid_cart = nearest_sync(gt_raw, cart_raw, args.max_dt)
    odom_idx, valid_odom = nearest_sync(gt_raw, odom_raw, args.max_dt)
    valid = valid_cart & valid_odom

    gt = gt_raw[valid]
    cart = cart_raw[cart_idx[valid]]
    odom = odom_raw[odom_idx[valid]]

    n = len(gt)
    if n < 100:
        raise RuntimeError(f"Too few synchronized samples: {n}")

    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)
    n_test = n - n_train - n_val

    if n_test <= 10:
        raise RuntimeError("Too few test samples. Adjust split ratios.")

    train_slice = slice(0, n_train)
    val_slice = slice(n_train, n_train + n_val)
    test_slice = slice(n_train + n_val, n)

    # Align Cartographer and Odometry to GT using training set only.
    # This avoids using test groundtruth for alignment.
    R_cart, t_cart, th_cart = fit_se2_alignment(cart[train_slice, 1:3], gt[train_slice, 1:3])
    R_odom, t_odom, th_odom = fit_se2_alignment(odom[train_slice, 1:3], gt[train_slice, 1:3])

    cart_aligned = apply_se2_alignment(cart, R_cart, t_cart, th_cart)
    odom_aligned = apply_se2_alignment(odom, R_odom, t_odom, th_odom)

    X, y = make_features(gt, cart_aligned, odom_aligned)

    X_train, y_train = X[train_slice], y[train_slice]
    X_val, y_val = X[val_slice], y[val_slice]
    X_test, y_test = X[test_slice], y[test_slice]

    x_mean = X_train.mean(axis=0)
    x_std = X_train.std(axis=0) + 1e-8
    y_mean = y_train.mean(axis=0)
    y_std = y_train.std(axis=0) + 1e-8

    X_train_n = (X_train - x_mean) / x_std
    X_val_n = (X_val - x_mean) / x_std
    X_test_n = (X_test - x_mean) / x_std

    y_train_n = (y_train - y_mean) / y_std
    y_val_n = (y_val - y_mean) / y_std

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(X.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    train_ds = TensorDataset(
        torch.tensor(X_train_n, dtype=torch.float32),
        torch.tensor(y_train_n, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    best_val = float("inf")
    best_state = None
    patience = 80
    bad = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            pred = model(xb)
            loss = loss_fn(pred, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item() * len(xb)

        model.eval()
        with torch.no_grad():
            xv = torch.tensor(X_val_n, dtype=torch.float32).to(device)
            yv = torch.tensor(y_val_n, dtype=torch.float32).to(device)
            val_loss = loss_fn(model(xv), yv).item()

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1

        if epoch % 50 == 0 or epoch == 1:
            train_loss = total / len(train_ds)
            print(f"Epoch {epoch:04d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

        if bad >= patience:
            print(f"Early stopping at epoch {epoch}. Best val_loss={best_val:.6f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        xt = torch.tensor(X_test_n, dtype=torch.float32).to(device)
        pred_res_n = model(xt).cpu().numpy()

    pred_res = pred_res_n * y_std + y_mean

    gt_test = gt[test_slice].copy()
    cart_test = cart_aligned[test_slice].copy()
    odom_test = odom_aligned[test_slice].copy()
    deep_test = cart_test.copy()

    deep_test[:, 1] = cart_test[:, 1] + pred_res[:, 0]
    deep_test[:, 2] = cart_test[:, 2] + pred_res[:, 1]
    deep_test[:, 4] = wrap_angle(cart_test[:, 4] + pred_res[:, 2])

    print("\nSynchronized samples:", n)
    print("Train / Val / Test:", n_train, n_val, n_test)
    print("\nInternal XY RMSE on TEST only:")
    print(f"  Odometry aligned:      {rmse_xy(odom_test, gt_test):.6f} m")
    print(f"  Cartographer aligned:  {rmse_xy(cart_test, gt_test):.6f} m")
    print(f"  DeepFusion MLP:        {rmse_xy(deep_test, gt_test):.6f} m")

    write_tum(os.path.join(args.out_dir, "home1-1_gt_test.tum"), gt_test)
    write_tum(os.path.join(args.out_dir, "home1-1_odom_aligned_test.tum"), odom_test)
    write_tum(os.path.join(args.out_dir, "home1-1_cartographer_aligned_test.tum"), cart_test)
    write_tum(os.path.join(args.out_dir, "home1-1_deepfusion_mlp_test.tum"), deep_test)

    print("\nSaved test trajectories to:")
    print(args.out_dir)


if __name__ == "__main__":
    main()