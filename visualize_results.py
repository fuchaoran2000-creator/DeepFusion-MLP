#!/usr/bin/env python3
"""
Visualization utilities for the DeepFusion embodied navigation experiment.
Run inside WSL, e.g.:
    conda activate deepfusion310
    python ~/cartographer_home1/visualize_results.py

Expected files:
    ~/cartographer_home1/results/deepfusion/home1-1_gt_test.tum
    ~/cartographer_home1/results/deepfusion/home1-1_odom_aligned_test.tum
    ~/cartographer_home1/results/deepfusion/home1-1_cartographer_aligned_test.tum
    ~/cartographer_home1/results/deepfusion/home1-1_deepfusion_mlp_test.tum
    ~/cartographer_home1/results/deepfusion_eval_strict/*.zip
"""
from __future__ import annotations

import math
import zipfile
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


BASE = Path.home() / "cartographer_home1"
DEEP_DIR = BASE / "results" / "deepfusion"
EVAL_DIR = BASE / "results" / "deepfusion_eval_strict"
FIG_DIR = BASE / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


TRAJ_FILES = {
    "Groundtruth": DEEP_DIR / "home1-1_gt_test.tum",
    "Odometry only": DEEP_DIR / "home1-1_odom_aligned_test.tum",
    "Cartographer": DEEP_DIR / "home1-1_cartographer_aligned_test.tum",
    "DeepFusion-MLP": DEEP_DIR / "home1-1_deepfusion_mlp_test.tum",
}

APE_ZIPS = {
    "Odometry only": EVAL_DIR / "ape_strict_odom.zip",
    "Cartographer": EVAL_DIR / "ape_strict_cartographer.zip",
    "DeepFusion-MLP": EVAL_DIR / "ape_strict_deepfusion_mlp.zip",
}

RPE_ZIPS = {
    "Odometry only": EVAL_DIR / "rpe_strict_odom.zip",
    "Cartographer": EVAL_DIR / "rpe_strict_cartographer.zip",
    "DeepFusion-MLP": EVAL_DIR / "rpe_strict_deepfusion_mlp.zip",
}


def read_tum(path: Path) -> np.ndarray:
    """Read TUM trajectory: timestamp tx ty tz qx qy qz qw."""
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    rows = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 8:
                continue
            rows.append([float(v) for v in parts])
    arr = np.array(rows, dtype=float)
    arr = arr[np.argsort(arr[:, 0])]
    return arr


def load_evo_zip(zip_path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load error_array, timestamps, and stats from an evo result zip."""
    if not zip_path.exists():
        raise FileNotFoundError(f"Missing evo result: {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        with zf.open("error_array.npy") as f:
            errors = np.load(f)
        with zf.open("timestamps.npy") as f:
            timestamps = np.load(f)
        import json
        with zf.open("stats.json") as f:
            stats = json.loads(f.read().decode("utf-8"))
    # Shift timestamps for plotting.
    timestamps = timestamps - timestamps[0]
    return timestamps, errors, stats


def save_trajectory_comparison() -> None:
    trajs = {name: read_tum(path) for name, path in TRAJ_FILES.items()}

    plt.figure(figsize=(7.2, 5.2), dpi=180)
    for name, arr in trajs.items():
        x = arr[:, 1]
        y = arr[:, 2]
        plt.plot(x, y, linewidth=1.8, label=name)

    plt.axis("equal")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("Trajectory comparison on home1-1 test split")
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    out = FIG_DIR / "trajectory_comparison.png"
    plt.savefig(out)
    plt.close()
    print(f"Saved {out}")


def save_rmse_bar_chart() -> None:
    labels = ["Odometry only", "Cartographer", "DeepFusion-MLP"]
    ape_rmse = [load_evo_zip(APE_ZIPS[m])[2]["rmse"] for m in labels]
    rpe_rmse = [load_evo_zip(RPE_ZIPS[m])[2]["rmse"] for m in labels]

    x = np.arange(len(labels))
    width = 0.36

    plt.figure(figsize=(7.2, 4.2), dpi=180)
    plt.bar(x - width / 2, ape_rmse, width, label="APE RMSE")
    plt.bar(x + width / 2, rpe_rmse, width, label="RPE RMSE")
    plt.xticks(x, labels)
    plt.ylabel("RMSE (m)")
    plt.title("Strict no-align APE/RPE RMSE comparison")
    plt.legend()
    plt.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.6)

    for i, v in enumerate(ape_rmse):
        plt.text(i - width / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    for i, v in enumerate(rpe_rmse):
        plt.text(i + width / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    out = FIG_DIR / "rmse_bar_chart.png"
    plt.savefig(out)
    plt.close()
    print(f"Saved {out}")


def save_error_curve(zip_dict: dict[str, Path], title: str, ylabel: str, out_name: str) -> None:
    plt.figure(figsize=(7.2, 4.2), dpi=180)
    for name, path in zip_dict.items():
        t, err, _ = load_evo_zip(path)
        plt.plot(t, err, linewidth=1.5, label=name)

    plt.xlabel("Time from start (s)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    out = FIG_DIR / out_name
    plt.savefig(out)
    plt.close()
    print(f"Saved {out}")


def save_residual_vectors(step: int = 20) -> None:
    gt = read_tum(TRAJ_FILES["Groundtruth"])
    cart = read_tum(TRAJ_FILES["Cartographer"])
    deep = read_tum(TRAJ_FILES["DeepFusion-MLP"])

    # Assume trajectories are already synchronized by the DeepFusion script.
    n = min(len(gt), len(cart), len(deep))
    gt, cart, deep = gt[:n], cart[:n], deep[:n]

    plt.figure(figsize=(7.2, 5.2), dpi=180)
    plt.plot(gt[:, 1], gt[:, 2], linewidth=2.0, label="Groundtruth")
    plt.plot(cart[:, 1], cart[:, 2], linewidth=1.5, label="Cartographer")
    plt.plot(deep[:, 1], deep[:, 2], linewidth=1.5, label="DeepFusion-MLP")

    idx = np.arange(0, n, step)
    dx = deep[idx, 1] - cart[idx, 1]
    dy = deep[idx, 2] - cart[idx, 2]
    plt.quiver(
        cart[idx, 1], cart[idx, 2], dx, dy,
        angles="xy", scale_units="xy", scale=1.0,
        width=0.003, alpha=0.75,
        label="Residual correction"
    )

    plt.axis("equal")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("DeepFusion residual correction vectors")
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    out = FIG_DIR / "deepfusion_residual_vectors.png"
    plt.savefig(out)
    plt.close()
    print(f"Saved {out}")


def save_cartographer_map() -> None:
    """Optional: visualize saved Cartographer map if the PGM exists."""
    candidates = [
        BASE / "results" / "maps" / "home1-1_noimu_map.pgm",
        BASE / "results" / "maps" / "home1-1_map.pgm",
    ]
    pgm = next((p for p in candidates if p.exists()), None)
    if pgm is None:
        print("No map .pgm found. Skip map visualization.")
        return

    img = plt.imread(str(pgm))
    plt.figure(figsize=(6.2, 6.2), dpi=180)
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.title("Generated Cartographer occupancy grid map")
    plt.tight_layout()
    out = FIG_DIR / "cartographer_map.png"
    plt.savefig(out)
    plt.close()
    print(f"Saved {out}")


def main() -> None:
    save_cartographer_map()
    save_trajectory_comparison()
    save_rmse_bar_chart()
    save_error_curve(APE_ZIPS, "APE error over time", "APE translation error (m)", "ape_error_over_time.png")
    save_error_curve(RPE_ZIPS, "RPE error over time", "RPE translation error (m)", "rpe_error_over_time.png")
    save_residual_vectors(step=25)
    print(f"\nAll figures are saved in: {FIG_DIR}")


if __name__ == "__main__":
    main()
