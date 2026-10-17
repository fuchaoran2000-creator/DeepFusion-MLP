# DeepFusion-MLP

Causal output-level residual correction for 2D LiDAR SLAM. This repository contains the experimental implementation, training and evaluation tools, recorded numeric inputs, checkpoints, and frozen paper-result checks.

[中文说明](README.zh-CN.md) · [Full experiment commands](docs/EXPERIMENTS.md) · [Data and protocols](docs/DATA.md) · [ROS setup](docs/ROS.md)

The primary C20 model uses 20 pose/motion/discrepancy features, a `20→64→64→32→6` MLP (7,782 parameters), Gaussian NLL, and validation-fixed correction/fallback rules. It operates after Cartographer and does not feed corrected poses into scan matching or the pose graph.

## Reproduce the main numbers

Python 3.8–3.10 on Linux was used. The minimal replay needs only NumPy; ROS and a GPU are not required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-replay.txt
python tools/reproduce_paper.py
```

This checks the three archived paired Home captures (Table 4), exact raw fallback, features, output poses, and the five archived transfer rows (Table 7). Reports go to `results/public_reproduction/verification.json`. The source artifacts are copied to a temporary directory before replay and remain unchanged.

| Paired Home metric | Raw Cartographer | C20 |
|---|---:|---:|
| Sequence-mean translational APE RMSE | 0.16314284 m | 0.10138135 m |
| Sequence-mean translational 1 s RPE RMSE | 0.04465594 m | 0.04468453 m |

The APE reduction is **37.9% on three same-scene Home test sequences**. Mean local-motion error is essentially unchanged. Office/Cafe/Corridor transfer worsens; NCLT mostly falls back. These results do not establish reliable cross-scene transfer, an isolated C20 gate benefit, or physical edge-device performance.

## Train and evaluate

```bash
# Full numerical stack; CPU installation can use the PyTorch CPU wheel index.
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v

# Verify the complete training path with two epochs (not paper accuracy).
python tools/reproduce_paper.py --train-smoke

# Retrain the complete Home P20/I32/C20 candidates on exact recorded inputs.
python mentor_benchmark.py \
  --manifest config/recorded/benchmark_home_causal_v1.json \
  --output results/mentor_benchmark/retrained_home \
  --configs full20 relative32 --seeds 11 22 33 \
  --no-extra-models --with-uncertainty
python tools/select_causal_deployment.py results/mentor_benchmark/retrained_home

# Re-run frozen C20 inference on recorded Office/Cafe/Corridor/NCLT inputs.
python tools/reproduce_paper.py --transfer
```

See [EXPERIMENTS.md](docs/EXPERIMENTS.md) for feature ablations, fixed scene splits, all 108 scene-rotation candidates, Q44 fault controls, original raw-data/ROS runs, and resource measurements. Fresh ROS arrivals and software/hardware differences may produce different trajectories and scores. The frozen replay verifies the archived execution; retraining and source-SLAM reproduction are separate operations.

## Repository layout

```text
causal_stream.py, portable_residual.py   Shared causal sampling/features/runtime
mentor_benchmark.py, residual_models.py Training, baselines and scoring
trajectory_quality.py                  Auxiliary Q44 input/runtime
trajectory_objective.py, gated_objective.py  Continuous/gate-aware losses
tools/                                 Data preparation, ROS, training and evaluation
config/                                Portable new-run manifests and SLAM configs
config/recorded/                        Manifests for included recorded numeric inputs
artifacts/recorded_inputs/              Actual pose-arrival captures and compressed references
artifacts/quality_inputs/               Source-causal scan-quality descriptors
artifacts/quality_replay/               Actual fault/frozen-NCLT callback captures
artifacts/checkpoints/                  Original frozen checkpoints and training records
artifacts/paper_tables/, artifacts/studies/  Score tables and split/selection evidence
reproduction/C20/, reproduction/transfer/    Frozen paper checks
tests/                                 Causality, masks, resets, objectives and metric checks
third_party/versions.json               Exact external SLAM source commits
```

## Data, attribution and limits

Raw ROS bags, images and laser scans are obtained separately from [OpenLORIS-Scene](https://lifelong-robotic-vision.github.io/dataset/scene.html) and [NCLT](https://robots.engin.umich.edu/nclt/). This repository includes low-dimensional experiment records and derived reference fixtures with source attribution; it does not distribute raw datasets or third-party build products. Dataset-derived material retains its source rights and is not relicensed by the software license.

Office reference poses come from OptiTrack. Home/Cafe/Corridor and NCLT references are SLAM-derived, so APE measures agreement with supplied references. Home1-1…5 are five sequences within one scene. Main training uses Home1-1, validation Home1-2, and testing Home1-3/4/5. Previously inspected development data are not a new blind holdout.

Original project source code is available under [MIT](LICENSE). External components retain their own licenses; see [NOTICE.md](NOTICE.md). The manuscript is not bundled with this code release. Please cite the underlying datasets and software when using their data or implementations.
