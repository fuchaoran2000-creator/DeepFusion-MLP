# Experimental execution

Run from the repository root. Use a fresh `--output` or `--label` to preserve original artifacts. Install `requirements.txt`; source-SLAM commands additionally need ROS and the external builds described in [ROS.md](ROS.md).

## Main C20/P20/I32 Home training

```bash
python mentor_benchmark.py --manifest config/recorded/benchmark_home_causal_v1.json \
  --output results/mentor_benchmark/retrained_home --configs full20 relative32 \
  --seeds 11 22 33 --no-extra-models --with-uncertainty
python tools/select_causal_deployment.py results/mentor_benchmark/retrained_home
```

Defaults: max 150 epochs, batch 256, patience 25. Normalization/clipping use training data; early stopping and alpha/gate/seed selection use validation. The original deployed C20 is `artifacts/checkpoints/home_causal_v1/mlp_full20_pose_uncertainty_seed11/model.pt`, SHA256 `f4a6f98f890cbfba755007f6b1e1bae9f9ba1e5d3f5be61a239e7e0dc4c5e3a2`. Its fixed alpha is 0.65. Three seeds need not reproduce the same selected seed in a different numerical environment.

The minimal L20 time-split study uses `deepfusion_mlp.py --gt ... --cart ... --odom ... --out_dir ...` on the original final-trajectory inputs. It is distinct from the causal complete-sequence experiments.

## Shared-input baselines and resources

The full `mentor_benchmark.py` run also implements CI, conservative EKF, fixed-lag/full linearized graphs, Ridge and bounded-basis GP adaptations. Remove `--no-extra-models` and select the configurations needed. These are documented adaptations, not full reproductions of every raw-sensor learning method.

```bash
python mentor_benchmark.py --manifest config/recorded/benchmark_home_causal_v1.json \
  --output results/mentor_benchmark/retrained_baselines --configs full20 relative32 --seeds 11 22 33 --with-uncertainty
python tools/benchmark_postprocessing_resources.py --run retrained_baselines \
  --methods cartographer covariance_intersection ekf_conservative_noise fixed_lag_factor_graph \
  full_trajectory_factor_graph ridge_pose_residual gp_pose_residual_adaptation mlp_full20_pose \
  --output results/baseline_resources.json
```

Each resource worker verifies its frozen output before timing. CPU time is loaded-input trajectory processing, amortized by frames; full-graph timing uses future poses and excludes fitting/loading/source SLAM. RSS includes the full worker/framework. Ridge/GP constants are fixed; alpha and permitted EKF/graph weights are validation-selected. Original measured values are in `artifacts/paper_tables/`.

## Controlled feature ablations

```bash
python tools/train_causal_feature_ablation.py --manifest config/recorded/benchmark_home_causal_v1.json \
  --output results/ablation_home
python tools/train_causal_feature_ablation.py --manifest config/recorded/benchmark_cross_scene_frontend_v2.json \
  --output results/ablation_scene
```

Each study trains 11 configurations × 3 seeds. Main masks keep 20 inputs and 7,683 parameters, zeroing a group before normalization. Compact/richer controls change dimension but keep the pose-residual target. P20 ablations do not prove C20 feature necessity. Seed variation is not between-sequence stability.

## Frozen Home transfer and synthetic input faults

```bash
# Re-evaluate clean target inputs; includes initial and adapted NCLT inputs separately.
python tools/reproduce_paper.py --transfer

# Same frozen C20, same gate_alpha for none/rule/uncertainty on synthetic pose-input faults.
python tools/benchmark_gate_stress.py --manifest config/recorded/benchmark_home_causal_v1.json \
  --checkpoints artifacts/checkpoints/home_causal_v1/mlp_full20_pose_uncertainty_seed11/model.pt \
  --output results/c20_synthetic_faults
```

Table 7 is clean frozen-C20 transfer. Pose-input synthetic faults leave SLAM itself unchanged. Q44 raw-input replay below is a different experiment; neither is an isolated C20 gate ablation of the main O result.

## Q44 training and all scene-rotation candidates

```bash
python tools/train_trajectory_quality.py --manifest config/recorded/benchmark_home_causal_v1.json \
  --quality-root artifacts/quality_inputs --output results/quality_home
python tools/train_trajectory_quality.py --manifest config/recorded/benchmark_cross_scene_frontend_v2.json \
  --quality-root artifacts/quality_inputs --output results/quality_scene
python tools/train_scene_rotation.py --manifest config/recorded/benchmark_cross_scene_frontend_v2.json \
  --quality-root artifacts/quality_inputs --output results/rotations
python tools/freeze_rotation_calibration.py results/rotations --quality-root artifacts/quality_inputs
```

Q44 adds scan/age proxies; training uses 50 increments and 51 underlying poses per 5 s window (stride 20 increments), batch 16 windows, max 100 epochs, patience 15, and 1/3/5 s path losses. Scene rotations train all 108 candidates and select by validation. Training/validation/test folds, original selections and scores are supplied in `artifacts/studies/scene_rotation_gated_v1`.

## Matched-strength Q44 fault replay and new-session replay

```bash
python tools/evaluate_frozen_quality_stream.py \
  --directory artifacts/quality_replay/home1-3_raw_fault_v2 \
  --checkpoint artifacts/checkpoints/home_trajectory_quality_v1/quality44_trajectory_seed22/model.pt \
  --label raw_fault_v2 --same-strength --output results/q44_faults.json

python tools/evaluate_frozen_quality_stream.py \
  --directory artifacts/quality_replay/nclt-2012-04-29 \
  --checkpoint artifacts/checkpoints/scene_rotation_gated_v1/test_corridor_val_cafe/quality44_continuous_seed33/model.pt \
  --label frozen_holdout_v1 --plan artifacts/studies/final_public_holdout_v1/plan.json \
  --output results/nclt_2012-04-29.json
```

Repeat the second command for `nclt-2013-04-05`. `--same-strength` explicitly uses gate_alpha (0.70 for Q44-F) for all policies, including ungated. Without it the historical ungated checkpoint alpha is 0.45; do not label that as an equal-strength control. Exact recorded features, causal scan scope and uncertainty output are checked before references are loaded. Optional covariance graph-node scoring requires original NCLT covariance CSVs in `data/nclt/`.

Q44-F gives APE −3.41%, 1 s RPE +12.40%, and only 0.0008 m better APE than rules; this does not isolate C20 gating or establish calibrated uncertainty. NCLT severe baseline failures remain negative results. Five-times NCLT replay is an accuracy test, not an online real-time claim.

## Latency and deployment exports

```bash
python tools/benchmark_gpu.py --root artifacts/checkpoints/home_causal_v1 --device cpu
# GPU comparison needs an appropriate installed PyTorch/CUDA build; use the same build for CPU/GPU.
python tools/benchmark_gpu.py --root artifacts/checkpoints/home_causal_v1 --device cuda
python tools/export_portable.py artifacts/checkpoints/home_causal_v1/mlp_full20_pose_uncertainty_seed11/model.pt \
  --output results/c20_numpy
```

Forward timing excludes features, transfers, gates and ROS. Callback P99 excludes consumer waiting. Original GPU measurements used Torch 2.7.1+cu128 rather than the training environment; do not infer physical edge performance from host results.
