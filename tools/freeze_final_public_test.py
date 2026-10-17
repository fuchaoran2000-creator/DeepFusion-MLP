#!/usr/bin/python3
"""Freeze new sessions and deployed artifacts before downloading references."""
import datetime
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
output = root / 'results/mentor_benchmark/final_public_holdout_v1'
output.mkdir(exist_ok=True)
plan_path = output / 'plan.json'
if plan_path.exists():
    raise SystemExit('Existing frozen plan preserved; do not replace it.')
checkpoint = root / 'results/mentor_benchmark/scene_rotation_gated_v1/test_corridor_val_cafe/quality44_continuous_seed33/model.pt'
configuration = root / 'config/nclt_diag_ceres10_loop.lua'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
plan = dict(
    frozen_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    sessions=['2012-04-29', '2013-04-05'],
    selection_reason='Previously unprocessed complete sessions available in the same documented 2D sensor format; no reference scores inspected.',
    checkpoint=str(checkpoint.relative_to(root)), checkpoint_sha256=sha(checkpoint),
    configuration=str(configuration.relative_to(root)), configuration_sha256=sha(configuration),
    selection_file='results/mentor_benchmark/scene_rotation_gated_v1/test_corridor_val_cafe/deployment_selection.json',
    selection_file_sha256=sha(root/'results/mentor_benchmark/scene_rotation_gated_v1/test_corridor_val_cafe/deployment_selection.json'),
    checkpoint_selection='Already selected using Cafe validation with Home and Office training; no new-session adaptation.',
    alpha=1.0, integration_tau_s=None, gates=['none', 'rule', 'uncertainty'],
    replay_rate=5.0, replay_label='frozen_holdout_v1', grid_hz=10.0,
    inference_input='Actual received Cartographer, provided fused odometry and scan geometry; never reference poses.',
    scores=['initial-pose-aligned planar APE RMSE', '1s/5s translational and yaw RPE',
            'reference-valid and input-available coverage', 'harmful correction over 5cm',
            'fallback fraction', 'graph-node-time APE if covariance timestamps available'],
    missing_policy='Preserve all grid outputs; evaluate every method on common reference-valid times; no RPE across gaps.',
    resource_scope='Accelerated replay is an accuracy/robustness test, not a real-time deployment claim.',
    inference_before_reference_scoring=True, retuning_after_scores=False,
    limits=['Same NCLT robot, source and campus; additional sessions do not isolate hardware effects.',
            'Reference is a SLAM graph; high-rate interpolation uses odometry and is not independent ground truth.',
            'No self-collected sequences or real edge-device measurements are available.'])
plan_path.write_text(json.dumps(plan, indent=2)+'\n')
print(plan_path)
