#!/usr/bin/python3
"""Second-stage source-code-driven diagnosis; no ground-truth parameter fitting."""
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.replay_cartographer import ROOT,run


if __name__=='__main__':
    output=ROOT/'results/mentor_benchmark/nclt_frontend_diagnostic_v2'
    output.mkdir(parents=True,exist_ok=True)
    plan=dict(stage=2,reason='stage 1 failed; upstream Ceres rotation prior uses initial_ceres_pose after correlative refinement',
        hypothesis='correlative angular search may amplify ambiguous planar scan matches despite reliable fused odometry',
        single_factor='use_online_correlative_scan_matching true -> false',
        settings_unchanged=['original beam timing','range 10m','all scan matcher weights','all odometry settings'],
        conditions=['ceres10_loop','ceres10_noloop'],duration='complete session',
        reference_parameter_optimization=False,
        sequence_status='previously inspected public development sequence; adaptive diagnosis, not new held-out test')
    path=output/'plan.json'
    if path.exists() and json.loads(path.read_text())!=plan:raise ValueError('Plan changed')
    path.write_text(json.dumps(plan,indent=2)+'\n')
    source=ROOT/'data/prepared/nclt-2013-01-10'
    for loops in [True,False]:
        name='ceres10_loop' if loops else 'ceres10_noloop'
        directory=output/f'nclt-2013-01-10_{name}';directory.mkdir(exist_ok=True)
        for filename in ['slam_inputs.bag','groundtruth.tum','odom.tum','static_transforms.json','preparation.json']:
            link=directory/filename
            if not link.exists():link.symlink_to((source/filename).resolve())
        text=(ROOT/'config/home1_2d_noimu.lua').read_text().replace(
            'TRAJECTORY_BUILDER_2D.use_online_correlative_scan_matching = true',
            'TRAJECTORY_BUILDER_2D.use_online_correlative_scan_matching = false')
        if not loops:
            text=text.replace('return options','POSE_GRAPH.constraint_builder.sampling_ratio = 0.0\nPOSE_GRAPH.global_sampling_ratio = 0.0\nreturn options')
        config=f'nclt_diag_{name}.lua';(ROOT/'config'/config).write_text(text)
        (directory/'condition.json').write_text(json.dumps(dict(name=name,inter_submap_constraints=loops,
            online_correlative=False,configuration_sha256=hashlib.sha256(text.encode()).hexdigest()),indent=2)+'\n')
        run([directory],['noimu'],11451,config)
