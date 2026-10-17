#!/usr/bin/python3
"""Predeclared full-session factor contrasts; never modify benchmark inputs.

The zero-beam-time condition tests the documented single timestamp per scan;
it is not a claim that the physical sensor samples all beams simultaneously.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rosbag
from tools.replay_cartographer import ROOT, run


VARIANTS = [
    ('original10_loop', False, 10, True),
    ('original10_noloop', False, 10, False),
    ('single_stamp10_loop', True, 10, True),
    ('single_stamp10_noloop', True, 10, False),
    ('single_stamp30_noloop', True, 30, False),
]


def prepare(output):
    source = ROOT/'data/prepared/nclt-2013-01-10'
    output.mkdir(parents=True, exist_ok=True)
    plan = {'purpose': 'development diagnostic, not held-out improvement',
            'reference_used_to_select_variants': False,
            'session': '2013-01-10', 'duration': 'complete session',
            'variants': [{'name': n, 'zero_beam_time': z, 'max_range_m': r,
                          'inter_submap_constraints': l} for n,z,r,l in VARIANTS],
            'contrasts': ['original10_loop vs original10_noloop: loop constraints',
                          'original10_loop vs single_stamp10_loop: beam timing',
                          'original10_noloop vs single_stamp10_noloop: beam timing without loops',
                          'single_stamp10_noloop vs single_stamp30_noloop: range without loops'],
            'beam_timing_caveat': 'scan start/end semantics and per-beam acquisition times are undocumented; zero is a sensitivity condition',
            'reference_caveat': 'SLAM-derived; dense reference uses supplied odometry for interpolation'}
    plan_path = output/'plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError('Existing diagnostic plan differs')
    plan_path.write_text(json.dumps(plan,indent=2)+'\n')
    zero = output/'single_stamp_inputs.bag'
    if not zero.exists():
        temporary = output/'single_stamp_inputs.tmp.bag'
        with rosbag.Bag(str(source/'slam_inputs.bag')) as inp, rosbag.Bag(str(temporary),'w') as out:
            for topic,msg,stamp in inp.read_messages():
                if topic == '/scan': msg.time_increment = 0.
                out.write(topic,msg,stamp)
        temporary.rename(zero)
    base = (ROOT/'config/home1_2d_noimu.lua').read_text()
    directories = []
    for name,zero_time,range_m,loops in VARIANTS:
        directory = output/f'nclt-2013-01-10_{name}'
        directory.mkdir(exist_ok=True)
        links = {'slam_inputs.bag': zero if zero_time else source/'slam_inputs.bag'}
        links.update({f:source/f for f in ['groundtruth.tum','odom.tum','static_transforms.json','preparation.json']})
        for filename,target in links.items():
            link = directory/filename
            if not link.exists(): link.symlink_to(target.resolve())
        config_name = f'nclt_diag_{name}.lua'
        text = base.replace('TRAJECTORY_BUILDER_2D.max_range = 10.0',
                            f'TRAJECTORY_BUILDER_2D.max_range = {range_m}.0')
        if not loops:
            text = text.replace('return options',
                'POSE_GRAPH.constraint_builder.sampling_ratio = 0.0\n'
                'POSE_GRAPH.global_sampling_ratio = 0.0\nreturn options')
        config_path = ROOT/'config'/config_name
        config_path.write_text(text)
        (directory/'condition.json').write_text(json.dumps(dict(
            name=name, zero_beam_time=zero_time, max_range_m=range_m,
            inter_submap_constraints=loops,
            configuration_sha256=hashlib.sha256(text.encode()).hexdigest()),indent=2)+'\n')
        directories.append((directory,config_name))
    return directories


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'results/mentor_benchmark/nclt_diagnostic_v1')
    p.add_argument('--port',type=int,default=11450);p.add_argument('--prepare-only',action='store_true')
    args=p.parse_args();directories=prepare(args.output.resolve())
    if not args.prepare_only:
        for directory,config_name in directories:
            run([directory],['noimu'],args.port,config_name)
