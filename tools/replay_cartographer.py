#!/usr/bin/python3
"""Isolated Cartographer replay with measured resources and explicit offline export."""
import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
import xmlrpc.client
from pathlib import Path

import numpy as np
import psutil
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
INSTALL = Path(os.environ.get('CARTOGRAPHER_INSTALL', str(ROOT/'third_party/cartographer_ws/install_isolated'))).expanduser()


def reference_transform(static, source, target):
    graph = {}
    for row in static:
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_quat(row['quaternion']).as_matrix()
        transform[:3, 3] = row['translation']
        graph.setdefault(row['parent'], []).append((row['child'], transform))
        graph.setdefault(row['child'], []).append((row['parent'], np.linalg.inv(transform)))
    queue = [(source, np.eye(4))]
    visited = set()
    while queue:
        frame, transform = queue.pop(0)
        if frame == target:
            return transform
        if frame in visited:
            continue
        visited.add(frame)
        queue.extend((n, transform @ t) for n, t in graph.get(frame, []) if n not in visited)
    raise ValueError(f'Missing static transform from {source} to {target}')


def export_poses(directory, mode, environment):
    import rosbag
    output = directory / f'cartographer_{mode}_final.tum'
    exported = directory / f'{mode}_poses.bag'
    subprocess.run([str(INSTALL / 'lib/cartographer_ros/cartographer_dev_pbstream_trajectories_to_rosbag'),
                    '-input', str(directory / f'{mode}.pbstream'), '-output', str(exported),
                    '-logtostderr'], env=environment, check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)
    static = json.loads((directory / 'static_transforms.json').read_text())
    tracking = 'nclt_imu' if directory.name.startswith('nclt-') else 't265_imu'
    extrinsic = reference_transform(static, tracking, 'base_link') if mode == 'imu' else np.eye(4)
    with rosbag.Bag(str(exported), 'r') as bag, output.open('w') as stream:
        for _, msg, _ in bag.read_messages(topics=['/tf']):
            for tr in msg.transforms:
                pose = np.eye(4)
                q = tr.transform.rotation
                p = tr.transform.translation
                pose[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
                pose[:3, 3] = [p.x, p.y, p.z]
                pose = pose @ extrinsic
                q = Rotation.from_matrix(pose[:3, :3]).as_quat()
                values = [tr.header.stamp.to_sec(), *pose[:3, 3], *q]
                stream.write(' '.join(f'{v:.9f}' for v in values) + '\n')


def run(directory, modes, port, config_basename=None):
    environment = dict(os.environ, ROS_MASTER_URI=f'http://127.0.0.1:{port}',
                       ROS_IP='127.0.0.1', ROS_HOME=str(ROOT / 'results/mentor_benchmark/ros_home'),
                       ROS_LOG_DIR=str(ROOT / 'results/mentor_benchmark/ros_logs'))
    Path(environment['ROS_HOME']).mkdir(parents=True, exist_ok=True)
    Path(environment['ROS_LOG_DIR']).mkdir(parents=True, exist_ok=True)
    master_log = (ROOT / 'results/mentor_benchmark/roscore.log').open('a')
    master = subprocess.Popen(['/opt/ros/noetic/bin/roscore', '-p', str(port)], env=environment,
                              stdout=master_log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        ready = False
        for _ in range(100):
            if master.poll() is not None:
                raise RuntimeError('roscore failed; see results/mentor_benchmark/roscore.log')
            try:
                if xmlrpc.client.ServerProxy(environment['ROS_MASTER_URI']).getUri('/benchmark')[0] == 1:
                    ready = True
                    break
            except OSError:
                time.sleep(0.1)
        if not ready:
            raise RuntimeError('roscore did not become ready')
        subprocess.run(['/opt/ros/noetic/bin/rosparam', 'set', '/use_sim_time', 'true'],
                       env=environment, check=True)
        for sequence in directory:
            for mode in modes:
                metrics_path = sequence / f'cartographer_{mode}_resources.json'
                if metrics_path.exists() and (sequence / f'cartographer_{mode}_final.tum').exists():
                    print(f'Already replayed {sequence.name} {mode}', flush=True)
                    continue
                config_name=config_basename or f'home1_2d_{mode}.lua'
                if config_basename is None and sequence.name.startswith('nclt-') and (mode=='imu' or sequence.name.endswith('_sensorconfig')):
                    config_name=f'nclt_2d_{mode}'+('_sensorconfig' if sequence.name.endswith('_sensorconfig') else '')+'.lua'
                    text=(ROOT/f'config/home1_2d_{mode}.lua').read_text().replace('"t265_imu"','"nclt_imu"')
                    if sequence.name.endswith('_sensorconfig'):
                        # Diagnostic physical-range and no-loop condition, not GT-based parameter fitting.
                        text=text.replace('TRAJECTORY_BUILDER_2D.max_range = 10.0',
                                          'TRAJECTORY_BUILDER_2D.max_range = 30.0')
                        text=text.replace('return options',
                            'POSE_GRAPH.constraint_builder.sampling_ratio = 0.0\n'
                            'POSE_GRAPH.global_sampling_ratio = 0.0\nreturn options')
                    (ROOT/'config'/config_name).write_text(text)
                config_path=ROOT/'config'/config_name
                if not config_path.is_file():
                    raise ValueError(f'Missing configuration: {config_path}')
                (sequence/f'cartographer_{mode}_configuration.lua').write_bytes(config_path.read_bytes())
                command = [str(INSTALL / 'lib/cartographer_ros/cartographer_offline_node'),
                           '-configuration_directory', str(ROOT / 'config'),
                           '-configuration_basenames', config_name,
                           '-bag_filenames', str(sequence / 'slam_inputs.bag'),
                           '-save_state_filename', str(sequence / f'{mode}.pbstream'),
                           '-logtostderr', '__name:=benchmark_cartographer',
                           'scan:=/scan', 'odom:=/odom', 'imu:=/imu_combined']
                start = time.perf_counter()
                peak = 0
                with (sequence / f'cartographer_{mode}.log').open('w') as log:
                    process = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
                    measured = psutil.Process(process.pid)
                    while process.poll() is None:
                        try:
                            peak = max(peak, measured.memory_info().rss)
                        except psutil.NoSuchProcess:
                            pass
                        time.sleep(0.02)
                    if process.returncode:
                        raise RuntimeError(f'Cartographer failed for {sequence.name}/{mode}; inspect its log')
                wall = time.perf_counter() - start
                export_poses(sequence, mode, environment)
                metrics_path.write_text(json.dumps({'command': command, 'wall_s': wall,
                                                    'peak_rss_mib': peak / 1024**2,
                                                    'rss_sample_period_ms': 20,
                                                    'trajectory_kind': 'offline_final_optimized',
                                                    'device': 'CPU', 'reference_point': 'base_link',
                                                    'includes_frontend_backend_and_final_optimization': True}, indent=2)+'\n')
                measured_resources=json.loads(metrics_path.read_text())
                measured_resources['configuration_sha256']=hashlib.sha256(config_path.read_bytes()).hexdigest()
                metrics_path.write_text(json.dumps(measured_resources,indent=2)+'\n')
                print(f'Replayed {sequence.name}/{mode}: {wall:.2f}s, peak RSS {peak/1024**2:.1f}MiB', flush=True)
    finally:
        if master.poll() is None:
            os.killpg(master.pid, signal.SIGINT)
            try:
                master.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(master.pid, signal.SIGTERM)
        master_log.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directories', nargs='+', type=Path)
    parser.add_argument('--modes', nargs='+', choices=['noimu', 'imu'], default=['noimu', 'imu'])
    parser.add_argument('--port', type=int, default=11431)
    parser.add_argument('--config-basename', help='Explicit configuration in config/; suppresses directory-name configuration generation')
    args = parser.parse_args()
    run([p.resolve() for p in args.directories], args.modes, args.port, args.config_basename)
