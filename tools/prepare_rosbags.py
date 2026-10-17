#!/usr/bin/python3
"""Read original bags without mutation; retain only SLAM inputs and reference poses."""
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
from pathlib import Path

import rosbag
from sensor_msgs.msg import Imu


def tum_line(stamp, position, orientation):
    return (f"{stamp.to_sec():.9f} {position.x:.9f} {position.y:.9f} "
            f"{position.z:.9f} {orientation.x:.9f} {orientation.y:.9f} "
            f"{orientation.z:.9f} {orientation.w:.9f}\n")


def prepare(source, output):
    target = output / source.stem
    target.mkdir(parents=True, exist_ok=True)
    completed = target / "preparation.json"
    if completed.exists():
        print(f"Already prepared: {source.stem}", flush=True)
        return
    counts = Counter()
    latest_gyro = None
    static = []
    started = time.perf_counter()
    topics = ['/scan', '/odom', '/gt', '/tf_static',
              '/t265/gyro/sample', '/t265/accel/sample']
    # Raw originals remain untouched; reference data never enters the SLAM bag.
    with rosbag.Bag(str(source), 'r') as bag, \
            rosbag.Bag(str(target / 'slam_inputs.bag'), 'w') as compact, \
            (target / 'groundtruth.tum').open('w') as gt, \
            (target / 'odom.tum').open('w') as odom:
        info = bag.get_type_and_topic_info().topics
        missing = [name for name in ['/scan', '/odom', '/gt', '/tf_static'] if name not in info]
        if missing:
            raise ValueError(f"{source}: required topics missing: {missing}")
        for topic, msg, stamp in bag.read_messages(topics=topics):
            counts[topic] += 1
            if topic in ['/scan', '/odom', '/tf_static']:
                compact.write(topic, msg, stamp)
            if topic == '/odom':
                if msg.child_frame_id != 'base_link':
                    raise ValueError(f"Unexpected odometry reference point: {msg.child_frame_id}")
                odom.write(tum_line(msg.header.stamp, msg.pose.pose.position, msg.pose.pose.orientation))
            elif topic == '/gt':
                for transform in msg.transforms:
                    if transform.child_frame_id == 'base_link':
                        gt.write(tum_line(transform.header.stamp,
                                          transform.transform.translation,
                                          transform.transform.rotation))
                        counts['gt_base_link'] += 1
            elif topic == '/tf_static':
                for transform in msg.transforms:
                    tr, q = transform.transform.translation, transform.transform.rotation
                    static.append({'parent': transform.header.frame_id, 'child': transform.child_frame_id,
                                   'translation': [tr.x, tr.y, tr.z], 'quaternion': [q.x, q.y, q.z, q.w]})
            elif topic == '/t265/gyro/sample':
                latest_gyro = msg
            elif topic == '/t265/accel/sample' and latest_gyro is not None:
                age = (msg.header.stamp - latest_gyro.header.stamp).to_sec()
                # Do not interpolate against future gyro samples.
                if 0 <= age <= 0.05:
                    merged = Imu()
                    merged.header = msg.header
                    merged.header.frame_id = 't265_imu'
                    merged.orientation.w = 1.0
                    merged.orientation_covariance[0] = -1
                    merged.angular_velocity = latest_gyro.angular_velocity
                    merged.angular_velocity_covariance = latest_gyro.angular_velocity_covariance
                    merged.linear_acceleration = msg.linear_acceleration
                    merged.linear_acceleration_covariance = msg.linear_acceleration_covariance
                    compact.write('/imu_combined', merged, stamp)
                    counts['imu_combined'] += 1
        duration = bag.get_end_time() - bag.get_start_time()
    if not counts['gt_base_link']:
        raise ValueError(f"No base_link reference poses in {source}")
    (target / 'static_transforms.json').write_text(json.dumps(static, indent=2) + '\n')
    completed.write_text(json.dumps({'source': str(source.resolve()), 'sequence': source.stem,
                                    'counts': dict(counts), 'bag_duration_s': duration,
                                    'preparation_wall_s': time.perf_counter() - started,
                                    'compact_bytes': (target / 'slam_inputs.bag').stat().st_size,
                                    'reference_used_by_slam': False,
                                    'imu_merge': 'latest non-future gyro, age <= 50ms'}, indent=2) + '\n')
    print(f"Prepared {source.stem}: {dict(counts)}", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('bags', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, default=Path('data/prepared'))
    parser.add_argument('--jobs', type=int, default=1)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    if args.jobs == 1:
        for path in args.bags:
            prepare(path, args.output)
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures=[pool.submit(prepare,path,args.output) for path in args.bags]
            for future in futures:
                future.result()
