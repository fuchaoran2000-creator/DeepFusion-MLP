#!/usr/bin/python3
"""Reconstruct source-causal scan geometry at each existing captured pose pair.

Only input bag scans and captured input timestamps are used. Actual callback
arrival order was not logged in old bags; this limitation is explicitly saved.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import rosbag
from trajectory_quality import ScanQualityBuffer,QUALITY_NAMES


def run(args):
    manifest=json.loads(args.manifest.read_text());args.output.mkdir(parents=True,exist_ok=True)
    for row in manifest['sequences']:
        output=args.output/(row['sequence_id']+'.npz')
        stream=Path(row['sampled_stream']);bag_path=Path(row['cart']).parent/'slam_inputs.bag'
        if output.exists():continue
        with np.load(stream) as d:t,cs,os=[d[k].copy() for k in ['t','cart_source_t','odom_source_t']]
        buffer=ScanQualityBuffer();quality=[];scan_times=[];descriptors=[]
        with rosbag.Bag(str(bag_path)) as bag:
            for _,msg,_ in bag.read_messages(topics=['/scan']):
                buffer.receive(msg);scan_times.append(buffer.rows[-1][0]);descriptors.append(buffer.rows[-1][1])
        scan_times=np.asarray(scan_times);descriptors=np.asarray(descriptors)
        order=np.argsort(scan_times,kind='stable');scan_times=scan_times[order];descriptors=descriptors[order]
        source=[]
        for tick,cart_stamp,odom_stamp in zip(t,cs,os):
            i=np.searchsorted(scan_times,tick,side='right')-1
            buffer.rows.clear()
            if i>=0:buffer.push(scan_times[i],descriptors[i])
            q,stamp=buffer.at(tick,cart_stamp,odom_stamp);quality.append(q);source.append(stamp)
        q=np.asarray(quality)
        np.savez_compressed(output,t=t,quality=q,scan_source_t=source,quality_names=QUALITY_NAMES)
        metadata=dict(sequence=row['sequence_id'],samples=len(t),quality_present_fraction=float(q[:,11].mean()),
            bag=str(bag_path.resolve()),captured_stream_sha256=hashlib.sha256(stream.read_bytes()).hexdigest(),
            input_uses_gt=False,quality_kind='source-causal geometric proxies and timestamp ages',
            arrival_limitation='historical scan callback arrival times unavailable; source timestamp causality only',
            scan_end='header + last sensor-valid beam index * time_increment, matching Cartographer conversion',
            cartographer_range_max_m=10.,quality_names=QUALITY_NAMES)
        output.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')
        print(row['sequence_id'],len(t),metadata['quality_present_fraction'],flush=True)
    (args.output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
