#!/usr/bin/python3
"""Create a new raw-input synthetic-fault bag without changing the source."""
import hashlib
import argparse
import json
from pathlib import Path
import rosbag
import rospy

parser=argparse.ArgumentParser();parser.add_argument('--monotonic-clock',action='store_true')
args=parser.parse_args()
root=Path(__file__).resolve().parents[1]
source=root/'data/prepared/home1-3'
output=root/('data/prepared/home1-3_raw_fault_v2' if args.monotonic_clock else 'data/prepared/home1-3_raw_fault_v1')
output.mkdir(exist_ok=True)
path=output/'slam_inputs.bag'
if path.exists():
    raise SystemExit('Existing fault bag preserved.')
with rosbag.Bag(str(source/'slam_inputs.bag')) as bag:
    start=bag.get_start_time();duration=bag.get_end_time()-start
events=[dict(kind='odometry_drop',topic='/odom',start_offset_s=15.,end_offset_s=17.),
        dict(kind='odometry_future_timestamp',topic='/odom',start_offset_s=35.,end_offset_s=37.,offset_s=.3),
        dict(kind='laser_no_returns',topic='/scan',start_offset_s=55.,end_offset_s=57.)]
assert duration>67
if args.monotonic_clock:
    events[1].update(kind='odometry_clock_bias_monotonic',ramp_s=.5,
                     interpretation='0 to +300ms in 0.5s, hold 1s, return in 0.5s; source timestamps remain ordered.')
plan=dict(sequence=output.name,source_sequence='home1-3',source_start_t=start,
          bag_duration_s=duration,events=events,model_selection='Frozen previously; no tuning on faults.',
          scope='Synthetic raw sensor faults through SLAM and output correction; not real human-motion annotations.',
          reference_in_slam_bag=False)
(output/'fault_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
counts=dict(odom_dropped=0,odom_shifted=0,scans_occluded=0)
with rosbag.Bag(str(source/'slam_inputs.bag')) as src,rosbag.Bag(str(path),'w') as dst:
    for topic,msg,t in src.read_messages():
        elapsed=t.to_sec()-start
        if topic=='/odom' and 15<=elapsed<17:
            counts['odom_dropped']+=1;continue
        if topic=='/odom' and 35<=elapsed<37:
            bias=.3
            if args.monotonic_clock:bias=.3*min((elapsed-35)/.5,(37-elapsed)/.5,1.)
            msg.header.stamp+=rospy.Duration.from_sec(bias);counts['odom_shifted']+=1
        if topic=='/scan' and 55<=elapsed<57:
            msg.ranges=[float('inf')]*len(msg.ranges);counts['scans_occluded']+=1
        assert topic!='/gt'
        dst.write(topic,msg,t)
for name in ['groundtruth.tum','odom.tum','static_transforms.json']:
    (output/name).write_bytes((source/name).read_bytes())
preparation=json.loads((source/'preparation.json').read_text())
preparation.update(sequence=output.name,original_sequence=source.name,
                   preparation_kind='synthetic raw-input fault replay',fault_plan='fault_plan.json')
preparation['counts']['/odom']-=counts['odom_dropped']
preparation['reference_messages_in_compact_bag']=0
(output/'preparation.json').write_text(json.dumps(preparation,indent=2)+'\n')
plan.update(counts=counts,prepared_bag_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
(output/'fault_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
print(json.dumps(plan,indent=2))
