#!/usr/bin/python3
"""Check every converted message against author-format raw inputs and TF math."""
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import rosbag
from scipy.spatial.transform import Rotation
from tools.prepare_nclt import convert_pose

ROOT=Path(__file__).resolve().parents[1]


def audit():
    source=ROOT/'data/nclt/2013-01-10';prepared=ROOT/'data/prepared/nclt-2013-01-10'
    odom=np.loadtxt(source/'odometry_mu_100hz.csv',delimiter=',')
    imu=np.loadtxt(source/'ms25.csv',delimiter=',')
    laser=np.fromfile(source/'hokuyo_30m.bin',dtype=[('t','<u8'),('r','<u2',(1081,))])
    counts={k:0 for k in ['/scan','/odom','/imu_combined']}
    errors={'timestamp_s':0.,'odom_position_m':0.,'odom_rotation_rad':0.,
            'range_quantization_m':0.,'acceleration_m_s2':0.,'angular_rate_rad_s':0.}
    flip=np.diag([1.,-1.,-1.]);finite_counts=[];beyond10=[];stamps={k:[] for k in counts}
    with rosbag.Bag(str(prepared/'slam_inputs.bag')) as bag:
        for topic,msg,record_time in bag.read_messages():
            if topic not in counts:
                if topic!='/tf_static':raise ValueError(f'Unexpected input topic: {topic}')
                continue
            i=counts[topic];counts[topic]+=1
            timestamp=laser[i]['t'] if topic=='/scan' else (odom if topic=='/odom' else imu)[i,0]
            errors['timestamp_s']=max(errors['timestamp_s'],abs(msg.header.stamp.to_sec()-float(timestamp)/1e6),
                                       abs(record_time.to_sec()-float(timestamp)/1e6))
            stamps[topic].append(msg.header.stamp.to_sec())
            if topic=='/scan':
                expected=laser[i]['r'].astype(float)*.005-100;valid=(expected>=.1)&(expected<=30)
                actual=np.array(msg.ranges)
                if not np.array_equal(np.isfinite(actual),valid):raise AssertionError('Invalid range handling differs')
                errors['range_quantization_m']=max(errors['range_quantization_m'],float(np.abs(actual[valid]-expected[valid]).max(initial=0)))
                np.testing.assert_allclose([msg.angle_min,msg.angle_max,msg.angle_increment],[-3*np.pi/4,3*np.pi/4,np.pi/720],atol=1e-7)
                finite_counts.append(int(valid.sum()));beyond10.append(int(((expected>10)&valid).sum()))
            elif topic=='/odom':
                pos,q=convert_pose(odom[i]);p=msg.pose.pose.position;r=msg.pose.pose.orientation
                actual=np.array([r.x,r.y,r.z,r.w])
                errors['odom_position_m']=max(errors['odom_position_m'],float(np.abs(pos-[p.x,p.y,p.z]).max()))
                errors['odom_rotation_rad']=max(errors['odom_rotation_rad'],float((Rotation.from_quat(actual).inv()*Rotation.from_quat(q)).magnitude()))
            else:
                a,w=msg.linear_acceleration,msg.angular_velocity
                errors['acceleration_m_s2']=max(errors['acceleration_m_s2'],float(np.abs(flip@imu[i,4:7]-[a.x,a.y,a.z]).max()))
                errors['angular_rate_rad_s']=max(errors['angular_rate_rad_s'],float(np.abs(flip@imu[i,7:10]-[w.x,w.y,w.z]).max()))
    assert counts=={'/scan':len(laser),'/odom':len(odom),'/imu_combined':len(imu)}
    assert errors['timestamp_s']<1e-6 and errors['range_quantization_m']<1e-5
    static={r['child']:r for r in json.loads((prepared/'static_transforms.json').read_text())}
    # Laser axes remain in the author laser frame: F R_body_laser = I.
    # IMU vectors are explicitly flipped: F R_body_imu F = I.
    expected_tf={'laser':(flip@np.array([.28,0,-.44]),flip@Rotation.from_euler('x',np.pi).as_matrix()),
                 'nclt_imu':(flip@np.array([-.11,-.18,-.71]),flip@np.eye(3)@flip)}
    for frame,(position,rotation) in expected_tf.items():
        np.testing.assert_allclose(static[frame]['translation'],position,atol=1e-12)
        np.testing.assert_allclose(Rotation.from_quat(static[frame]['quaternion']).as_matrix(),rotation,atol=1e-12)
    result={'passed':True,'counts':counts,'maximum_decode_errors':errors,
        'static_transform_checks_passed':True,'pose_euler_order':'Rz(yaw) Ry(pitch) Rx(roll)',
        'timestamps':{k:dict(nonincreasing=int((np.diff(v)<=0).sum()),
            dt_quantiles_s=np.quantile(np.diff(v),[0,.01,.5,.99,1]).tolist()) for k,v in stamps.items()},
        'valid_beams_quantiles':np.quantile(finite_counts,[0,.01,.5,.99,1]).tolist(),
        'valid_return_fraction':float(np.mean(finite_counts)/1081),
        'valid_returns_between_10_and_30_m_fraction_of_all_beams':float(np.mean(beyond10)/1081),
        'imu_acceleration_norm_quantiles_m_s2':np.quantile(np.linalg.norm(imu[:,4:7],axis=1),[.01,.5,.99]).tolist(),
        'odom_pitch_quantiles_deg':np.degrees(np.quantile(odom[:,5],[.01,.5,.99])).tolist(),
        'unresolved':['scan timestamp start/end convention','true per-beam sampling interval',
                      'hardware clock calibration residuals; equal converted timestamps do not prove physical synchronization'],
        'legacy_assumption':dict(scan_time_s=.025,time_increment_s=.025/1081,
                                 last_beam_offset_s=.025*1080/1081),
        'paper_url':'https://robots.engin.umich.edu/nclt/nclt.pdf',
        'author_format_scripts':['data/nclt/python/read_hokuyo_30m.py','data/nclt/python/read_ms25.py'],
        'paper_binary_timestamp_typo':'paper says four bytes; author struct <Q and observed 2170-byte records support eight bytes',
        'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [
            ROOT/'tools/prepare_nclt.py',ROOT/'tools/audit_nclt_inputs.py',
            ROOT/'data/nclt/python/read_hokuyo_30m.py',ROOT/'data/nclt/python/read_ms25.py']}}
    output=ROOT/'results/mentor_benchmark/nclt_diagnostic_v1/input_audit.json'
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':audit()
