#!/usr/bin/python3
"""Convert one complete NCLT session, explicitly preserving its odometry provenance.

Coordinate conventions and laser format follow the NCLT paper and author examples.
The supplied odometry includes wheel encoders, FOG and IMU; it is not pure wheel odometry.
"""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from nclt_binary import read_checked_hokuyo

import numpy as np
import rosbag
import rospy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, LaserScan
from tf2_msgs.msg import TFMessage
from scipy.spatial.transform import Rotation


def convert_pose(row):
    # Original world/body conventions are NED and forward/right/down.
    flip=np.diag([1.,-1.,-1.])
    position=flip@row[1:4]
    rotation=flip@Rotation.from_euler('xyz',row[4:7]).as_matrix()@flip
    return position,Rotation.from_matrix(rotation).as_quat()


def assign_pose(msg,position,q):
    msg.position.x,msg.position.y,msg.position.z=position.tolist()
    msg.orientation.x,msg.orientation.y,msg.orientation.z,msg.orientation.w=q.tolist()


def write_tum(stream,stamp,position,q):
    stream.write(' '.join(f'{v:.9f}' for v in [stamp,*position,*q])+'\n')


def run(args):
    source=args.root/args.session
    output=args.output/f'nclt-{args.session}'
    output.mkdir(parents=True,exist_ok=True)
    if (output/'preparation.json').exists():
        print('Already prepared:',output,flush=True);return
    odom=np.loadtxt(source/'odometry_mu_100hz.csv',delimiter=',')
    imu=np.loadtxt(source/'ms25.csv',delimiter=',')
    truth=np.loadtxt(args.root/f'groundtruth_{args.session}.csv',delimiter=',')
    truth=truth[np.isfinite(truth).all(axis=1)]
    laser_path=source/'hokuyo_30m.bin'
    # Author format has fixed-size records. Some archives contain a malformed
    # suffix. Detect it using other sensor timestamps only, never reference data.
    laser,binary_audit=read_checked_hokuyo(laser_path,odom[:,0],imu[:,0])
    (output/'laser_binary_audit.json').write_text(json.dumps(binary_audit,indent=2)+'\n')
    events=[(int(row[0]),'odom',i) for i,row in enumerate(odom)]
    events.extend((int(row[0]),'imu',i) for i,row in enumerate(imu))
    events.extend((int(row['timestamp']),'scan',i) for i,row in enumerate(laser))
    events.sort()
    static_rows=[{'parent':'base_link','child':'laser','translation':[.28,0,.44],
                  'quaternion':[0.,0.,0.,1.]},
                 {'parent':'base_link','child':'nclt_imu','translation':[-.11,.18,.71],
                  'quaternion':[0.,0.,0.,1.]}]
    with rosbag.Bag(str(output/'slam_inputs.bag'),'w') as bag, (output/'odom.tum').open('w') as stream:
        static=TFMessage()
        for row in static_rows:
            msg=TransformStamped();msg.header.frame_id=row['parent'];msg.child_frame_id=row['child']
            msg.header.stamp=rospy.Time.from_sec(events[0][0]/1e6)
            msg.transform.translation.x,msg.transform.translation.y,msg.transform.translation.z=row['translation']
            msg.transform.rotation.w=1.
            static.transforms.append(msg)
        bag.write('/tf_static',static,rospy.Time.from_sec(events[0][0]/1e6))
        for micros,kind,index in events:
            stamp=rospy.Time.from_sec(micros/1e6)
            if kind=='scan':
                msg=LaserScan();msg.header.frame_id='laser';msg.header.stamp=stamp
                msg.angle_min=-3*np.pi/4;msg.angle_max=3*np.pi/4;msg.angle_increment=np.pi/720
                msg.range_min=.1;msg.range_max=30.;msg.scan_time=.025
                msg.time_increment=.025/1081
                ranges=laser[index]['ranges'].astype(float)*.005-100
                msg.ranges=np.where((ranges>=.1)&(ranges<=30),ranges,np.inf).tolist()
                bag.write('/scan',msg,stamp)
            elif kind=='odom':
                msg=Odometry();msg.header.frame_id='base_odom';msg.child_frame_id='base_link';msg.header.stamp=stamp
                position,q=convert_pose(odom[index]);assign_pose(msg.pose.pose,position,q)
                bag.write('/odom',msg,stamp);write_tum(stream,micros/1e6,position,q)
            else:
                row=imu[index]
                msg=Imu();msg.header.frame_id='nclt_imu';msg.header.stamp=stamp
                msg.orientation.w=1.;msg.orientation_covariance[0]=-1
                acceleration=row[4:7]*[1.,-1.,-1.];angular=row[7:10]*[1.,-1.,-1.]
                msg.linear_acceleration.x,msg.linear_acceleration.y,msg.linear_acceleration.z=acceleration.tolist()
                msg.angular_velocity.x,msg.angular_velocity.y,msg.angular_velocity.z=angular.tolist()
                bag.write('/imu_combined',msg,stamp)
    with (output/'groundtruth.tum').open('w') as stream:
        for row in truth:
            position,q=convert_pose(row);write_tum(stream,row[0]/1e6,position,q)
    (output/'static_transforms.json').write_text(json.dumps(static_rows,indent=2)+'\n')
    (output/'preparation.json').write_text(json.dumps({
        'sequence':f'nclt-{args.session}','source':str(source.resolve()),
        'counts':{'scan':len(laser),'odom':len(odom),'imu':len(imu),'gt':len(truth)},
        'laser_binary_audit':binary_audit,
        'bag_duration_s':(events[-1][0]-events[0][0])/1e6,
        'reference_used_by_slam':False,'odom_kind':'provided wheel/FOG/IMU EKF odometry',
        'reference_kind':'SLAM graph with odometry-interpolated high-rate poses',
        'coordinate_conversion':'NED and forward/right/down to north/west/up and forward/left/up',
        'documentation':'https://robots.engin.umich.edu/nclt/nclt.pdf'},indent=2)+'\n')
    print(f'Prepared complete NCLT session: {output}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path('data/nclt'))
    parser.add_argument('--session',default='2013-01-10')
    parser.add_argument('--output',type=Path,default=Path('data/prepared'))
    run(parser.parse_args())
