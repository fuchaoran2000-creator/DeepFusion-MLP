#!/usr/bin/python3
"""One-timespeed ROS replay: capture published poses, resources and optional correction.

No reference topic is present in input bags. Algorithms run separately for timing.
"""
import argparse
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
import xmlrpc.client
from collections import deque
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import psutil
import yaml
from scipy.spatial.transform import Rotation
from tools.replay_cartographer import ROOT, INSTALL, reference_transform


def terminate(process):
    if process is None or process.poll() is not None:return
    os.killpg(process.pid,signal.SIGINT)
    try:process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid,signal.SIGTERM)
        process.wait(timeout=10)


def transform_pose(transform):
    p,q=transform.translation,transform.rotation
    matrix=np.eye(4)
    matrix[:3,:3]=Rotation.from_quat([q.x,q.y,q.z,q.w]).as_matrix()
    matrix[:3,3]=[p.x,p.y,p.z]
    return matrix


def write_pose(stream,stamp,matrix):
    q=Rotation.from_matrix(matrix[:3,:3]).as_quat()
    stream.write(' '.join(f'{v:.9f}' for v in [stamp,*matrix[:3,3],*q])+'\n')


def serialize_metrics(response):
    families=[]
    for family in response.metric_families:
        values=[]
        for metric in family.metrics:
            values.append({'type':metric.type,'labels':{v.key:v.value for v in metric.labels},
                'value':metric.value,'buckets':[{'upper':v.bucket_boundary if math.isfinite(v.bucket_boundary) else None,
                                               'count':v.count} for v in metric.counts_by_bucket]})
        families.append({'name':family.name,'description':family.description,'metrics':values})
    return {'status_code':response.status.code,'status_message':response.status.message,'families':families}


class OnlineCorrection:
    def __init__(self, checkpoint, publisher, stream):
        import torch
        from residual_models import Regressor
        self.torch=torch
        torch.set_num_threads(1)
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if state.get('stream_protocol'):
            raise ValueError('Versioned checkpoint requires --capture-stream to use its matching causal grid runtime')
        if state['kind']!='mlp' or (state['config'],state['target']) not in [('relative32','increment'),('full20','pose')]:
            raise ValueError('Online runner supports relative32 increment or full20 pose MLP checkpoints')
        self.config,self.target=state['config'],state['target']
        self.model=Regressor(len(state['norm']['xm']),'mlp',state['uncertainty']);self.model.load_state_dict(state['model_state_dict'])
        self.model.eval()
        self.norm={k:v.numpy() for k,v in state['norm'].items()}
        self.gate_parameters=state['gate_parameters']
        self.input_kind=sorted({row.get('trajectory_kind','unspecified') for row in state['manifest']['sequences']})
        self.alpha=state['alpha'] if self.gate_parameters is None else self.gate_parameters['gate_alpha']
        self.publisher,self.stream=publisher,stream
        self.history=deque(maxlen=6)
        self.origin_cart=self.origin_odom=None
        self.last_output=None
        self.last_time=-float('inf')
        self.timings=[];self.fallbacks=0

    def apply(self,stamp,cart_matrix,odom_msg):
        from mentor_benchmark import Sequence,features,increments,add_increment,wrap_angle
        if stamp-self.last_time < .099:return
        start=time.perf_counter_ns()
        if odom_msg is None or not 0<=stamp-odom_msg.header.stamp.to_sec()<=.15:
            # Publish the original pose even when the second estimator is unavailable.
            self.history.clear();self.last_output=None;self.last_time=stamp
            self._publish(stamp,cart_matrix,start,True)
            return
        p,q=odom_msg.pose.pose.position,odom_msg.pose.pose.orientation
        odom_matrix=np.eye(4);odom_matrix[:3,:3]=Rotation.from_quat([q.x,q.y,q.z,q.w]).as_matrix()
        odom_matrix[:3,3]=[p.x,p.y,p.z]
        if self.origin_cart is None:
            self.origin_cart=cart_matrix.copy();self.origin_odom=odom_matrix.copy()
        def planar(matrix):return np.array([matrix[0,3],matrix[1,3],np.arctan2(matrix[1,0],matrix[0,0])])
        def local(matrix,origin):
            p,o=planar(matrix),planar(origin);c,s=np.cos(o[2]),np.sin(o[2])
            d=p[:2]-o[:2]
            return np.array([c*d[0]+s*d[1],-s*d[0]+c*d[1],wrap_angle(p[2]-o[2])])
        cp,op=local(cart_matrix,self.origin_cart),local(odom_matrix,self.origin_odom)
        self.history.append((stamp,cp,op));self.last_time=stamp
        if len(self.history)<2:
            self.last_output=cp.copy()
            self._publish(stamp,cart_matrix,start,True)
            return
        hist=list(self.history)
        seq=Sequence('stream','online',np.array([v[0] for v in hist]),
                     np.array([v[1] for v in hist]),np.array([v[2] for v in hist]),
                     np.zeros((len(hist),3)),np.ones(len(hist),dtype=bool),'online',{})
        x=((features(seq,self.config)[-1]-self.norm['xm'])/self.norm['xs']).astype(np.float32)
        with self.torch.inference_mode():output=self.model(self.torch.from_numpy(x[None])).numpy()[0]
        residual=np.clip(output[:3]*self.norm['ys']+self.norm['ym'],-self.norm['clip'],self.norm['clip'])
        gate=True
        if self.gate_parameters:
            sigma=np.exp(.5*np.clip(output[3:],-8,6)).mean()
            gate=sigma<=self.gate_parameters['sigma_threshold'] and np.abs(x).max()<=self.gate_parameters['ood_threshold']
        if not gate or self.alpha==0:
            self.last_output=cp.copy()
        else:
            if self.target=='pose':
                self.last_output=cp+self.alpha*residual;self.last_output[2]=wrap_angle(self.last_output[2])
            else:
                self.last_output=add_increment(self.last_output,increments(seq.cart)[-1]+self.alpha*residual)
        pose=cart_matrix.copy()
        if gate and self.alpha!=0:
            origin=planar(self.origin_cart);c,s=np.cos(origin[2]),np.sin(origin[2])
            pose[:2,3]=origin[:2]+np.array([[c,-s],[s,c]])@self.last_output[:2]
            pose[:3,:3]=Rotation.from_euler('z',wrap_angle(self.last_output[2]-cp[2])).as_matrix()@cart_matrix[:3,:3]
        self._publish(stamp,pose,start,not gate or self.alpha==0)

    def _publish(self,stamp,pose,start,fallback):
        from geometry_msgs.msg import PoseStamped
        self.fallbacks+=int(fallback)
        write_pose(self.stream,stamp,pose)
        msg=PoseStamped();msg.header.frame_id='map'
        import rospy
        msg.header.stamp=rospy.Time.from_sec(stamp)
        msg.pose.position.x,msg.pose.position.y,msg.pose.position.z=pose[:3,3]
        quaternion=Rotation.from_matrix(pose[:3,:3]).as_quat()
        msg.pose.orientation.x,msg.pose.orientation.y,msg.pose.orientation.z,msg.pose.orientation.w=quaternion
        self.publisher.publish(msg)
        self.timings.append((time.perf_counter_ns()-start)/1e6)


class UnifiedOnlineCorrection:
    """Versioned grid runtime, also usable to capture training pairs without a model."""
    def __init__(self, checkpoint, publisher, stream, hz=10.):
        from causal_stream import GridSampler, FeatureStream, ResidualStream, PROTOCOL
        self.publisher, self.stream = publisher, stream
        self.protocol, self.hz = PROTOCOL, hz
        self.has_model = checkpoint is not None
        self.sampler = GridSampler(hz)
        self.timings = []; self.fallbacks = 0; self.records = []
        self.input_kind = ['captured_online_arrivals']
        self.engine = None
        self.config = 'relative32'
        if checkpoint:
            import torch
            from residual_models import Regressor
            torch.set_num_threads(1)
            state = torch.load(checkpoint, map_location='cpu', weights_only=False)
            if state.get('stream_protocol') != PROTOCOL or state['kind'] != 'mlp' or state['grid_hz'] != hz:
                raise ValueError('Checkpoint does not match shared runtime protocol')
            model = Regressor(len(state['norm']['xm']), 'mlp', state['uncertainty'])
            model.load_state_dict(state['model_state_dict']); model.eval()
            def forward(z):
                with torch.inference_mode():
                    return model(torch.from_numpy(z[None])).numpy()[0]
            self.config = state['config']
            norm = {k: v.numpy() for k, v in state['norm'].items()}
            self.engine = ResidualStream(self.config, state['target'], norm, forward,
                                         state['alpha'], state['gate_parameters'], state.get('integration_tau_s'))
        self.extractor = FeatureStream(self.config)

    @staticmethod
    def planar(matrix):
        return np.array([matrix[0,3], matrix[1,3], np.arctan2(matrix[1,0], matrix[0,0])])

    def receive_odom(self, msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        quat = np.array([q.x, q.y, q.z, q.w])
        pose = np.full(3, np.nan)
        if np.isfinite(quat).all() and np.linalg.norm(quat) > 1e-12:
            rotation = Rotation.from_quat(quat).as_matrix()
            pose = np.array([p.x, p.y, np.arctan2(rotation[1,0], rotation[0,0])])
        self.sampler.push_odom(msg.header.stamp.to_sec(), pose)

    def apply(self, stamp, matrix, odom_msg=None):
        from causal_stream import wrap
        start = time.perf_counter_ns()
        rows = self.sampler.push_cart(stamp, matrix)
        for tick, raw_matrix, odom, available, cart_stamp, odom_stamp in rows:
            raw = self.planar(raw_matrix)
            if self.engine:
                result, status = self.engine.step(tick, raw, odom, available)
            else:
                _, _, x, ready = self.extractor.step(tick, raw, odom, available)
                result, status = raw.copy(), {'feature': x, 'ready': ready, 'fallback': True}
            pose = raw_matrix.copy()
            if not status['fallback']:
                pose[:2,3] = result[:2]
                pose[:3,:3] = Rotation.from_euler('z', wrap(result[2]-raw[2])).as_matrix() @ raw_matrix[:3,:3]
            self.records.append((tick, raw, odom, available, cart_stamp, odom_stamp,
                                 status['feature'].copy(), status['ready'], status['fallback'], result.copy()))
            OnlineCorrection._publish(self, tick, pose, start, status['fallback'])
            start = time.perf_counter_ns()

    def save(self, path):
        keys = ['t','cart','odom','available','cart_source_t','odom_source_t','features','ready','fallback','output']
        arrays = {k: np.array([r[i] for r in self.records]) for i,k in enumerate(keys)}
        np.savez_compressed(path, protocol=self.protocol, hz=self.hz, config=self.config,
                            has_model=self.has_model, **arrays)


class QualityOnlineCorrection(UnifiedOnlineCorrection):
    """Record scan descriptors chosen from actual received scan callbacks."""
    def __init__(self, checkpoint, publisher, stream, hz=10.):
        from trajectory_quality import ScanQualityBuffer, QualityFeatureStream, QualityResidualStream, PROTOCOL
        super().__init__(None,publisher,stream,hz)
        self.feature_protocol=PROTOCOL
        self.quality_buffer=ScanQualityBuffer()
        self.scan_timings=[];self.quality_records=[]
        self.config='quality44'
        self.has_model=checkpoint is not None
        if checkpoint:
            import torch
            from residual_models import Regressor
            torch.set_num_threads(1)
            state=torch.load(checkpoint,map_location='cpu',weights_only=False)
            if state.get('stream_protocol')!=PROTOCOL or state['kind']!='mlp' or state['grid_hz']!=hz:
                raise ValueError('Checkpoint does not match scan-quality runtime')
            self.config=state['feature_config']
            model=Regressor(44,'mlp',True)
            model.load_state_dict(state['model_state_dict']);model.eval()
            def forward(z):
                with torch.inference_mode():return model(torch.from_numpy(z[None])).numpy()[0]
            norm={k:v.numpy() for k,v in state['norm'].items()}
            self.engine=QualityResidualStream(self.config,norm,forward,state['alpha'],
                state['gate_parameters'],state.get('integration_tau_s'))
        self.extractor=QualityFeatureStream(self.config)

    def receive_scan(self,msg):
        started=time.perf_counter_ns()
        self.quality_buffer.receive(msg)
        self.scan_timings.append((time.perf_counter_ns()-started)/1e6)

    def apply(self,stamp,matrix,odom_msg=None):
        from causal_stream import wrap
        start=time.perf_counter_ns()
        for tick,raw_matrix,odom,available,cart_stamp,odom_stamp in self.sampler.push_cart(stamp,matrix):
            raw=self.planar(raw_matrix)
            quality,scan_stamp=self.quality_buffer.at(tick,cart_stamp,odom_stamp)
            if self.engine:
                result,status=self.engine.step(tick,raw,odom,quality,available)
            else:
                self.extractor.quality=quality
                _,_,x,ready=self.extractor.step(tick,raw,odom,available)
                result,status=raw.copy(),dict(feature=x,ready=ready,fallback=True)
            pose=raw_matrix.copy()
            if not status['fallback']:
                pose[:2,3]=result[:2]
                pose[:3,:3]=Rotation.from_euler('z',wrap(result[2]-raw[2])).as_matrix()@raw_matrix[:3,:3]
            self.records.append((tick,raw,odom,available,cart_stamp,odom_stamp,
                status['feature'].copy(),status['ready'],status['fallback'],result.copy()))
            self.quality_records.append((quality.copy(),scan_stamp))
            OnlineCorrection._publish(self,tick,pose,start,status['fallback'])
            start=time.perf_counter_ns()

    def save(self,path):
        keys=['t','cart','odom','available','cart_source_t','odom_source_t','features','ready','fallback','output']
        arrays={k:np.array([r[i] for r in self.records]) for i,k in enumerate(keys)}
        np.savez_compressed(path,protocol=self.protocol,feature_protocol=self.feature_protocol,
            hz=self.hz,config=self.config,has_model=self.has_model,
            quality=np.array([r[0] for r in self.quality_records]),
            scan_source_t=np.array([r[1] for r in self.quality_records]),
            quality_arrival_scope='actual received scan callbacks',**arrays)


def run(args):
    env=dict(os.environ,ROS_MASTER_URI=f'http://127.0.0.1:{args.port}',ROS_IP='127.0.0.1',
             ROS_HOME=str(ROOT/'results/mentor_benchmark/online_ros_home'),
             ROS_LOG_DIR=str(ROOT/'results/mentor_benchmark/online_ros_logs'))
    for key in ('ROS_HOME','ROS_LOG_DIR'):Path(env[key]).mkdir(parents=True,exist_ok=True)
    os.environ.update({k:env[k] for k in ['ROS_MASTER_URI','ROS_IP','ROS_HOME','ROS_LOG_DIR']})
    master_log=(ROOT/'results/mentor_benchmark/online_roscore.log').open('a')
    master=subprocess.Popen(['/opt/ros/noetic/bin/roscore','-p',str(args.port)],env=env,
                            stdout=master_log,stderr=subprocess.STDOUT,start_new_session=True)
    node=player=None
    try:
        for _ in range(100):
            if master.poll() is not None:raise RuntimeError('Online ROS master failed')
            try:
                if xmlrpc.client.ServerProxy(env['ROS_MASTER_URI']).getUri('/benchmark')[0]==1:break
            except OSError:time.sleep(.1)
        import rospy
        from tf2_msgs.msg import TFMessage
        from nav_msgs.msg import Odometry
        from geometry_msgs.msg import PoseStamped
        from sensor_msgs.msg import LaserScan
        rospy.init_node('online_benchmark_observer',disable_signals=True)
        rospy.set_param('/use_sim_time',True)
        for directory in args.directories:
            directory=directory.resolve()
            static=json.loads((directory/'static_transforms.json').read_text())
            for algorithm in args.algorithms:
                tag='_'+args.label if args.label else ''
                output=directory/f'{algorithm}_online_resources{tag}.json'
                if output.exists():
                    print(f'Already measured {directory.name}/{algorithm}',flush=True);continue
                if algorithm=='2dliw':
                    config=yaml.safe_load((ROOT/'third_party/2dliw_workspace/src/2DLIW-SLAM-main/config/params.yaml').read_text())
                    config.update(imu_topic='/imu_combined',output_tum=True,output_dir=str(directory)+'/',
                                  enable_camera=False,enable_camera_vis=False,enable_laser_vis=False)
                    for name,target in [('T_imu_to_wheel','base_link'),('T_imu_to_laser','laser'),('T_imu_to_camera','d400_color')]:
                        config[name]=reference_transform(static,'t265_imu',target).reshape(-1).tolist()
                    for key,value in config.items():rospy.set_param('/online_2dliw/'+key,value)
                    (directory/'2dliw_config.yaml').write_text(yaml.safe_dump(config))
                    command=[str(ROOT/'third_party/2dliw_workspace/devel/lib/lvio_2d/lvio_2d_node'),
                             '__name:=online_2dliw']
                    extrinsic=reference_transform(static,'t265_imu','base_link')
                else:
                    mode='imu' if algorithm=='cartographer_imu' else 'noimu'
                    config_basename=args.config_basename or (f'nclt_2d_{mode}.lua'
                        if directory.name.startswith('nclt-') and mode=='imu' else f'home1_2d_{mode}.lua')
                    config_path=ROOT/'config'/config_basename
                    if not config_path.is_file():raise ValueError(f'Missing configuration: {config_path}')
                    (directory/f'{algorithm}_configuration{tag}.lua').write_bytes(config_path.read_bytes())
                    command=[str(INSTALL/'lib/cartographer_ros/cartographer_node'),
                             '-configuration_directory',str(ROOT/'config'),
                             '-configuration_basename',config_basename,'-logtostderr',
                             '__name:=online_cartographer','scan:=/scan','odom:=/odom','imu:=/imu_combined']
                    extrinsic=np.eye(4)
                    if args.collect_metrics:command.append('-collect_metrics=true')
                state={'poses':0,'last_stamp':-1,'odom':None};lock=threading.Lock()
                poses=(directory/f'{algorithm}_online{tag}.tum').open('w')
                corrected_stream=(directory/f'{algorithm}_corrected_online{tag}.tum').open('w')
                correction=None
                if (args.checkpoint or args.capture_stream or args.quality_stream) and algorithm.startswith('cartographer'):
                    publisher=rospy.Publisher('/deepfusion_pose',PoseStamped,queue_size=10)
                    if args.quality_stream:
                        correction=QualityOnlineCorrection(args.checkpoint,publisher,corrected_stream,args.grid_hz)
                    elif args.capture_stream:
                        correction=UnifiedOnlineCorrection(args.checkpoint,publisher,corrected_stream,args.grid_hz)
                    else:
                        correction=OnlineCorrection(args.checkpoint,publisher,corrected_stream)
                def odom_callback(msg):
                    with lock:
                        state['odom']=msg
                        if isinstance(correction,UnifiedOnlineCorrection):correction.receive_odom(msg)
                def callback(msg):
                    with lock:
                        for transform in msg.transforms:
                            selected=(transform.header.frame_id=='world' and transform.child_frame_id=='imu') if algorithm=='2dliw' else (transform.header.frame_id=='map' and transform.child_frame_id=='base_link')
                            if not selected:continue
                            stamp=transform.header.stamp.to_sec()
                            if stamp<=state['last_stamp']:continue
                            matrix=transform_pose(transform.transform)@extrinsic
                            write_pose(poses,stamp,matrix)
                            state['last_stamp']=stamp;state['poses']+=1
                            if correction:correction.apply(stamp,matrix,state['odom'])
                def scan_callback(msg):
                    with lock:
                        if isinstance(correction,QualityOnlineCorrection):correction.receive_scan(msg)
                sub=rospy.Subscriber('/tf',TFMessage,callback,queue_size=10000)
                odom_sub=rospy.Subscriber('/odom',Odometry,odom_callback,queue_size=1000)
                scan_sub=rospy.Subscriber('/scan',LaserScan,scan_callback,queue_size=1000) if args.quality_stream else None
                with (directory/f'{algorithm}_online{tag}.log').open('w') as log:
                    node=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    time.sleep(1)
                    started=time.perf_counter();peak=0;observer_peak=0
                    observer=psutil.Process(os.getpid())
                    player=subprocess.Popen(['/opt/ros/noetic/lib/rosbag/play','--clock','--rate',str(args.rate),
                                             '--delay','0.5',str(directory/'slam_inputs.bag')],
                                            env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    process=psutil.Process(node.pid)
                    initial_cpu=process.cpu_times()
                    read_metrics=None;metric_samples=[];last_metric_poll=started
                    if args.collect_metrics and algorithm.startswith('cartographer'):
                        from cartographer_ros_msgs.srv import ReadMetrics
                        rospy.wait_for_service('/read_metrics',timeout=10)
                        read_metrics=rospy.ServiceProxy('/read_metrics',ReadMetrics)
                    while player.poll() is None:
                        if node.poll() is not None:raise RuntimeError(f'{algorithm} exited early; inspect log')
                        try:peak=max(peak,process.memory_info().rss)
                        except psutil.NoSuchProcess:pass
                        if args.quality_stream:observer_peak=max(observer_peak,observer.memory_info().rss)
                        if read_metrics and time.perf_counter()-last_metric_poll>=1:
                            response=read_metrics();serialized=serialize_metrics(response)
                            gauges={f['name']:f['metrics'][0]['value'] for f in serialized['families']
                                    if f['metrics'] and f['metrics'][0]['type']==1 and any(k in f['name'] for k in ['latency','real_time_ratio','queue_length'])}
                            metric_samples.append({'wall_elapsed_s':time.perf_counter()-started,'gauges':gauges})
                            last_metric_poll=time.perf_counter()
                        time.sleep(.02)
                    if player.returncode:raise RuntimeError('ROS bag replay failed')
                    # Allow queued measurements to drain; retain online publications only.
                    time.sleep(2)
                    wall=time.perf_counter()-started
                    if read_metrics:
                        metrics_artifact={'poll_period_s':1,'gauge_scope':'one Hz snapshots of last local-SLAM update; not per-frame latency samples',
                                          'samples':metric_samples,'before_shutdown':serialize_metrics(read_metrics())}
                        (directory/f'{algorithm}_metrics{tag}.json').write_text(json.dumps(metrics_artifact,indent=2)+'\n')
                    final_cpu=process.cpu_times()
                    cpu_user=max(0.,final_cpu.user-initial_cpu.user)
                    cpu_system=max(0.,final_cpu.system-initial_cpu.system)
                    terminate(node);shutdown_code=node.returncode;node=None;player=None
                sub.unregister();odom_sub.unregister()
                if scan_sub:scan_sub.unregister()
                with lock:poses.close();corrected_stream.close()
                if isinstance(correction,UnifiedOnlineCorrection):
                    correction.save(directory/f'{algorithm}_sampled_stream{tag}.npz')
                measurement={'algorithm':algorithm,'trajectory_kind':'online','reference_point':'base_link',
                             'replay_rate':args.rate,'wall_s':wall,'peak_rss_mib':peak/1024**2,
                             'pose_count':state['poses'],'output_only_correction':bool(correction),
                             'correction_changes_slam_state':False,'reference_used_as_input':False,
                             'source_command':command,'resource_scope':'SLAM child process; observer separate',
                             'rss_sample_period_ms':20}
                measurement['shutdown_returncode']=shutdown_code
                measurement.update(slam_cpu_user_s=cpu_user,slam_cpu_system_s=cpu_system,
                    slam_cpu_total_s=cpu_user+cpu_system,
                    cpu_time_scope='SLAM child user+system CPU seconds across all threads, playback and 2s drain; excludes 1s startup and shutdown final optimization. Not paced replay wall time.')
                measurement['shutdown_success']=shutdown_code in (0,-signal.SIGINT)
                if args.quality_stream:
                    measurement.update(observer_peak_rss_mib=observer_peak/1024**2,
                        observer_resource_scope='ROS observer, Torch, NumPy and retained capture records; entire observer process')
                if correction and correction.timings:
                    measurement['correction']={'count':len(correction.timings),
                        'mean_ms':float(np.mean(correction.timings)),'p95_ms':float(np.quantile(correction.timings,.95)),
                        'p99_ms':float(np.quantile(correction.timings,.99)),
                        'fallback_fraction':correction.fallbacks/len(correction.timings),
                        'timing_scope':'feature extraction, CPU model, confidence, integration, file write and publisher submission',
                        'checkpoint':str(args.checkpoint.resolve()) if args.checkpoint else None,'checkpoint_input_kind':correction.input_kind}
                    if isinstance(correction,UnifiedOnlineCorrection):
                        measurement['correction'].update(stream_protocol=correction.protocol, grid_hz=correction.hz,
                            sampling='first Cartographer timestamp + integer/Hz; causal received message buffers',
                            capture_only=args.checkpoint is None)
                    if isinstance(correction,QualityOnlineCorrection):
                        measurement['correction'].update(feature_protocol=correction.feature_protocol,
                            quality_arrival_scope='actual received scan callbacks',
                            scan_descriptor_count=len(correction.scan_timings),
                            scan_descriptor_mean_ms=float(np.mean(correction.scan_timings)),
                            scan_descriptor_p99_ms=float(np.quantile(correction.scan_timings,.99)),
                            scan_timing_scope='geometry descriptor and received-scan buffer insertion; separate scan callback, excluded from correction timing')
                output.write_text(json.dumps(measurement,indent=2)+'\n')
                print(f'Online replay {directory.name}/{algorithm}: {state["poses"]} poses, {wall:.1f}s',flush=True)
    finally:
        terminate(player);terminate(node);terminate(master);master_log.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('directories',nargs='+',type=Path)
    parser.add_argument('--algorithms',nargs='+',choices=['cartographer_noimu','cartographer_imu','2dliw'],
                        default=['cartographer_noimu','2dliw'])
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--port',type=int,default=11432)
    parser.add_argument('--rate',type=float,default=1.0)
    parser.add_argument('--label',default='')
    parser.add_argument('--collect-metrics',action='store_true')
    parser.add_argument('--capture-stream',action='store_true',help='use versioned causal grid runtime and save exact sampled input pairs')
    parser.add_argument('--quality-stream',action='store_true',help='capture actual received laser quality descriptors and use causal_quality_v1 checkpoints')
    parser.add_argument('--grid-hz',type=float,default=10.)
    parser.add_argument('--config-basename',help='Explicit Cartographer configuration in config/; saved beside each capture')
    args=parser.parse_args();run(args)
