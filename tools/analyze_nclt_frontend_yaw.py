#!/usr/bin/python3
"""Compare local-SLAM heading changes to input odometry and gyroscope, without GT."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mentor_benchmark import load_pose,interpolate

ROOT=Path(__file__).resolve().parents[1]
V1=ROOT/'results/mentor_benchmark/nclt_diagnostic_v1'
V2=ROOT/'results/mentor_benchmark/nclt_frontend_diagnostic_v2'

if __name__=='__main__':
    original_t,original=load_pose(V1/'nclt-2013-01-10_original10_loop/noimu_local_nodes.tum')
    ceres_t,ceres=load_pose(V2/'nclt-2013-01-10_ceres10_loop/noimu_local_nodes.tum')
    ot,odom=load_pose(ROOT/'data/prepared/nclt-2013-01-10/odom.tum')
    raw=np.loadtxt(ROOT/'data/nclt/2013-01-10/ms25.csv',delimiter=',')
    start=max(original_t[0],ceres_t[0],ot[0],raw[0,0]/1e6)
    end=min(original_t[-1],ceres_t[-1],ot[-1],raw[-1,0]/1e6)
    t=start+np.arange(int((end-start)*10)+1)/10.
    angular={}
    for name,pt,p in [('correlative',original_t,original),('ceres_only',ceres_t,ceres),('provided_ekf',ot,odom)]:
        yaw=np.unwrap(interpolate(t,pt,p)[:,2]);angular[name]=np.degrees(np.diff(yaw)/np.diff(t))
    # Raw angular velocity is in forward/right/down; ROS sign for z is negative.
    angular['imu_body_z']=np.degrees(np.interp((t[1:]+t[:-1])/2,raw[:,0]/1e6,-raw[:,9]))
    reports={}
    for name,rate in angular.items():
        diff=rate-angular['provided_ekf']
        reports[name]=dict(absolute_rate_p99_deg_s=float(np.quantile(np.abs(rate),.99)),
            absolute_difference_from_ekf_p99_deg_s=float(np.quantile(np.abs(diff),.99)),
            rate_difference_rmse_deg_s=float(np.sqrt(np.mean(diff**2))),
            rate_correlation_with_ekf=float(np.corrcoef(rate,angular['provided_ekf'])[0,1]))
    early=(t[1:]-t[0])<=60
    fig,axes=plt.subplots(2,1,figsize=(10,6),constrained_layout=True)
    for name,rate in angular.items():
        axes[0].plot((t[1:]-t[0])[early],rate[early],label=name,linewidth=1)
        axes[1].plot(t[1:]-t[0],np.cumsum(rate*np.diff(t)),label=name,linewidth=1)
    axes[0].set(xlabel='elapsed time (s)',ylabel='yaw rate (deg/s)',title='first 60 s; no reference poses used')
    axes[1].set(xlabel='elapsed time (s)',ylabel='integrated heading change (deg)')
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8)
    for ext in ['png','pdf']:fig.savefig(V2/f'yaw_input_agreement.{ext}',dpi=180)
    plt.close(fig)
    result=dict(reference_used=False,grid_hz=10,samples=len(t),statistics=reports,
        caveat='EKF already uses gyro/FOG/IMU; agreement with gyro is not independent accuracy; yaw rate differs from body z on tilted platform',
        causal_interpretation='controlled correlative on/off contrast, not a GT-labelled false-match count')
    (V2/'yaw_input_agreement.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
