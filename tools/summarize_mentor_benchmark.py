#!/usr/bin/python3
"""Build auditable tables and figures from measured artifacts, without model selection."""
import csv
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1]/'results/matplotlib_cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from mentor_benchmark import (Sequence, load_sequence, load_pose, interpolate, metrics,
                              wrap_angle, dump, features, corrected)
from residual_models import Regressor

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/mentor_benchmark'
OUT=BASE/'report'
RUNS=['home_complete_sequences_v2','cross_scene_and_nclt','cross_scene_and_nclt_imu',
      'home_online_inputs','home_online_pose_uncertainty','cross_scene_pose_uncertainty']
SELECTED=['wheel_odometry','cartographer','covariance_intersection','ekf_conservative_noise',
          'fixed_lag_factor_graph','full_trajectory_factor_graph','ridge_pose_residual',
          'gp_pose_residual_adaptation','mlp_full20_pose','mlp_relative14_increment',
          'mlp_relative32_increment','gru_relative32_increment','tcn_relative32_increment',
          'mlp_relative32_increment_uncertainty','mlp_relative32_increment_uncertainty_hard_fallback']


def read(path):return json.loads(path.read_text())
def csv_write(path,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(rows)
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |',
                      *['| '+' | '.join(str(v) for v in row)+' |' for row in rows]])+'\n'
def f(value,digits=4):return '—' if value is None else f'{value:.{digits}f}'
def fallback(row):
    # Early immutable runs recorded confidence rejection only, even when alpha=0.
    return 1.0 if row.get('gate_alpha')==0 else row.get('fallback_fraction')
def macro(rows,key):
    groups=defaultdict(list)
    for row in rows:groups[row.get('seed')].append(key(row))
    values=np.asarray([np.mean(v) for v in groups.values()])
    return float(values.mean()),float(values.std(ddof=1)) if len(values)>1 else 0.
def rigid_alignment(estimate,truth,valid):
    a,b=estimate[valid,:2],truth[valid,:2]
    u,_,vt=np.linalg.svd((a-a.mean(0)).T@(b-b.mean(0)))
    sign=np.diag([1.,np.linalg.det(u@vt)])
    rot=u@sign@vt
    result=estimate.copy();result[:,:2]=(estimate[:,:2]-a.mean(0))@rot+b.mean(0)
    result[:,2]=wrap_angle(result[:,2]+np.arctan2(rot[0,1],rot[0,0]))
    return result
def anchor_alignment(estimate,truth,index):
    angle=truth[index,2]-estimate[index,2];c,s=np.cos(angle),np.sin(angle)
    result=estimate.copy();result[:,:2]=(estimate[:,:2]-estimate[index,:2])@np.array([[c,s],[-s,c]])+truth[index,:2]
    result[:,2]=wrap_angle(result[:,2]+angle)
    return result
def shared_anchor_alignment(estimate,anchor_source,truth,index):
    angle=truth[index,2]-anchor_source[index,2];c,s=np.cos(angle),np.sin(angle)
    result=estimate.copy();result[:,:2]=(estimate[:,:2]-anchor_source[index,:2])@np.array([[c,s],[-s,c]])+truth[index,:2]
    result[:,2]=wrap_angle(result[:,2]+angle)
    return result
def at_times(path,t):
    st,sp=load_pose(path);indices=np.searchsorted(st,t,side='right')-1
    indices=np.clip(indices,0,len(st)-1)
    valid=(t>=st[0])&(t<=st[-1])&((t-st[indices])<=.15)
    return sp[indices],valid


def online_evaluation():
    rows=[];resource_rows=[]
    for scene,names,label in [('home',['home1-3','home1-4','home1-5'],'v2'),
                             ('corridor',['corridor1-3','corridor1-4','corridor1-5'],'crossscene')]:
        for name in names:
            directory=ROOT/'data/prepared'/name
            sources={a:directory/f'{a}_online_{label}.tum' for a in ['cartographer_noimu','cartographer_imu','2dliw']}
            confidence_path=directory/'cartographer_noimu_corrected_online_confidence20.tum'
            if confidence_path.exists() and (directory/'cartographer_noimu_online_resources_confidence20.json').exists():
                sources['cartographer_noimu']=directory/'cartographer_noimu_online_confidence20.tum'
                sources['mlp20_confidence_online']=confidence_path
            complete=all((directory/f'{a}_online_resources_{label}.json').exists() for a in ['cartographer_noimu','cartographer_imu','2dliw'])
            if not complete or not all(path.exists() and path.stat().st_size for path in sources.values()):continue
            gt,gp=load_pose(directory/'groundtruth.tum');ot,op=load_pose(directory/'odom.tum')
            t=np.arange(max(ot[0],gt[0]),min(ot[-1],gt[-1]),.1)
            truth=interpolate(t,gt,gp);ri=np.clip(np.searchsorted(gt,t),0,len(gt)-1);li=np.maximum(ri-1,0)
            reference=(gt[ri]-gt[li])<=.25
            sources['wheel_odometry']=directory/'odom.tum'
            samples={a:at_times(p,t) for a,p in sources.items()}
            common=reference.copy()
            for pose,mask in samples.values():common&=mask
            if common.sum()<20:raise ValueError('Insufficient common online coverage: '+name)
            anchor=int(np.flatnonzero(common)[0]);base_seq=Sequence(name,scene,t,truth,truth,truth,common,'online',{})
            for algorithm,(pose,mask) in samples.items():
                aligned=anchor_alignment(pose,truth,anchor)
                stat=metrics(base_seq,aligned)
                row={'scene':scene,'sequence':name,'method':algorithm,'trajectory_kind':'online',
                     'alignment':'each estimate origin at same common valid time; SE2, no scale',
                     'full_input_coverage':float(mask.mean()),'reference_coverage':float(reference.mean()),
                     'common_coverage':float(common.mean()),'common_frames':int(common.sum()),
                     'ape_rmse_m':stat['ape_m']['rmse'],'ape_p95_m':stat['ape_m']['p95'],
                     'full_se2_ape_rmse_m':metrics(base_seq,rigid_alignment(aligned,truth,common))['ape_m']['rmse']}
                for interval,values in stat['rpe'].items():
                    row['rpe_'+interval+'_m']=values['translation_m']['rmse']
                    row['rpe_'+interval+'_deg']=values['rotation_deg']['rmse']
                if algorithm!='wheel_odometry':
                    parent='cartographer_noimu' if algorithm=='mlp20_confidence_online' else algorithm
                    resource_label='confidence20' if parent=='cartographer_noimu' and 'mlp20_confidence_online' in sources else label
                    resource_path=directory/f'{parent}_online_resources_{resource_label}.json'
                    res=read(resource_path)
                    row.update(replay_wall_s=res['wall_s'],slam_peak_rss_mib=res['peak_rss_mib'],
                               slam_memory_scope=res['resource_scope'],input_budget='lidar + wheel odometry'+(' + IMU' if parent!='cartographer_noimu' else ''))
                    if algorithm=='mlp20_confidence_online':
                        row.update(res['correction'])
                        # Both poses are published in the same map frame; also score the pair
                        # with the original Cartographer's transform, without re-aligning correction.
                        shared=shared_anchor_alignment(pose,samples['cartographer_noimu'][0],truth,anchor)
                        row['paired_cart_map_alignment_ape_m']=metrics(base_seq,shared)['ape_m']['rmse']
                    if algorithm!='mlp20_confidence_online':
                        resource_rows.append(dict(res,sequence=name,scene=scene,label=resource_label))
                rows.append(row)
    csv_write(OUT/'online_common_time.csv',rows);dump(OUT/'online_resources.json',resource_rows)
    return rows


def collect():
    data={};flat=[];supplement=[];node_times=np.loadtxt(ROOT/'data/nclt/cov_2013-01-10.csv',delimiter=',')[:,0]/1e6
    fresh=read(BASE/'postprocessing_fresh_process_resources.json')
    for run in RUNS:
        directory=BASE/run;data[run]=read(directory/'results.json');manifest=read(directory/'manifest.json')
        seqs={r['sequence_id']:load_sequence(r) for r in manifest['sequences'] if r['sequence_id'] in manifest['test']}
        for row in data[run]:
            row['_run']=run
            seq=seqs[row['sequence']];path=directory/f"{row['method']}_seed{row['seed']}"/(seq.name+'.tum')
            times,pose=load_pose(path)
            if not np.allclose(times,seq.t,atol=1e-7,rtol=0):raise ValueError('Saved output timeline mismatch')
            aligned=rigid_alignment(pose,seq.gt,seq.valid)
            extra={'run':run,'method':row['method'],'sequence':seq.name,'seed':row['seed'],
                   'origin_ape_rmse_m':row['ape_m']['rmse'],
                   'full_se2_ape_rmse_m':metrics(seq,aligned)['ape_m']['rmse']}
            if seq.scene=='nclt':
                nt=node_times[(node_times>=seq.t[0])&(node_times<=seq.t[-1])]
                ri=np.minimum(np.searchsorted(seq.t,nt),len(seq.t)-1);li=np.maximum(ri-1,0)
                idx=np.where(abs(seq.t[li]-nt)<abs(seq.t[ri]-nt),li,ri)
                keep=seq.valid[idx]&(abs(seq.t[idx]-nt)<=.051);idx=idx[keep]
                error=np.linalg.norm(pose[idx,:2]-seq.gt[idx,:2],axis=1)
                extra.update(nclt_graph_node_count=len(idx),nclt_graph_node_nearest_grid_ape_rmse_m=float(np.sqrt(np.mean(error**2))))
            supplement.append(extra)
            record={**extra,'scene':row['scene'],'trajectory_kind':row['trajectory_kind'],
                    'ape_p95_m':row['ape_m']['p95'],'yaw_ape_deg':row['yaw_ape_deg']['rmse'],
                    'input_coverage':row['coverage']['input_available_fraction'],
                    'reference_valid_coverage':row['coverage']['reference_valid_fraction'],
                    'postprocessing_wall_s':row.get('postprocessing_wall_s'),
                    'rss_mib_cumulative_benchmark':row.get('process_peak_rss_mib'),
                    'rss_scope':row.get('rss_scope'),'actual_fallback_fraction':fallback(row),
                    'confidence_rejected_fraction':row.get('confidence_rejected_fraction',row.get('fallback_fraction')),
                    'alpha':row.get('alpha'),'gate_alpha':row.get('gate_alpha'),
                    'parameter_count':row.get('parameter_count'),
                    'nominal_95pct_coverage':row.get('marginal_95pct_coverage'),
                    'normalized_gaussian_nll':row.get('normalized_gaussian_nll')}
            isolated=next((r for r in fresh if r['run']==run and r['sequence']==seq.name and r['method']==row['method'] and r['seed']==row['seed']),None)
            if isolated:
                record.update(fresh_process_peak_rss_mib=isolated['fresh_process_peak_rss_mib'],
                              fresh_process_memory_scope=isolated['memory_scope'],
                              measured_postprocess_mean_ms_per_frame=isolated['mean_ms_per_frame'])
            if row['method']=='cartographer' and row['trajectory_kind']=='offline_final_optimized':
                input_row=next(r for r in manifest['sequences'] if r['sequence_id']==seq.name)
                mode='noimu' if 'noimu' in input_row['cart'] else 'imu'
                res=read(Path(input_row['cart']).with_name(f'cartographer_{mode}_resources.json'))
                record.update(offline_slam_wall_s=res['wall_s'],offline_slam_peak_rss_mib=res['peak_rss_mib'])
            for interval,values in row['rpe'].items():
                record['rpe_'+interval+'_m']=values['translation_m']['rmse']
                record['rpe_'+interval+'_deg']=values['rotation_deg']['rmse']
            flat.append(record)
    csv_write(OUT/'all_methods.csv',flat);csv_write(OUT/'alignment_and_nclt_nodes.csv',supplement)
    return data,supplement


def paired_intervals(data):
    records=[];rng=np.random.default_rng(2026)
    for run,rows in data.items():
        for scene in sorted({r['scene'] for r in rows}):
            baselines={r['sequence']:r['ape_m']['rmse'] for r in rows if r['scene']==scene and r['method']=='cartographer'}
            if len(baselines)<2:continue
            for method in sorted({r['method'] for r in rows if 'anomaly' not in r['method']}):
                groups=defaultdict(list)
                for row in rows:
                    if row['scene']==scene and row['method']==method:groups[row['sequence']].append(row['ape_m']['rmse'])
                if set(groups)!=set(baselines):continue
                differences=np.array([np.mean(groups[name])-baselines[name] for name in sorted(groups)])
                boot=rng.choice(differences,size=(10000,len(differences)),replace=True).mean(1)
                records.append({'run':run,'scene':scene,'method':method,'independent_sequences':len(differences),
                    'delta_ape_m':float(differences.mean()),'ci95_low_m':float(np.quantile(boot,.025)),
                    'ci95_high_m':float(np.quantile(boot,.975)),
                    'scope':'conditional sequence bootstrap after seed averaging; three sequences give weak interval precision'})
    csv_write(OUT/'paired_sequence_intervals.csv',records)


def anomalies(data):
    records=[];models={}
    for run,rows in data.items():
        manifest=read(BASE/run/'manifest.json');seqs={}
        for row in rows:
            if 'anomaly' not in row['method']:continue
            name=row['sequence']
            if name not in seqs:seqs[name]=load_sequence(next(r for r in manifest['sequences'] if r['sequence_id']==name))
            seq=seqs[name];_,pose=load_pose(BASE/run/f"{row['method']}_seed{row['seed']}"/(name+'.tum'))
            region=row['anomaly_region'];mask=seq.valid.copy();mask[:region[0]]=False;mask[region[1]:]=False
            err=np.linalg.norm(pose[:,:2]-seq.gt[:,:2],axis=1);base=np.linalg.norm(seq.cart[:,:2]-seq.gt[:,:2],axis=1)
            model_name=row['method'].split('_anomaly_')[0];model_key=(run,model_name,row['seed'])
            if model_key not in models:
                state=torch.load(BASE/run/f'{model_name}_seed{row["seed"]}/model.pt',map_location='cpu',weights_only=False)
                norm={k:v.numpy() for k,v in state['norm'].items()}
                model=Regressor(len(norm['xm']),state['kind'],state['uncertainty']);model.load_state_dict(state['model_state_dict']);model.eval()
                models[model_key]=(model,norm,state)
            model,norm,state=models[model_key]
            altered=Sequence(**vars(seq));altered.odom=seq.odom.copy();region_slice=slice(*region)
            if row['anomaly']=='wheel_scale':altered.odom[region_slice,:2]*=1.3
            elif row['anomaly']=='wheel_jump':altered.odom[region_slice,0]+=2
            x=((features(altered,state['config'])-norm['xm'])/norm['xs']).astype(np.float32)
            if row['anomaly']=='feature_outlier':x[region_slice]*=10
            with torch.inference_mode():output=model(torch.from_numpy(x)).numpy()
            mean=np.clip(output[:,:3]*norm['ys']+norm['ym'],-norm['clip'],norm['clip'])
            ungated_same_alpha=corrected(seq,mean,row['gate_alpha'],state['target'])
            ungated_error=np.linalg.norm(ungated_same_alpha[:,:2]-seq.gt[:,:2],axis=1)
            consecutive=seq.valid[1:]&seq.valid[:-1]
            records.append({'run':run,'method':row['method'],'sequence':name,'scene':row['scene'],'seed':row['seed'],
                'anomaly':row['anomaly'],'region_start':region[0],'region_stop':region[1],
                'segment_cart_ape_m':float(np.sqrt(np.mean(base[mask]**2))),
                'segment_gated_ape_m':float(np.sqrt(np.mean(err[mask]**2))),
                'segment_harmful_fraction_005m':float(((err-base)>.05)[mask].mean()),
                'full_gated_harmful_fraction_005m':row['harmful_frame_fraction'],
                'full_ungated_harmful_fraction_005m':row['ungated_harmful_frame_fraction'],
                'full_ungated_same_alpha_harmful_fraction_005m':float(((ungated_error-base)>.05)[seq.valid].mean()),
                'segment_ungated_same_alpha_ape_m':float(np.sqrt(np.mean(ungated_error[mask]**2))),
                'max_correction_offset_step_m':float(np.linalg.norm(np.diff(pose[:,:2]-seq.cart[:,:2],axis=0),axis=1)[consecutive].max()),
                'actual_full_fallback_fraction':fallback(row),
                'max_output_step_m':float(np.linalg.norm(np.diff(pose[:,:2],axis=0),axis=1)[consecutive].max()),
                'scope':'controlled postprocessor input only; no SLAM rerun or real wheel-slip annotation'})
    csv_write(OUT/'anomaly_segments.csv',records)
    return records


def figure(path):
    plt.tight_layout();plt.savefig(OUT/(path+'.png'),dpi=180);plt.savefig(OUT/(path+'.pdf'));plt.close()
def plots(data):
    methods=['cartographer','mlp_full20_pose','mlp_relative32_increment','mlp_relative32_increment_uncertainty_hard_fallback']
    labels=['Cartographer','20D pose MLP','32D increment MLP','32D + fallback']
    fig,axes=plt.subplots(1,3,figsize=(13,3.6))
    for ax,(run,scene,title) in zip(axes,[('home_complete_sequences_v2','home','Same-scene complete sequences'),('cross_scene_and_nclt','corridor','Held-out corridor'),('cross_scene_and_nclt','nclt','Held-out NCLT session')]):
        means=[];stds=[]
        for method in methods:
            mean,std=macro([r for r in data[run] if r['method']==method and r['scene']==scene],lambda r:r['ape_m']['rmse']);means.append(mean);stds.append(std)
        ax.bar(np.arange(4),means,yerr=stds,capsize=3,color=['#777777','#3178ae','#e58d3b','#4c9463'])
        ax.set_xticks(np.arange(4));ax.set_xticklabels(labels,rotation=25,ha='right',fontsize=8)
        ax.set_title(title,fontsize=10);ax.set_ylabel('Origin-aligned APE RMSE (m)')
    figure('generalization')
    ablation=['full20','no_P_C','no_P_O','no_D_C','no_D_O','no_increments','no_E','poses8','motion_difference12']
    plt.figure(figsize=(9,4))
    values=[];std=[]
    for config in ablation:
        a,b=macro([r for r in data['home_complete_sequences_v2'] if r['method']=='mlp_'+config+'_pose'],lambda r:r['ape_m']['rmse']);values.append(a);std.append(b)
    plt.bar(np.arange(len(ablation)),values,yerr=std,capsize=3);plt.xticks(np.arange(len(ablation)),ablation,rotation=25,ha='right');plt.ylabel('Macro APE RMSE (m), mean +/- seed SD')
    figure('feature_ablation')
    plt.figure(figsize=(5,4));plt.plot([.5,.68,.9,.95],[.5,.68,.9,.95],'k--',label='Nominal')
    for run,scene,label in [('home_complete_sequences_v2','home','Offline home / 32D'),('home_online_pose_uncertainty','home','Online home / 20D'),('cross_scene_and_nclt','corridor','Corridor / 32D'),('cross_scene_and_nclt','nclt','NCLT / 32D')]:
        rows=[r for r in data[run] if r['scene']==scene and r['method'].endswith('_hard_fallback')]
        plt.plot([.5,.68,.9,.95],[np.mean([r['nominal_interval_coverage'][str(p)] for r in rows]) for p in [.5,.68,.9,.95]],'o-',label=label)
    plt.xlabel('Nominal marginal coverage');plt.ylabel('Observed marginal coverage');plt.legend(fontsize=8);plt.xlim(.45,1);plt.ylim(.45,1.02)
    figure('uncertainty_coverage')
    run='home_online_pose_uncertainty';name='home1-3';methods=['cartographer','mlp_full20_pose_uncertainty_hard_fallback','mlp_full20_pose_uncertainty_anomaly_wheel_jump']
    seq=load_sequence(next(r for r in read(BASE/run/'manifest.json')['sequences'] if r['sequence_id']==name))
    plt.figure(figsize=(9,3.5))
    for method in methods:
        seed=None if method=='cartographer' else 22;_,pose=load_pose(BASE/run/f'{method}_seed{seed}'/(name+'.tum'))
        error=np.linalg.norm(pose[:,:2]-seq.gt[:,:2],axis=1);error[~seq.valid]=np.nan
        plt.plot(seq.t-seq.t[0],error,label=method.replace('mlp_full20_pose_uncertainty_',''))
    n=len(seq.t);plt.axvspan(seq.t[n//3]-seq.t[0],seq.t[2*n//3]-seq.t[0],alpha=.12,color='red',label='Injected wheel jump')
    plt.xlabel('Time (s)');plt.ylabel('Translation error (m)');plt.legend(fontsize=8)
    figure('anomaly_error')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,(run,name,title) in zip(axes,[('home_complete_sequences_v2','home1-3','Home: complete held-out sequence'),('cross_scene_and_nclt','nclt-2013-01-10','NCLT: held-out session')]):
        manifest=read(BASE/run/'manifest.json');seq=load_sequence(next(r for r in manifest['sequences'] if r['sequence_id']==name))
        for pose,label in [(seq.gt,'Reference'),(seq.cart,'Cartographer'),(seq.odom,'Provided odometry')]:
            xy=pose[:,:2].copy();xy[~seq.valid]=np.nan;ax.plot(xy[:,0],xy[:,1],label=label,linewidth=1)
        ax.set_title(title,fontsize=10);ax.set_xlabel('x (m)');ax.set_ylabel('y (m)');ax.axis('equal');ax.legend(fontsize=8)
    figure('trajectory_examples')


def benchmark_table(rows,methods,scene=None):
    result=[]
    for method in methods:
        selected=[r for r in rows if r['method']==method and (scene is None or r['scene']==scene)]
        if not selected:continue
        mean,std=macro(selected,lambda r:r['ape_m']['rmse'])
        rpe,_=macro(selected,lambda r:r['rpe']['1s']['translation_m']['rmse'])
        rotation,_=macro(selected,lambda r:r['rpe']['1s']['rotation_deg']['rmse'])
        result.append([method,f(mean)+(' ± '+f(std) if selected[0]['seed'] is not None else ''),f(rpe),f(rotation,3)])
    return table(['方法','APE RMSE/m（种子均值±SD）','RPE 1s/m','RPE 1s/°'],result)


def system_impact():
    records=[]
    for label in ['baseline','v2','causal','confidence20']:
        path=ROOT/'data/prepared/home1-3'/f'cartographer_noimu_online_resources_{label}.json'
        if not path.exists():continue
        res=read(path);log=path.with_name(f'cartographer_noimu_online_{label}.log').read_text(errors='replace')
        scores=re.findall(r'Score histogram:\s*Count: (\d+)',log)
        nodes=set(re.findall(r'Node \((\d+), (\d+)\) with',log))
        records.append({'label':label,'poses':res['pose_count'],'wall_s':res['wall_s'],
                        'slam_peak_rss_mib':res['peak_rss_mib'],
                        'last_constraint_score_histogram_count':int(scores[-1]) if scores else None,
                        'distinct_nodes_in_constraint_log':len(nodes),
                        'correction_changes_slam_state':res['correction_changes_slam_state'],
                        'interpretation':'descriptive repeated replay; logs duplicate ROS/glog lines; histogram is not loop precision/recall'})
    csv_write(OUT/'frontend_backend_observations.csv',records)
    return records


def intrinsic_metrics():
    records=[]
    for label in ['metrics_baseline','metrics_confidence20']:
        directory=ROOT/'data/prepared/home1-3'
        path=directory/f'cartographer_noimu_metrics_{label}.json'
        data=read(path)
        if data['before_shutdown']['status_code']!=0:raise ValueError('Cartographer metrics unavailable')
        record={'label':label,'snapshots':len(data['samples'])}
        for name,short in [('mapping_2d_local_trajectory_builder_latency','local_update_wall_interval'),
                           ('mapping_2d_local_trajectory_builder_real_time_ratio','local_wall_real_time_ratio'),
                           ('mapping_2d_local_trajectory_builder_cpu_real_time_ratio','local_thread_cpu_real_time_ratio'),
                           ('mapping_constraints_constraint_builder_2d_queue_length','constraint_queue_length')]:
            values=np.array([r['gauges'][name] for r in data['samples'] if name in r['gauges'] and np.isfinite(r['gauges'][name])])
            record[short+'_p50']=float(np.median(values))
            record[short+'_p95']=float(np.quantile(values,.95))
            record[short+'_max']=float(values.max())
        family=next(r for r in data['before_shutdown']['families'] if r['name']=='mapping_constraints_constraint_builder_2d_constraints')
        for item in family['metrics']:
            record['constraints_'+item['labels']['search_region']+'_'+item['labels']['matcher']]=item['value']
        score_family=next(r for r in data['before_shutdown']['families'] if r['name']=='mapping_2d_local_trajectory_builder_scores')
        record['local_scan_match_observations']=sum(b['count'] for m in score_family['metrics'] for b in m['buckets'])
        resource=read(directory/f'cartographer_noimu_online_resources_{label}.json')
        record.update(slam_peak_rss_mib=resource['peak_rss_mib'],wall_s=resource['wall_s'],
                      correction_p99_ms=resource.get('correction',{}).get('p99_ms'))
        records.append(record)
    csv_write(OUT/'cartographer_intrinsic_metrics.csv',records)
    return records


def report(data,supplement,online,anomaly,impact):
    lines=['# 导师建议实验执行报告','',
    '本报告只汇总实际生成的轨迹、模型和测量文件。已完成两个公共数据来源、18条完整序列的数据准备与Cartographer有/无IMU回放，完整序列/场景隔离训练、特征消融、传统及学习基线、紧耦合在线对比、CPU/GPU测时和置信度硬回退。自采数据、独立跨建筑/跨LiDAR受控实验、真实边缘设备与状态反馈闭环实验仍未完成。',
    '',table(['导师建议','已执行内容','仍需补齐'],[
        ['1 多数据/防止泄漏','2个公共数据来源、18条完整序列、场景与来源留出','独立自采、建筑身份、单因素跨LiDAR/轮里程计、动态标签'],
        ['2 统一对比','紧耦合2DLIW、GP适配、GRU/TCN、EKF/因子图；APE/RPE/资源','已发表学习方法完整配置复现与更多独立序列'],
        ['3 特征消融','20维分组、6/9/14/32维、目标交叉及3种子','质量特征和相同参数预算复核'],
        ['4 实时/部署/系统影响','CPU/GPU、ROS1倍速、输出接入日志、NumPy导出','真实边缘设备、完整端到端等待计时、反馈接入及闭环标签'],
        ['5 不确定性/回退','方差头、验证校准、精确硬回退、3类输入异常、同alpha对照','真实异常整链路、检测/恢复与软回退扩展']]),
    '', '## 1. 数据、划分与适用范围','',
    table(['来源','序列','本地位置','用途'],[
        ['已有 OpenLORIS home','home1-1…5（5条）','/home/fuchaoran/datasets/home1_rosbag','只读输入，原始文件未改动'],
        ['新增 OpenLORIS office','office1-1…7（7条）','data/openloris','跨场景训练'],
        ['新增 OpenLORIS cafe','cafe1-1…2（2条）','data/openloris','跨场景验证'],
        ['新增 OpenLORIS corridor','corridor1-3…5（3条）','data/openloris','场景留出测试与在线对比'],
        ['新增 NCLT','2013-01-10（1条完整会话）','data/nclt/2013-01-10','跨数据来源/平台留出测试']]),
    'OpenLORIS的office参考来自OptiTrack，其他场景来自离线激光SLAM；同一场景的传感器配置不变。不同场景不能直接当作不同建筑或不同LiDAR的受控对比。[官方数据说明](https://lifelong-robotic-vision.github.io/dataset/scene.html)。',
    '', 'NCLT本次使用Segway平台的Hokuyo UTM-30LX平面激光与公开融合里程计，未使用Velodyne；该里程计已融合轮编码器、FOG及IMU，表中“wheel_odometry”在NCLT应理解为提供的融合里程计。高频参考位姿使用里程计在图节点之间补全，因此NCLT短间隔RPE并非独立真值检验。补充在cov文件的图节点时间附近评价APE，仍不将SLAM参考当作独立动捕。[NCLT论文](https://robots.engin.umich.edu/nclt/nclt.pdf)。',
    '', '新增OpenLORIS归档位于`data/downloads`，经下载源SHA-256核验并解包；校验记录在各归档的`.verified.json`旁文件中。[当前官方下载说明](https://github.com/lifelong-robotic-vision/OpenLORIS-Scene/blob/master/download.md)。NCLT压缩包与作者转换例程在`data/nclt`，预处理输出统一在`data/prepared/<sequence>`。',
    '', table(['协议','训练','验证','测试','输入'],[
        ['同域完整序列','home1-1','home1-2','home1-3/4/5','离线最终Cartographer，无IMU'],
        ['跨场景/数据来源','home全部5条+office全部7条','cafe两条','corridor三条+NCLT一条','分别训练无IMU/有IMU条件'],
        ['同域因果在线','home1-1在线轨迹','home1-2在线轨迹','home1-3/4/5在线轨迹','在线Cartographer，无IMU']]),
    '三种协议独立训练，不能把跨场景训练用到home测试序列说成同域协议泄漏，也不能把不同协议的测试称为共同未见测试。home原有开发历史未知，所有结果作为开发基准；本轮观察过测试结果后新增的20维置信度模型也属于开发扩展，最终论文还应保留新采集的冻结测试集。场景协议对train/validation/test做scene_id互斥检查；NCLT平台整体留出，但场景、里程计及来源同时变化，不能拆分单一硬件因素。',
    '', '## 2. 评分与测量协议','',
    '采用10Hz估计器时间网格；网格、输入归一化与历史缓存不依赖真值。在线取时间戳不晚于网格时刻的最近输入，最大年龄0.15s；真值只供监督和评分，插值跨度超过0.25s排除。SLAM输入bag不含/gt。所有输出转换到base_link参考点和共同平面坐标约定。',
    '', '主指标为起点对齐的平面APE及1s/5s、1m/5m的平移/航向RPE，无尺度拟合；RPE配对不能跨缺失段。训练基准保留固定Cartographer监督坐标原点；不同有/无IMU条件首段航向差会影响其他基线的原点评分，不能把两种条件直接当作同一参考原点。`alignment_and_nclt_nodes.csv`补充全轨迹SE(2)刚体对齐，供评分分析，不参与推理。在线紧耦合表在各方法共同有效时刻独立做同一起点对齐，同时报告每种方法完整覆盖率，避免失败片段被静默删除。',
    '', '10Hz后处理保留输入时间网格并排除评分中的不可用点，32维统计窗口目前仍会包含缺失输入的保持值；真实ROS运行在里程计失效时清空窗口。这两种缓存处理尚未完全一致，缺失段附近的32维结果应作为开发结果。20维位姿模型不采用滚动统计，仍需补充独立丢帧与时间异常实验。',
    '', '学习方法3个种子11/22/33，按序列等权后再按种子汇总；归一化/限幅仅训练集拟合，早停、融合alpha、方差校准、阈值及部署种子仅验证集选择。`paired_sequence_intervals.csv`采用序列配对bootstrap；每场景只有3条测试轨迹，其区间精度有限。不得用相关帧数充当独立样本量。',
    '', '运行时间明确区分：离线回放墙钟（包含完整优化）、在线1倍速回放墙钟（受播放速率约束，不等于算法纯计算耗时）、后处理批量耗时、batch=1模型前向、在线修正回调。SLAM峰值RSS只测子进程；训练基准RSS是累计进程峰值，不能称单模型独立内存。部分ROS回放在独立master上同时运行，负载并非严格隔离，开销对照只能作描述性测量。',
    '', '## 3. 方法对比：离线最终优化输入','',
    '传统方法包含保守协方差交叉融合、膨胀噪声EKF、固定滞后与全轨迹线性化因子图；因子图不是重新运行原始扫描匹配或引入额外真值闭环。Cartographer与其输入里程计相关，噪声使用验证集选择的保守代理，不能宣称最优协方差估计。学习基线包含岭回归、256基函数高斯过程适配、GRU和因果TCN；GP是共享特征适配，GRU/TCN为自建架构，尚不等同于某篇论文完整复现。',
    '', '### home：同域完整序列','',benchmark_table(data['home_complete_sequences_v2'],SELECTED),
    '### corridor：完全留出场景，无IMU输入','',benchmark_table(data['cross_scene_and_nclt'],SELECTED,'corridor'),
    '这一条件下20维模型在验证集选择的alpha=0，因此其测试输出等于原始轨迹；相同APE不意味着学到了跨场景修正。32维部分种子会放大测试误差；门控输出接近原轨迹也必须结合真实回退比例解释。',
    '### corridor：有IMU输入','',benchmark_table(data['cross_scene_and_nclt_imu'],SELECTED,'corridor'),
    '### NCLT：独立数据来源的失败与退化案例','',benchmark_table(data['cross_scene_and_nclt'],SELECTED,'nclt'),
    'NCLT仅有一条会话，不能给出多会话泛化结论。沿用home室内10m配置时Cartographer全局误差显著，学习修正也不能可靠修复。保留这些结果；另做30m物理量程且关闭跨子图约束采样的诊断回放，不替换主结果、不称为盲测配置。诊断误差如下：','']
    diagnostic=[]
    for dirname in ['nclt-2013-01-10','nclt-2013-01-10_sensorconfig']:
        for mode in ['noimu','imu']:
            directory=ROOT/'data/prepared'/dirname
            seq=load_sequence({'sequence_id':dirname,'scene_id':'nclt','cart':str(directory/f'cartographer_{mode}_final.tum'),
                               'odom':str(directory/'odom.tum'),'gt':str(directory/'groundtruth.tum')})
            diagnostic.append([dirname,mode,f(metrics(seq,seq.cart)['ape_m']['rmse'],3),f(metrics(seq,rigid_alignment(seq.cart,seq.gt,seq.valid))['ape_m']['rmse'],3)])
    lines += [table(['配置','IMU条件','起点APE/m','全轨迹SE2 APE/m'],diagnostic),
    '这些数字反映当前适配与配置的表现，尚不能单独判断漂移来自场景退化、传感器模型、时间补偿或闭环约束。后续应从标定与无回环前端逐项验证，不能通过测试集调参后覆盖原始失败记录。NCLT适配采用论文外参和作者距离编码，平面激光按25ms周期近似束时间；原始文件没有逐束时间，时间补偿假设仍需单独验证。',
    '', '![跨场景误差](../results/mentor_benchmark/report/generalization.png)',
    '图中误差棒为三个种子的宏平均标准差；NCLT只有一条会话，误差棒不是会话泛化置信区间。',
    '', '![轨迹示例](../results/mentor_benchmark/report/trajectory_examples.png)',
    '', '## 4. 特征消融与目标交叉对照','',
    '20维分组为P_O/P_C（当前位姿）、D_O/D_C（帧间增量）、E（估计器差），每组x/y/sinθ/cosθ。去掉一组重新训练；紧凑表示为6/9/14维机体系运动特征；32维增加因果窗口均值/标准差和运动状态。固定MLP隐藏层64/64/32，输入维数改变参数量，尚未完成相同参数预算复核。',
    '',benchmark_table(data['home_complete_sequences_v2'],['mlp_'+c+'_pose' for c in ['full20','no_P_C','no_P_O','no_D_C','no_D_O','no_increments','no_E','poses8','motion_difference12']]),
    '',benchmark_table(data['home_complete_sequences_v2'],['mlp_relative6_increment','mlp_relative9_increment','mlp_relative14_increment','mlp_relative32_increment','mlp_no_window_mean_increment','mlp_no_window_std_increment','mlp_no_motion_state_increment','mlp_full20_increment','mlp_relative32_pose']),
    '跨场景全部消融结果在`all_methods.csv`中。不能由home的20维优势直接推断跨场景优势，也不能由更丰富维数推断更高精度。',
    '', '![特征消融](../results/mentor_benchmark/report/feature_ablation.png)',
    '', '## 5. 因果在线训练与实际在线运行','',benchmark_table(data['home_online_inputs'],SELECTED),
    '20维置信度追加开发实验：','',benchmark_table(data['home_online_pose_uncertainty'],['cartographer','mlp_full20_pose_uncertainty','mlp_full20_pose_uncertainty_hard_fallback']),
    '实际ROS在线比较使用相同扫描、轮里程计与IMU输入bag；Cartographer_noimu信息较少，Cartographer_imu与2DLIW的信息预算匹配。2DLIW运行作者实现并依据bag外参转回base_link；采用实际发布TF的当前轨迹，未用最后优化文件冒充在线结果。[作者实现](https://github.com/LittleDang/2DLIW-SLAM)。',
    '', '2DLIW回放结束的SIGINT退出中发现上游线程清理异常，轨迹是在完整回放时捕获；需保留退出故障，不能报告为无故障部署。部分早期试跑因私有参数命名空间不正确失败，正式v2/crossscene结果使用修正后的命名空间，失败日志保留。','']
    for scene in ['home','corridor']:
        values=[]
        for method in sorted({r['method'] for r in online if r['scene']==scene}):
            a=[r for r in online if r['scene']==scene and r['method']==method]
            values.append([method,f(np.mean([r['ape_rmse_m'] for r in a])),f(np.mean([r['rpe_1s_m'] for r in a])),f(np.mean([r['rpe_1s_deg'] for r in a]),3),f(np.mean([r['full_input_coverage'] for r in a]),3),f(np.mean([r['common_coverage'] for r in a]),3)])
        lines += ['### '+scene+'：真实在线发布轨迹，共同时间评分','',table(['方法','APE/m','RPE1s/m','RPE1s/°','完整输出覆盖率','共同覆盖率'],values)]
    paired=[]
    for name in ['home1-3','home1-4','home1-5']:
        raw=next(r for r in online if r['sequence']==name and r['method']=='cartographer_noimu')
        corrected=next(r for r in online if r['sequence']==name and r['method']=='mlp20_confidence_online')
        paired.append([name,f(raw['ape_rmse_m']),f(corrected['paired_cart_map_alignment_ape_m']),f(corrected['ape_rmse_m']),f(corrected['p99_ms'],3),f(corrected['fallback_fraction'],3)])
    lines += ['原始与修正位姿发布在同一map坐标系。下表补充保留原始Cartographer对齐变换的配对评价；与上表各方法独立重新对齐的主评分并列，不据此覆盖较差结果。','',
    table(['序列','原始APE/m','同map对齐修正APE/m','独立起点对齐修正APE/m','回调P99/ms','实际回退比例'],paired),
    '20维实际在线修正在同map评价中降低误差，但独立起点评分在home1-3/4上变差。独立起点对齐会移除共同有效首帧的修正平移与航向，并把该航向差旋转到整段轨迹；可见收益依赖坐标与评分条件，不能直接宣称全面优于原方法。训练输入固定10Hz网格，实际ROS回调使用约10Hz到达时间，初始参考帧与采样抖动也有差异，尚需新采集冻结序列进一步复核。',
    '', '在线20维方差头的部署种子仅按home1-2验证APE选择（seed22），模型已冻结。先前32维seed11在线训练模型的验证gate_alpha=0，实际回放会全程选择原始轨迹；这属于关闭修正的结果，不能称高效改善定位。旧离线训练32维在线试跑单独标注为域变化试验。',
    '', '## 6. 不确定性、异常与回退','',
    '使用异方差高斯回归输出三维残差均值和方差，按训练尺度标准化；方差校准、90% sigma阈值与99%最大特征z阈值由验证集拟合。输出是方差/异常分数，并非“修正正确概率”。增量模型在硬回退时直接重置为当前原始Cartographer位姿，清除此前积分偏移；在线输入里程计失效时发布原始位姿并重置缓存。alpha=0同样属于真实全回退。早期results.json的fallback_fraction只记录门控拒绝，汇总CSV已依据gate_alpha修正真实回退比例，同时保留拒绝比例字段。','']
    calibration=[]
    for run,scene in [('home_complete_sequences_v2','home'),('home_online_inputs','home'),('home_online_pose_uncertainty','home'),('cross_scene_and_nclt','corridor'),('cross_scene_and_nclt','nclt')]:
        rows=[r for r in data[run] if r['scene']==scene and r['method'].endswith('_hard_fallback')]
        calibration.append([run,scene,f(np.mean([fallback(r) for r in rows]),3),f(np.mean([r['marginal_95pct_coverage'] for r in rows]),3),f(np.mean([r['normalized_gaussian_nll'] for r in rows]),3),f(np.mean([r['sigma_error_correlation'] for r in rows if r['sigma_error_correlation'] is not None]),3)])
    lines += [table(['实验','测试域','真实回退比例','95%边际区间覆盖','归一化NLL','sigma与误差相关'],calibration),
    '置信区间有时过宽；高覆盖不等于方差头识别了分布外风险。大量回退也不能作为泛化成功证据。位姿/增量标签及其归一化尺度不同，表中NLL不用于跨目标排序。没有进行软回退、集成不确定性及完整规则门控单独对照。',
    '', '![边际区间覆盖](../results/mentor_benchmark/report/uncertainty_coverage.png)',
    '', '受控异常只改变后处理输入：轮里程计尺度1.3、x跳变2m、归一化特征放大10倍；异常在中间1/3段注入。Cartographer输入未扰动，不能替代真实轮滑/遮挡下整套SLAM的稳健性检验。有害帧定义为相对原始Cartographer平移误差增大超过0.05m；它是逐帧开发指标，不是带持续时间的失效事件。','']
    ar=[]
    for kind in ['wheel_scale','wheel_jump','feature_outlier']:
        a=[r for r in anomaly if r['run']=='home_online_pose_uncertainty' and r['anomaly']==kind]
        ar.append([kind,f(np.mean([r['segment_cart_ape_m'] for r in a])),f(np.mean([r['segment_gated_ape_m'] for r in a])),f(np.mean([r['segment_harmful_fraction_005m'] for r in a]),3),f(np.mean([r['full_ungated_harmful_fraction_005m'] for r in a]),3),f(np.mean([r['full_gated_harmful_fraction_005m'] for r in a]),3)])
    lines += [table(['异常','异常段原始APE/m','异常段门控APE/m','异常段有害比例','全序列无门控有害比例','全序列门控有害比例'],ar),
    '上表最初的无门控与门控版本分别按验证集选alpha，差异也包含修正强度的变化。补充在同一个gate_alpha下只改变门控的32维异常对照：','']
    matched=[]
    for run,scene in [('home_complete_sequences_v2','home'),('cross_scene_and_nclt','corridor')]:
        for kind in ['wheel_scale','wheel_jump','feature_outlier']:
            rows=[r for r in anomaly if r['run']==run and r['scene']==scene and r['anomaly']==kind]
            matched.append([scene,kind,f(np.mean([r['full_ungated_same_alpha_harmful_fraction_005m'] for r in rows]),3),f(np.mean([r['full_gated_harmful_fraction_005m'] for r in rows]),3)])
    lines += [table(['32维测试域','异常','同alpha无门控有害比例','同alpha门控有害比例'],matched),
    '门控仍可能引入有害修正和轨迹跳变，不能声称在异常下保证不出错。完整逐序列统计、最大输出步长及修正偏移步长在`anomaly_segments.csv`，后续还需报告检测/恢复延迟和专门的回退边界跳变量。',
    '', '![异常段误差](../results/mentor_benchmark/report/anomaly_error.png)',
    '', '## 7. CPU/GPU、内存与部署','',
    '当前CPU为Intel Core i7-14650HX，GPU为RTX5060 Laptop；硬件清单在`hardware_inventory.json`。CPU/GPU测量不是边缘设备测量。','']
    cpu=read(BASE/'home_complete_sequences_v2/cpu_matched_runtime_latency.json');gpu=read(BASE/'home_complete_sequences_v2/gpu_latency.json')
    latencies=[]
    for device,result in [('CPU',cpu),('RTX5060 Laptop GPU',gpu)]:
        for row in result['models']:latencies.append([device,row['model'],f(row['mean_ms'],3),f(row['p95_ms'],3),f(row['p99_ms'],3),f(row.get('peak_allocated_mib'),2)])
    lines += [table(['设备','模型','均值/ms','P95/ms','P99/ms','Torch峰值allocated/MiB'],latencies),
    '上述CPU/GPU使用同一Torch2.7.1、batch=1、CPU线程1、预热30次后500次测量；GPU完成同步。仅包含模型前向，不包含数据传输/特征/同步等待/SLAM。早期Torch2.4.1不支持此GPU计算能力的失败记录保留。该小模型在本次测量中CPU前向更快，GPU启动开销不能忽略。','']
    fresh=read(BASE/'postprocessing_fresh_process_resources.json');csv_write(OUT/'postprocessing_resources.csv',fresh)
    lines += ['各后处理方法还在独立新进程中实测资源，并逐点核对其输出与已保存轨迹一致。下面以home1-3在线输入为代表（学习方法seed11），墙钟按完整已加载轨迹计算，框架、数据加载与基线拟合均可能计入进程RSS峰值；所有方法使用同一Torch/Scipy测量脚本，RSS包含公共测量框架，并非每种部署实现的最小内存。Cartographer行仅复制已加载输入轨迹，其SLAM耗时/RSS另见原始资源JSON及统一CSV。其余corridor1-3与NCLT测量在`postprocessing_resources.csv`。','',
        table(['方法','整段后处理/s','均值/帧 ms','独立进程峰值RSS/MiB'],[[r['method'],f(r['mean_trajectory_wall_s'],5),f(r['mean_ms_per_frame'],4),f(r['fresh_process_peak_rss_mib'],1)] for r in fresh if r['sequence']=='home1-3']), '']
    pr=read(BASE/'portable_online_selected/portable_recorded_latency.json')
    lines += [f'仅依赖NumPy的32维便携模型权重为{pr["weights_bytes"]}字节；真实已记录在线输入的主机独立进程测量：均值{pr["mean_ms"]:.3f}ms、P99 {pr["p99_ms"]:.3f}ms、进程峰值RSS {pr["process_peak_rss_mib"]:.2f}MiB，回退比例{pr["fallback_fraction"]:.3f}。接受分支样本{pr["branch_latency"]["accepted"]["count"]}、回退分支样本{pr["branch_latency"]["fallback"]["count"]}；输入查找和ROS发布不计入。该导出模型使用真实在线输入训练，seed22按验证集门控APE在三个32维种子中选择。导出与Torch的100组随机输出一致性检查误差保存于portable.json。',
    '', 'ROS修正回调的P95/P99和峰值RSS在`online_resources.json`及各原始资源JSON中；计时包含特征、CPU前向、门控、积分、写文件和发布提交，不含消息到达等待/ROS排队/消费者接收延迟。回调低于100ms不等于完整链路端到端硬实时。没有边缘设备、功耗与温度实测，NumPy可移植文件只是部署准备。',
    '', '## 8. 对前端及闭环的影响','',
    '当前修正在输出端发布/deepfusion_pose，代码未将修正反馈给扫描匹配或Cartographer状态。新增不带模型的基线回放，结合带模型的资源与后端约束日志进行描述性比较：','',
    table(['home1-3运行','输出位姿数','回放墙钟/s','SLAM RSS/MiB','最终分数直方图计数','日志中不同节点数'],[[r['label'],r['poses'],f(r['wall_s'],2),f(r['slam_peak_rss_mib'],2),r['last_constraint_score_histogram_count'],r['distinct_nodes_in_constraint_log']] for r in impact]),
    '日志计数不是闭环正确率，重复回放又受线程调度和同时负载影响，不能据此声称回环检测得到改善。再在其他基准完成后串行运行baseline与置信度修正，并开启Cartographer内置指标，以1Hz读取当前前端更新间隔、CPU实时比及约束队列：','']
    intrinsic=intrinsic_metrics()
    lines += [table(['串行运行','快照数','前端更新间隔P50/ms','更新间隔P95/ms','线程CPU实时比P50','约束批次条目最大','local searched/found','global searched/found'],[
        [r['label'],r['snapshots'],f(r['local_update_wall_interval_p50']*1000,3),f(r['local_update_wall_interval_p95']*1000,3),f(r['local_thread_cpu_real_time_ratio_p50'],2),r['constraint_queue_length_max'],
         f(r['constraints_local_searched'],0)+'/'+f(r['constraints_local_found'],0),f(r['constraints_global_searched'],0)+'/'+f(r['constraints_global_found'],0)] for r in intrinsic]),
    '已核对安装版本源码：名为latency的Gauge记录相邻local SLAM完成更新的墙钟间隔，包含1倍速数据流周期；不是单次扫描匹配纯计算耗时。CPU实时比为传感器时间增量/线程CPU时间增量；这些是一秒一次的末次更新快照，不是逐帧时延分布。queue_length为当前constraints容器大小，包含已计算但待批次回调清空的结果，不等于未完成任务或ROS消息积压。searched/found为实际约束搜索统计，不是闭环精确率/召回率。串行对照仅一条序列、一次重复，不能得出统计显著的无负载影响结论。前端扫描匹配观测次数及全部原始直方图另行保存。',
    '', '已核实输出接入路径并记录共享负载下的前端更新与约束计算；逐扫描匹配纯耗时、真实闭环精确率/召回率和将修正用作前端先验/后端约束的反馈实验仍需新增仪器化与独立闭环标签。',
    '', '## 9. 文件与复现','',
    table(['产物','位置'],[
        ['全部离线/因果后处理逐序列指标','results/mentor_benchmark/report/all_methods.csv'],
        ['实际在线共同时间指标','results/mentor_benchmark/report/online_common_time.csv'],
        ['全轨迹SE2与NCLT节点补充','results/mentor_benchmark/report/alignment_and_nclt_nodes.csv'],
        ['配对序列区间','results/mentor_benchmark/report/paired_sequence_intervals.csv'],
        ['异常段指标','results/mentor_benchmark/report/anomaly_segments.csv'],
        ['图（PNG/PDF）','results/mentor_benchmark/report/{generalization,feature_ablation,uncertainty_coverage,anomaly_error,trajectory_examples}'],
        ['权重/训练/轨迹/软件与代码快照','results/mentor_benchmark/<run>/<method>_seed<seed>/，各run根目录'],
        ['复现命令与边界','docs/实验复现说明.md']]),
    '', f'本报告读取{sum(len(v) for v in data.values())}条已测模型/序列/种子/异常结果行及{len(online)}条在线方法/序列结果。它们不是同等数量的独立序列。旧版常量通道归一化导致全回退的试跑保留在home_complete_sequences及早期在线文件，主表采用v2与明确在线训练版本。',
    '', '尚需提供独立自采bag目录、设备/标定/真值信息和边缘设备访问；之后按固定清单接入，不使用现有公共测试结果重新选择最终测试集参数。多建筑标识、真实跨LiDAR单因素对比、动态程度标签、已发表学习方法完整复现、边缘实测及前端/闭环反馈验证在完成前均不能写成论文已完成实验。','']
    (ROOT/'docs/实验执行报告.md').write_text('\n'.join(lines))


def main():
    torch.set_num_threads(1)
    OUT.mkdir(parents=True,exist_ok=True)
    for name in ['corridor1-3','corridor1-4','corridor1-5']:
        for algorithm in ['cartographer_noimu','cartographer_imu','2dliw']:
            if not (ROOT/'data/prepared'/name/f'{algorithm}_online_resources_crossscene.json').exists():
                raise RuntimeError('Online measurement still pending: '+name+'/'+algorithm)
    for label in ['metrics_baseline','metrics_confidence20']:
        if not (ROOT/'data/prepared/home1-3'/f'cartographer_noimu_metrics_{label}.json').exists():
            raise RuntimeError('Cartographer intrinsic metric run pending: '+label)
    if len(read(BASE/'postprocessing_fresh_process_resources.json'))!=36:
        raise RuntimeError('Fresh-process measurements are not all complete')
    data,supplement=collect();paired_intervals(data);anomaly=anomalies(data)
    online=online_evaluation();impact=system_impact();plots(data);report(data,supplement,online,anomaly,impact)
    print('Saved measured report, CSV tables and PNG/PDF figures to',OUT)
if __name__=='__main__':main()
