"""Paired, same-weight/same-alpha gates on captured causal postprocessor inputs.

All policies share availability guards and the exact same predictions. Reference
coverage stays frozen from the clean stream; synthetic missing frames are scored.
"""
import copy
import numpy as np
import torch

from causal_stream import extract, wrap
from mentor_benchmark import corrected, metrics

MODES = ('none', 'rule', 'uncertainty')


def corrupt(data, name, parameters):
    result={k:v.copy() for k,v in data.items()}
    n=len(data['t']);start,end=n//3,2*n//3
    if 'duration_s' in parameters:
        end=min(end,start+max(1,int(round(parameters['duration_s']*float(data['hz'])))))
    affected=np.zeros(n,bool)
    if name!='clean':affected[start:end]=True
    if name=='wheel_scale':result['odom'][affected,:2]*=parameters['xy_scale']
    elif name=='wheel_jump':result['odom'][affected,0]+=parameters['x_offset_m']
    elif name=='odom_drop':
        dropped=np.arange(start,end,parameters['every_nth_sample'])
        result['available'][dropped]=False;result['odom'][dropped]=np.nan
    elif name=='odom_outage':
        result['available'][affected]=False;result['odom'][affected]=np.nan
    elif name=='cart_outage':
        # During a publication outage only the last already seen pose is known.
        result['cart'][affected]=result['cart'][start-1]
        result['available'][affected]=False
    elif name=='odom_delay':
        indices=np.arange(start,end);lag=max(1,int(round(parameters['lag_s']*float(data['hz']))))
        previous=np.maximum(indices-lag,0)
        result['odom'][indices]=data['odom'][previous]
        result['available'][indices] &= data['available'][previous]
    elif name=='odom_stale_stamp':
        result['odom_source_t'][affected]+=parameters['source_timestamp_offset_s']
        ages=result['t']-result['odom_source_t']
        result['available'] &= (ages>=0)&(ages<=parameters['age_guard_s'])
    elif name not in ('clean','feature_outlier','wheel_scale','wheel_jump'):
        raise ValueError('Unknown corruption '+name)
    return result,affected,start,end


def gate_masks(ready, normalized, output, parameters):
    common=ready & np.isfinite(output).all(1) & np.isfinite(normalized).all(1)
    rule=common & (np.abs(normalized).max(1)<=parameters['ood_threshold'])
    sigma=np.exp(.5*np.clip(output[:,3:],-8,6)).mean(1)
    return {'none':common,'rule':rule,
            'uncertainty':rule & (sigma<=parameters['sigma_threshold'])}


def first_streak(accepted,start,t,required=3):
    for i in range(start,len(accepted)-required+1):
        if accepted[i:i+required].all():return float(t[i]-t[start])
    return None


def new_rejection_latency(fallback,clean_fallback,affected,t):
    rejected=np.flatnonzero(fallback & ~clean_fallback & affected)
    region=np.flatnonzero(affected)
    return None if not len(rejected) or not len(region) else float(t[rejected[0]]-t[region[0]])


def fraction(values,mask):
    return float(np.asarray(values)[mask].mean()) if np.any(mask) else None


def prepare(seq,data,state,model,anomaly,parameters):
    altered,affected,start,end=corrupt(data,anomaly,parameters)
    cart,odom,x,ready,_=extract(altered['t'],altered['cart'],altered['odom'],
                               altered['available'],state['config'])
    working=copy.copy(seq);working.cart=cart;working.odom=odom
    working.available=altered['available'];working.ready=ready
    # GT and validity mask deliberately stay identical to the clean stream.
    norm={k:v.numpy() for k,v in state['norm'].items()}
    z=((x[1:]-norm['xm'])/norm['xs']).astype(np.float32)
    if anomaly=='feature_outlier':z[affected[1:]]*=parameters['normalized_scale']
    with torch.inference_mode():output=model(torch.from_numpy(z)).numpy()
    prediction=np.clip(output[:,:3]*norm['ys']+norm['ym'],-norm['clip'],norm['clip'])
    prediction[:,2]=wrap(prediction[:,2])
    masks=gate_masks(ready[1:],z,output,state['gate_parameters'])
    alpha=state['gate_parameters']['gate_alpha']
    poses={mode:corrected(working,prediction,alpha,state['target'],mask) for mode,mask in masks.items()}
    accepted={mode:np.r_[False,mask & (alpha!=0)] for mode,mask in masks.items()}
    return working,poses,accepted,affected,start,end,alpha


def evaluate(seq,data,state,model,protocol,save=None):
    clean_working,clean_poses,clean_accept,_,_,_,alpha=prepare(seq,data,state,model,'clean',{})
    original_error=np.linalg.norm(seq.cart[:,:2]-seq.gt[:,:2],axis=1)
    records=[]
    for anomaly,parameters in protocol['corruptions'].items():
        work,poses,accepted,affected,start,end,_=prepare(seq,data,state,model,anomaly,parameters)
        corrupted_raw_error=np.linalg.norm(work.cart[:,:2]-seq.gt[:,:2],axis=1)
        none_error=np.linalg.norm(poses['none'][:,:2]-seq.gt[:,:2],axis=1)
        errors={mode:np.linalg.norm(pose[:,:2]-seq.gt[:,:2],axis=1) for mode,pose in poses.items()}
        for mode in MODES:
            estimate=poses[mode];active=accepted[mode];fallback=~active
            clean_fallback=~clean_accept[mode]
            # Every fallback (including removal of integrated offsets) is exact.
            np.testing.assert_array_equal(estimate[fallback],work.cart[fallback])
            score=metrics(seq,estimate)
            recovery=first_streak(active,end,seq.t) if anomaly!='clean' else None
            clean_recovery=first_streak(clean_accept[mode],end,seq.t) if anomaly!='clean' else None
            harmful=(errors[mode]-corrupted_raw_error)>protocol['harm_threshold_m']
            beneficial=(corrupted_raw_error-none_error)>protocol['harm_threshold_m']
            eligible=seq.valid & work.ready
            region_valid=seq.valid & affected
            offsets=estimate[:,:2]-work.cart[:,:2]
            consecutive_valid=seq.valid[1:] & seq.valid[:-1]
            row={'sequence':seq.name,'scene':seq.scene,'config':state['config'],'seed':state['seed'],
                'anomaly':anomaly,'gate_mode':mode,'alpha':alpha,'alpha_disabled':alpha==0,
                'affected_grid_interval':[start,end] if anomaly!='clean' else None,
                'samples':len(seq.t),'scored_samples':int(seq.valid.sum()),
                'ape_m':score['ape_m'],'rpe':score['rpe'],'coverage':score['coverage'],
                'raw_clean_ape_m':metrics(seq,seq.cart)['ape_m']['rmse'],
                'raw_corrupted_ape_m':metrics(seq,work.cart)['ape_m']['rmse'],
                'harmful_fraction_vs_corrupted_raw':fraction(harmful,seq.valid),
                'harmful_fraction_vs_clean_raw':fraction((errors[mode]-original_error)>protocol['harm_threshold_m'],seq.valid),
                'region_harmful_fraction_vs_corrupted_raw':fraction(harmful,region_valid),
                'harmful_fraction_among_accepted':fraction(harmful,seq.valid & active),
                'region_rmse_m':float(np.sqrt(np.mean(errors[mode][region_valid]**2))) if region_valid.any() else None,
                'fallback_fraction_all':float(fallback.mean()),
                'fallback_fraction_region':fraction(fallback,affected),
                'clean_rejected_fraction_ready':fraction(clean_fallback,seq.valid & clean_working.ready),
                'newly_rejected_fraction_region':fraction(fallback & ~clean_fallback,affected),
                'rejection_of_beneficial_none_fraction':fraction(fallback & beneficial,eligible),
                'new_rejection_latency_s':new_rejection_latency(fallback,clean_fallback,affected,seq.t),
                'recovery_s':recovery,'recovery_censored':anomaly!='clean' and recovery is None,
                'clean_recovery_s':clean_recovery,
                'additional_recovery_delay_s':recovery-clean_recovery if recovery is not None and clean_recovery is not None else None,
                'max_pose_step_m':float(np.linalg.norm(np.diff(estimate[:,:2],axis=0),axis=1)[consecutive_valid].max()) if consecutive_valid.any() else None,
                'max_correction_offset_step_m':float(np.linalg.norm(np.diff(offsets,axis=0),axis=1).max()),
                'sigma_additional_rejection_fraction':fraction(accepted['rule'] & ~accepted['uncertainty'],eligible),
                'scope':protocol['scope'],'gate_selection_split':'validation','same_clean_gt_mask':True}
            if save:save(anomaly,mode,work,estimate,active)
            records.append(row)
    return records
