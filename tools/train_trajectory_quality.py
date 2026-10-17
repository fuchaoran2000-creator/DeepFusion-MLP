#!/usr/bin/python3
"""Controlled feature/loss study; freeze all choices before reading evaluation GT."""
import argparse
import copy
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from causal_stream import wrap
from mentor_benchmark import load_sequence,increments,targets,corrected,metrics,save_tum,validate_manifest
from residual_models import Regressor
from trajectory_quality import extract_quality,QualityResidualStream,PROTOCOL
from trajectory_objective import path_loss

SEEDS=[11,22,33]
HORIZONS=(10,30,50)
TAUS=[None,5.,30.]
ALPHAS=np.linspace(0,1,21)


def dump(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def load(row,quality_root):
    seq=load_sequence(row)
    with np.load(quality_root/(seq.name+'.npz')) as d:
        np.testing.assert_array_equal(d['t'],seq.t);q=d['quality'].copy()
    meta=json.loads((quality_root/(seq.name+'.json')).read_text())
    assert meta['captured_stream_sha256']==hashlib.sha256(Path(row['sampled_stream']).read_bytes()).hexdigest()
    xs={};common_ready=None
    for config in ['padded44','quality44']:
        x,ready=extract_quality(seq.t,seq.cart,seq.odom,seq.available,q,config)
        xs[config]=x[1:]
        if common_ready is not None:np.testing.assert_array_equal(ready,common_ready)
        common_ready=ready
    seq.ready=common_ready
    return dict(seq=seq,quality=q,x=xs)


def training_norm(data,config):
    xs=[];ys=[]
    for d in data:
        y,valid=targets(d['seq'],'increment');xs.append(d['x'][config][valid]);ys.append(y[valid])
    x,y=np.concatenate(xs),np.concatenate(ys)
    std=x.std(0)
    scale=np.maximum(y.std(0),1e-6)
    return dict(xm=x.mean(0),xs=np.where(std<1e-5,1.,std),ym=y.mean(0),ys=scale,
                clip=np.maximum(np.quantile(np.abs(y),.995,axis=0)*1.5,3*scale))


def windows(data,config,norm):
    pools=[];sizes=[]
    for d in data:
        s=d['seq'];y,valid=targets(s,'increment');n=len(y)
        starts=np.arange(0,max(n-50+1,0),20)
        starts=np.array([i for i in starts if valid[i:i+50].all()],dtype=int)
        if not len(starts):raise ValueError(f'{s.name} has no complete valid 5s window')
        indices=starts[:,None]+np.arange(50)
        x=((d['x'][config]-norm['xm'])/norm['xs']).astype(np.float32)
        ci,gi=increments(s.cart).astype(np.float32),increments(s.gt).astype(np.float32)
        values=[x[indices],((y-norm['ym'])/norm['ys']).astype(np.float32)[indices],ci[indices],gi[indices]]
        pools.append(tuple(torch.from_numpy(v) for v in values));sizes.append(dict(sequence=s.name,windows=len(starts)))
    return pools,sizes


def objective(output,y,ci,gi,norm,with_path):
    mean,logvar=output[...,:3],output[...,3:].clamp(-8,6)
    nll=(.5*((mean-y)**2*torch.exp(-logvar)+logvar)).mean()
    physical=mean*norm['ys']+norm['ym']
    physical=torch.maximum(torch.minimum(physical,norm['clip']),-norm['clip'])
    path=path_loss(physical,ci,gi,torch.maximum(norm['ys'],torch.tensor([.005,.005,.001])),HORIZONS)
    return nll+(path if with_path else 0),path


def train(train_data,val_data,config,with_path,seed,epochs):
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    norm=training_norm(train_data,config)
    train_pools,train_counts=windows(train_data,config,norm);val_pools,val_counts=windows(val_data,config,norm)
    tn={k:torch.from_numpy(np.asarray(v,dtype=np.float32)) for k,v in norm.items()}
    model=Regressor(44,'mlp',True);optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=1e-5)
    steps=max(1,math.ceil(sum(len(p[0]) for p in train_pools)/16))
    best=math.inf;best_state=None;stale=0;history=[];started=time.perf_counter()
    for epoch in range(epochs):
        model.train();losses=[]
        for _ in range(steps):
            # Equal sequence probability prevents longer trajectories dominating.
            seq_indices=rng.integers(len(train_pools),size=16)
            samples=[(train_pools[j],rng.integers(len(train_pools[j][0]))) for j in seq_indices]
            x,y,ci,gi=[torch.stack([pool[k][i] for pool,i in samples]) for k in range(4)]
            optimizer.zero_grad();out=model(x.reshape(-1,44)).reshape(16,50,6)
            loss,_=objective(out,y,ci,gi,tn,with_path)
            if not torch.isfinite(loss):raise ValueError('Non-finite training objective')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step();losses.append(float(loss.detach()))
        model.eval();scores=[]
        with torch.inference_mode():
            for pool in val_pools:
                score=[]
                for i in range(0,len(pool[0]),32):
                    x,y,ci,gi=[v[i:i+32] for v in pool]
                    out=model(x.reshape(-1,44)).reshape(len(x),50,6)
                    _,path=objective(out,y,ci,gi,tn,False);score.append((float(path),len(x)))
                scores.append(sum(a*n for a,n in score)/sum(n for a,n in score))
        val=float(np.mean(scores))
        history.append(dict(epoch=epoch+1,train_objective=float(np.mean(losses)),validation_common_path_loss=val))
        if val<best-1e-5:
            best=val;stale=0;best_epoch=epoch+1;best_state=copy.deepcopy(model.state_dict())
        else:stale+=1
        if stale>=15:break
    model.load_state_dict(best_state);model.eval()
    return model,norm,dict(seed=seed,feature_config=config,trajectory_loss=with_path,best_epoch=best_epoch,
        best_validation_path_loss=best,early_stop_criterion='same validation relative 1/3/5s path loss for both losses',
        epochs=len(history),wall_s=time.perf_counter()-started,history=history,
        train_windows=train_counts,validation_windows=val_counts,parameter_count=sum(p.numel() for p in model.parameters()))


def prediction(model,norm,d,config):
    z=((d['x'][config]-norm['xm'])/norm['xs']).astype(np.float32)
    with torch.inference_mode():out=model(torch.from_numpy(z)).numpy()
    mean=np.clip(out[:,:3]*norm['ys']+norm['ym'],-norm['clip'],norm['clip']);mean[:,2]=wrap(mean[:,2])
    sigma=np.exp(.5*np.clip(out[:,3:],-8,6))
    return mean,sigma,z


def score_ape(seq,pose):
    return float(np.sqrt(np.mean(np.sum((pose[seq.valid,:2]-seq.gt[seq.valid,:2])**2,axis=1))))


def choose(data,predictions,masks):
    candidates=[]
    for tau in TAUS:
        for alpha in ALPHAS:
            ape=float(np.mean([score_ape(d['seq'],corrected(d['seq'],p[0],float(alpha),'increment',mask,tau))
                for d,p,mask in zip(data,predictions,masks)]))
            candidates.append(dict(alpha=float(alpha),tau_s=tau,validation_macro_ape_m=ape))
    # Candidate order preserves unlimited integration for exact ties.
    return min(candidates,key=lambda r:r['validation_macro_ape_m']),candidates


def fit_gate(data,predictions,norm):
    errors=[];sigma=[];ood=[]
    for d,p in zip(data,predictions):
        y,valid=targets(d['seq'],'increment');sigma.append(p[1][valid].mean(1));ood.append(np.abs(p[2][valid]).max(1))
        errors.append(((y[valid]-p[0][valid])/np.maximum(p[1][valid]*norm['ys'],1e-8))**2)
    return dict(sigma_threshold=float(np.quantile(np.concatenate(sigma),.9)),
        ood_threshold=float(np.quantile(np.concatenate(ood),.99)),
        variance_calibration_scale=float(np.sqrt(np.mean(np.concatenate(errors)))),fit_split='validation')


def masks(d,p,gate):
    ready=d['seq'].ready[1:]
    rule=ready&(np.abs(p[2]).max(1)<=gate['ood_threshold'])
    return dict(none=ready,rule=rule,uncertainty=rule&(p[1].mean(1)<=gate['sigma_threshold']))


def run(args):
    torch.set_num_threads(1);args.output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads(args.manifest.read_text());validate_manifest(manifest)
    rows={r['sequence_id']:r for r in manifest['sequences']}
    plan=dict(train=manifest['train'],validation=manifest['validation'],test=manifest['test'],
        features=['padded44','quality44'],losses=['per_frame_nll','per_frame_nll_plus_path'],seeds=SEEDS,
        horizons_frames=list(HORIZONS),grid_hz=10,window_frames=50,window_stride_frames=20,
        trajectory_weight=1.,epochs=args.epochs,patience=15,alpha_candidates=ALPHAS.tolist(),tau_candidates_s=TAUS,
        parameter_control='44 input channels for both; padded variant appends twelve zeros; same readiness guard',
        quality_arrival_scope='source-causal bag reconstruction; historical scan callback arrivals not known',
        evaluation_status='previously inspected public development sequences, NCLT frontend already adapted',
        reference_in_inference=False,selection_uses_test=False)
    path=args.output/'plan.json'
    if path.exists() and json.loads(path.read_text())!=plan:raise ValueError('Study plan changed')
    dump(path,plan);dump(args.output/'manifest.json',manifest)
    # Training and all deployment selection complete before evaluation GT loads.
    training=[load(rows[n],args.quality_root) for n in manifest['train']]
    validation=[load(rows[n],args.quality_root) for n in manifest['validation']]
    states=[];curves=[]
    for config in plan['features']:
        for with_path in [False,True]:
            for seed in SEEDS:
                name=f'{config}_'+('trajectory' if with_path else 'frame')+f'_seed{seed}'
                directory=args.output/name;directory.mkdir(exist_ok=True)
                if (directory/'model.pt').exists():
                    state=torch.load(directory/'model.pt',map_location='cpu',weights_only=False)
                    states.append((directory,state));print('Existing frozen',name,flush=True);continue
                print('Training',name,flush=True)
                model,norm,info=train(training,validation,config,with_path,seed,args.epochs)
                vp=[prediction(model,norm,d,config) for d in validation];gate=fit_gate(validation,vp,norm)
                vm=[masks(d,p,gate) for d,p in zip(validation,vp)]
                ungated,u_curve=choose(validation,vp,[m['none'] for m in vm])
                safe,s_curve=choose(validation,vp,[m['uncertainty'] for m in vm])
                gate.update(gate_alpha=safe['alpha'],gate_validation_ape_m=safe['validation_macro_ape_m'])
                state=dict(model_state_dict=model.state_dict(),norm={k:torch.from_numpy(np.asarray(v)) for k,v in norm.items()},
                    stream_protocol=PROTOCOL,input_stream_protocol='causal_grid_v1',grid_hz=10.,kind='mlp',target='increment',
                    uncertainty=True,feature_config=config,trajectory_loss=with_path,seed=seed,alpha=ungated['alpha'],
                    integration_tau_s=safe['tau_s'],gate_parameters=gate,ungated_parameters=ungated,manifest=manifest,
                    training_selection=dict(validation_path_loss=info['best_validation_path_loss'],test_used=False))
                torch.save(state,directory/'model.pt');dump(directory/'training.json',info)
                dump(directory/'selection.json',dict(ungated=ungated,safe=safe,gate=gate,test_used=False))
                dump(directory/'validation_curves.json',dict(ungated=u_curve,safe=s_curve))
                states.append((directory,state));print(name,'val_safe',safe,'seconds',round(info['wall_s'],2),flush=True)
    all_selection=[dict(model=d.name,seed=s['seed'],feature_config=s['feature_config'],trajectory_loss=s['trajectory_loss'],
        validation_ape_m=s['gate_parameters']['gate_validation_ape_m'],alpha=s['gate_parameters']['gate_alpha'],
        tau_s=s['integration_tau_s']) for d,s in states]
    deployment=min(all_selection,key=lambda r:(r['validation_ape_m'],r['model']))
    dump(args.output/'deployment_selection.json',dict(selected=deployment,candidates=all_selection,test_used=False))
    # First access to evaluation reference files occurs here.
    evaluation=[load(rows[n],args.quality_root) for n in manifest['test']]
    results=[];verification=[]
    for directory,state in states:
        norm={k:v.numpy() for k,v in state['norm'].items()};model=Regressor(44,'mlp',True)
        model.load_state_dict(state['model_state_dict']);model.eval();config=state['feature_config'];gate=state['gate_parameters']
        for d in evaluation:
            seq=d['seq'];p=prediction(model,norm,d,config);ms=masks(d,p,gate)
            specifications=[('ungated_selected',ms['none'],state['ungated_parameters']['alpha'],state['ungated_parameters']['tau_s']),
                ('none_same_alpha',ms['none'],gate['gate_alpha'],state['integration_tau_s']),
                ('rule_same_alpha',ms['rule'],gate['gate_alpha'],state['integration_tau_s']),
                ('uncertainty_selected',ms['uncertainty'],gate['gate_alpha'],state['integration_tau_s'])]
            for method,mask,alpha,tau in specifications:
                pose=corrected(seq,p[0],alpha,'increment',mask,tau);m=metrics(seq,pose)
                raw_error=np.linalg.norm(seq.cart[:,:2]-seq.gt[:,:2],axis=1)
                error=np.linalg.norm(pose[:,:2]-seq.gt[:,:2],axis=1)
                active=np.r_[False,mask&(alpha!=0)];np.testing.assert_array_equal(pose[~active],seq.cart[~active])
                results.append(dict(model=directory.name,feature_config=config,trajectory_loss=state['trajectory_loss'],seed=state['seed'],
                    sequence=seq.name,scene=seq.scene,method=method,alpha=alpha,tau_s=tau,
                    raw_ape_m=score_ape(seq,seq.cart),ape_m=m['ape_m']['rmse'],rpe_1s_m=m['rpe']['1s']['translation_m']['rmse'],
                    harmful_fraction=float(((error-raw_error)[seq.valid]>.05).mean()),fallback_fraction=float((~active).mean()),
                    alpha_disabled=alpha==0,max_offset_m=float(np.linalg.norm(pose[:,:2]-seq.cart[:,:2],axis=1).max()),
                    test_used_for_selection=False))
                save_tum(directory/f'{seq.name}_{method}.tum',seq,pose)
                if method=='uncertainty_selected':
                    def forward(z):
                        with torch.inference_mode():return model(torch.from_numpy(z[None])).numpy()[0]
                    engine=QualityResidualStream(config,norm,forward,state['alpha'],gate,tau)
                    out=[];fallback=[]
                    for t,c,o,q,a in zip(seq.t,seq.cart,seq.odom,d['quality'],seq.available):
                        v,status=engine.step(t,c,o,q,bool(a));out.append(v);fallback.append(status['fallback'])
                    delta=np.array(out)-pose;delta[:,2]=wrap(delta[:,2])
                    max_error=float(np.abs(delta).max());np.testing.assert_allclose(delta,0,atol=3e-5,rtol=0)
                    np.testing.assert_array_equal(fallback,~active)
                    verification.append(dict(model=directory.name,sequence=seq.name,max_error=max_error,passed=True))
    with (args.output/'results.csv').open('w',newline='') as out:
        writer=csv.DictWriter(out,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)
    dump(args.output/'equivalence.json',verification)
    source_files=['trajectory_quality.py','trajectory_objective.py','tools/train_trajectory_quality.py','mentor_benchmark.py','causal_stream.py','residual_models.py']
    sha={}
    for name in source_files:
        path=Path(__file__).resolve().parents[1]/name;sha[name]=hashlib.sha256(path.read_bytes()).hexdigest()
        (args.output/('source_'+path.name)).write_bytes(path.read_bytes())
    dump(args.output/'provenance.json',dict(source_sha256=sha,models=len(states),result_rows=len(results),equivalence_checks=len(verification)))
    print('Finished',args.output,len(results),'rows',len(verification),'equivalence checks',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--quality-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--epochs',type=int,default=100);run(p.parse_args())
