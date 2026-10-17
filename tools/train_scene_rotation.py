#!/usr/bin/python3
"""Scene-disjoint rotation and runtime-policy training; no external GT selection."""
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
from residual_models import Regressor
from trajectory_quality import QualityResidualStream,PROTOCOL
from trajectory_objective import path_loss
from gated_objective import gated_path_loss
from mentor_benchmark import validate_manifest,targets,corrected,metrics,save_tum
from causal_stream import wrap
from tools.train_trajectory_quality import load,training_norm,windows,prediction,masks,score_ape,dump,ALPHAS,TAUS

SCENES=['home','office','cafe','corridor']
SEEDS=[11,22,33]
LOSSES=['frame','continuous','gated_reset']


def make_folds(manifest):
    folds=[]
    for test in SCENES:
        for val in SCENES:
            if val==test:continue
            fold=copy.deepcopy(manifest)
            fold['train']=[r['sequence_id'] for r in manifest['sequences'] if r['scene_id'] in SCENES and r['scene_id'] not in [val,test]]
            fold['validation']=[r['sequence_id'] for r in manifest['sequences'] if r['scene_id']==val]
            fold['test']=[r['sequence_id'] for r in manifest['sequences'] if r['scene_id']==test]
            fold['isolation_field']='scene_id';fold['external_test']=['nclt-2013-01-10']
            fold['fold_id']=f'test_{test}_val_{val}'
            validate_manifest(fold);folds.append(fold)
    return folds


def policy_gate(train,model,norm):
    sigma=[];ood=[]
    for d in train:
        _,s,z=prediction(model,norm,d,'quality44');ready=d['seq'].ready[1:]
        sigma.append(s[ready].mean(1));ood.append(np.abs(z[ready]).max(1))
    return dict(sigma_threshold=float(np.quantile(np.concatenate(sigma),.9)),
        ood_threshold=float(np.quantile(np.concatenate(ood),.99)),fit_split='training_warmup_input_only',
        warmup_epochs=5,thresholds_frozen_before_validation=True)


def draw_batch(pools,rng):
    indices=rng.integers(len(pools),size=16)
    samples=[(pools[j],rng.integers(len(pools[j][0]))) for j in indices]
    return [torch.stack([p[k][i] for p,i in samples]) for k in range(4)]


def objective(out,x,y,ci,gi,norm,gate,loss,alpha=1.,tau_s=30.,reject=None):
    mean,logvar=out[...,:3],out[...,3:].clamp(-8,6)
    nll=(.5*((mean-y)**2*torch.exp(-logvar)+logvar)).mean()
    residual=torch.maximum(torch.minimum(mean*norm['ys']+norm['ym'],norm['clip']),-norm['clip'])
    scale=torch.maximum(norm['ys'],torch.tensor([.005,.005,.001]))
    accepted=((torch.exp(.5*logvar.detach()).mean(-1)<=gate['sigma_threshold'])&
        (x.abs().amax(-1)<=gate['ood_threshold']))
    if reject is not None:accepted=accepted&~reject
    if loss=='continuous':extra=path_loss(residual,ci,gi,scale)
    elif loss=='gated_reset':extra=gated_path_loss(residual,ci,gi,accepted,scale,alpha,tau_s)
    else:extra=nll*0
    return nll+extra,accepted


def macro_score(data,poses,relative=False):
    scenes=sorted({d['seq'].scene for d in data});values=[]
    for scene in scenes:
        group=[]
        for d,pose in zip(data,poses):
            if d['seq'].scene==scene:
                score=score_ape(d['seq'],pose)
                if relative:score/=max(score_ape(d['seq'],d['seq'].cart),1e-8)
                group.append(score)
        values.append(float(np.mean(group)))
    return float(np.mean(values))


def choose(data,predictions,accepted,relative=False):
    curves=[]
    for tau in TAUS:
        for alpha in ALPHAS:
            poses=[corrected(d['seq'],p[0],float(alpha),'increment',a,tau) for d,p,a in zip(data,predictions,accepted)]
            curves.append(dict(alpha=float(alpha),tau_s=tau,validation_score=macro_score(data,poses,relative)))
    return min(curves,key=lambda r:r['validation_score']),curves


def selected_model(candidates,loss=None):
    if loss is not None:candidates=[r for r in candidates if r['loss']==loss]
    return min(candidates,key=lambda r:(r['validation_score'],r['model']))


def train_fold(fold,root,quality_root,epochs):
    rows={r['sequence_id']:r for r in fold['sequences']}
    train=[load(rows[n],quality_root) for n in fold['train']]
    val=[load(rows[n],quality_root) for n in fold['validation']]
    norm=training_norm(train,'quality44');pools,counts=windows(train,'quality44',norm)
    tn={k:torch.tensor(v,dtype=torch.float32) for k,v in norm.items()}
    steps=max(1,math.ceil(sum(len(p[0]) for p in pools)/16))
    states=[];selection=[]
    for seed in SEEDS:
        torch.manual_seed(seed);rng=np.random.default_rng(seed)
        warmup=Regressor(44,'mlp',True);opt=torch.optim.Adam(warmup.parameters(),lr=.001,weight_decay=1e-5)
        for _ in range(5):
            for _ in range(steps):
                x,y,ci,gi=draw_batch(pools,rng)
                opt.zero_grad();out=warmup(x.reshape(-1,44)).reshape(16,50,6)
                mean,lv=out[...,:3],out[...,3:].clamp(-8,6)
                nll=(.5*((mean-y)**2*torch.exp(-lv)+lv)).mean()
                nll.backward();torch.nn.utils.clip_grad_norm_(warmup.parameters(),5);opt.step()
        warmup.eval();gate=policy_gate(train,warmup,norm)
        warmup_hash=hashlib.sha256(b''.join(v.numpy().tobytes() for v in warmup.state_dict().values())).hexdigest()
        rng_state=copy.deepcopy(rng.bit_generator.state)
        for loss in LOSSES:
            name=f'quality44_{loss}_seed{seed}';directory=root/name;directory.mkdir(exist_ok=True)
            if (directory/'model.pt').exists():
                state=torch.load(directory/'model.pt',map_location='cpu',weights_only=False)
                if state['training_selection']['warmup_sha256']!=warmup_hash:raise ValueError('Warmup differs from frozen model')
                choice=json.loads((directory/'selection.json').read_text())
                states.append((directory,state));selection.append(choice['candidate']);print('Existing frozen',fold['fold_id'],name,flush=True);continue
            model=copy.deepcopy(warmup);optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=1e-5)
            optimizer.load_state_dict(copy.deepcopy(opt.state_dict()));rng=np.random.default_rng();rng.bit_generator.state=copy.deepcopy(rng_state)
            policy_rng=np.random.default_rng(seed+7919)
            best=math.inf;best_state=None;stale=0;history=[];started=time.perf_counter()
            for epoch in range(5,epochs):
                model.train();values=[];acceptance=[]
                for _ in range(steps):
                    x,y,ci,gi=draw_batch(pools,rng)
                    # Same sampled batches for all losses; independent policy RNG.
                    alpha=float(policy_rng.choice(ALPHAS[1:]));tau=TAUS[int(policy_rng.integers(len(TAUS)))]
                    reject=np.zeros((16,50),bool)
                    for i in range(16):
                        if policy_rng.random()<.2:
                            start=int(policy_rng.integers(5,45));reject[i,start:start+2]=True
                    optimizer.zero_grad();out=model(x.reshape(-1,44)).reshape(16,50,6)
                    total,mask=objective(out,x,y,ci,gi,tn,gate,loss,alpha,tau,torch.from_numpy(reject))
                    if not torch.isfinite(total):raise ValueError('Nonfinite objective')
                    total.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
                    values.append(float(total.detach()));acceptance.append(float(mask.float().mean()))
                model.eval();vp=[prediction(model,norm,d,'quality44') for d in val]
                vm=[masks(d,p,gate)['uncertainty'] for d,p in zip(val,vp)]
                poses=[corrected(d['seq'],p[0],1.,'increment',m,30.) for d,p,m in zip(val,vp,vm)]
                score=macro_score(val,poses)
                history.append(dict(epoch=epoch+1,training_objective=float(np.mean(values)),training_policy_acceptance=float(np.mean(acceptance)),
                    validation_fixed_policy_ape_m=score,validation_acceptance=float(np.mean([m.mean() for m in vm]))))
                if score<best-1e-6:best=score;best_state=copy.deepcopy(model.state_dict());best_epoch=epoch+1;stale=0
                else:stale+=1
                if stale>=15:break
            model.load_state_dict(best_state);model.eval()
            vp=[prediction(model,norm,d,'quality44') for d in val];vm=[masks(d,p,gate) for d,p in zip(val,vp)]
            safe,curve=choose(val,vp,[m['uncertainty'] for m in vm])
            g=dict(gate,gate_alpha=safe['alpha'],gate_validation_ape_m=safe['validation_score'])
            state=dict(model_state_dict=model.state_dict(),norm={k:torch.from_numpy(np.asarray(v)) for k,v in norm.items()},
                stream_protocol=PROTOCOL,input_stream_protocol='causal_grid_v1',grid_hz=10.,kind='mlp',target='increment',
                uncertainty=True,feature_config='quality44',trajectory_loss=loss!='frame',loss=loss,seed=seed,
                alpha=safe['alpha'],integration_tau_s=safe['tau_s'],gate_parameters=g,manifest=fold,
                training_selection=dict(test_used=False,warmup_sha256=warmup_hash,training_thresholds=gate,
                    common_early_stop='complete validation APE, same frozen gate, alpha=1, tau=30s'))
            torch.save(state,directory/'model.pt')
            info=dict(loss=loss,seed=seed,best_epoch=best_epoch,epochs=len(history)+5,warmup_epochs=5,
                warmup_sha256=warmup_hash,wall_s=time.perf_counter()-started,train_windows=counts,history=history,
                parameter_count=sum(p.numel() for p in model.parameters()))
            dump(directory/'training.json',info)
            candidate=dict(model=name,seed=seed,loss=loss,validation_score=safe['validation_score'],alpha=safe['alpha'],tau_s=safe['tau_s'])
            dump(directory/'selection.json',dict(candidate=candidate,gate=g,test_used=False));dump(directory/'validation_curves.json',curve)
            states.append((directory,state));selection.append(candidate)
            print(fold['fold_id'],name,'val',round(safe['validation_score'],6),'alpha',safe['alpha'],'tau',safe['tau_s'],
                'seconds',round(info['wall_s'],2),flush=True)
    dump(root/'deployment_selection.json',dict(selected=selected_model(selection),candidates=selection,
        selected_by_loss={loss:selected_model(selection,loss) for loss in LOSSES},test_used=False))
    return states


def score_state(directory,state,data,protocol='single_scene_validation',selection_role='candidate',override=None):
    norm={k:v.numpy() for k,v in state['norm'].items()};model=Regressor(44,'mlp',True)
    model.load_state_dict(state['model_state_dict']);model.eval()
    gate=state['gate_parameters'] if override is None else override['gate']
    tau=state['integration_tau_s'] if override is None else override['choice']['tau_s']
    result=[];verification=[]
    for d in data:
        seq=d['seq'];p=prediction(model,norm,d,'quality44');ms=masks(d,p,gate)
        raw_metrics=metrics(seq,seq.cart);raw=score_ape(seq,seq.cart);rawerr=np.linalg.norm(seq.cart[:,:2]-seq.gt[:,:2],axis=1)
        for method,mask in ms.items():
            pose=corrected(seq,p[0],gate['gate_alpha'],'increment',mask,tau);m=metrics(seq,pose)
            active=np.r_[False,mask&(gate['gate_alpha']!=0)]
            np.testing.assert_array_equal(pose[~active],seq.cart[~active])
            err=np.linalg.norm(pose[:,:2]-seq.gt[:,:2],axis=1)
            result.append(dict(fold=state['manifest']['fold_id'],model=directory.name,loss=state['loss'],seed=state['seed'],
                sequence=seq.name,scene=seq.scene,gate=method,validation_protocol=protocol,selection_role=selection_role,
                alpha=gate['gate_alpha'],tau_s=tau,raw_ape_m=raw,ape_m=m['ape_m']['rmse'],ape_ratio=m['ape_m']['rmse']/raw,
                raw_rpe_1s_m=raw_metrics['rpe']['1s']['translation_m']['rmse'],rpe_1s_m=m['rpe']['1s']['translation_m']['rmse'],
                yaw_ape_deg=m['yaw_ape_deg']['rmse'],rpe_1s_deg=m['rpe']['1s']['rotation_deg']['rmse'],
                harmful_fraction=float(((err-rawerr)[seq.valid]>.05).mean()),fallback_fraction=float((~active).mean()),
                alpha_disabled=gate['gate_alpha']==0,max_offset_m=float(np.linalg.norm(pose[:,:2]-seq.cart[:,:2],axis=1).max()),
                selection_uses_outer_or_nclt=False))
            if method=='uncertainty' and selection_role=='validation_best':
                def forward(z):
                    with torch.inference_mode():return model(torch.from_numpy(z[None])).numpy()[0]
                engine=QualityResidualStream('quality44',norm,forward,state['alpha'],gate,tau)
                output=[];fallback=[]
                for t,c,o,q,a in zip(seq.t,seq.cart,seq.odom,d['quality'],seq.available):
                    v,status=engine.step(t,c,o,q,bool(a));output.append(v);fallback.append(status['fallback'])
                delta=np.array(output)-pose;delta[:,2]=wrap(delta[:,2])
                np.testing.assert_allclose(delta,0,atol=3e-5,rtol=0);np.testing.assert_array_equal(fallback,~active)
                verification.append(dict(fold=state['manifest']['fold_id'],model=directory.name,protocol=protocol,
                    sequence=seq.name,max_pose_error=float(np.abs(delta).max()),exact_fallback=True))
                save_tum(directory/f'{protocol}_{seq.name}.tum',seq,pose)
    return result,verification


def balanced_quantile(groups,quantile):
    values=[];weights=[]
    for group in groups:
        for a in group:
            values.extend(a);weights.extend(np.full(len(a),1/(len(groups)*len(group)*len(a))))
    values=np.asarray(values);weights=np.asarray(weights);order=np.argsort(values)
    return float(values[order][np.searchsorted(np.cumsum(weights[order]),quantile,side='left')])


def pooled_selection(folds,quality_root,output):
    # Same Home+Office trained weights; change only calibration/alpha/tau/ranking.
    root=output/'test_corridor_val_cafe';selection=json.loads((root/'deployment_selection.json').read_text())
    states=[(root/r['model'],torch.load(root/r['model']/'model.pt',map_location='cpu',weights_only=False)) for r in selection['candidates']]
    rowmap={r['sequence_id']:r for r in folds[0]['sequences']}
    data=[load(r,quality_root) for r in rowmap.values() if r['scene_id'] in ['cafe','corridor']]
    candidates=[];overrides={}
    for directory,state in states:
        norm={k:v.numpy() for k,v in state['norm'].items()};model=Regressor(44,'mlp',True)
        model.load_state_dict(state['model_state_dict']);model.eval();ps=[prediction(model,norm,d,'quality44') for d in data]
        sg=[];zg=[]
        for scene in ['cafe','corridor']:
            ss=[];zz=[]
            for d,p in zip(data,ps):
                if d['seq'].scene==scene:
                    ready=d['seq'].ready[1:];ss.append(p[1][ready].mean(1));zz.append(np.abs(p[2][ready]).max(1))
            sg.append(ss);zg.append(zz)
        pooled=dict(sigma_threshold=balanced_quantile(sg,.9),ood_threshold=balanced_quantile(zg,.99),
            fit_split='cafe_corridor_validation',weights='equal scenes, equal sequences within scene, equal ready frames within sequence')
        for relative in [False,True]:
            policy='pooled_absolute' if not relative else 'pooled_relative'
            for gate_kind,g in [('training_gate',state['gate_parameters']),('pooled_gate',pooled)]:
                protocol=policy+'_'+gate_kind
                accepted=[masks(d,p,g)['uncertainty'] for d,p in zip(data,ps)]
                choice,curve=choose(data,ps,accepted,relative)
                gate=dict(g,gate_alpha=choice['alpha'])
                candidate=dict(protocol=protocol,model=directory.name,loss=state['loss'],seed=state['seed'],**choice)
                candidates.append(candidate);overrides[(protocol,directory.name)]=dict(choice=choice,gate=gate)
    selected={}
    for protocol in sorted({r['protocol'] for r in candidates}):
        best=min([r for r in candidates if r['protocol']==protocol],key=lambda r:(r['validation_score'],r['model']))
        selected[protocol]=best
    dump(output/'pooled_selection.json',dict(selected=selected,candidates=candidates,
        weight_scope='same nine checkpoints, trained Home/Office with Cafe-only early stopping; no retraining',
        validation_scenes=['cafe','corridor'],external_test=['nclt-2013-01-10'],nclt_used_for_selection=False))
    dump(output/'pooled_calibration.json',{protocol+'/'+name:dict(value,
        checkpoint_sha256=hashlib.sha256((root/name/'model.pt').read_bytes()).hexdigest())
        for (protocol,name),value in overrides.items()})
    return states,overrides,selected


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def run(args):
    torch.set_num_threads(1);args.output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads(args.manifest.read_text());folds=make_folds(manifest)
    plan=dict(folds=[{k:f[k] for k in ['fold_id','train','validation','test','external_test']} for f in folds],
        feature='quality44',losses=LOSSES,seeds=SEEDS,parameter_count=9318,epochs=args.epochs,warmup_epochs=5,
        warmup='identical model/optimizer/batch order per seed before loss branches',
        gate_thresholds='training input-only warmup P90 sigma/P99 max-z, frozen for training and deployment',
        common_early_stop='complete validation APE under fixed alpha=1/tau=30s and frozen gate',
        horizon_frames=[10,30,50],window_frames=50,stride_frames=20,gated_training_alphas=ALPHAS[1:].tolist(),
        gated_training_tau_candidates_s=TAUS,rejection_augmentation='20% of windows receive one two-frame confidence rejection block; features unchanged',
        runtime_match='same gate comparisons, clip, SE2 recursion, tau damping and exact raw reset; each training window starts at raw origin',
        limitations=['all public development scenes previously inspected','rotating validation changes training scenes too',
            'short-window objective excludes pre-window accumulated offset','quality is source-causal reconstructed geometry, not match score',
            'no new independently collected data or actual edge hardware'],
        external_nclt_parameters_frozen_before_reference_load=True)
    path=args.output/'plan.json'
    if path.exists() and json.loads(path.read_text())!=plan:raise ValueError('Rotation plan changed')
    dump(path,plan)
    all_states=[]
    for fold in folds:
        root=args.output/fold['fold_id'];root.mkdir(exist_ok=True);dump(root/'manifest.json',fold)
        all_states.extend(train_fold(fold,root,args.quality_root,args.epochs))
    pool_states,overrides,pool_selected=pooled_selection(folds,args.quality_root,args.output)
    # All model and selection artifacts above freeze before first NCLT reference access.
    dump(args.output/'selection_freeze.json',dict(model_count=len(all_states),fold_selections=[f['fold_id'] for f in folds],
        pooled_protocols=list(pool_selected),nclt_reference_loaded=False))
    rowmap={r['sequence_id']:r for r in manifest['sequences']}
    external=[load(rowmap['nclt-2013-01-10'],args.quality_root)]
    rows=[];checks=[];held_data={}
    for directory,state in all_states:
        fold=state['manifest'];fold_id=fold['fold_id']
        if fold_id not in held_data:held_data[fold_id]=[load(rowmap[n],args.quality_root) for n in fold['test']]
        selection=json.loads((directory.parent/'deployment_selection.json').read_text())
        role='validation_best' if directory.name==selection['selected']['model'] else 'candidate'
        rr,cc=score_state(directory,state,held_data[fold_id],selection_role=role);rows+=rr;checks+=cc
        rr,cc=score_state(directory,state,external,selection_role=role);rows+=rr;checks+=cc
    for directory,state in pool_states:
        for protocol,best in pool_selected.items():
            role='validation_best' if directory.name==best['model'] else 'candidate'
            rr,cc=score_state(directory,state,external,protocol,role,overrides[(protocol,directory.name)]);rows+=rr;checks+=cc
    write_csv(args.output/'results.csv',rows);dump(args.output/'equivalence.json',checks)
    sources=['gated_objective.py','trajectory_objective.py','trajectory_quality.py','causal_stream.py','mentor_benchmark.py',
        'residual_models.py','tools/train_scene_rotation.py','tools/train_trajectory_quality.py']
    sha={}
    for name in sources:
        path=Path(__file__).resolve().parents[1]/name;sha[name]=hashlib.sha256(path.read_bytes()).hexdigest()
        (args.output/('source_'+path.name)).write_bytes(path.read_bytes())
    dump(args.output/'provenance.json',dict(source_sha256=sha,models=len(all_states),rows=len(rows),stream_checks=len(checks)))
    print('Finished',args.output,len(all_states),'models',len(rows),'rows',len(checks),'stream checks',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--quality-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--epochs',type=int,default=100);run(p.parse_args())
