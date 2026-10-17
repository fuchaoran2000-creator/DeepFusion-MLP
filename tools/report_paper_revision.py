#!/usr/bin/python3
"""Build manuscript tables from saved evidence, never manually enter scores."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/paper_revision'


def csvrows(path):
    with path.open() as f:return list(csv.DictReader(f))


def aggregate_ablation(root,scene):
    rows=csvrows(root/'results.csv');result=[]
    for config in dict.fromkeys(r['config'] for r in rows):
        selected=[r for r in rows if r['config']==config and r['scene']==scene]
        means=[np.mean([float(r['ape_m']) for r in selected if r['seed']==str(s)]) for s in [11,22,33]]
        result.append(dict(config=config,input_dim=int(selected[0]['input_dim']),parameters=int(selected[0]['parameters']),
                           ape_m=float(np.mean(means)),seed_sample_sd_m=float(np.std(means,ddof=1)),
                           rpe_1s_m=float(np.mean([float(r['rpe_1s_m']) for r in selected])),
                           disabled_seeds=sum(float(next(r for r in selected if r['seed']==str(s))['alpha'])==0 for s in [11,22,33])))
    return result


def summarize_metrics(path,label):
    d=json.loads(path.read_text());families={f['name']:f for f in d['before_shutdown']['families']}
    counts=[]
    for name in ['mapping_constraints_constraint_builder_2d_constraints','mapping_2d_pose_graph_constraints']:
        for metric in families[name]['metrics']:
            counts.append(dict(family=name,labels=metric['labels'],count=metric['value']))
    snapshots=[]
    key='mapping_2d_local_trajectory_builder_latency'
    for sample in d['samples']:
        if key in sample['gauges'] and sample['gauges'][key]>0:snapshots.append(sample['gauges'][key]*1000)
    # Prometheus buckets in this capture are disjoint, not cumulative.
    histograms={}
    for name in ['mapping_2d_local_trajectory_builder_costs','mapping_2d_local_trajectory_builder_residuals']:
        histograms[name]=families[name]['metrics']
    return dict(label=label,source=str(path.relative_to(ROOT)),counts=counts,
                last_local_update_snapshot_count=len(snapshots),
                snapshot_latency_p99_ms=float(np.quantile(snapshots,.99)) if snapshots else None,
                latency_scope='One-Hz samples of latest local SLAM update, not per-frame latency.',histograms=histograms)


def run():
    OUT.mkdir(exist_ok=True)
    base=ROOT/'results/mentor_benchmark'
    home=aggregate_ablation(base/'home_causal_feature_ablation_v1','home')
    cross=aggregate_ablation(base/'cross_causal_feature_ablation_v1','corridor')
    causal=csvrows(base/'home_causal_v1/report/causal_methods.csv')
    resources=json.loads((OUT/'causal_resources.json').read_text())
    unified=[]
    for r in causal:
        rr=[x for x in resources if x['method']==r['method']]
        if not rr:continue
        sd=next((x['seed_sample_sd_m'] for x in home if r['method']=='mlp_'+x['config']+'_pose'),float(r['seed_macro_std_m'])*np.sqrt(1.5))
        unified.append(dict(method=r['method'],ape_m=float(r['ape_m']),seed_sample_sd_m=sd,
            rpe_1s_m=float(r['rpe_1s_m']),rpe_1s_deg=float(r['rpe_1s_deg']),
            postprocess_ms_per_frame=float(np.mean([x['mean_ms_per_frame'] for x in rr])),
            worker_peak_rss_mib=max(x['fresh_process_peak_rss_mib'] for x in rr),
            time_scope='Batch postprocessing, loaded causal inputs, seed11 when applicable; excludes source SLAM and loading.',
            memory_scope='Whole independent Python/Torch/Scipy worker, not incremental model memory.',
            accuracy_scope='Three held-out Home sequences; three seeds for learned models.'))
    paired=csvrows(base/'home_causal_v1/report/live_paired.csv');live=[]
    for model in ['unified20','unified32']:
        r=[x for x in paired if x['model']==model]
        live.append(dict(model=model,**{k:float(np.mean([float(x[k]) for x in r])) for k in
            ['raw_ape_m','shared_map_corrected_ape_m','raw_rpe_1s_m','corrected_rpe_1s_m','fallback_fraction']},
            callback_p99_ms_max=max(float(x['callback_p99_ms']) for x in r)))
    slam=[];sr=csvrows(OUT/'online_unified_common_time.csv')
    for scene in ['home','corridor']:
        for method in (['cartographer_noimu','cartographer_imu','2dliw','mlp20_confidence'] if scene=='home' else ['cartographer_noimu','cartographer_imu','2dliw']):
            r=[x for x in sr if x['scene']==scene and x['method']==method]
            slam.append(dict(scene=scene,method=method,**{key:float(np.mean([float(x[key]) for x in r])) for key in
                ['ape_rmse_m','rpe_1s_m','replay_wall_s']},peak_slam_rss_mib=max(float(x['slam_peak_rss_mib']) for x in r),
                common_coverage_min=min(float(x['common_coverage']) for x in r),
                common_coverage_max=max(float(x['common_coverage']) for x in r),
                input_budget=r[0]['input_budget'],
                scope='Current selected Home correction and independently replayed SLAM alternatives on shared valid input times; different sensor budgets. Replay wall time includes rate pacing.',
                mask_note='New common mask over the stated methods only; raw/corrected Home share one source run, with paired map APE also retained.'))
    frontend=[]
    for path,label in [
        ('data/prepared/home1-3/cartographer_noimu_metrics_metrics_baseline.json','clean baseline'),
        ('data/prepared/home1-3/cartographer_noimu_metrics_quality_path_v1.json','clean with quality correction'),
        ('data/prepared/home1-3_raw_fault_v2/cartographer_noimu_metrics_raw_fault_v2.json','raw input faults with quality correction')]:
        frontend.append(summarize_metrics(ROOT/path,label))
    result=dict(causal_ablation_home=home,causal_ablation_corridor=cross,
                unified_postprocessing=unified,live_paired=live,slam_comparison=slam,
                frontend_metrics=frontend,scene_rotation=json.loads((base/'scene_rotation_gated_v1/report/summary.json').read_text()),
                raw_fault=json.loads((OUT/'raw_fault_evaluation.json').read_text()),
                raw_timestamp_failure=json.loads((OUT/'raw_timestamp_failure.json').read_text()),
                cpu=json.loads((OUT/'cpu_frozen_latency_serial.json').read_text()),
                gpu=json.loads((OUT/'gpu_frozen_latency_serial.json').read_text()))
    cpu_path=OUT/'slam_cpu_computation.json'
    if cpu_path.exists():
        cpu_rows=json.loads(cpu_path.read_text())
        for r in cpu_rows:
            actual=json.loads((ROOT/r['source']).read_text())
            r.update(shutdown_returncode=actual['shutdown_returncode'],shutdown_success=actual['shutdown_success'])
        cpu_path.write_text(json.dumps(cpu_rows,indent=2)+'\n')
        result['slam_cpu_computation']=cpu_rows
    for seq in ['nclt-2012-04-29','nclt-2013-04-05']:
        path=base/'final_public_holdout_v1'/(seq+'.json')
        if path.exists():result.setdefault('frozen_holdout',[]).append(json.loads(path.read_text()))
    (OUT/'evidence_summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    for name,rows in [('causal_ablation_home',home),('causal_ablation_corridor',cross),('unified_postprocessing',unified),('slam_comparison',slam)]:
        with (OUT/(name+'.csv')).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    for study in ['home_causal_feature_ablation_v1','cross_causal_feature_ablation_v1']:
        provenance=json.loads((base/study/'provenance.json').read_text())
        for source in ['tools/train_causal_feature_ablation.py','mentor_benchmark.py','causal_stream.py']:
            snapshot=base/study/('source_'+Path(source).name)
            if not snapshot.exists():snapshot.write_bytes((ROOT/source).read_bytes())
            assert hashlib.sha256(snapshot.read_bytes()).hexdigest()==provenance['source_sha256'][source]
    print('Updated evidence summary; new frozen sessions:',len(result.get('frozen_holdout',[])))


if __name__=='__main__':run()
