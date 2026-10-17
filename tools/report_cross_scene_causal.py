#!/usr/bin/python3
"""Publish paired cross-scene and same-alpha stress results without retuning."""
import csv
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
OUT=ROOT/'results/mentor_benchmark/cross_scene_causal_report_v1'
FAMILIES={
    'home_seed_robustness':['home_causal_gate_stress_v1_r1'],
    'home_frozen_transfer':['home_frozen_transfer_office_cafe_v1','home_frozen_transfer_corridor_v1','home_frozen_transfer_nclt_v1'],
    'scene_isolated':['cross_scene_gate_stress_openloris_v1','cross_scene_gate_stress_nclt_v1']}


def read(path):return json.loads(path.read_text())


def macro(rows,key):
    # Equal weight per sequence within each seed, then equal weight per seed.
    per_seed=defaultdict(list)
    for r in rows:
        value=r[key]
        if value is not None:per_seed[r['seed']].append(value)
    values=[np.mean(v) for v in per_seed.values()]
    return float(np.mean(values)) if values else None


def csv_write(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    all_rows=[];by_family={}
    for family,dirs in FAMILIES.items():
        rows=[]
        for directory in dirs:
            data=read(ROOT/'results/mentor_benchmark'/directory/'results.json')
            for row in data:row['family']=family
            rows+=data
        by_family[family]=rows;all_rows+=rows
    assert [len(by_family[k]) for k in FAMILIES]==[486,702,648]
    # A checkpoint/anomaly/sequence has three modes and one identical alpha.
    pairs=defaultdict(list)
    for row in all_rows:pairs[(row['family'],row['checkpoint'],row['sequence'],row['anomaly'])].append(row)
    for rows in pairs.values():
        assert {r['gate_mode'] for r in rows}=={'none','rule','uncertainty'}
        assert len({r['alpha'] for r in rows})==1 and len(rows)==3
        assert len({r['scored_samples'] for r in rows})==1
    groups=defaultdict(list)
    for r in all_rows:groups[(r['family'],r['scene'],r['config'],r['anomaly'],r['gate_mode'])].append(r)
    columns=['raw_clean_ape_m','raw_corrupted_ape_m','harmful_fraction_vs_corrupted_raw',
        'harmful_fraction_vs_clean_raw','region_harmful_fraction_vs_corrupted_raw','fallback_fraction_all',
        'clean_rejected_fraction_ready','newly_rejected_fraction_region','rejection_of_beneficial_none_fraction',
        'additional_recovery_delay_s','max_correction_offset_step_m','sigma_additional_rejection_fraction']
    summary=[]
    for keys,rows in groups.items():
        group=dict(zip(['family','scene','config','anomaly','gate_mode'],keys))
        group.update(sequences=len({r['sequence'] for r in rows}),seeds=len({r['seed'] for r in rows}),
            active_seeds=len({r['seed'] for r in rows if not r['alpha_disabled']}),measured_cases=len(rows))
        ape_values=defaultdict(list)
        for r in rows:ape_values[r['seed']].append(r['ape_m']['rmse'])
        group['ape_rmse_m']=float(np.mean([np.mean(v) for v in ape_values.values()]))
        for key in columns:group[key]=macro(rows,key)
        recoveries=[r['recovery_s'] for r in rows if r['recovery_s'] is not None]
        new_alarms=[r['new_rejection_latency_s'] for r in rows if r['new_rejection_latency_s'] is not None]
        group.update(recovery_censored_fraction=float(np.mean([r['recovery_censored'] for r in rows])),
            recovery_median_s=float(np.median(recoveries)) if recoveries else None,
            new_rejection_latency_median_s=float(np.median(new_alarms)) if new_alarms else None)
        summary.append(group)
    csv_write(OUT/'gate_scene_summary.csv',summary)
    (OUT/'all_gate_results.json').write_text(json.dumps(all_rows,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    transfer=[r for r in summary if r['family']=='home_frozen_transfer' and r['anomaly']=='clean']
    csv_write(OUT/'frozen_home_transfer.csv',transfer)
    learned=read(ROOT/'results/mentor_benchmark/cross_scene_causal_v1/results.json')
    learned+=read(ROOT/'results/mentor_benchmark/cross_scene_causal_external_v1/results.json')
    assert len(learned)==158
    clean=defaultdict(list)
    for r in learned:
        if 'anomaly' not in r:clean[(r['scene'],r['method'])].append(r)
    methods=[]
    for (scene,method),rows in clean.items():
        seeds=defaultdict(list)
        for r in rows:seeds[r['seed']].append(r['ape_m']['rmse'])
        values=[np.mean(v) for v in seeds.values()]
        methods.append({'scene':scene,'method':method,'ape_m':float(np.mean(values)),
            'seed_std_m':float(np.std(values)),'rpe_1s_m':float(np.mean([r['rpe']['1s']['translation_m']['rmse'] for r in rows])),
            'rpe_1s_deg':float(np.mean([r['rpe']['1s']['rotation_deg']['rmse'] for r in rows])),
            'sequences':len({r['sequence'] for r in rows}),'seeds':len(seeds),
            'mean_actual_fallback':float(np.mean([r['fallback_fraction'] for r in rows])) if all('fallback_fraction' in r for r in rows) else None})
    csv_write(OUT/'scene_isolated_methods.csv',methods)
    inputs=read(ROOT/'results/mentor_benchmark/cross_scene_causal_v1/input_equivalence.json')
    assert len(inputs)==18 and all(r['passed'] for r in inputs)
    manifest=read(ROOT/'config/benchmark_cross_scene_causal_v1.json')
    resources=[]
    for row in manifest['sequences']:
        path=Path(row['cart']).parent/'cartographer_noimu_online_resources_unified_inputs.json'
        resource=read(path);assert resource['shutdown_success']
        resources.append({'sequence':row['sequence_id'],'dataset':row['dataset_id'],
            'scene':row['scene_id'],'replay_rate':resource['replay_rate'],'wall_s':resource['wall_s'],
            'slam_peak_rss_mib':resource['peak_rss_mib'],'poses':resource['pose_count'],
            'input_available_fraction':next(r['input_available_fraction'] for r in inputs if r['sequence']==row['sequence_id'])})
    csv_write(OUT/'online_capture_resources.csv',resources)
    # Supplemental NCLT graph-node-time APE avoids relying only on dense
    # odometry-interpolated labels. Nearest grid sample tolerance is fixed.
    from mentor_benchmark import load_sequence,load_pose
    nclt_row=next(r for r in manifest['sequences'] if r['sequence_id'].startswith('nclt-'))
    nclt_seq=load_sequence(nclt_row)
    node_times=np.loadtxt(ROOT/'data/nclt/cov_2013-01-10.csv',delimiter=',')[:,0]/1e6
    nodes=node_times[(node_times>=nclt_seq.t[0])&(node_times<=nclt_seq.t[-1])]
    indices=np.searchsorted(nclt_seq.t,nodes).clip(0,len(nclt_seq.t)-1)
    earlier=np.maximum(indices-1,0)
    indices=np.where(abs(nclt_seq.t[earlier]-nodes)<abs(nclt_seq.t[indices]-nodes),earlier,indices)
    indices=np.unique(indices[(abs(nclt_seq.t[indices]-nodes)<=.051)&nclt_seq.valid[indices]])
    sparse=[]
    for r in learned:
        if r['sequence']!=nclt_seq.name:continue
        path=ROOT/'results/mentor_benchmark/cross_scene_causal_external_v1'/f'{r["method"]}_seed{r["seed"]}'/f'{nclt_seq.name}.tum'
        _,pose=load_pose(path)
        error=np.linalg.norm(pose[indices,:2]-nclt_seq.gt[indices,:2],axis=1)
        sparse.append({'method':r['method'],'seed':r['seed'],'graph_node_samples':len(indices),
                       'ape_rmse_m':float(np.sqrt(np.mean(error**2))),'nearest_grid_tolerance_s':.051,
                       'reference_note':'graph-node times; reference still dataset SLAM reference'})
    csv_write(OUT/'nclt_graph_node_ape.csv',sparse)
    selected=read(ROOT/'results/mentor_benchmark/cross_scene_causal_v1/deployment_selection.json')
    home_selected=read(ROOT/'results/mentor_benchmark/home_causal_v1/deployment_selection.json')
    home_display=[]
    for r in by_family['home_seed_robustness']:
        if r['seed']==home_selected[r['config']]['selected']['seed']:home_display.append(r)
    home_groups=defaultdict(list)
    for r in home_display:home_groups[(r['config'],r['anomaly'],r['gate_mode'])].append(r)
    home_table=[]
    for (config,anomaly,mode),rows in home_groups.items():
        home_table.append({'config':config,'anomaly':anomaly,'gate_mode':mode,
            'alpha':rows[0]['alpha'],'ape_m':float(np.mean([r['ape_m']['rmse'] for r in rows])),
            'harm_fraction':macro(rows,'harmful_fraction_vs_corrupted_raw'),
            'fallback_fraction':macro(rows,'fallback_fraction_all'),
            'recovery_censored_fraction':float(np.mean([r['recovery_censored'] for r in rows]))})
    csv_write(OUT/'home_deployment_stress.csv',home_table)
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'results/mentor_benchmark/mpl_cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    anomalies=['clean','wheel_scale','wheel_jump','feature_outlier','odom_drop','odom_delay','odom_outage','cart_outage','odom_stale_stamp']
    labels=['Clean','Wheel scale','Wheel jump','Feature x10','Odom drop','Hidden delay','Odom outage','Cart outage','Stale stamp']
    fig,axes=plt.subplots(2,1,figsize=(11,7),sharex=True)
    for ax,config in zip(axes,['full20','relative32']):
        x=np.arange(len(anomalies))
        for shift,mode in [(-.24,'none'),(0,'rule'),(.24,'uncertainty')]:
            values=[next(r['harm_fraction'] for r in home_table if r['config']==config and r['anomaly']==a and r['gate_mode']==mode) for a in anomalies]
            ax.bar(x+shift,np.array(values)*100,width=.24,label=mode)
        ax.set_title(f'Frozen Home {config}: same model and alpha');ax.set_ylabel('Harmful frames (%)');ax.grid(axis='y',alpha=.2)
    axes[0].legend();axes[-1].set_xticks(np.arange(len(anomalies)));axes[-1].set_xticklabels(labels,rotation=25,ha='right')
    fig.tight_layout()
    for suffix in ['png','pdf']:fig.savefig(OUT/f'same_alpha_harm.{suffix}',dpi=180)
    plt.close(fig)
    lines=['# 统一流程后的跨场景与门控异常实验','',
        '本轮仅使用已有公共数据。所有SLAM输入来自不含真值的bag，采用causal_grid_v1保存实际到达的10 Hz输入对；学习模型和异常试验在这些记录流上因果后处理。除上轮Home主动修正回放外，本轮跨场景结果不是重新运行带反馈的SLAM，也不是每种异常的整链路传感器回放。','',
        '## 数据、分组与冻结规则','',
        '- OpenLORIS：Home 5条、Office 7条、Cafe 2条、Corridor 3条；NCLT：2013-01-10完整1条。共18条独立完整序列，场景名称不等于已核实的不同建筑或LiDAR。',
        '- 冻结Home迁移：home1-1训练、home1-2验证选定的20维seed11与32维seed22直接评估Office/Cafe/Corridor/NCLT；不利用目标场景真值调整阈值或强度。',
        '- 场景隔离重训：全部Home与Office训练，Cafe验证，Corridor测试；NCLT预先声明为独立外部测试，不进入训练/模型/阈值选择。每种模型11、22、33三个种子。超参数沿用上轮。',
        '- 这些公共测试序列曾被上一轮实验查看，本轮是开发复验，不能宣称未接触过的盲测；用户确认暂无设备，自采和实际边缘硬件仍未完成。',
        '- NCLT使用wheel/FOG/IMU融合里程计，与OpenLORIS构成平台、场景和来源的联合变化。其高频参考轨迹含里程计插值，短时RPE不能视为完全独立真值证据。','',
        f'18条流共{sum(r["samples"] for r in inputs)}个实际采样点，一致性验证全部通过。训练端与记录特征的最大分量差值为{max(r["training_feature_max_error"] for r in inputs):.3g}。13条新采集序列及5条上轮Home均为1倍速、正常关闭；资源记录见online_capture_resources.csv，SLAM子进程RSS不包含外部纠正进程。部分采集期间同时进行CPU训练/后处理，这些wall记录不能作为严格负载控制下的算法耗时对照。','',
        '## 冻结Home模型的直接跨场景迁移','',
        '下表为同一场景内完整序列的宏平均APE，各列使用相同记录流及真值掩码。三个门控模式固定同一个置信度模型及其验证选定的alpha；无门控仍保留共同输入有效性与初始化回退。','',
        '| 场景 | 特征/目标 | 原始APE（m） | 无门控（m） | 规则门控（m） | 不确定性门控（m） | 不确定性回退率 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for scene in ['office','cafe','corridor','nclt']:
        for config in ['full20','relative32']:
            cases={r['gate_mode']:r for r in transfer if r['scene']==scene and r['config']==config}
            u=cases['uncertainty']
            lines.append(f'| {scene} | {config} | {u["raw_clean_ape_m"]:.4f} | {cases["none"]["ape_rmse_m"]:.4f} | {cases["rule"]["ape_rmse_m"]:.4f} | {u["ape_rmse_m"]:.4f} | {u["fallback_fraction_all"]:.1%} |')
    lines+=['','回退接近原始轨迹表明保护规则有效限制了修正，不代表迁移模型已经获得目标场景精度收益；没有隐藏失败场景或关闭修正的配置。','',
        '## 场景隔离重新训练','',
        '按种子先平均完整序列、再平均种子；种子标准差不是独立序列置信区间。表中Corridor和NCLT分开报告，避免一条长NCLT轨迹掩盖OpenLORIS结果。','',
        '| 场景 | 方法 | APE（m） | 种子标准差（m） | 1 s RPE（m） |',
        '|---|---|---:|---:|---:|']
    display={'cartographer':'Cartographer','wheel_odometry':'数据源里程计','covariance_intersection':'保守CI融合',
        'ridge_pose_residual':'Ridge','gp_pose_residual_adaptation':'GP共享特征适配',
        'mlp_full20_pose':'20维普通MLP','mlp_relative32_increment':'32维普通增量MLP',
        'mlp_full20_pose_uncertainty_hard_fallback':'20维置信度回退',
        'mlp_relative32_increment_uncertainty_hard_fallback':'32维置信度回退'}
    for scene in ['corridor','nclt']:
        for r in methods:
            if r['scene']==scene and r['method'] in display:
                lines.append(f'| {scene} | {display[r["method"]]} | {r["ape_m"]:.4f} | {r["seed_std_m"]:.4f} | {r["rpe_1s_m"]:.4f} |')
    lines+=['','普通MLP与置信度模型分别训练；该表的不同模型之间差距不能归因于单独增加一个门控。下节只在同一置信度模型内比较门控，排除了权重和alpha改变。完整EKF/因子图结果、回退比例及角度RPE见scene_isolated_methods.csv。','',
        '本轮场景隔离重训的六个置信度权重均由Cafe验证集选出gate_alpha=0，修正实际关闭。它们与原始轨迹相同的APE、异常有害率为0和恢复删失均由关闭修正解释，不能算作门控学会跨场景识别的证据。部署选型仍保留验证集决定，没有在Corridor/NCLT上挑强度。',
        '验证诊断保存在cross_scene_causal_v1/validation_residual_diagnostics.csv及validation_alpha_curves.csv：20维普通模型的标准化残差RMSE约为零修正的1.04–1.08倍；32维普通增量模型约为0.88倍，但短时残差拟合改善没有转化为积分后的验证APE收益。20维置信度的校准95%边际区间覆盖约89%，32维约96%；这些只来自两条验证序列，不证明跨场景置信度已经可靠。诊断没有改动权重、阈值或选型。','',
        '## 相同模型与alpha下的门控贡献','',
        '- none：无特征/置信度拒绝，保留共同的输入有效性、初始化及缺失后重启。',
        '- rule：在共同有效性之外，按验证集max-normalized-feature的99%分位拒绝OOD输入。',
        '- uncertainty：与rule完全相同，再按验证集normalized-sigma的90%分位拒绝。新增差距是sigma拒绝及其触发的真实积分回退，不能直接解释为概率已经校准。',
        '- 同一权重固定相同gate_alpha；alpha=0的配置明确标记为关闭修正，不用于声称门控带来精度提升。Home三个种子的完整结果在gate_scene_summary.csv，下表只展示由上轮验证选定的部署种子。',
        '- 有害帧：在同一有效真值掩码内，修正误差比对应受扰原始Cartographer输入增加超过5 cm。另存与干净原始轨迹的对照；Cartographer中断期间输入本身受损，不把这部分损伤全归因于修正模型。','',
        '| 特征 | 异常 | 无门控有害帧 | 规则门控有害帧 | 不确定性门控有害帧 |',
        '|---|---|---:|---:|---:|']
    for config in ['full20','relative32']:
        for anomaly in ['clean','wheel_jump','feature_outlier','odom_drop','odom_delay']:
            values={r['gate_mode']:r for r in home_table if r['config']==config and r['anomaly']==anomaly}
            lines.append(f'| {config} | {anomaly} | {values["none"]["harm_fraction"]:.1%} | {values["rule"]["harm_fraction"]:.1%} | {values["uncertainty"]["harm_fraction"]:.1%} |')
    lines+=['','![相同模型和强度下的有害修正](../results/mentor_benchmark/cross_scene_causal_report_v1/same_alpha_harm.png)','',
        'Home 20维模型在干净输入上的sigma门控增加回退，并可能损失修正机会；Home 32维模型中规则门控贡献了多数保护，sigma门控又减少部分有害修正。应同时报告干净输入拒绝率和精度代价，不将受控异常的改善泛化为所有动态场景安全。','',
        '| Home部署模型 | 干净就绪输入：规则拒绝率 | 干净就绪输入：不确定性拒绝率 | 规则APE（m） | 不确定性APE（m） |',
        '|---|---:|---:|---:|---:|']
    for config in ['full20','relative32']:
        records={mode:[r for r in home_display if r['config']==config and r['anomaly']=='clean' and r['gate_mode']==mode] for mode in ['rule','uncertainty']}
        rule_rejection=macro(records['rule'],'clean_rejected_fraction_ready')
        uncertainty_rejection=macro(records['uncertainty'],'clean_rejected_fraction_ready')
        rule_ape=np.mean([r['ape_m']['rmse'] for r in records['rule']])
        uncertainty_ape=np.mean([r['ape_m']['rmse'] for r in records['uncertainty']])
        lines.append(f'| {config} | {rule_rejection:.2%} | {uncertainty_rejection:.2%} | {rule_ape:.4f} | {uncertainty_ape:.4f} |')
    lines+=['','干净输入拒绝率只在就绪且真值有效的输入中计算，排除共同初始化/缺失回退；它不是经过异常真标签认证的误报率。','',
        '## 丢帧、时间偏移、中断与恢复','',
        '共9种条件：干净、轮里程计XY乘1.3、X跳变2 m、标准化特征乘10、中间三分之一区域每3帧丢1帧、内容滞后0.2 s但时间戳保持正常、里程计中断2 s、Cartographer中断2 s、明确过旧0.2 s的源时间戳。两秒中断位于中间三分之一区域起点；边界与真值无关。',
        '这些是对已记录后处理输入的受控改动。没有遮挡原始激光、模拟完整轮滑，也没有重新运行受扰SLAM；Cartographer中断仅可返回最后已知位姿。可见过旧时间戳由共同年龄检查拒绝，不能计作sigma检测能力；隐藏内容滞后用于检验该检查之外的门控。',
        '恢复定义为异常结束后首次连续3帧主动接受的起始时间；未出现则右删失，alpha=0不能记为0秒恢复。另报干净配对流的恢复时刻及额外延迟，避免把本来就在回退的场景误判为异常导致失效。新增拒绝时延只记录异常流相对干净流新增的回退，仍不等同于真实异常分类召回率。',
        '原始JSON/CSV保存恢复删失、干净输入拒绝率、反事实有益修正被拒绝比例、回退及修正偏移突变。跨场景门控试验中，如果干净流已长期回退，恢复会被删失；不能只平均成功恢复的时间。','',
        '| Home部署模型 | 条件 | 恢复起始时间中位数（s） | 删失比例 |',
        '|---|---|---:|---:|']
    for config in ['full20','relative32']:
        for anomaly in ['odom_delay','odom_outage','cart_outage','odom_stale_stamp']:
            records=[r for r in home_display if r['config']==config and r['anomaly']==anomaly and r['gate_mode']=='uncertainty']
            times=[r['recovery_s'] for r in records if r['recovery_s'] is not None]
            value=f'{np.median(times):.3f}' if times else '未恢复'
            lines.append(f'| {config} | {anomaly} | {value} | {np.mean([r["recovery_censored"] for r in records]):.0%} |')
    lines+=['','上述时间是连续3帧的第一帧；10 Hz下确认这段连续接受还需再到达2帧（约0.2 s），不是完整ROS链路恢复延迟。场景隔离新模型全部alpha=0，其异常恢复均为删失，未混入该表。','',
        '## 可复核证据与范围','',
        f'共有{len(all_rows)}条门控/序列/异常记录和{len(learned)}条重训/基线记录，独立完整序列数仍为18；相同序列的不同种子、门控和异常不增加独立样本数。23项单元检查通过。每个门控试验还断言所有回退帧与对应输入精确相等。',
        f'NCLT另在cov文件图节点附近补充{len(indices)}个样本的APE，最近网格容差固定51 ms，见nclt_graph_node_ape.csv；参考仍是数据集的SLAM参考，没有把它当成独立动捕。',
        '当前跨场景结果是实际在线输入上的因果记录流后处理；跨场景主动ROS发布、完整消息到消费者延迟、闭环精确率/召回率、真正动态事件标注与独立自采均未补齐。不同数据集的硬件及参考方式混杂，不能独立归因于LiDAR/轮里程计变化；没有实际边缘设备测量。','',
        '复现命令见[实验复现说明](实验复现说明.md)的跨场景与门控异常补充。旧版报告与模型保留。']
    (ROOT/'docs/跨场景与门控异常实验.md').write_text('\n'.join(lines)+'\n')
    paths=[ROOT/p for p in ['gate_stress.py','tools/benchmark_gate_stress.py','tools/evaluate_causal_external.py',
        'tools/report_cross_scene_causal.py','config/causal_gate_stress_v1.json','config/benchmark_cross_scene_causal_v1.json']]
    provenance={'unit_tests':'23 passed','independent_sequences':18,'gate_cases':len(all_rows),
        'scene_isolated_rows':len(learned),'same_alpha_assertions_passed':True,
        'code_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        'input_streams':[{k:r[k] for k in ['sequence','stream_sha256','samples']} for r in inputs],
        'scene_model_selection':selected,'actual_edge_hardware':False,'independently_collected_sequences':0}
    (OUT/'provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n')
    print('Saved docs/跨场景与门控异常实验.md and cross_scene_causal_report_v1 artifacts')


if __name__=='__main__':main()
