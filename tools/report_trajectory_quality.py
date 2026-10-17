#!/usr/bin/python3
"""Assemble descriptive ablations, negative transfer, actual ROS and provenance."""
import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from mentor_benchmark import validate_manifest,load_sequence,metrics

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/mentor_benchmark'
STUDIES=['home_trajectory_quality_v1','cross_scene_trajectory_quality_v1']
LABELS={('padded44',False):'32维+零填充 / 逐帧',('padded44',True):'32维+零填充 / 轨迹',
        ('quality44',False):'44维质量 / 逐帧',('quality44',True):'44维质量 / 轨迹'}


def csvrows(path):
    with path.open() as f:return list(csv.DictReader(f))


def macro_seed(rows,column):
    return np.array([np.mean([float(r[column]) for r in rows if int(r['seed'])==s]) for s in [11,22,33]])


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
        ['| '+' | '.join(map(str,row))+' |' for row in rows])+'\n'


def run():
    output=BASE/'trajectory_quality_report_v1';output.mkdir(exist_ok=True)
    summaries=[];tables=[];plots=[];selected_tables=[];equal_tables=[];stress_tables=[]
    models=[];equivalence=[];portable=[];stress_cases=0;reset_checks=0;reference=[];result_rows=0
    for name in STUDIES:
        study=BASE/name;manifest=json.loads((study/'manifest.json').read_text());validate_manifest(manifest)
        rows=csvrows(study/'results.csv');scenes=sorted(set(r['scene'] for r in rows))
        assert len(rows)==12*len(manifest['test'])*4
        result_rows+=len(rows)
        assert len(list(study.glob('*/model.pt')))==12
        equivalence+=json.loads((study/'equivalence.json').read_text())
        portable+=json.loads((study/'portable_verification.json').read_text())
        for row in manifest['sequences']:
            if row['sequence_id'] in manifest['test']:
                seq=load_sequence(row);m=metrics(seq,seq.cart)
                reference.append(dict(study=name,scene=seq.scene,sequence=seq.name,
                    ape_m=m['ape_m']['rmse'],rpe_1s_m=m['rpe']['1s']['translation_m']['rmse']))
        for scene in scenes:
            raw=[r for r in reference if r['study']==name and r['scene']==scene]
            lines=[['原Cartographer',f"{np.mean([r['ape_m'] for r in raw]):.6f}",f"{np.mean([r['rpe_1s_m'] for r in raw]):.6f}",'0','100%']]
            for config,loss in LABELS:
                group=[r for r in rows if r['scene']==scene and r['feature_config']==config and
                    r['trajectory_loss']==str(loss) and r['method']=='uncertainty_selected']
                ape=macro_seed(group,'ape_m');rpe=macro_seed(group,'rpe_1s_m')
                summary=dict(study=name,scene=scene,config=config,trajectory_loss=loss,
                    seed_macro_ape_m=ape.tolist(),ape_mean_m=float(ape.mean()),ape_seed_std_m=float(ape.std(ddof=1)),
                    rpe_mean_m=float(rpe.mean()),rpe_seed_std_m=float(rpe.std(ddof=1)),
                    harmful_fraction=float(macro_seed(group,'harmful_fraction').mean()),
                    fallback_fraction=float(macro_seed(group,'fallback_fraction').mean()))
                summaries.append(summary)
                lines.append([LABELS[config,loss],f'{ape.mean():.6f} ± {ape.std(ddof=1):.6f}',
                    f'{rpe.mean():.6f} ± {rpe.std(ddof=1):.6f}',f"{summary['harmful_fraction']*100:.2f}%",f"{summary['fallback_fraction']*100:.2f}%"])
            tables.append(f"### {'Home完整序列留出' if scene=='home' else '跨场景'+scene}\n\n"+
                table(['特征/损失','APE(m)','1s RPE(m)','有害修正比例','回退比例'],lines))
        selection=json.loads((study/'deployment_selection.json').read_text())
        model=selection['selected']['model']
        for scene in scenes:
            rr=[r for r in rows if r['model']==model and r['scene']==scene and r['method']=='uncertainty_selected']
            selected_tables.append([name.replace('_trajectory_quality_v1',''),model,scene,
                f"{selection['selected']['validation_ape_m']:.6f}",f"{float(rr[0]['alpha']):.2f}",
                f"{np.mean([float(r['raw_ape_m']) for r in rr]):.6f}",f"{np.mean([float(r['ape_m']) for r in rr]):.6f}"])
        names=[r['model'] for r in json.loads((study/'portable_verification.json').read_text())]
        for model in names:
            for scene in scenes:
                vals=[]
                for method in ['none_same_alpha','rule_same_alpha','uncertainty_selected']:
                    group=[r for r in rows if r['model']==model and r['scene']==scene and r['method']==method]
                    vals.append(f"{np.mean([float(r['ape_m']) for r in group]):.6f}")
                equal_tables.append([model,scene,*vals])
        stress=csvrows(study/'quality_stress.csv');stress_cases+=len(stress)
        protocol=json.loads((study/'quality_stress_protocol.json').read_text());reset_checks+=len(protocol['reset_checks'])
        for model in names:
            for scene in scenes:
                rr=[r for r in stress if r['model']==model and r['scene']==scene and r['case']=='scan_no_returns_2s']
                stress_tables.append([model,scene,*[f"{np.mean([float(r['interval_fallback_fraction']) for r in rr if r['gate']==g])*100:.2f}%" for g in ['none','rule','uncertainty']]])
        for path in study.glob('*/training.json'):
            info=json.loads(path.read_text());assert info['parameter_count']==9318
            models.append(dict(study=name,model=path.parent.name,epochs=info['epochs'],best_epoch=info['best_epoch'],
                train_windows=sum(r['windows'] for r in info['train_windows']),validation_windows=sum(r['windows'] for r in info['validation_windows'])))
    os.environ.setdefault('MPLCONFIGDIR',str(output/'matplotlib_cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(10.5,7))
    for ax,(scene,column,title) in zip(axes.flat,[('home','seed_macro_ape_m','Home APE (m)'),('corridor','seed_macro_ape_m','Corridor APE (m)'),
        ('nclt','seed_macro_ape_m','NCLT APE (m)'),('nclt','rpe_mean_m','NCLT 1s RPE (m)')]):
        ss=[r for r in summaries if r['scene']==scene]
        values=[r[column] if column=='seed_macro_ape_m' else r['rpe_mean_m'] for r in ss]
        mean=[float(np.mean(v)) for v in values]
        err=[float(np.std(v,ddof=1)) if isinstance(v,list) else r['rpe_seed_std_m'] for v,r in zip(values,ss)]
        ax.bar(np.arange(4),mean,yerr=err,capsize=4,color=['#8198ad','#4a7096','#9ac5b8','#398c76'])
        if column=='seed_macro_ape_m':
            for i,v in enumerate(values):ax.scatter(np.full(3,i)+[-.1,0,.1],v,color='#333333',s=15,zorder=3)
        raw=[r for r in reference if r['scene']==scene]
        baseline=np.mean([r['ape_m' if column=='seed_macro_ape_m' else 'rpe_1s_m'] for r in raw])
        ax.axhline(baseline,color='#cc593e',linestyle='--',label='Raw Cartographer')
        ax.set_xticks(np.arange(4));ax.set_xticklabels(['Pad/frame','Pad/path','Quality/frame','Quality/path'],fontsize=8)
        ax.set_title(title);ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8)
        if scene=='nclt' and column=='seed_macro_ape_m':ax.set_ylim(16.2,17.1)
    fig.suptitle('Public development sequences; validation-selected gates; 3 seeds (mean ± sample SD)',fontsize=10)
    fig.tight_layout();fig.savefig(output/'ablations.png',dpi=180);fig.savefig(output/'ablations.pdf');plt.close(fig)
    online=json.loads((BASE/'home_trajectory_quality_v1/actual_ros_verification.json').read_text())
    resources=json.loads((ROOT/'data/prepared/home1-3/cartographer_noimu_online_resources_quality_path_v1.json').read_text())
    cpu=json.loads((BASE/'trajectory_quality_cpu_v1.json').read_text());runtime=resources['correction']
    isolated=json.loads((BASE/'trajectory_quality_package_v1.json').read_text())
    assert len(isolated)==4 and all(r['isolated_import'] and r['imports_use_exported_source'] for r in isolated)
    portable_checks=[r for p in portable for r in p['verification']['captured_sequences']]
    details=dict(summaries=summaries,raw_reference=reference,models=models,result_rows=result_rows,
        equivalent_streams=len(equivalence),equivalence_max_pose_error=max(r['max_error'] for r in equivalence),
        portable_models=len(portable),portable_sequence_checks=len(portable_checks),
        portable_max_pose_error=max(r['max_pose_error'] for r in portable_checks),
        quality_perturbation_cases=stress_cases,exact_reset_checks=reset_checks)
    (output/'summary.json').write_text(json.dumps(details,indent=2)+'\n')
    with (output/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
    links=[f"- [{n}结果](../results/mentor_benchmark/{n}/results.csv)，[训练与评估计划](../results/mentor_benchmark/{n}/plan.json)，[验证集选择](../results/mentor_benchmark/{n}/deployment_selection.json)" for n in STUDIES]
    text=f'''# 轨迹级训练与质量特征实验（2026-10-05）

完成两套划分、24个模型和336条评分记录。新特征及轨迹训练还没有带来稳定的跨场景提升：Home有小幅改善，但验证集选出的跨场景最佳模型在走廊上变差，NCLT无门控残差仍会累积出巨大偏差。因此本轮候选仅用于实验，保留既有运行模型和原始轨迹。

## 1. 数据与冻结协议

Home：home1-1训练、home1-2验证、home1-3/4/5评估。跨场景：Home全部5条＋Office全部7条训练、Cafe2条验证、Corridor3条＋NCLT一条评估。完整序列不按时间切分，跨场景清单按scene_id隔离。Home划分只支持同场景完整序列留出，不代表跨场景/机器人隔离。

每套实验先保存plan，训练12个模型并确定checkpoint、sigma/OOD阈值、alpha、积分时间常数和候选排序，然后加载本套评估参考文件。阈值、alpha和tau都由验证集确定，归一化与物理裁剪尺度仅由训练集确定。本轮所有评估序列此前已查看，因此属于开发复验，不能当作未接触的最终测试。NCLT基础前端沿用已诊断的Ceres配置，没有在这一轮搜索其SLAM配置。

18条激光质量缓存位于`results/mentor_benchmark/quality_inputs_frontend_v1`。旧采样记录没有激光callback到达记录，故质量采用bag中源时间不晚于网格时刻的扫描重建；这是源时间因果重建，不能代表当时的真实callback顺序。本轮Home实际ROS复验另外记录了真实收到的扫描及所选质量值，按同一条实际输入流验证一致性。两种范围在元数据中明确区分。

## 2. 消融控制与目标

2种特征×2种损失×3种子(11/22/33)，每套12个模型。所有模型为同一MLP、44输入、6输出(均值与方差)，9318个参数。对照使用原32维特征＋12个零；扩展使用32维＋12个质量代理。零输入通道不提供有效信息，所以相同名义参数量不能证明有效容量完全相同。两组共享质量缺失保护和历史重置，避免把保护逻辑收益误记为质量特征收益。

12个代理包含Cartographer/里程计/激光输入年龄、10m内回波比例、10m外回波比例、传感器有效回波比例、近距离中位数/IQR、12扇区覆盖、近回波质心、协方差各向同性以及扫描存在标志。**这些是激光几何及输入新鲜度代理，并非Cartographer逐帧匹配评分、匹配残差或真实定位协方差。** 量程阈值与当前两类SLAM配置一致。扫描时间采用Cartographer最后有效束规则；保留既有逐束时间假设，未证明物理同步准确。

两个损失均使用相同50帧窗口、20帧步长、按序列均衡采样。对照为逐帧高斯NLL；扩展为NLL加权重1的可微SE(2)积分端点损失，比较1/3/5秒窗口内相对轨迹，按sqrt(horizon)归一。监督参考仅进入训练标签，推理只读两路位姿及激光质量。这个目标不是整段全局APE，也未在训练目标内模拟完整门控重置。两个损失都用同一验证相对轨迹损失早停，最大100轮、耐心15轮；训练窗口数量和最佳epoch保存在各模型training.json。

alpha搜索0–1、步长0.05；积分tau候选None/5/30秒；以验证序列平均APE选择。无门控自选参数另外保存；评估置信度效果时固定相同alpha和tau。方差头阈值为验证集经验百分位，`variance_calibration_scale`仅作为诊断记录，未参与在线阈值计算，不能宣称输出已校准概率或置信区间。

## 3. 三种子完整消融

下表先在每个种子内按完整序列平均，再统计三个种子的均值±样本标准差。该标准差只描述优化随机性，不代表建筑/机器人泛化的置信区间。所有学习行均使用各自验证集选择的不确定性门控；有害修正比例是评估参考有效帧中位置误差比同帧原Cartographer增大超过5cm的比例。回退包含初始化、缺失、拒绝和alpha=0。

{chr(10).join(tables)}

![完整消融，均值与种子波动](../results/mentor_benchmark/trajectory_quality_report_v1/ablations.png)

图的NCLT APE纵轴从16.2m起始，以显示小差异；原始数值以上表和CSV为准。[PDF图](../results/mentor_benchmark/trajectory_quality_report_v1/ablations.pdf)。

质量＋轨迹损失在走廊平均APE接近原轨迹，改善仅约0.2mm，且包含一个alpha=0的种子。不能据此声称有效泛化。NCLT的质量＋轨迹模型平均回退约83%，APE虽稍降，1s RPE仍明显变差。只报告APE会掩盖局部精度问题。

## 4. 验证集选择的结果与门控贡献

以下是每套研究按验证APE排序第一的模型，完整报告其评估表现；没有根据走廊/NCLT结果重新选种子。

{table(['划分','验证选中模型','评估场景','验证APE(m)','alpha','原APE(m)','修正APE(m)'],selected_tables)}

下表补充全局候选与“质量＋轨迹”组内验证候选。各行三种门控使用同一个验证alpha和tau，所以可以检查门控影响。

{table(['模型','场景','无门控APE(m)','规则门控APE(m)','不确定性门控APE(m)'],equal_tables)}

跨场景全局候选在NCLT无门控APE约346m；质量＋轨迹组内候选无门控约382m。门控大幅压低恶化，但仍不能保证有收益。Home中门控可以减小部分有害修正，却也会损失某些平均APE改善。不能把大量回退解释为泛化成功。

## 5. 真实ROS回放与CPU可行性

对Home“质量＋轨迹”组内验证选中模型quality44_trajectory_seed22，在home1-3做1倍速完整ROS回放；该模型不是Home全局验证冠军。实际发布{resources['pose_count']}个原始位姿、{runtime['count']}个10Hz修正样本，{runtime['scan_descriptor_count']}个激光描述子callback。只发布后处理位姿，不反馈到SLAM前端/图优化，不据此推断改善前端或闭环。

- 实际特征重算最大差为0；Torch逐帧复算最大差为0；批量轨迹与实际发布最大差{online['checks']['batch_vs_publication_error_max']:.3g}；NumPy与实际发布最大差{online['checks']['numpy_replay_pose_error_max']:.3g}。低置信度与缺失时逐元素精确回到原Cartographer。
- 同一运行的原APE {online['raw_metrics']['ape_m']['rmse']:.6f}→{online['corrected_metrics']['ape_m']['rmse']:.6f}m；1s RPE {online['raw_metrics']['rpe']['1s']['translation_m']['rmse']:.6f}→{online['corrected_metrics']['rpe']['1s']['translation_m']['rmse']:.6f}m；yaw APE {online['raw_metrics']['yaw_ape_deg']['rmse']:.4f}→{online['corrected_metrics']['yaw_ape_deg']['rmse']:.4f}°。有害修正{online['harmful_fraction_over_5cm']*100:.2f}%，回退{runtime['fallback_fraction']*100:.2f}%。这次回放输入与旧回放会有差异，因此只做同一运行配对评分。
- 在线CPU修正均值{runtime['mean_ms']:.3f}ms、P99 {runtime['p99_ms']:.3f}ms；包含特征、质量查找、Torch、门控、积分、文件与publish提交。激光描述子callback另计均值{runtime['scan_descriptor_mean_ms']:.3f}ms、P99 {runtime['scan_descriptor_p99_ms']:.3f}ms，不能把两个不同callback的P99简单相加当作端到端P99。修正计时也不包含传感器采集、传输与ROS排队等待。
- SLAM子进程峰值{resources['peak_rss_mib']:.2f}MiB；包含Torch和采集记录的ROS观察进程峰值{resources['observer_peak_rss_mib']:.2f}MiB。内存范围明确区分，观察进程RSS不是纯模型占用。回放墙钟{resources['wall_s']:.2f}s包含启动、播放和排空等待，不宣称严格实时期限保证。

[实际发布一致性与APE/RPE](../results/mentor_benchmark/home_trajectory_quality_v1/actual_ros_verification.json)。

4个验证选中候选已导出独立NumPy包，400个随机前向检查和{len(portable_checks)}条捕获序列复核均通过，位姿最大差{details['portable_max_pose_error']:.3g}。另外4项隔离Python检查从/tmp导入包内两份源码，未导入Torch或ROS，确认不依赖项目目录中的运行模块。另在独立进程、单线程BLAS中测纯NumPy运行，未导入Torch或ROS，硬件为{cpu['hardware']['cpu']}；表中RSS包含Python、NumPy与该条输入序列驻留。

{table(['划分/候选','CPU均值(ms)','CPU P99(ms)','进程峰值(MiB)','压缩权重(KiB)'],[[r['study'].replace('_trajectory_quality_v1','')+'/'+r['model'],f"{r['mean_ms']:.4f}",f"{r['p99_ms']:.4f}",f"{r['process_peak_rss_mib']:.2f}",f"{r['weights_bytes']/1024:.2f}"] for r in cpu['results']])}

纯NumPy耗时仅覆盖batch=1特征/模型/门控/积分，不含扫描描述子、ROS、加载和I/O；本机CPU预算充足，仍未完成GPU新模型测时、真实边缘设备实测或完整闭环接入。[CPU记录](../results/mentor_benchmark/trajectory_quality_cpu_v1.json)。

## 6. 质量缺失及无回波对照

完成{stress_cases}条固定候选评分：clean、轨迹中部2秒缺扫描标志、2秒零有效回波描述子，分别比较无门控/规则门控/不确定性门控，alpha和tau相同、参考有效mask固定。这是后处理质量输入扰动，未重跑受损激光下的SLAM，不能当作真实遮挡整链路测试。

缺扫描期间与第一个恢复样本的精确原轨迹回退通过{reset_checks}项检查；这来自共享输入保护，三种门控都会执行，不能作为方差头的独立贡献。无回波时保留“已收到扫描”标志，测试的是几何特征和门控反应，异常窗口回退如下：

{table(['模型','场景','无门控回退','规则门控回退','不确定性门控回退'],stress_tables)}

质量特征模型能在这些人工无回波窗口通过OOD规则回退；零填充对照不会感知回波变化。该结果只验证此类输入处理，不能证明全面异常检测或全链路安全。

## 7. 复现、证据与剩余工作

{chr(10).join(links)}
- [聚合CSV](../results/mentor_benchmark/trajectory_quality_report_v1/summary.csv)、[证据哈希](../results/mentor_benchmark/trajectory_quality_report_v1/provenance.json)。各study保存训练时的源文件快照；后续加入NumPy类和划分校验的当前运行源码另存入本报告目录，旧训练快照保持原样。
- 84项批量/逐帧完整输入流复核全部通过，最大误差{details['equivalence_max_pose_error']:.3g}；37项单元检查通过，涵盖历史重置、未来消息、SE(2)积分梯度和ROS质量记录。
- 导师第3项新增质量代理/同参数量消融和轨迹训练；第4项新增真实质量输入ROS回放及NumPyCPU/RSS；第5项新增相同alpha的质量异常对照。真实匹配质量、独立跨建筑/单硬件改变、自采/边缘硬件、反馈SLAM及闭环效果仍未完成。用户无设备，后续继续优先公开数据。
- NCLT只有一个已诊断会话，且提供里程计为wheel/FOG/IMU EKF；密集参考有图优化和里程计插值，RPE不能视为完全独立真值。场景身份不能替代建筑身份。

下一步应优先做多场景轮换验证，检验Cafe验证排序为何不能预测Corridor表现；重新设计与实际门控重置一致的训练积分目标，再用新增未调参公开序列检验冻结模型。当前证据不足以用新候选替换已有输出。

### 复现命令

在项目根目录运行，训练会复用已有冻结checkpoint；若要重新训练，使用新的输出目录。训练默认100轮，与本轮一致。

```bash
/usr/bin/python3 -u tools/prepare_quality_features.py --manifest config/benchmark_cross_scene_frontend_v2.json --output results/mentor_benchmark/quality_inputs_frontend_v1
/usr/bin/python3 -u tools/train_trajectory_quality.py --manifest config/benchmark_cross_scene_frontend_v2.json --quality-root results/mentor_benchmark/quality_inputs_frontend_v1 --output results/mentor_benchmark/cross_scene_trajectory_quality_v1
/usr/bin/python3 -u tools/train_trajectory_quality.py --manifest config/benchmark_home_causal_v1.json --quality-root results/mentor_benchmark/quality_inputs_frontend_v1 --output results/mentor_benchmark/home_trajectory_quality_v1
/usr/bin/python3 -u tools/export_trajectory_quality.py results/mentor_benchmark/home_trajectory_quality_v1 --quality-root results/mentor_benchmark/quality_inputs_frontend_v1
/usr/bin/python3 -u tools/export_trajectory_quality.py results/mentor_benchmark/cross_scene_trajectory_quality_v1 --quality-root results/mentor_benchmark/quality_inputs_frontend_v1
/usr/bin/python3 -u tools/online_ros_benchmark.py data/prepared/home1-3 --algorithms cartographer_noimu --checkpoint results/mentor_benchmark/home_trajectory_quality_v1/quality44_trajectory_seed22/model.pt --quality-stream --label quality_path_v1 --port 11453 --collect-metrics
/usr/bin/python3 -u tools/verify_quality_online.py --checkpoint results/mentor_benchmark/home_trajectory_quality_v1/quality44_trajectory_seed22/model.pt --capture data/prepared/home1-3/cartographer_noimu_sampled_stream_quality_path_v1.npz --sequence home1-3 --portable results/mentor_benchmark/home_trajectory_quality_v1/portable/quality44_trajectory_seed22 --output results/mentor_benchmark/home_trajectory_quality_v1/actual_ros_verification.json
/usr/bin/python3 -u tools/stress_trajectory_quality.py results/mentor_benchmark/home_trajectory_quality_v1 --quality-root results/mentor_benchmark/quality_inputs_frontend_v1
/usr/bin/python3 -u tools/stress_trajectory_quality.py results/mentor_benchmark/cross_scene_trajectory_quality_v1 --quality-root results/mentor_benchmark/quality_inputs_frontend_v1
/usr/bin/python3 -u tools/benchmark_quality_portable.py --studies results/mentor_benchmark/home_trajectory_quality_v1 results/mentor_benchmark/cross_scene_trajectory_quality_v1 --quality-root results/mentor_benchmark/quality_inputs_frontend_v1 --output results/mentor_benchmark/trajectory_quality_cpu_v1.json
/usr/bin/python3 -u tools/verify_quality_packages.py --studies results/mentor_benchmark/home_trajectory_quality_v1 results/mentor_benchmark/cross_scene_trajectory_quality_v1 --quality-root results/mentor_benchmark/quality_inputs_frontend_v1 --output results/mentor_benchmark/trajectory_quality_package_v1.json
/usr/bin/python3 tools/report_trajectory_quality.py
/usr/bin/python3 -m unittest discover -s tests -v
```
'''
    (ROOT/'docs/轨迹级训练与质量特征实验.md').write_text(text)
    paths=[]
    for name in STUDIES:
        study=BASE/name
        paths+=[study/p for p in ['plan.json','manifest.json','results.csv','deployment_selection.json','equivalence.json','portable_verification.json','quality_stress.csv','quality_stress_protocol.json']]
        paths+=list(study.glob('*/model.pt'))+list(study.glob('*/selection.json'))
        paths+=list(study.glob('portable/*/portable.json'))+list(study.glob('portable/*/portable_weights.npz'))
        paths+=list(study.glob('portable/*/trajectory_quality.py'))+list(study.glob('portable/*/causal_stream.py'))
    paths+=list((BASE/'quality_inputs_frontend_v1').glob('*.npz'))+list((BASE/'quality_inputs_frontend_v1').glob('*.json'))
    paths+=[BASE/'home_trajectory_quality_v1/actual_ros_verification.json',BASE/'trajectory_quality_cpu_v1.json',BASE/'trajectory_quality_package_v1.json',
        ROOT/'data/prepared/home1-3/cartographer_noimu_sampled_stream_quality_path_v1.npz',
        ROOT/'data/prepared/home1-3/cartographer_noimu_online_resources_quality_path_v1.json']
    sources=['trajectory_quality.py','trajectory_objective.py','tools/train_trajectory_quality.py','tools/online_ros_benchmark.py',
        'tools/prepare_quality_features.py','tools/export_trajectory_quality.py','tools/verify_quality_online.py',
        'tools/stress_trajectory_quality.py','tools/benchmark_quality_portable.py','tools/verify_quality_packages.py','tools/report_trajectory_quality.py','tests/test_trajectory_quality.py']
    for name in sources:
        path=ROOT/name;paths.append(path);(output/('source_'+path.name)).write_bytes(path.read_bytes())
    provenance={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print('Wrote docs/轨迹级训练与质量特征实验.md',len(provenance),'hashes',flush=True)


if __name__=='__main__':run()
