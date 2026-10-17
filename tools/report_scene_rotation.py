#!/usr/bin/python3
"""Report all scene rotations; ranking diagnostics never change selections."""
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[1]
STUDY=ROOT/'results/mentor_benchmark/scene_rotation_gated_v1'
LOSSES=['frame','continuous','gated_reset']
SCENES=['home','office','cafe','corridor']


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
        ['| '+' | '.join(map(str,r))+' |' for r in rows])+'\n'


def average(rows,key):return float(np.mean([float(r[key]) for r in rows]))


def run():
    report=STUDY/'report';report.mkdir(exist_ok=True)
    with (STUDY/'results.csv').open() as f:rows=list(csv.DictReader(f))
    assert len(rows)==1809
    clean=[r for r in rows if r['validation_protocol']=='single_scene_validation' and r['gate']=='uncertainty']
    details=[];chosen=[];ranks=[];loss_selected=[];external=[]
    for fold_path in sorted(STUDY.glob('test_*/manifest.json')):
        fold=json.loads(fold_path.read_text());name=fold['fold_id'];val_scene=name.split('_val_')[1];test_scene=name.split('_val_')[0].replace('test_','')
        selection=json.loads((fold_path.parent/'deployment_selection.json').read_text())
        scored=[r for r in clean if r['fold']==name and r['scene']==test_scene]
        ex=[r for r in clean if r['fold']==name and r['scene']=='nclt']
        val=[];test=[]
        for candidate in selection['candidates']:
            rr=[r for r in scored if r['model']==candidate['model']]
            val.append(candidate['validation_score']);test.append(average(rr,'ape_m'))
            details.append(dict(fold=name,validation_scene=val_scene,test_scene=test_scene,model=candidate['model'],
                loss=candidate['loss'],seed=candidate['seed'],validation_ape_m=candidate['validation_score'],
                raw_ape_m=average(rr,'raw_ape_m'),ape_m=average(rr,'ape_m'),ape_ratio=average(rr,'ape_m')/average(rr,'raw_ape_m'),
                raw_rpe_1s_m=average(rr,'raw_rpe_1s_m'),rpe_1s_m=average(rr,'rpe_1s_m'),
                harmful_fraction=average(rr,'harmful_fraction'),fallback_fraction=average(rr,'fallback_fraction'),
                alpha=candidate['alpha'],tau_s=candidate['tau_s']))
        rho=None if np.ptp(val)==0 or np.ptp(test)==0 else float(spearmanr(val,test).statistic)
        picked=selection['selected'];rr=[r for r in scored if r['model']==picked['model']]
        selected_ape=average(rr,'ape_m');posthoc_min=min(test)
        rank=dict(fold=name,test_scene=test_scene,validation_scene=val_scene,spearman=rho,
            selected_test_ape_m=selected_ape,posthoc_min_test_ape_m=posthoc_min,
            posthoc_selection_gap_m=selected_ape-posthoc_min,diagnostic_only=True,changes_model_selection=False)
        ranks.append(rank)
        chosen.append(dict(fold=name,validation_scene=val_scene,test_scene=test_scene,**picked,
            raw_ape_m=average(rr,'raw_ape_m'),ape_m=selected_ape,ape_ratio=selected_ape/average(rr,'raw_ape_m'),
            raw_rpe_1s_m=average(rr,'raw_rpe_1s_m'),rpe_1s_m=average(rr,'rpe_1s_m'),
            harmful_fraction=average(rr,'harmful_fraction'),fallback_fraction=average(rr,'fallback_fraction')))
        er=[r for r in ex if r['model']==picked['model']]
        external.append(dict(fold=name,model=picked['model'],loss=picked['loss'],
            raw_ape_m=average(er,'raw_ape_m'),ape_m=average(er,'ape_m'),rpe_1s_m=average(er,'rpe_1s_m'),
            raw_rpe_1s_m=average(er,'raw_rpe_1s_m'),harmful_fraction=average(er,'harmful_fraction'),fallback_fraction=average(er,'fallback_fraction')))
        for loss,candidate in selection['selected_by_loss'].items():
            rr=[r for r in scored if r['model']==candidate['model']]
            loss_selected.append(dict(fold=name,test_scene=test_scene,loss=loss,model=candidate['model'],
                ape_m=average(rr,'ape_m'),ape_ratio=average(rr,'ape_m')/average(rr,'raw_ape_m'),rpe_1s_m=average(rr,'rpe_1s_m'),
                harmful_fraction=average(rr,'harmful_fraction'),fallback_fraction=average(rr,'fallback_fraction')))
    pool=json.loads((STUDY/'pooled_selection.json').read_text());pooled=[]
    for protocol,candidate in pool['selected'].items():
        rr=[r for r in rows if r['validation_protocol']==protocol and r['model']==candidate['model'] and r['gate']=='uncertainty']
        assert len(rr)==1
        r=rr[0];pooled.append(dict(**candidate,ape_m=float(r['ape_m']),rpe_1s_m=float(r['rpe_1s_m']),
            fallback_fraction=float(r['fallback_fraction']),harmful_fraction=float(r['harmful_fraction'])))
    summaries=[]
    for scene in SCENES:
        for loss in LOSSES:
            rr=[r for r in details if r['test_scene']==scene and r['loss']==loss];assert len(rr)==9
            ape=np.array([r['ape_m'] for r in rr]);rpe=np.array([r['rpe_1s_m'] for r in rr])
            summaries.append(dict(scene=scene,loss=loss,raw_ape_m=float(np.mean([r['raw_ape_m'] for r in rr])),
                ape_mean_m=float(ape.mean()),ape_conditional_sd_m=float(ape.std(ddof=1)),rpe_mean_m=float(rpe.mean()),
                raw_rpe_m=float(np.mean([r['raw_rpe_1s_m'] for r in rr])),
                harmful_fraction=float(np.mean([r['harmful_fraction'] for r in rr])),
                fallback_fraction=float(np.mean([r['fallback_fraction'] for r in rr])),
                disabled_models=sum(r['alpha']==0 for r in rr),conditional_models=9))
    checks=json.loads((STUDY/'equivalence.json').read_text());assert len(checks)==67
    counts=dict(models=108,folds=12,rows=len(rows),stream_checks=len(checks),stream_error_max=max(r['max_pose_error'] for r in checks),
        improved_global_selections=sum(r['ape_m']<r['raw_ape_m']-1e-6 for r in chosen),
        worsened_global_selections=sum(r['ape_m']>r['raw_ape_m']+1e-6 for r in chosen),
        unchanged_global_selections=sum(abs(r['ape_m']-r['raw_ape_m'])<=1e-6 for r in chosen),
        negative_rank_folds=sum(r['spearman'] is not None and r['spearman']<0 for r in ranks),
        undefined_rank_folds=sum(r['spearman'] is None for r in ranks))
    warmups=[]
    for fold_path in STUDY.glob('test_*/manifest.json'):
        for seed in [11,22,33]:
            info=[json.loads((fold_path.parent/f'quality44_{loss}_seed{seed}'/'training.json').read_text()) for loss in LOSSES]
            assert len({r['warmup_sha256'] for r in info})==1
            assert all(r['parameter_count']==9318 and r['warmup_epochs']==5 for r in info)
            warmups.append(dict(fold=fold_path.parent.name,seed=seed,shared_warmup_verified=True))
    all_details=dict(counts=counts,summaries=summaries,selected_folds=chosen,ranking_diagnostics=ranks,
        selected_by_loss=loss_selected,external_nclt=external,pooled_external=pooled,shared_warmup_checks=warmups)
    (report/'summary.json').write_text(json.dumps(all_details,indent=2,allow_nan=False)+'\n')
    for filename,data in [('fold_models.csv',details),('selected_folds.csv',chosen),('ranking_diagnostics.csv',ranks),('loss_summary.csv',summaries),('external_nclt.csv',external)]:
        with (report/filename).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    os.environ.setdefault('MPLCONFIGDIR',str(report/'matplotlib_cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'frame':'#7e99b1','continuous':'#478c78','gated_reset':'#c48352'}
    fig,axes=plt.subplots(2,1,figsize=(12,7))
    for i,r in enumerate(chosen):
        axes[0].bar(i,r['ape_m']/r['raw_ape_m'],color=colors[r['loss']])
        rank=ranks[i]
        if rank['spearman'] is not None:axes[1].bar(i,rank['spearman'],color=colors[r['loss']])
    axes[0].axhline(1,color='#b74e40',linestyle='--');axes[0].set_ylabel('Held scene APE / raw APE')
    axes[1].axhline(0,color='#444444',linewidth=.8);axes[1].set_ylabel('Validation/test Spearman rho');axes[1].set_ylim(-1.05,1.05)
    ticks=np.arange(12);labels=[r['test_scene']+' / '+r['validation_scene'] for r in chosen]
    for ax in axes:ax.set_xticks(ticks);ax.set_xticklabels(labels,rotation=25,ha='right',fontsize=8);ax.grid(axis='y',alpha=.2)
    from matplotlib.patches import Patch
    axes[0].legend(handles=[Patch(color=colors[loss],label=loss) for loss in LOSSES],fontsize=9)
    axes[0].set_title('Validation-selected candidates; held scene / validation scene')
    axes[1].set_title('Ranking diagnostic only; undefined correlations left blank')
    fig.tight_layout();fig.savefig(report/'rotation.png',dpi=180);fig.savefig(report/'rotation.pdf');plt.close(fig)
    scene_rows=[]
    for r in summaries:
        scene_rows.append([r['scene'],r['loss'],f"{r['raw_ape_m']:.6f}",f"{r['ape_mean_m']:.6f} ± {r['ape_conditional_sd_m']:.6f}",
            f"{r['raw_rpe_m']:.6f}→{r['rpe_mean_m']:.6f}",f"{r['harmful_fraction']*100:.2f}%",f"{r['fallback_fraction']*100:.2f}%",f"{r['disabled_models']}/9"])
    fold_rows=[]
    for r,rank in zip(chosen,ranks):
        fold_rows.append([r['test_scene'],r['validation_scene'],r['model'],f"{r['alpha']:.2f}",str(r['tau_s']),
            f"{r['raw_ape_m']:.6f}→{r['ape_m']:.6f}",f"{r['raw_rpe_1s_m']:.6f}→{r['rpe_1s_m']:.6f}",
            '未定义' if rank['spearman'] is None else f"{rank['spearman']:.3f}"])
    external_rows=[[r['fold'],r['model'],f"{r['ape_m']:.6f}",f"{r['rpe_1s_m']:.6f}",f"{r['harmful_fraction']*100:.2f}%",f"{r['fallback_fraction']*100:.2f}%"] for r in external]
    paired_baseline=next(r for r in external if r['fold']=='test_corridor_val_cafe')
    pooled_rows=[['Cafe单验证/训练门槛',paired_baseline['model'],'见原轮选择',f"{paired_baseline['ape_m']:.6f}",f"{paired_baseline['rpe_1s_m']:.6f}",f"{paired_baseline['harmful_fraction']*100:.2f}%",f"{paired_baseline['fallback_fraction']*100:.2f}%"]]
    pooled_rows.extend([[r['protocol'],r['model'],f"{r['alpha']:.2f}/{r['tau_s']}",f"{r['ape_m']:.6f}",f"{r['rpe_1s_m']:.6f}",f"{r['harmful_fraction']*100:.2f}%",f"{r['fallback_fraction']*100:.2f}%"] for r in pooled])
    text=f'''# 场景轮换与门控重置训练实验（2026-10-05）

完成12组按场景隔离的训练/验证/留出轮换、108个模型、1809条相同强度门控评分和67条完整流一致性复核。验证选出的模型在5组留出条件中改善APE、7组恶化；门控重置目标在四个留出场景上的平均APE均未优于连续目标，尚无稳定精度收益。本轮新目标直接模拟拒绝修正后的原轨迹重置，以及在线SE(2)积分、裁剪和有限记忆；结果以完整留出场景的APE/RPE、有害修正和回退评价。以下所有公开序列之前均已查看，属于开发研究，不能替代未触及的最终泛化测试。

## 1. 划分和冻结

OpenLORIS四场景：Home5条、Office7条、Cafe2条、Corridor3条。每轮完整留出一个场景，剩下三个场景各轮流作为验证场景，其余两个训练，共4×3=12组；每组3种损失×3种子(11/22/33)。NCLT不进入任何训练或场景轮换验证，沿用已诊断的基础前端，仅在108个模型及全部选择结果冻结后读取其参考。每个轮次按scene_id校验完整序列无重叠，原始数据未修改。

每组更换验证场景时，训练场景也会变化。因此这12组展示选择敏感性，不能把差异完全归因于验证场景。本报告第5节另外固定同一组权重，只改变验证/校准策略，以区分两种影响。各轮都保存manifest、模型、完整验证曲线和选择文件。[预先保存的计划](../results/mentor_benchmark/scene_rotation_gated_v1/plan.json)、[外部参考访问前的选择冻结记录](../results/mentor_benchmark/scene_rotation_gated_v1/selection_freeze.json)。

## 2. 与实际回退一致的训练

所有模型固定quality44输入、同一MLP均值/方差头、9318参数。质量仍是源时间因果重建的激光几何和输入年龄代理，不是逐帧匹配评分，也不是旧callback到达的精确还原。每个种子的三种损失共享相同5轮NLL预热权重、Adam状态和采样顺序，36项预热哈希对照通过，减少初始化和采样差异。

- frame：逐帧高斯NLL。
- continuous：NLL＋原1/3/5秒连续积分端点损失，窗口内不模拟置信拒绝和重置。
- gated_reset：NLL＋按当前sigma/OOD硬门控的1/3/5秒端点损失。拒绝时整个累计偏移归零并精确回到原Cartographer；接受时按相同SE(2)公式、物理裁剪和tau指数衰减递推。

门槛取预热模型在训练输入上的sigma P90和max-z P99，之后冻结，主轮次训练、验证与推理使用同一门槛。门槛估计不使用验证/留出参考。硬门控不对sigma阈值比较求导，方差头仍由NLL训练；不能宣称输出是校准概率。训练时alpha从0.05–1步长0.05采样，tau从None/5/30秒采样。20%训练窗口额外插入两帧置信拒绝，输入特征不变；这是决策拒绝训练，不是受损传感器重放。

两个轨迹目标均为50帧窗口、20帧步长、按完整序列均衡采样，完整有效标签窗相同。每窗从原轨迹起点初始化，尚不包含窗前长期累计偏差；因此只证明窗口内递推和重置机制一致。在线输入缺失后的历史保护沿用已有运行模块，单元对照覆盖首个恢复样本。三种损失共用完整验证轨迹APE早停：冻结门槛、alpha=1、tau=30秒；之后仅用验证集搜索最终alpha和tau。最大100轮(含预热)、耐心15轮。

新增4项可微积分检查通过：与现有运行流在角度绕回、输入缺失/恢复、置信拒绝和三种tau下一致；拒绝会切断更早残差的梯度；全拒绝/alpha=0逐元素回到原轨迹；连续接受及有限记忆下梯度有限。最终冻结候选还进行了{len(checks)}条实际捕获输入上的批量/逐帧复核，最大误差{counts['stream_error_max']:.3g}。本轮仅改训练与选择，没有修改现有运行协议或替换部署模型。

## 3. 全部损失消融

以下每行汇总该留出场景下3种验证配置×3种子=9个模型：先在模型内按完整序列平均，再按模型平均。±为这些相关条件模型的样本标准差，混合了训练/验证配置和优化随机性，不是9次独立采集的置信区间。RPE为1秒平移RMSE；有害修正是同帧位置误差比原Cartographer增大超过5cm；alpha=0和大量回退不能当作泛化成功。

{table(['留出场景','损失','原APE(m)','修正APE(m)','原→修正RPE(m)','有害修正','回退','alpha=0'],scene_rows)}

完整评分保留三种门控none/rule/uncertainty，各候选在它们之间使用相同验证alpha、tau，不把降低强度误记为门控收益。[全部评分](../results/mentor_benchmark/scene_rotation_gated_v1/results.csv)。

## 4. 验证集排序能否预测留出表现

各组只按验证APE选模型，下表完整显示所选模型在留出场景上的表现；没有根据留出结果重新选种子、损失、alpha或tau。

{table(['留出','验证','验证选中模型','alpha','tau(s)','原→修正APE(m)','原→修正RPE(m)','排序ρ'],fold_rows)}

12组验证选择中，APE改善{counts['improved_global_selections']}组、恶化{counts['worsened_global_selections']}组、不变{counts['unchanged_global_selections']}组。ρ为同轮9个候选的验证APE与留出APE Spearman相关，仅用于完成训练后的诊断；负相关{counts['negative_rank_folds']}组、因并列而未定义{counts['undefined_rank_folds']}组。模型共享数据和初始化，这些ρ不用于独立显著性检验。排序诊断和事后最小APE差距均未写回模型选择。[排序诊断CSV](../results/mentor_benchmark/scene_rotation_gated_v1/report/ranking_diagnostics.csv)。

问题并不只来自Cafe验证：Home留出时使用三个不同验证场景都恶化，而Cafe留出时三个条件都有小幅APE改善。当前质量代理和短窗残差训练尚不足以稳定区分这些场景的修正需求；这是本轮结果支持的范围，不把它扩展成单一因果解释。

![场景轮换与验证排序](../results/mentor_benchmark/scene_rotation_gated_v1/report/rotation.png)

[PDF图](../results/mentor_benchmark/scene_rotation_gated_v1/report/rotation.pdf)。图中每根APE柱是一个按验证选出的条件模型，原轨迹为1；不能把12个相关条件当作12条独立外部机器人测试。

## 5. 相同权重下的多场景验证与NCLT

先完整报告12个全局验证候选在同一NCLT会话上的结果。原NCLT APE {paired_baseline['raw_ape_m']:.6f}m，1s RPE {paired_baseline['raw_rpe_1s_m']:.6f}m。这是12种参数/训练条件对同一外部开发会话的复验，不是12条新的数据。

{table(['来源轮次','冻结候选','NCLT APE(m)','1s RPE(m)','有害修正','回退'],external_rows)}

另取test_corridor_val_cafe的9个Home＋Office训练权重，全部保持不变，比较Cafe单验证和Cafe＋Corridor双场景验证。**此对照中Corridor明确转为校准/选择场景，只用NCLT作外部评价，不再把Corridor叫测试集。** 这些权重仍由Cafe单场景早停得到，故不是完整多场景训练重跑。

双场景采用场景等权、场景内序列等权。2×2对照：验证目标为绝对APE场景平均或相对原轨迹APE比例场景平均；门槛保留训练预热门槛或用双验证场景等权分位数重估。最终模型/alpha/tau仅依据相应验证目标确定。pooled_gate门槛是额外校准控制，与原训练门槛不同，不计作严格相同门槛训练。四种策略均在读取NCLT参考前冻结。

{table(['验证/校准策略','选中权重','alpha/tau','NCLT APE(m)','1s RPE(m)','有害修正','回退'],pooled_rows)}

训练门槛下规则门控已使NCLT回退约99.24%，加方差头后为99.25%；这里的保护主要来自输入OOD判断，不能独立归功于方差头。重估双场景门槛后APE稍降到16.475666m，1s RPE却升到0.300312m，有害修正约24.47%。因此不能用这项APE下降宣布整体性能改善。

[双场景选择文件](../results/mentor_benchmark/scene_rotation_gated_v1/pooled_selection.json)保留全部36项候选选择结果(9权重×4策略)，[门槛及权重哈希](../results/mentor_benchmark/scene_rotation_gated_v1/pooled_calibration.json)记录对应sigma/OOD门槛和alpha/tau。首轮评分时这些门槛已在内存中冻结；随后只用Cafe/Corridor输入恢复并保存门槛记录，没有改选择结果，也未读取NCLT参考。后续驱动直接在外部评价前保存该记录。不根据NCLT结果挑其中一种作最终方法。外部会话的密集参考含图优化及里程计插值，且输入里程计为wheel/FOG/IMU EKF，RPE不能视为完全独立真值。

## 6. 复核范围与下一步

44项单元检查通过；新增场景轮换、均衡校准和相对APE目标检查，加上拒绝积分及梯度检查。主轮次结果满足场景隔离、相同名义参数量、相同预热、冻结门槛、相同强度门控及精确回退。所有模型和关键文件具有SHA-256及源码快照，[证据记录](../results/mentor_benchmark/scene_rotation_gated_v1/report/provenance.json)。

本轮没有新增数据下载、自采、GPU新模型测时、边缘设备或真实SLAM反馈闭环测试。上轮CPU/ROS结果只说明同一运行架构的已有测量，不能代替本轮每个新模型的设备实测。现有结果属于公开开发序列轮换；不同场景不自动等同于不同建筑，单次NCLT联合平台/场景变化也不是单因素硬件消融。

下一步应依据完整轮换结果诊断仍然恶化的条件，并在新增公开完整序列上验证最终冻结方案；只有少数条件改善或大量关闭修正时，仍不能据此宣称导师要求的泛化已完成。

### 复现命令

```bash
/usr/bin/python3 -u tools/train_scene_rotation.py --manifest config/benchmark_cross_scene_frontend_v2.json --quality-root results/mentor_benchmark/quality_inputs_frontend_v1 --output results/mentor_benchmark/scene_rotation_gated_v1
/usr/bin/python3 tools/freeze_rotation_calibration.py results/mentor_benchmark/scene_rotation_gated_v1 --quality-root results/mentor_benchmark/quality_inputs_frontend_v1
/usr/bin/python3 tools/report_scene_rotation.py
/usr/bin/python3 -m unittest discover -s tests -v
```

已有checkpoint会按预热哈希校验后复用；要从头复训请用新输出目录。此前18条质量缓存的准备命令见[上轮报告](轨迹级训练与质量特征实验.md#复现命令)。
'''
    (ROOT/'docs/场景轮换与门控重置训练实验.md').write_text(text)
    paths=[STUDY/p for p in ['plan.json','selection_freeze.json','pooled_selection.json','pooled_calibration.json','results.csv','equivalence.json','provenance.json']]
    paths+=list(STUDY.glob('test_*/manifest.json'))+list(STUDY.glob('test_*/deployment_selection.json'))
    paths+=list(STUDY.glob('test_*/*/model.pt'))+list(STUDY.glob('test_*/*/selection.json'))+list(STUDY.glob('test_*/*/training.json'))
    paths+=list(report.glob('*.csv'))+[report/'summary.json']
    for name in ['gated_objective.py','tools/train_scene_rotation.py','tools/report_scene_rotation.py','tools/freeze_rotation_calibration.py','tests/test_gated_objective.py','tests/test_scene_rotation.py']:
        path=ROOT/name;paths.append(path);(report/('source_'+path.name)).write_bytes(path.read_bytes())
    sha={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (report/'provenance.json').write_text(json.dumps(sha,indent=2)+'\n')
    print(json.dumps(counts,indent=2),flush=True)


if __name__=='__main__':run()
