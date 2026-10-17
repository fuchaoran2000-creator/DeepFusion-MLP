#!/usr/bin/python3
"""Supplemental report for v1; never overwrites frozen mentor experiment tables."""
import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/mentor_benchmark/home_causal_v1'
OUT=BASE/'report'


def read(path):return json.loads(path.read_text())


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    rows=read(BASE/'results.json')
    assert len(rows)==132
    grouped=defaultdict(list)
    for row in rows:
        if 'anomaly' not in row:grouped[row['method']].append(row)
    summary=[]
    for method,group in grouped.items():
        seeds=defaultdict(list)
        for row in group:seeds[row['seed']].append(row['ape_m']['rmse'])
        seed_means=[np.mean(v) for v in seeds.values()]
        summary.append({'method':method,'ape_m':float(np.mean(seed_means)),
            'seed_macro_std_m':float(np.std(seed_means)),
            'rpe_1s_m':float(np.mean([r['rpe']['1s']['translation_m']['rmse'] for r in group])),
            'rpe_1s_deg':float(np.mean([r['rpe']['1s']['rotation_deg']['rmse'] for r in group])),
            'measured_rows':len(group),'seeds':len(seeds),'sequences':3})
    with (OUT/'causal_methods.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    inputs=read(BASE/'input_equivalence.json')
    portable=read(BASE/'portable_equivalence.json')
    live20=read(BASE/'live20_equivalence.json');live32=read(BASE/'live32_equivalence.json')
    assert len(inputs)==5 and len(live20)==len(live32)==len(portable)==3
    assert all(r['passed'] for r in inputs+portable+live20+live32)
    live=[]
    for label,reports in [('unified20',live20),('unified32',live32)]:
        for report in reports:
            path=ROOT/'data/prepared'/report['sequence']/f'cartographer_noimu_online_resources_{label}.json'
            resource=read(path)
            assert resource['shutdown_success']
            assert 'saved_live_vs_step_max_error' in report
            item={'model':label,'sequence':report['sequence'],'samples':report['samples'],
                'raw_ape_m':report['raw_metrics']['ape_m']['rmse'],
                'shared_map_corrected_ape_m':report['corrected_metrics']['ape_m']['rmse'],
                'independent_origin_corrected_ape_m':report['independent_origin_metrics']['ape_m']['rmse'],
                'raw_rpe_1s_m':report['raw_metrics']['rpe']['1s']['translation_m']['rmse'],
                'corrected_rpe_1s_m':report['corrected_metrics']['rpe']['1s']['translation_m']['rmse'],
                'fallback_fraction':report['fallback_fraction'],
                'input_available_fraction':report['input_available_fraction'],
                'reference_valid_fraction':report['raw_metrics']['coverage']['reference_valid_fraction'],
                'callback_p99_ms':resource['correction']['p99_ms'],
                'batch_vs_step_max_error':report['batch_vs_step_max_error'],
                'published_tum_max_error':report['published_tum_max_error']}
            live.append(item)
    with (OUT/'live_paired.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(live[0]));writer.writeheader();writer.writerows(live)
    portable_resources=read(ROOT/'results/mentor_benchmark/portable_causal_v1/portable_recorded_latency.json')
    selection=read(BASE/'deployment_selection.json')
    # Scientific figure: same-run paired APE for each configuration, both alignments.
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'results/mentor_benchmark/mpl_cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
    for ax,label in zip(axes,['unified20','unified32']):
        cases=[r for r in live if r['model']==label];x=np.arange(3)
        for offset,key,title in [(-.23,'raw_ape_m','Raw Cartographer'),(0,'shared_map_corrected_ape_m','Correction: shared map'),(.23,'independent_origin_corrected_ape_m','Correction: independent origin')]:
            ax.bar(x+offset,[r[key] for r in cases],width=.23,label=title)
        ax.set_xticks(x);ax.set_xticklabels([r['sequence'] for r in cases])
        ax.set_title(label);ax.set_ylabel('Planar APE RMSE (m)');ax.grid(axis='y',alpha=.2)
    axes[0].legend(fontsize=8);fig.tight_layout()
    for suffix in ['png','pdf']:fig.savefig(OUT/f'live_paired.{suffix}',dpi=170)
    plt.close(fig)
    lines=['# 在线训练与推理一致性验证：causal_grid_v1','',
        '这是上一轮实验的独立补充，旧版结果和表格保留。本轮使用OpenLORIS Home已有开发序列，不是新增盲测集，也不构成跨场景泛化验证。','',
        '## 已完成的修改','',
        '- 训练、ROS和NumPy部署共用`causal_stream.py`。第一条Cartographer发布确定10 Hz时钟：起点时间 + 整数序号/10；不按回调到达间隔决定采样。',
        '- 两路位姿仅使用已经收到且源时间不晚于采样时间、年龄不超过150 ms的消息；缓冲未来时间戳，保留仍可用的过去消息。在线实际选中的输入保存为NPZ，训练直接读取这些配对，避免事后猜测消息到达顺序。',
        '- 第一对可用输入分别确定两个估计器的平面坐标原点，缺失后的恢复不会改动原点。缺失会清空运动窗口、移除累积修正；恢复第一帧原样回退，第二帧开始重新提取增量。低置信度回退移除累积偏移，同时保留有效输入窗口。',
        '- 输入可用性、特征是否就绪、真值是否可用于监督/评分分别保存。真值缺失不会清空输入缓存；训练归一化只使用训练集有效目标，门控阈值从验证集就绪输入确定。',
        '- 本版本支持20维位姿MLP和32维增量MLP；旧版消融与时序网络仍使用原协议。版本权重不能静默送入旧ROS接口。','',
        '## 一致性证据','',
        f'- 5条序列共{sum(r["samples"] for r in inputs)}个实际在线采样点：保存特征与训练重算特征最大差值均为0。',
        f'- 两个模型共6次1倍速ROS测试回放全部正常关闭；同一回放内，实际保存的模型输出与逐帧重算最大差值为{max(r["saved_live_vs_step_max_error"] for r in live20+live32):.3g}。',
        f'- 批量预测后按相同状态规则积分，与逐帧预测最大坐标/角度分量差值为{max(r["batch_vs_step_max_error"] for r in live20+live32):.3g}；发布TUM和逐帧重算最大分量差值为{max(r["published_tum_max_error"] for r in live20+live32):.3g}。位置单位为米，角度单位为弧度；TUM保留9位小数。',
        '- 所有回退帧的平面输出与对应原始输入逐元素相等。测试另外覆盖未来里程计、丢帧后的窗口统计重启、固定原点、积分偏移移除、改变真值时间覆盖后输入特征不变。',
        f'- 32维NumPy与Torch在3条记录输入上的最大分量差值为{max(r["portable_vs_torch_max_error"] for r in portable):.3g}，门控决定完全相同；另有100组随机前向导出核对。',
        '- 一致性针对同一回放的输入流；不同SLAM回放的异步优化、消息到达和轨迹波动仍可能造成不同结果。','',
        '## 重新训练与记录流后处理','',
        'home1-1训练，home1-2验证，home1-3/4/5开发测试；每种MLP使用11、22、33三个种子。最多150轮，验证损失早停25轮。以下为3序列先平均、再按种子平均的平面APE/RPE；标准差是种子间宏平均标准差，不是置信区间。132条记录包括基线、普通/置信度模型与受控输入异常，不是132条独立序列。','',
        '| 方法 | APE RMSE（m） | 种子标准差（m） | 1 s RPE（m） | 1 s RPE（°） |',
        '|---|---:|---:|---:|---:|']
    names={'cartographer':'Cartographer','wheel_odometry':'轮里程计','covariance_intersection':'保守CI融合',
        'ridge_pose_residual':'Ridge位姿残差','gp_pose_residual_adaptation':'GP共享特征适配',
        'mlp_full20_pose':'20维普通MLP','mlp_relative32_increment':'32维普通增量MLP',
        'mlp_full20_pose_uncertainty_hard_fallback':'20维置信度回退',
        'mlp_relative32_increment_uncertainty_hard_fallback':'32维置信度回退'}
    for r in summary:
        if r['method'] in names:lines.append(f'| {names[r["method"]]} | {r["ape_m"]:.4f} | {r["seed_macro_std_m"]:.4f} | {r["rpe_1s_m"]:.4f} | {r["rpe_1s_deg"]:.3f} |')
    lines+=['','20维与32维同时改变了特征和预测目标，二者的直接差距不能解释为“增加维数”的独立因果贡献；旧版交叉目标与消融仍需参照原报告。本轮置信度版本与普通版本分别训练，其差距不能单独证明门控贡献。','',
        '## 新的实际ROS发布结果','',
        '每行都使用对应同一次回放的原始轨迹、修正轨迹、时间网格和有效真值掩码，先保证配对比较。共享map对齐保留修正的起始偏移，独立起点对齐为每个方法重新消除起始位置和航向；两种评分均报告。模型选择只依据home1-2验证集门控APE，不依据这张表。','']
    for key in ['full20','relative32']:
        s=selection[key]['selected']
        lines.append(f'- {key}选择seed{s["seed"]}，验证APE {s["validation_ape_m"]:.4f} m，门控修正强度alpha={s["gate_alpha"]:.2f}。')
    lines+=['','| 模型 | 序列 | 原始APE（m） | 修正APE：共享map（m） | 修正APE：独立起点（m） | 回退率 | 回调P99（ms） |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in live:lines.append(f'| {r["model"]} | {r["sequence"]} | {r["raw_ape_m"]:.4f} | {r["shared_map_corrected_ape_m"]:.4f} | {r["independent_origin_corrected_ape_m"]:.4f} | {r["fallback_fraction"]:.1%} | {r["callback_p99_ms"]:.3f} |')
    lines+=['','| 模型宏平均 | 原始APE（m） | 共享map修正APE（m） | 独立起点修正APE（m） |',
        '|---|---:|---:|---:|']
    for label in ['unified20','unified32']:
        cases=[r for r in live if r['model']==label]
        means=[np.mean([r[key] for r in cases]) for key in ['raw_ape_m','shared_map_corrected_ape_m','independent_origin_corrected_ape_m']]
        lines.append(f'| {label} | {means[0]:.4f} | {means[1]:.4f} | {means[2]:.4f} |')
    lines+=['','本轮两个对齐分数相同，是因为评分起点处模型仍原样回退，修正轨迹与原始轨迹在该锚点一致；不能据此认为两种对齐规则普遍等价。旧版共同时段表同时要求其他SLAM方法可用，起点与本轮配对评价不同；新旧分数不构成控制其他因素后的流程修复效果。','']
    for label in ['unified20','unified32']:
        cases=[r for r in live if r['model']==label]
        raw_rpe=np.mean([r['raw_rpe_1s_m'] for r in cases])
        corrected_rpe=np.mean([r['corrected_rpe_1s_m'] for r in cases])
        lines.append(f'{label}的1 s平移RPE宏平均从{raw_rpe:.4f} m变为{corrected_rpe:.4f} m。')
    lines+=['因此20维的主要收益是APE下降，局部RPE基本持平；32维APE略降，但局部RPE变差。仅凭APE不能宣称全部轨迹指标改善。']
    lines+=['', '![同一次回放的配对APE](../results/mentor_benchmark/home_causal_v1/report/live_paired.png)','',
        '回退率包含初始化、输入缺失后的重启、置信度拒绝和alpha为0的情况；不把回退率低或大量回退直接解释为泛化成功。完整覆盖率、RPE和一致性误差见`live_paired.csv`。','',
        '## 部署与测量范围','',
        f'32维NumPy在本机CPU的记录流测量：{portable_resources["repeats"]}个计时点，平均{portable_resources["mean_ms"]:.3f} ms，P99 {portable_resources["p99_ms"]:.3f} ms，进程峰值RSS {portable_resources["process_peak_rss_mib"]:.2f} MiB，压缩权重{portable_resources["weights_bytes"]}字节。',
        '该计时包括共享特征、前向、置信度和积分，排除输入检索、ROS排队与接收端。ROS回调计时只覆盖发出网格样本的调用，包括该次采样、特征、Torch前向、门控、积分、轨迹写入和发布提交；没有计算完整消息到消费者延迟，也没有覆盖所有不发出网格样本的TF回调开销。当前没有真实边缘硬件实测。',
        '可移植目录包含portable.json、portable_weights.npz、portable_residual.py、causal_stream.py，运行依赖为NumPy；调用者必须传入相同GridSampler产生的实际输入对，并设置available。','',
        '## 仍未解决的导师要求','',
        '用户已确认暂无设备，后续范围只使用已有公共数据中的机器人和传感器配置，自采与实际边缘硬件仍记为未完成。本轮完成在线流程的一致性修复与同域重训、真实回放核对；公共数据上仍需跨场景复验、相同alpha下的门控比较及丢帧/时间偏移异常评分、NCLT前端诊断和闭环质量评估。不同公共数据的场景与平台联合变化不能单独证明硬件影响。当前输出仍只发布/deepfusion_pose，没有反馈到Cartographer状态；数值一致性通过不等于跨场景精度或闭环质量改善。','',
        '## 复现','',
        '完整命令见[实验复现说明](实验复现说明.md)的causal_grid_v1补充。结果目录保存manifest、源码快照、训练记录、验证选型证据、三类一致性JSON及CSV；历史结果表格不被该报告工具覆盖。']
    (ROOT/'docs/在线流程一致性验证.md').write_text('\n'.join(lines)+'\n')
    paths=[ROOT/p for p in ['causal_stream.py','mentor_benchmark.py','portable_residual.py',
        'tools/online_ros_benchmark.py','tools/verify_causal_stream.py','tools/report_causal_stream.py',
        'config/benchmark_home_causal_v1.json']]
    provenance={'protocol':'causal_grid_v1','unit_tests':'18 passed',
        'code_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        'input_streams':[{k:r[k] for k in ['sequence','stream_sha256','samples']} for r in inputs],
        'live_paired_executions':6,'independent_self_collected_sequences':0,'edge_device_tested':False}
    (OUT/'provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n')
    print('Saved docs/在线流程一致性验证.md and causal v1 supplemental report artifacts')


if __name__=='__main__':main()
