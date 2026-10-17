#!/usr/bin/python3
"""Write the measured diagnostic report, distinguishing frontend/model gains."""
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'results/mentor_benchmark'
V1=RESULTS/'nclt_diagnostic_v1';V2=RESULTS/'nclt_frontend_diagnostic_v2'
HOME=RESULTS/'home_increment_memory_v1';CROSS=RESULTS/'cross_scene_increment_memory_v1'


def csv_rows(path):return list(csv.DictReader(path.open()))


def run():
    audit=json.loads((V1/'input_audit.json').read_text())
    v1=csv_rows(V1/'common_grid_metrics.csv');v2=csv_rows(V2/'common_grid_metrics.csv')
    summary=csv_rows(HOME/'scene_summary.csv');selections=json.loads((HOME/'selection.json').read_text())
    yaw=json.loads((V2/'yaw_input_agreement.json').read_text())['statistics']
    lines=['# NCLT前端诊断与增量积分偏差实验（2026-10-05）','',
        '本轮先诊断基础SLAM，再检查残差积分。原数据、旧模型与既有评分保留，配置对照使用独立目录；本报告中的新NCLT结果属于已检查公共序列上的开发诊断，不能作为新的独立泛化测试。用户目前没有自采设备，本轮继续使用公开机器人数据。','',
        '## 1. 输入审计','',
        f'逐条核查{audit["counts"]["/scan"]}帧激光、{audit["counts"]["/odom"]}帧融合里程计和{audit["counts"]["/imu_combined"]}帧IMU，原始数据与bag时间戳最大差为0，量程序列最大浮点误差{audit["maximum_decode_errors"]["range_quantization_m"]:.3g}米；各话题无非递增时间戳。角度起点−135°、步长0.25°、1081束，以及量程5mm量化和−100m偏移，与作者例程一致。','',
        '外参和姿态转换核查通过。原世界及车体为NED/前右下；转换矩阵F=diag(1,−1,−1)。车辆姿态使用F·R·F，激光保留原传感器角度定义，因此外参为F·R_body_laser=I；IMU向量已乘F，外参为F·I·F=I。对应激光平移(0.28,0,0.44)m、IMU平移(−0.11,0.18,0.71)m。[作者论文，第3节、表4与第7.4节](https://robots.engin.umich.edu/nclt/nclt.pdf)。','',
        '作者公开格式只给整帧扫描时间戳。旧转换设time_increment=0.025/1081，因此末束被Cartographer置于头时间戳之后约24.98ms。这是未验证的逐束时间假设，不能仅从40Hz帧率确定扫描起点/终点和有效扫描时长。零逐束时间作为敏感性条件，不宣称物理上同时采样。时间戳逐条一致也不能证明传感器硬件时钟误差为零。','',
        f'有效激光束占{100*audit["valid_return_fraction"]:.1f}%，单帧中位数为493束；10–30m回波仅占全部束{100*audit["valid_returns_between_10_and_30_m_fraction_of_all_beams"]:.1f}%。里程计俯仰角1%/99%分位为−10.72°/+8.00°。这些是场景和运动诊断，未将其直接当作失准原因。','',
        '审计记录：`results/mentor_benchmark/nclt_diagnostic_v1/input_audit.json`。作者论文中的时间戳“四字节”描述与作者`struct.unpack("<Q", ...)`例程不符；实际二进制及本次转换采用八字节时间戳。','',
        '## 2. 时间补偿、回环和量程的完整序列对照','',
        '第一阶段在评分前保存五组方案。全部输入bag完整回放，取各节点轨迹共同覆盖的1021.6秒，使用同一10Hz评分网格、同一真值掩码和共同评分起点处的各轨迹SE2原点对齐，10217个样本。节点轨迹经过插值，用于离线诊断；它不是在线发布轨迹。','',
        '| 条件 | 局部节点APE(m) | 最终节点APE(m) | 最终1秒RPE(m) | 跨子图约束数 | 离线总耗时(s) | 峰值RSS(MiB) |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    names={'original10_loop':'原逐束时间，10m，回环开','original10_noloop':'原逐束时间，10m，回环关',
        'single_stamp10_loop':'零逐束时间，10m，回环开','single_stamp10_noloop':'零逐束时间，10m，回环关',
        'single_stamp30_noloop':'零逐束时间，30m，回环关'}
    for name,label in names.items():
        a=next(r for r in v1 if r['condition']==name and r['trajectory']=='local')
        b=next(r for r in v1 if r['condition']==name and r['trajectory']=='global')
        lines.append(f'| {label} | {float(a["ape_m"]):.2f} | {float(b["ape_m"]):.2f} | {float(b["rpe_1s_m"]):.4f} | {b["inter_submap_constraints"]} | {float(b["wall_s"]):.2f} | {float(b["peak_rss_mib"]):.1f} |')
    lines+=['','原配置局部误差在17.7秒首次超过1m、20.7秒超过5m，表明失准已经出现在前端。关闭回环后局部轨迹完全一致，最终APE从152.82升至254.25m；当前数据不支持“回环是唯一原因”。零逐束时间及单独增大量程也没有恢复可靠轨迹。约束数不是回环精确率/召回率。','',
        '上述耗时涵盖离线前后端和最终优化，部分诊断分析同时运行，不是隔离性能测试，也不能据此直接声称实时性。不同网格起点使这里152.82m与旧完整输出153.00m略有不同，旧在线315.55m也不是同一种轨迹。','',
        '![第一阶段单因素对照](../results/mentor_benchmark/nclt_diagnostic_v1/factor_contrasts.png)','',
        '## 3. 前端角度初值对照','',
        '第二阶段是第一阶段完成后的自适应诊断。阅读实际Cartographer源码发现：在线相关扫描匹配先改变initial_ceres_pose，Ceres的旋转先验围绕该初值。因此可靠里程计外推的角度仍可能被相关匹配的搜索初值带偏。新方案仅把use_online_correlative_scan_matching由true改为false，保留旧逐束时间、10m量程、里程计、权重与其他设置，分别对照回环开/关；没有按真值搜索权重。','',
        '| 条件 | 局部节点APE(m) | 最终节点APE(m) | 最终1秒RPE(m) | 跨子图约束数 | 离线总耗时(s) |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name,label in [('ceres10_loop','Ceres初值直接来自外推，回环开'),('ceres10_noloop','Ceres初值直接来自外推，回环关')]:
        a=next(r for r in v2 if r['condition']==name and r['trajectory']=='local')
        b=next(r for r in v2 if r['condition']==name and r['trajectory']=='global')
        lines.append(f'| {label} | {float(a["ape_m"]):.2f} | {float(b["ape_m"]):.2f} | {float(b["rpe_1s_m"]):.4f} | {b["inter_submap_constraints"]} | {float(b["wall_s"]):.2f} |')
    lines+=['','关闭该前端环节后，局部APE由306.99降至22.49m，最终APE由152.82降至15.64m；这项改善来自基础SLAM配置。共同网格下公开融合里程计APE为16.17m。两个条件的pbstream最终节点与Cartographer自带ROS导出工具一致，最大差2.38×10⁻⁷；比较四元数时已处理q和−q等价。','',
        f'不用真值的输入一致性检查：相关匹配开启/关闭的局部航向角速度绝对值P99分别为{yaw["correlative"]["absolute_rate_p99_deg_s"]:.2f}/{yaw["ceres_only"]["absolute_rate_p99_deg_s"]:.2f}°/s，融合里程计为{yaw["provided_ekf"]["absolute_rate_p99_deg_s"]:.2f}°/s；与里程计角速度的相关系数从{yaw["correlative"]["rate_correlation_with_ekf"]:.3f}升至{yaw["ceres_only"]["rate_correlation_with_ekf"]:.3f}。这支持角度搜索放大匹配歧义的解释；融合里程计已使用IMU/FOG，不能将IMU一致性当作独立真值验证。','',
        '局部误差仍有22m，不宣称问题完全消失。这是一个会话上的配置诊断，没有证明该配置在所有场景、机器人或LiDAR上都更优。新配置未覆盖旧基准。','',
        '![前端配置对照](../results/mentor_benchmark/nclt_frontend_diagnostic_v2/frontend_contrasts.png)','',
        '![前端与输入航向一致性](../results/mentor_benchmark/nclt_frontend_diagnostic_v2/yaw_input_agreement.png)','',
        '## 4. 1倍速在线复验与冻结模型复核','']
    online=V2/'online_common_time_metrics.csv'
    if online.exists():
        common=csv_rows(online);individual=csv_rows(V2/'online_individual_metrics.csv')
        lines+=['新配置以--rate 1.0完整回放，未加载学习模型，也没有纠正位姿反馈进SLAM。下表对两个不同ROS执行的原始发布流取共同时间网格与真值掩码；不是把两个不同执行当作数值一致性测试。','',
            '| 在线配置 | 共同网格APE(m) | 1秒RPE(m) | 航向APE(°) | 样本 |',
            '| --- | ---: | ---: | ---: | ---: |']
        for r in common:lines.append(f'| {r["configuration"]} | {float(r["ape_m"]):.2f} | {float(r["rpe_1s_m"]):.4f} | {float(r["yaw_ape_deg"]):.2f} | {r["common_valid_samples"]} |')
        n=next(r for r in individual if r['configuration']=='ceres_only')
        lines+=['',f'各自原采样流的评分保留在online_individual_metrics.csv：新配置APE为{float(n["ape_m"]):.2f}m、1秒RPE为{float(n["rpe_1s_m"]):.4f}m，输入可用率{100*float(n["input_available_fraction"]):.2f}%，SLAM进程峰值RSS为{float(n["slam_peak_rss_mib"]):.1f}MiB。graph节点附近的补充APE在online_graph_node_ape.csv。','',
            f'--rate 1.0命令的观测墙钟耗时为{float(n["wall_s"]):.1f}s，bag名义时长1026.0s。完整回放、有效输入覆盖和CPU指标支持主机运行可行性，但本轮没有精确的消息到达→最终发布端到端延迟记录，不能把1倍速参数或100%可用率等同于严格无延迟保证。','',
            '指标解释：Cartographer的local trajectory latency表示相邻更新间隔，不是逐帧计算耗时；real-time ratio是传感器时间/计算时间比例；constraint queue length包含批次内已算结果，不是未处理扫描积压。记录了这些指标，但没有把它们变成端到端延迟或闭环准确率。','',
            '![在线原轨迹对照](../results/mentor_benchmark/nclt_frontend_diagnostic_v2/online_frontend_comparison.png)','']
    else:lines+=['1倍速完整在线复验正在执行，尚未写入评分；不得用离线节点改善替代在线结果。','']
    external=RESULTS/'cross_scene_causal_frontend_external_v2/results.json'
    stress_home=RESULTS/'home_frozen_transfer_nclt_frontend_v2/results.json'
    stress_cross=RESULTS/'cross_scene_gate_stress_nclt_frontend_v2/results.json'
    if external.exists():
        evaluated=json.loads(external.read_text())
        lines+=['冻结原Home/Office训练、Cafe验证的权重与参数，按预声明NCLT外部清单重新评分新前端输入，没有重训或调整alpha/阈值。全部六个不确定性门控模型原gate_alpha=0，仍原样保留；基础前端改善不能归功于它们。传统/学习基线的结果单独放在cross_scene_causal_frontend_external_v2，旧26行外部评分保留。','']
    if stress_home.exists() and stress_cross.exists():
        h=json.loads(stress_home.read_text());c=json.loads(stress_cross.read_text())
        lines+=['另将原验证集选出的两个Home部署权重迁移到新NCLT输入，并对六个场景隔离权重重复同alpha的九种输入条件。','',
            '| 冻结Home模型，干净输入 | 原Cartographer APE(m) | 无门控APE(m) | 规则门控APE(m) | 不确定性门控APE(m) | 不确定性回退比例 |',
            '| --- | ---: | ---: | ---: | ---: | ---: |']
        for config in ['full20','relative32']:
            clean=[r for r in h if r['config']==config and r['anomaly']=='clean']
            modes={r['gate_mode']:r for r in clean};u=modes['uncertainty']
            lines.append(f'| {config} | {u["raw_clean_ape_m"]:.3f} | {modes["none"]["ape_m"]["rmse"]:.3f} | {modes["rule"]["ape_m"]["rmse"]:.3f} | {u["ape_m"]["rmse"]:.3f} | {100*u["fallback_fraction_all"]:.1f}% |')
        incremental=next(r for r in h if r['config']=='relative32' and r['anomaly']=='clean' and r['gate_mode']=='none')
        gated=next(r for r in h if r['config']=='relative32' and r['anomaly']=='clean' and r['gate_mode']=='uncertainty')
        lines+=['',f'原Home32维模型在恢复后的输入上仍有危险的积分漂移：相同alpha=1时，无门控APE达到{incremental["ape_m"]["rmse"]:.2f}m；不确定性门控后约{gated["ape_m"]["rmse"]:.3f}m，回退{100*gated["fallback_fraction_all"]:.1f}%。门控降低破坏，但有害修正（误差增加>5cm）的比例仍为{100*gated["harmful_fraction_vs_corrupted_raw"]:.2f}%，不是零风险或泛化成功。','',
            f'本轮新增{len(h)+len(c)}条模式/异常/种子结果，仍只有同一条NCLT独立序列；异常施加在捕获的后处理输入，未重跑受扰前端。高回退或关闭修正不能当作成功泛化。','']
    lines+=['## 5. 32维增量残差的积分偏差','',
        '新实现一个可选指数记忆：先按原SE2增量规则积分，再把相对原Cartographer的xy偏移及最短航向偏移乘exp(−Δt/τ)。τ=None完全保持旧实现。有限τ能够限制恒定偏差的累积，但不保证轨迹准确。门控拒绝、输入缺失及恢复首帧仍精确返回原轨迹，并清空累积偏移。','',
        '权重、归一化、物理裁剪、门控阈值和alpha冻结。候选τ为无限、0.5/1/2/5/10/30秒，只在验证集选择，结果先写selection.json，再读取测试参考。Home三种子选择：seed11保留无限、seed22选择30s、seed33因原alpha=0保留无限。场景隔离模型三种子原alpha均为0，全部保留无限。','',
        '原验证集已经选中的Home seed22使用30s记忆时，验证APE由0.0947495降至0.0946878m，改善仅约0.062mm，没有统计显著性证据。下表来自同一实际捕获输入流的复验，不是新的主动ROS纠正发布测试。','',
        '| 场景 | 序列数 | 原Cartographer APE(m) | 原门控积分APE(m) | 30s记忆APE(m) | 原/记忆1秒RPE(m) |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for scene in ['home','office','cafe','corridor','nclt']:
        a=next(r for r in summary if r['scene']==scene and r['seed']=='22' and r['method']=='original')
        b=next(r for r in summary if r['scene']==scene and r['seed']=='22' and r['method']=='selected_memory')
        lines.append(f'| {scene} | {a["sequences"]} | {float(a["raw_macro_ape_m"]):.4f} | {float(a["macro_ape_m"]):.4f} | {float(b["macro_ape_m"]):.4f} | {float(a["macro_rpe_1s_m"]):.4f}/{float(b["macro_rpe_1s_m"]):.4f} |')
    lines+=['','NCLT一行使用旧前端输入，与第4节的新前端复验分开。Home/Office/Cafe/Corridor虽略有缓解，跨场景仍普遍弱于原轨迹；新记忆没有解决泛化问题，因此未替换正式部署选择。','',
        '验证集诊断发现：局部残差MSE改善不等于积分后APE改善。Home seed22的归一化残差RMSE为零修正的0.970倍，残差误差平均约−0.000730m/步（前向）、−0.000311m/步（侧向）和−0.0271°/步（航向）。持续误差经旋转传播及门控重置影响轨迹；这些均值本身不证明某一轴是唯一原因。固定alpha=1且无门控的验证诊断中，seed22无限积分APE为0.2755m，5s记忆降到0.0803m；这条强制曲线不参与参数选择，也未据此在测试集强制开启修正。','',
        '![仅验证集的记忆时间常数对照](../results/mentor_benchmark/home_increment_memory_v1/validation_memory_curves.png)','',
        '## 6. 验证与复现','',
        '31项检查通过，包括恒定偏差收敛、最短航向偏移、可变Δt、缺失后精确回退、批量/逐帧一致、门控对照读取记忆参数，以及pbstream字段/时间纪元/截断校验。Home记忆实验48组、场景隔离实验12组实际捕获流的批量/逐帧推理一致性检查通过。','',
        '保留一个实验候选candidate_seed22_memory30.pt及NumPy便携包。三条Home实际记录流的NumPy与Torch最大轨迹差约5.25×10⁻⁹，随机100次前向最大误差7.45×10⁻⁸；已验证计算一致性，尚未用该候选主动重跑ROS发布，也没有真实边缘设备实测。旧便携包与旧模型未替换。','',
        '完整命令见[实验复现说明](实验复现说明.md)。第一阶段目录nclt_diagnostic_v1，第二阶段nclt_frontend_diagnostic_v2，记忆实验目录home_increment_memory_v1/cross_scene_increment_memory_v1。每个条件有输入软链接、配置快照、pbstream、节点导出、TUM和资源文件；数值表、PNG/PDF和源文件校验可复核。','',
        '下一轮应使用已经验证的NCLT前端配置，检查学习模型在正常基础轨迹上的收益，再考虑匹配质量、里程计可靠性等输入特征和轨迹级损失。新序列应在进一步调参之前明确留出。设备限制下，自采与真实边缘测试仍未完成；也没有闭环真值标签或纠正轨迹反馈进SLAM的系统收益证明。','']
    (ROOT/'docs/NCLT诊断与增量积分偏差实验.md').write_text('\n'.join(lines))
    files=['causal_stream.py','mentor_benchmark.py','portable_residual.py','gate_stress.py',
           'tools/benchmark_increment_memory.py','tools/online_ros_benchmark.py',
           'tools/run_nclt_frontend_diagnostics.py','tools/report_nclt_frontend_diagnostics.py',
           'tools/report_nclt_online_frontend.py','tools/analyze_nclt_frontend_yaw.py','tools/write_nclt_diagnostic_report.py']
    hashes={}
    for name in files:
        path=ROOT/name;hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest()
        (V2/('source_'+path.name)).write_bytes(path.read_bytes())
    (V2/'report_provenance.json').write_text(json.dumps(dict(source_sha256=hashes,
        unit_checks=31,recorded_stream_equivalence_checks=60,portable_sequence_equivalence_checks=3,
        diagnostic_slam_conditions=7,independent_nclt_sessions=1,
        online_scoring_complete=online.exists()),indent=2)+'\n')
    print('Updated docs/NCLT诊断与增量积分偏差实验.md')


if __name__=='__main__':run()
