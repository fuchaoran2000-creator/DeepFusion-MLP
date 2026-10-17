# DeepFusion-MLP 实验复现

本仓库包含训练、推理、传统基线、消融、跨场景、故障、资源评测和ROS回放代码，以及固定配置、检查点与数值复现记录。

## 最快验证论文主结果

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-replay.txt
python tools/reproduce_paper.py
```

只需NumPy，即可核对表4三条Home配对输入的特征、回退、输出和APE/RPE，并重算表7迁移汇总。输出为`results/public_reproduction/verification.json`。

Home平均APE从0.16314284 m变为0.10138135 m，改善37.9%；1 s RPE基本不变。37.9%仅属于三个同场景Home测试序列，不能写成多场景提升。Office/Cafe/Corridor迁移退化，NCLT近乎回退。

## 重新训练和完整实验

安装`requirements.txt`后，执行`python -m unittest discover -s tests -v`，再运行`python tools/reproduce_paper.py --train-smoke`检查两轮训练流程。完整训练命令见英文[README](README.md)及[实验命令](docs/EXPERIMENTS.md)。`--transfer`会用冻结C20在已有捕获输入上重新推理，不是重新跑SLAM。

`config/recorded`使用仓库内的数值输入；`config`中的原始实验清单使用`data/prepared`，供下载原始数据、准备ROS输入并重新回放后使用。所有命令在仓库根目录运行。原机器绝对路径已从执行配置中移除，原检查点/归档可能保留历史来源路径作为元数据。

权重、阈值及结果所属协议必须对应。P20特征消融不证明C20各特征必要，Q44故障实验不隔离C20门控效果，固定Home迁移与重新训练的场景轮换属于不同研究。没有自采、实体边缘设备或受控更换单一硬件的验证。

ROS环境和外部SLAM编译见[ROS.md](docs/ROS.md)，数据与参考来源见[DATA.md](docs/DATA.md)。原始bag、图像、扫描和论文文档不包含在仓库中。
