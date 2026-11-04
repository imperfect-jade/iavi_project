# 视角收益模型、仿真与训练 `server/`

[English](README.en.md) · [实验指导](../实验指导.md)


本模块训练 24 动作的 PointNet 风格候选收益模型。`arm_pose_id=0/1` 分别为上/下位姿，`action_id = arm_pose_id * 12 + azimuth_index`，每个位姿 12 个转台角度。模型输入必须只含已扫描点云；完整参考点云只用于离线标签和评估。ModelNet 仿真是几何可见性近似，不等同于实机结构光成像。

## 文件

| 文件 | 作用 |
| --- | --- |
| `model.py` | 基础/V2 收益预测网络、掩码损失与动作选择 |
| `data.py` | NPZ 样本格式验证及按物体划分检查 |
| `build_dataset.py` | 从已对齐的逐动作物体点云生成监督状态、覆盖表与训练集 |
| `train.py` | 读取 YAML 训练、验证、保存最佳模型并支持续训 |
| `predict_next_view.py` | 对当前已观测 `(N,3)` 毫米点云作一次动作预测；不控制硬件 |
| `simulate_modelnet.py` | 从 ModelNet OFF 网格生成 24 个虚拟视角点云 |
| `evaluate_sim.py` | 在仿真测试物体上回放学习、随机及基线策略 |
| `make_smoke_data.py` | 生成仅用于检查训练管线的合成样本 |
| `run_train_200.sh` | 生成 200 个多类别 CAD 物体数据、训练 V2 并评估的 Linux 脚本 |
| `sim_config.json` | 虚拟相机、投影仪、物体及采样参数示例 |
| `config_smoke.yaml`, `config_sim.yaml`, `config_sim_v2.yaml`, `config_real.yaml` | 对应试跑、基础仿真、V2 仿真和实测数据的训练配置模板 |
| `tests/` | 数据格式、仿真、训练模型、推理与评估测试 |

## 环境与试跑

在 `server/` 内执行以下命令。需 `numpy scipy pyyaml torch pytest`；仿真另需 `open3d`。按机器的 CUDA/CPU 环境安装 PyTorch，`config_*.yaml` 中的 `runtime.device` 需与之对应。

```bash
cd server
python make_smoke_data.py
python train.py --config config_smoke.yaml
python -m pytest tests -q
```

试跑输出在 `runs/`，只验证数据加载和训练流程，不可作为模型效果报告。

## ModelNet 仿真与真实数据

先自行取得 ModelNet40 并解压，目录需有 `<类别>/train/*.off` 与 `<类别>/test/*.off`。小规模示例：

```bash
python simulate_modelnet.py --modelnet-root /path/to/ModelNet40 --output sim_data --categories bottle bowl --max-train 10 --max-val 3 --max-test 3
python train.py --config config_sim.yaml
python evaluate_sim.py --sim-root sim_data --config config_sim.yaml
```

运行前核对 `config_sim.yaml` 的 `data.train`、`data.val`、`runtime.output_dir` 与实际生成位置。完整多类别实验在 Linux GPU 环境使用 `MODELNET_ROOT=/path/to/ModelNet40 bash run_train_200.sh`；脚本检查 CUDA，并固定为 160 训练、20 验证、20 测试物体。大规模仿真耗时且占用磁盘。

真实数据需每个物体 24 个已对齐至 `object_at_turntable_zero` 的 `(P,3)` `.npy` 点云，单位毫米。清单列出 `objects[].views` 的 24 个相对路径，并提供 `unit`、`coordinate_frame`、`action_count`、`initial_views`、`tolerance_mm` 等元数据；具体字段验证见 `build_dataset.py`。相机坐标系里的 `cloud_left.ply` 不能直接使用。

```bash
python build_dataset.py --manifest dataset_manifest.json --output datasets
python train.py --config config_real.yaml
python train.py --config config_real.yaml --resume
python predict_next_view.py --config config_real.yaml --points current_observed.npy --visited 0 6
```

训练输出含 `best_model.pt`、`last.pt`、配置与历史记录；部署使用与配置匹配的最佳模型。`config_real.yaml` 中工作区中心和尺度为待测模板值，训练前须替换。训练/验证/测试按物体划分。详见 [`../PointNet/README.md`](../PointNet/README.md) 中的本地执行与回放。
