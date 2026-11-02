# AutoDL 视角规划训练

本目录与实验室电脑上的 `project/code/` 采集、重建程序分开，包含候选视角收益预测网络、训练数据生成器、ModelNet 几何仿真器和仿真回放评估器。`smoke_data/` 是检查程序流程的合成样本，不能当作实验结果。

## 一、在 AutoDL 上准备环境

先在 AutoDL 终端创建项目目录：

```bash
mkdir -p /root/autodl-tmp/nbv_project
```

再从**本地 PowerShell** 上传整个 `server/` 目录；替换实例的 SSH 端口和主机名：

```powershell
cd 'D:\vscode\visual information\project'
$nbvPort = 12345  # 替换为 SSH 端口
$nbvHost = 'your-instance-host'  # 替换为实例主机名
scp -P $nbvPort -r .\server "root@${nbvHost}:/root/autodl-tmp/nbv_project/"
```

使用已验证的 PyTorch 2.12.1+cu130 镜像，不必重装 torch。在 **AutoDL Linux 终端**执行：

```bash
cd /root/autodl-tmp/nbv_project/server
python -m pip install numpy pyyaml pytest
python -m pip check
python -m pytest tests -q
python make_smoke_data.py
python train.py --config config_smoke.yaml
```

试跑成功时会输出两轮训练记录，并生成 `runs/smoke_24/best_model.pt`、`last.pt`、`model_config.json` 和 `history.jsonl`。24 动作的输出与旧版 36 动作结果分开。`best_model.pt` 只含部署所需的模型 `state_dict`；`last.pt` 含优化器和随机状态，仅用于从**可信的本机训练文件**续训，不要加载来源不明的完整 checkpoint。

## 二、用 ModelNet40 仿真训练

从 [Princeton 官方页面](https://modelnet.cs.princeton.edu/download.html)自行下载 ModelNet40，阅读其学术用途条款，并解压到 AutoDL 数据盘。目录应包含 `类别/train/*.off` 和 `类别/test/*.off`。选择与实验室实物形状相近的类别；命令中的 `bottle`、`bowl` 只有在解压目录确实包含这些类别时才能使用。

仿真器使用 [Open3D 射线投射](https://www.open3d.org/docs/latest/tutorial/geometry/ray_casting.html)求网格表面的首次交点。物体绕转盘轴转动 12 个角度，两台虚拟相机组成刚性双目组件并取两个固定高度；每个动作只保留**左右相机和固定投影仪共同可见**的表面，再把点变换回物体坐标系。因此每个 CAD 物体生成 24 份单视角点云。

在 AutoDL 的现有 Python 环境安装 Open3D 和 SciPy。[Open3D 0.20.0](https://pypi.org/project/open3d/0.20.0/)提供 Python 3.12 的 Linux wheel；安装前仍应确认实例的系统库与 wheel 兼容。先用少量物体试跑：

```bash
cd /root/autodl-tmp/nbv_project/server
python -m pip install open3d scipy
python simulate_modelnet.py --modelnet-root /root/autodl-tmp/ModelNet40 \
  --output sim_data --categories bottle bowl --max-train 10 --max-val 3 --max-test 3
python train.py --config config_sim.yaml
python evaluate_sim.py --sim-root sim_data --config config_sim.yaml
```

上述上限为训练 10 个、验证 3 个、测试 3 个 CAD 物体。`sim_config.json` 控制模型尺寸、两个相机高度、双目基线、固定投影仪位置、视场角、噪声、丢点率和覆盖容差；**这些目前都是示例参数，并非设备实测值**。脚本输出每个物体的 24 份 `.npy` 点云、清单、独立参考与覆盖表，以及 `train.npz`、`val.npz`、`test.npz`。

物体按 CAD 模型划分：ModelNet 原有的 `test` 模型只进入测试集，验证模型从原有 `train` 模型中划出。`evaluate_sim.py` 在仿真测试模型上输出学习、随机、固定均匀和参考贪心策略的覆盖率曲线与 AUC。仿真只近似几何可见性，**没有模拟格雷码、相移解码、材质反射、真实标定误差或机械重复性**；仿真结果不能证明真实扫描效果，实机结论仍需独立真实物体验证。

## 三、用真实扫描点云制作训练集

在 AutoDL 安装 SciPy：`python -m pip install scipy`。每个物体、每个动作准备一份非空的 `(P,3)` XYZ `.npy` 点云，单位为毫米，并已对齐到 `object_at_turntable_zero` 坐标系。**不能直接把 `cloud_left.ply` 作为输入**，因为它仍在相机坐标系中。导出前应去除背景并检查各视角配准质量。数据生成器会拒绝缺失视角；失败的扫描不能简单标为零收益。

创建 JSON 清单。`views` 必须依照 `view_id = arm_pose_id*12 + azimuth_index` 的动作顺序列出 **24 个相对路径**：上姿态为 0～11，下姿态为 12～23。顶层字段包括 `unit: "mm"`、`coordinate_frame: "object_at_turntable_zero"`、`action_count: 24`、`initial_views: [0,6]`、`tolerance_mm`、`voxel_mm`、`point_count: 2048`、`reference_max_points: 20000`、`states_per_object: 200`、`seed: 42` 和 `objects`。只有初始扫描位置确实固定为上姿态 0°、180° 时，才使用 `[0,6]`。

每个 `objects` 条目包含 `object_id`、`split`（`train`、`val` 或 `test`）和 `views`。同一物体的所有会话及变体应留在同一数据划分。体素尺寸与覆盖容差须依据实测重复误差确定，不能直接照抄下面的示例值。旧 36 动作数据与 checkpoint 不兼容，须重新生成。

```json
{
  "unit": "mm", "coordinate_frame": "object_at_turntable_zero",
  "action_count": 24, "initial_views": [0, 6],
  "tolerance_mm": 2.0, "voxel_mm": 1.0,
  "point_count": 2048, "reference_max_points": 20000,
  "states_per_object": 200, "seed": 42,
  "objects": [
    {"object_id": "object_001", "split": "train",
     "views": ["clouds/object_001/view_000.npy", "clouds/object_001/view_001.npy"]}
  ]
}
```

示例为便于阅读只写了两个 `views` 路径，**实际运行前必须补齐全部 24 个**。生成器从已访问视角构造随机及扰动贪心状态，集合大小为 2～9；输出 `train.npz`、`val.npz`、可选的 `test.npz`、`metadata.json`，以及独立存放参考点云和布尔覆盖表的 `references/<object_id>.npz`。参考只用于制作标签或离线评估，不能进入策略输入。生成器拒绝覆盖非空输出目录。

```bash
cd /root/autodl-tmp/nbv_project/server
python build_dataset.py --manifest dataset_manifest.json --output datasets
```

按下方数据格式制作文件后，将实测得到的**固定**工作区中心和尺度填入 `config_real.yaml`，再运行：

```bash
python train.py --config config_real.yaml
```

中断后可在相同配置和数据上加 `--resume` 续训；若配置或数据文件的 SHA-256 已改变，训练器会拒绝续训。数据和输出应保存在数据盘；长时间训练建议在 `tmux` 中运行。

## 四、NPZ 训练样本格式

每个数据划分对应一个压缩或未压缩的 `.npz` 文件。各数组的第一维 `N` 是状态样本数：

| 字段 | 形状／类型 | 含义 |
| --- | --- | --- |
| `points` | `(N,2048,3)` float32 | 当前**已扫描**视角融合后的物体坐标点云，单位毫米；不足时补零 |
| `point_mask` | `(N,2048)` bool | 哪些点是真实观测点 |
| `visited_mask` | `(N,24)` bool | 已扫描动作编号 |
| `available_mask` | `(N,24)` bool | 当前可执行动作编号 |
| `gain_target` | `(N,24)` float32，取值 0～1 | 各动作离线计算的新增覆盖比例 |
| `current_action_id` | `(N,)` int64 | 当前物理动作编号，必须已访问 |
| `used_scans` | `(N,)` float32 | 已完成的扫描次数，包含初始扫描 |
| `object_id` | `(N,)` Unicode 字符串 | 物体编号；按物体划分数据集 |

数据加载器自动以 `available_mask & ~visited_mask` 生成损失掩膜。状态输入不得包含完整参考点云或尚未访问视角的点云。收益标签按《实验方案》3.9 节，在冻结的动作表和覆盖容差下离线生成。训练与验证物体编号必须互不重叠；生成数据时应记录动作表、图案协议、位姿和重建版本。

当前训练器根据**验证集上带掩膜的 Smooth L1 损失**选择最佳 checkpoint。仿真回放评估由 `evaluate_sim.py` 执行；真实扫描数据的回放评估仍需接入真实逐视角点云。合成试跑的低损失不代表视角规划有效。
