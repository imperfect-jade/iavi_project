# 双目结构光三维重建与主动视角规划

[English](instruction.md) · [实验指导（中文）](实验指导.md) · [Experiment guide (English)](instruction.md)

本仓库实现一套实验性桌面物体扫描系统：固定投影仪投射随机密集点阵，两台随机械臂移动的相机采集图像；双目重建得到单视角点云，利用静止 ChArUco 板和转台轴标定将不同视角对齐。另有基于 PointNet 风格收益预测的下一视角选择、ModelNet 几何仿真、训练和离线回放。实验目标、器材、完整流程和命令见[实验指导](实验指导.md)。

> 当前代码仅支持 `dense_dots` 采集与重建。仿真覆盖率和软件回放不能直接作为实物扫描效果证明。

## 目录

| 路径 | 职责 | 说明 |
| --- | --- | --- |
| [`code/`](code/README.md) | 相机采集、双目标定、单视角重建、点云融合与颜色清理 | 基础视觉命令 |
| [`arm/`](arm/README.md) | 机械臂控制、固定 ChArUco 板生成与定位 | 两个预设相机位姿 |
| [`turntable/`](turntable/README.md) | 转台控制、四面 ArUco 标靶与转轴标定 | 12 个 30° 候选角度 |
| [`system/`](system/README.md) | 固定角度序列的采集、重建与融合调度 | 完整 24 视角扫描 |
| [`server/`](server/README.md) | 数据集、ModelNet 仿真、模型训练和评估 | 可在 Linux GPU 环境运行 |
| [`PointNet/`](PointNet/README.md) | 实机主动扫描、单次预测与历史会话回放 | 加载训练好的模型 |
| `configs/` | 色卡角点示例配置 | `color_chart_corners.json` |
| `tests/` 与各模块 `tests/` | 单元及流程测试 | `python -m pytest tests arm/tests turntable/tests system/tests PointNet/tests server/tests -q` |

`raw/`、`processed/`、`calibration/`、`models/` 及训练输出存放本地数据或生成物，已在 `.gitignore` 中排除。克隆仓库后须自行准备相机标定、轴标定、扫描数据及模型权重；README 的示例路径不代表仓库自带这些文件。

## 环境与动作定义

本地建议 Python 3.11，并先安装与相机兼容的 Basler pylon。仓库使用 ChArUco/ArUco，OpenCV 需安装 `opencv-contrib-python`；不要在同一环境中同时安装 `opencv-python`。在项目根目录执行：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install numpy opencv-contrib-python scipy pypylon pyserial pyyaml pillow open3d pytest
```

训练与 PointNet 推理还需 PyTorch。按操作系统与 CPU/CUDA 环境从 [PyTorch 官方安装页](https://pytorch.org/get-started/locally/)选择对应命令；CPU 环境可执行：

```powershell
.\.venv\Scripts\python.exe -m pip install torch
```

本仓库未提供统一锁定的依赖文件；训练配置中的 `runtime.device` 要与实际 PyTorch 设备一致。以下示例中的 `python` 指已激活虚拟环境的 Python，也可替换为 `.\.venv\Scripts\python.exe`。

在本目录运行命令。设备示例使用转台 `COM6`、机械臂 `COM7`，需换成实际串口。`arm_pose_id=0` 为上位姿，`1` 为下位姿；动作 `action_id = arm_pose_id × 12 + turntable_deg / 30`，共 24 个动作。几何长度单位为毫米。

## 最短流程

```powershell
# 1. 采集并分别标定下、上位姿双目相机；详见 code/README.md
python code\capture.py calibration --session-id session_01
python code\calibrate_stereo.py --pose lower --train raw\calibration\session_01 --validation raw\calibration\validation_01 --square-mm 12

# 2. 在下位姿拍摄转台标靶与静止板，求本次转轴
python turntable\capture_axis.py --port COM6 --session-id axis_01
python turntable\calibrate_axis.py --images raw\turntable_axis\axis_01

# 3. 固定 24 视角扫描；需先完成上下位姿标定与独立验证
python system\scan_session.py --port COM6 --arm-port COM7 --start-angle 0 --end-angle 330 --step-angle 30 --object-id object_001 --session-id session_01 --fusion-axis-images raw\turntable_axis\axis_01
```

第一步的 `validation_01` 需另行采集，完整上下位姿命令见 [`code/README.md`](code/README.md)。采用学习策略的逐次扫描见 [`PointNet/README.md`](PointNet/README.md)；训练见 [`server/README.md`](server/README.md)。

## 输出与限制

原始照片位于 `raw/<物体>/<会话>/view_###/repeat_###/`；单视角重建位于 `processed/<物体>/<会话>/view_###/repeat_###/`；最终融合点云通常位于 `processed/fused/<物体>/<会话>/all_views_axis_calibrated/fused.ply`。固定板在转轴拍摄与物体扫描期间须保持相对转台静止。运动控制发送命令后按设定时间等待，串口写入不构成实际到位反馈。
