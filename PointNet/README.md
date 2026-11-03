# PointNet 主动扫描

此目录包含独立的主动扫描流水线、历史数据回放和单次选角工具。新流水线复用 `code/`、`arm/`、`turntable/` 中的设备和几何程序；`system/` 的 24 视角流水线仍可按原入口运行。

## 单次选角工具

`predict.py` 从融合结果的 `fusion.json` 和 `aligned/*.ply` 中，**只读取 `--visited` 指定的视角**，生成模型所需的物体坐标系点云，并通过 `policy.py` 加载已训练模型。它输出 `action_id`、`arm_pose_id`、`turntable_deg`、候选动作分数和最高分。动作编号为 `arm_pose_id * 12 + turntable_deg / 30`。

首次调用需要 `--visited 0 6`，与训练时的初始视角一致。程序保存 `object_frame.json`，其中包含转轴、相机 0° 定义的水平轴、桌面高度、初始视角确定的 XY 中心和尺度。以后调用同一 `--output-dir`，可依次增加已完成的视角，坐标变换保持固定。`observed.npy`、`prediction.json` 等文件都写在该输出目录。

这里要求输入已经经过标定、重建和逐视角转台归零对齐。单个 `processed/<object>/<session>/view_.../cloud_left.ply` 仍在相机坐标系，不能直接输入。`fused.ply` 已含所有视角，不能用于逐步选视。

## 本地运行示例（PowerShell）

在项目根目录执行：

```powershell
cd "D:\vscode\visual information\project"
..\.venv\Scripts\python.exe PointNet\predict.py `
  --fusion-dir processed\fused\object_002\session_01\all_views_axis_calibrated `
  --raw-session raw\object_002\session_01 `
  --visited 0 6 `
  --table-height-axis-mm -95 `
  --output-dir PointNet\outputs\object_002\session_01
```

完成动作 15 的扫描、重建和对齐后：

```powershell
..\.venv\Scripts\python.exe PointNet\predict.py `
  --fusion-dir processed\fused\object_002\session_01\all_views_axis_calibrated `
  --raw-session raw\object_002\session_01 `
  --visited 0 6 15 `
  --output-dir PointNet\outputs\object_002\session_01
```

`--available` 可限制机械臂实际能到达的动作，例如 `--available 0 6 12 15 21`。只想检查点云转换时加 `--prepare-only`，无需加载 PyTorch。

`predict.py` 仅执行一次选角，不控制设备。完整闭环由下面的 `scan_pipeline.py` 执行；它在每个新视角完成采集和重建后，仅融合本次已访问的视角。

`--table-height-axis-mm` 是桌面平面在标定转轴方向上的坐标，单位毫米。示例 `-95` 只是这批数据的探索性估值，必须通过实际桌面测量后替换。该值会影响裁剪和高度归一化；已保存的 `object_frame.json` 不允许在同一输出目录内更换桌面高度。若要重新标定，使用新的输出目录。

目前的 200 物体仿真模型尚未证明能根据真实点云自适应选视。输出可用于接线测试和采集实验，正式效果需另行评估。

## 逐次选视扫描

`scan_pipeline.py` 是独立于 `system/scan_session.py` 的实机入口；它直接调用 `code/`、`arm/`、`turntable/` 已有的采集、重建、融合和控制程序。先扫描动作 0、6，再逐轮融合**当前已访问**视角、准备点云、由模型选下一动作。默认最多 10 次成功扫描；模型原始收益分数会保存，但默认**不按分数提前停止**。结束后将本轮全部成功视角融合到 `processed/fused/<物体>/<会话>/all_views_axis_calibrated/`。

模型结果是 JSON：`action_id`、`arm_pose_id`、`turntable_deg`、`scores_by_action`、`best_score` 等。`arm_pose_id=0/1` 对应 `upper/lower`；角度必须在转台支持的 0°、30°、…、330° 内。新调度器负责发送运动命令、等待并拍固定板定位图；模型结果本身不是到位反馈。

### 历史数据离线回放

以下例子只读取既有扫描，不连接相机或控制器。`-95` 是 `object_002` 历史探索用桌面高度，仅用于检查接口，不能作为新物体实机扫描参数。

```powershell
cd "D:\vscode\visual information\project"
..\.venv\Scripts\python.exe PointNet\replay.py `
  --raw-session raw\object_002\session_01 `
  --processed-root processed `
  --axis-calibration processed\fused\object_002\session_01\fusion_calibration\axis_calibration.json `
  --table-height-axis-mm -95 `
  --output PointNet\outputs\replay_object_002 `
  --budget 10
```

运行后 `state.json` 记录动作顺序和停止原因，`steps/step_XX/` 保存该轮融合、适配点云与分数，`final_fusion/` 保存最终点云。重复运行已有输出时加 `--resume`；已完成回放会直接返回，不会重选视角。程序只把清单和重建状态均完整的历史视角列为候选；每轮融合只传入当时的已访问列表。

现有 `object_002/session_01` 的 10 次预算回放结果保存在 `PointNet/outputs/replay_object_002_original_venv/`：动作依次为 `[0, 6, 15, 21, 16, 20, 13, 23, 18, 12]`，最终融合报告包含 10 个视角和 52,150 个点。首次预测的动作 15 对应下位姿与 90°。这组结果仅验证软件流程和结构化控制参数，未验证实机选角效果。

### 实机入口与运行前检查

先测量桌面沿本次转台轴方向的坐标 `table_height_axis_mm`。使用与本次固定板位置相符的完整转轴采集图像；程序会在本次会话的 `processed/fused/<物体>/<会话>/fusion_calibration/axis_calibration.json` 生成轴结果，并在全部步骤复用它。需检查固定板与转台的相对位置、上下位姿标定、机械臂路径、相机和投影仪状态。两个流水线不能同时占用设备。

统一使用上一级的 `D:\vscode\visual information\.venv`。该环境现在包含 PyTorch、OpenCV、pypylon、pyserial、PyYAML 和 pytest，可直接运行测试、历史回放和扫描入口，无需设置 `PYTHONPATH`。由于当前工作站无法连接 PyPI，缺少的包从本机已安装的 Python 环境离线补齐；换机器时需重新安装相应依赖。

```powershell
$tableHeightMm = [double]::Parse((Read-Host '输入实测桌面轴向高度，单位 mm'))
$axisImages = Read-Host '输入本次完整转轴采集图像目录'
..\.venv\Scripts\python.exe PointNet\scan_pipeline.py `
  --object-id object_new --session-id session_01 `
  --axis-images $axisImages `
  --table-height-axis-mm $tableHeightMm `
  --port COM6 --arm-port COM7
```

转轴目录应类似 `raw\turntable_axis\axis_cube_01`。命令先进行标定与文件检查，再打开设备；预览窗口按空格才开始运动，按 Q 取消。采集失败会保存会话状态和失败目录；修复原因后使用同样参数加 `--resume --repeat-id 2` 继续，已有成功动作不会重拍。后续再次失败要使用新的 repeat 编号。

`--score-stop-enabled --score-threshold <数值>` 目前只作为可配置机制。V2 模型输出未约束在 0～1，不是已校准概率；阈值必须先在独立仿真验证数据和实机试扫上核对，默认保持关闭。点云裁剪可用 `--fusion-radius-mm`、`--fusion-height-mm`、`--fusion-voxel-mm` 调整，但参数变化要使用新会话，不要在续跑中改变。
