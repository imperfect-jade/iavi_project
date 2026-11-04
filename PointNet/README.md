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

`--table-height-axis-mm` 是转台顶面在本次标定转轴方向上的坐标，单位毫米。实机扫描不传此参数时，程序根据转轴标定立方体的底面自动计算；接受预览并开始会话后保存到 `PointNet/outputs/<物体>/<会话>/table_height_estimate.json`。标定立方体须直接放在转台顶面；其 12 个角度的底面高度不一致时，自动高度会报错，可配合本次坐标系的手动高度使用。若标定结果提示转轴方向与立方体向上方向相反，应先检查转台旋向及标定，不能仅靠手动高度修正。这个坐标不能从转台离地高度直接换算，也不能跨不同转轴标定会话沿用。它会影响融合下限、去绿布和 PointNet 高度归一化；已保存的 `object_frame.json` 不允许在同一输出目录内更换高度。

目前的 200 物体仿真模型尚未证明能根据真实点云自适应选视。输出可用于接线测试和采集实验，正式效果需另行评估。

## 逐次选视扫描

`scan_pipeline.py` 是独立于 `system/scan_session.py` 的实机入口；它直接调用 `code/`、`arm/`、`turntable/` 已有的采集、重建、融合和控制程序。先扫描动作 0、6，再逐轮融合**当前已访问**视角、准备点云、由模型选下一动作。默认最多 10 次成功扫描；模型原始收益分数会保存，但默认**不按分数提前停止**。结束后将本轮全部成功视角融合到 `processed/fused/<物体>/<会话>/all_views_axis_calibrated/`。每轮融合默认调用 `code/clean_green.py` 生成去绿布的 `fused.ply` 与 `aligned/*.ply`，未过滤结果保存在同目录的 `unfiltered/`。自动半径由初始两视角估计并在本次会话固定；估计不可靠时使用宽松裁剪半径。

PointNet 默认使用清理后的 `aligned/*.ply` 决策；需要对照原始输入时可加 `--use-unfiltered-model-input`。历史回放也支持同名选项，便于离线比较。`--fusion-radius-mm` 是自动半径的宽松上限，首次估计和固定后的实际半径记录在 `PointNet/outputs/<物体>/<会话>/green_filter_radius.json`；台面以下的裁剪依据 `--table-height-axis-mm`。

模型结果是 JSON：`action_id`、`arm_pose_id`、`turntable_deg`、`scores_by_action`、`best_score` 等。`arm_pose_id=0/1` 对应 `upper/lower`；角度必须在转台支持的 0°、30°、…、330° 内。新调度器负责发送运动命令、等待并拍固定板定位图；模型结果本身不是到位反馈。

### 历史数据离线回放

以下例子只读取既有扫描，不连接相机或控制器。`-95` 是 `object_002` 历史探索用桌面高度，仅用于检查接口，不能作为新物体实机扫描参数。新的轴标定结果包含可靠顶面高度时，可以省略 `--table-height-axis-mm`。回放与实机使用同一高度放宽融合下限。

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

回放新会话时可传 `--fusion-radius-mm`、`--fusion-height-mm MIN MAX`、`--fusion-voxel-mm`，使历史回放的宽松裁剪与当时实机参数一致。

现有 `object_002/session_01` 的 10 次预算回放结果保存在 `PointNet/outputs/replay_object_002_original_venv/`：动作依次为 `[0, 6, 15, 21, 16, 20, 13, 23, 18, 12]`，最终融合报告包含 10 个视角和 52,150 个点。首次预测的动作 15 对应下位姿与 90°。这组结果仅验证软件流程和结构化控制参数，未验证实机选角效果。

### 实机入口与运行前检查

使用与本次固定板位置相符的完整转轴采集图像，且标定立方体底面直接接触转台顶面；程序会在本次会话的 `processed/fused/<物体>/<会话>/fusion_calibration/axis_calibration.json` 生成轴和顶面高度，在全部步骤复用，保留放在转台上的垫块。需检查固定板与转台的相对位置、上下位姿标定、机械臂路径、相机和投影仪状态。两个流水线不能同时占用设备。

统一使用上一级的 `D:\vscode\visual information\.venv`。该环境现在包含 PyTorch、OpenCV、pypylon、pyserial、PyYAML 和 pytest，可直接运行测试、历史回放和扫描入口，无需设置 `PYTHONPATH`。由于当前工作站无法连接 PyPI，缺少的包从本机已安装的 Python 环境离线补齐；换机器时需重新安装相应依赖。

```powershell
$axisImages = Read-Host '输入本次完整转轴采集图像目录'
..\.venv\Scripts\python.exe PointNet\scan_pipeline.py `
  --object-id object_new --session-id session_01 `
  --axis-images $axisImages `
  --port COM6 --arm-port COM7
```

转轴目录应类似 `raw\turntable_axis\axis_cube_01`。命令先进行标定与文件检查，再打开设备；预览窗口按空格才开始运动，按 Q 取消。若标定立方体未直接放在转台上，使用本次坐标系的实测 `--table-height-axis-mm`。采集失败会保存会话状态和失败目录；修复原因后使用同样参数加 `--resume --repeat-id 2` 继续，已有成功动作不会重拍。后续再次失败要使用新的 repeat 编号。

每个视角发送机械臂位姿命令后默认等待 6 秒，再发送转台命令；转台普通移动默认等待 6 秒，从非零角度回到 0° 或最终回零等待 12 秒。可分别通过 `--arm-settle-s`、`--motion-settle-s`、`--return-settle-s` 调整。

`--score-stop-enabled --score-threshold <数值>` 目前只作为可配置机制。V2 模型输出未约束在 0～1，不是已校准概率；阈值必须先在独立仿真验证数据和实机试扫上核对，默认保持关闭。点云裁剪可用 `--fusion-radius-mm`、`--fusion-height-mm`、`--fusion-voxel-mm` 调整，但参数变化要使用新会话，不要在续跑中改变。

### 去除成品点云中的绿布

现有会话或手动融合结果也可单独调用展示清理程序。它按 HSV 绿色、转轴半径和台面以上高度筛点，生成 `fused_no_green.ply` 与同名 JSON 报告；原始点云保持不变。下例中的 45 mm 半径适合 `object_010`，其他物体要按实际宽度调整或省略 `--radius-mm`。

```powershell
..\.venv\Scripts\python.exe code\clean_green.py `
  --fusion-dir processed\fused\object_010\session_01\all_views_axis_calibrated `
  --table-height-axis-mm -101.09 --min-object-height-mm 3 `
  --radius-mm 45
```

绿色物体表面也可能被颜色规则移除；可通过 `--hue-min`、`--hue-max`、`--min-saturation`、`--min-value` 调整，或直接使用未清理的原始点云。输出路径已存在时用 `--output` 指定新文件名。
