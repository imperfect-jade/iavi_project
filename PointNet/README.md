# PointNet 主动扫描 `PointNet/`

[English](README.en.md) · [实验指导](../实验指导.md)


本模块将 `server/` 训练出的 24 动作收益模型接入现有设备和重建流程。实机闭环先扫描上位姿 0°、180°（动作 0、6），然后只融合**已访问**视角，预测下一动作并继续扫描。与 [`system/`](../system/README.md) 的固定角度序列是两个独立入口。

## 文件

| 文件 | 作用 |
| --- | --- |
| `adapter.py` | 读取已对齐 PLY，将已访问视角变换为模型物体坐标系 |
| `fusion.py` | 本次会话轴标定与已访问视角融合 |
| `policy.py` | 校验 checkpoint 和配置、推理候选收益 |
| `predict.py` | 对已有融合结果做单次选角或只准备模型输入 |
| `replay.py` | 基于历史完整扫描离线逐次回放，无需连接设备 |
| `scan_step.py` | 执行单个动作的运动、定位、采集和重建 |
| `scan_pipeline.py` | 实机主动扫描主入口、状态保存与续跑 |
| `tests/` | 适配、策略、回放、扫描步骤及闭环测试 |

`models/` 不随仓库发布；需自行放入训练得到的配置和权重。默认配置路径为 `models/diverse_200_v2_seed42_24/config_local.yaml`，也可显式使用 `--config`。

## 单次选角

从项目根目录运行。输入必须是同一会话的 `fusion.json`、`aligned/*.ply`、原始会话 `camera_poses.json` 和轴/台面信息；单视角 `cloud_left.ply` 仍在相机坐标系，不能直接输入。第一次需传本次轴坐标系中的台面高度，之后同一输出目录复用保存的 `object_frame.json`。

```powershell
python PointNet\predict.py --fusion-dir processed\fused\object_001\session_01\all_views_axis_calibrated --raw-session raw\object_001\session_01 --visited 0 6 --table-height-axis-mm -95 --output-dir PointNet\outputs\object_001\session_01 --config models\diverse_200_v2_seed42_24\config_local.yaml
```

`-95` 只是示例，须换成本次轴标定测得值。加 `--prepare-only` 可仅生成 `observed.npy` 和 `object_frame.json`，不加载模型；完整预测另写 `prediction.json`。扫描新动作后在 `--visited` 后追加其编号，保持同一输出目录。`--available` 可限制可执行动作。

## 离线回放与实机闭环

历史回放要求原始会话与逐视角重建均完整：

```powershell
python PointNet\replay.py --raw-session raw\object_001\session_01 --processed-root processed --axis-calibration calibration\turntable_axis\axis_01\axis_calibration.json --output PointNet\outputs\replay_object_001 --budget 10 --config models\diverse_200_v2_seed42_24\config_local.yaml
```

`state.json` 记录动作和停止原因，`steps/` 保存每轮输入及分数，`final_fusion/` 为最终融合。重复运行相同输出时用 `--resume`。实机入口需已完成上下位姿双目标定，轴图像必须是本次固定板位置下的完整采集：

```powershell
python PointNet\scan_pipeline.py --object-id object_001 --session-id session_02 --axis-images raw\turntable_axis\axis_02 --port COM6 --arm-port COM7 --config models\diverse_200_v2_seed42_24\config_local.yaml --max-scans 10
```

预览按空格启动，Q 取消。失败后保持原参数并使用 `--resume --repeat-id 2`；已完成动作不会重拍。结果位于 `PointNet/outputs/<物体>/<会话>/` 和 `processed/fused/<物体>/<会话>/`。`--score-stop-enabled` 默认关闭，模型分数未经实机概率校准。绿色物体可能被默认去绿布规则误删，可用 `--use-unfiltered-model-input` 对照，成品清理细节见 [`../code/README.md`](../code/README.md)。运行测试：`python -m pytest PointNet/tests -q`。
