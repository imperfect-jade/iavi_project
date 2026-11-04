# 固定视角扫描调度 `system/`

[English](README.en.md) · [实验指导](../实验指导.md)


`scan_session.py` 调度机械臂、转台、相机、投影仪、重建和融合。默认先拍上位姿、再拍下位姿，每个位姿遍历指定角度范围；完整 0°～330°、步长 30° 对应 24 个动作。每视角以 black 图定位静止板，然后拍密集点阵和 white。固定板必须在轴标定和物体扫描之间保持不动。

## 文件

| 文件 | 作用 |
| --- | --- |
| `scan_session.py` | 会话采集、预览、续拍、重建、状态汇总与融合入口 |
| `board_stereo_correction.py` | 从左右固定板定位图估计会话内双目外参修正，保持原内参与基线长度 |
| `fusion_pipeline.py` | 检查转轴采集、准备会话轴几何、调用融合及绿布清理 |
| `tests/` | 会话恢复、双目修正与融合流程测试 |

## 运行

先完成上下位姿双目标定与独立验证，并在下位姿采集完整轴标定图像。项目根目录下：

```powershell
python system\scan_session.py --port COM6 --arm-port COM7 --start-angle 0 --end-angle 330 --step-angle 30 --object-id object_001 --session-id session_01 --fusion-axis-images raw\turntable_axis\axis_01
```

预览按空格开始，Q/Esc 取消。仅拍一个位姿可加 `--arm-pose-id 0` 或 `1`。只完成重建可加 `--no-fusion`。拍摄中断后保持相同会话参数，使用新的 repeat 编号：

```powershell
python system\scan_session.py --port COM6 --arm-port COM7 --start-angle 0 --end-angle 330 --step-angle 30 --object-id object_001 --session-id session_01 --fusion-axis-images raw\turntable_axis\axis_01 --resume --repeat-id 2
```

`--resume` 跳过已完成视角并重算定位、补做所需重建与融合。若自动深度范围失败，可加 `--reconstruction-min-depth-mm` 与 `--reconstruction-max-depth-mm`。裁剪可用 `--fusion-radius-mm`、`--fusion-height-mm MIN MAX`、`--fusion-voxel-mm` 调节；`--no-green-filter` 保留未清理成品。

原始图像在 `raw/<物体>/<会话>/view_###/repeat_###/`，`camera_poses.json` 位于会话根目录；单视角 `cloud_left.ply`、`qc.json` 与状态文件在对应 `processed/` 目录；最终 `fused.ply`、`fusion.json`、`aligned/` 与本次轴结果在 `processed/fused/<物体>/<会话>/` 下。失败时保留原始图像并以非零状态退出。运行测试：`python -m pytest system/tests -q`。
