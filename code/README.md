# 基础视觉模块 `code/`

[English](README.en.md) · [实验指导](../实验指导.md)


从项目根目录运行下列命令。当前投影协议只有 `dense_dots`：默认每视角采集 black、5 张点阵与 white 的左右图像。所有几何单位为毫米。相机需安装 pylon/pypylon，采集需要 OpenCV 图形窗口。

## 文件

| 文件 | 作用 |
| --- | --- |
| `capture.py` | 枚举相机/显示器，投影测试，棋盘或色卡成对拍摄，点阵扫描并写清单与原始图像 |
| `generate_stereo_board.py` | 生成双目标定用 A4 棋盘打印文件 |
| `calibrate_stereo.py` | 下位姿完整双目标定；上位姿沿用下位姿内参与基线并拟合外参，输出 NPZ/JSON 与验证报告 |
| `upper_stereo_fit.py` | 上位姿外参拟合和误差检查，供 `calibrate_stereo.py` 使用 |
| `reconstruct_view.py` | 图像校正、密集点阵匹配、视差重投影，输出单视角 PLY 与质量报告 |
| `fuse_views.py` | 按相机位姿及转台轴将多个视角转换到共同坐标系，裁剪并体素融合 |
| `estimate_table_height.py` | 从轴标定结果读取转台顶面高度的辅助函数 |
| `clean_green.py` | 对融合点云按颜色、半径和台面高度去除绿布；也可单独处理成品 PLY |
| `calibrate_color.py` | 可选的左右相机 BGR 色彩映射；不参与点阵几何匹配 |
| `tests/` | 标定、采集、重建、融合与上位姿拟合测试 |

## 标定与采集

```powershell
python code\generate_stereo_board.py
python code\capture.py list-cameras
python code\capture.py list-displays
python code\capture.py projector-test --display 1
# 下位姿：分别拍训练和独立验证组；预览按空格保存，Q 退出
python code\capture.py calibration --session-id session_01
python code\capture.py calibration --session-id validation_01
python code\calibrate_stereo.py --pose lower --train raw\calibration\session_01 --validation raw\calibration\validation_01 --square-mm 12
# 上位姿：同样单独拍两组
python code\capture.py calibration --session-id session_02
python code\capture.py calibration --session-id validation_02
python code\calibrate_stereo.py --pose upper --train raw\calibration\session_02 --validation raw\calibration\validation_02 --square-mm 12
```

`--square-mm` 填实测棋盘边长。默认标定输出在 `calibration/stereo_pose_lower/` 和 `calibration/stereo_pose_upper/`；上位姿依赖已完成的下位姿结果。`capture.py` 的 `--left-serial`、`--right-serial`、`--display` 和曝光参数需与设备一致。手动单视角扫描示例：

```powershell
python code\capture.py scan --object-id object_001 --session-id session_01 --view-id 0 --arm-pose-id 0 --turntable-deg 0 --display 1
python code\reconstruct_view.py --view raw\object_001\session_01\view_000\repeat_001
```

单视角扫描要求物体与双目组件在整组图案采集期间静止。`reconstruct_view.py` 按清单中的位姿选择上下标定；必要时用 `--calibration` 覆盖。输出包含 `cloud_left.ply` 与 `qc.json`，点云仍处于左相机坐标系。

## 融合与清理

通常由 `system/scan_session.py` 自动调用。独立融合时须提供同一会话的 `--camera-poses` 和轴文件；轴已在静止板坐标系时加 `--axis-already-in-reference`：

```powershell
python code\fuse_views.py --session-root raw\object_001\session_01 --processed-root processed --camera-poses raw\object_001\session_01\camera_poses.json --axis-calibration calibration\turntable_axis\axis_01\axis_calibration.json --axis-already-in-reference --output processed\fused\object_001\session_01\manual
python code\clean_green.py --fusion-dir processed\fused\object_001\session_01\manual
```

融合前需核查轴坐标系、相机定位、物体裁剪半径与视角重建状态。可选色彩标定：`python code\calibrate_color.py --images raw\color\session_01 --corners configs\color_chart_corners.json`。各入口的完整可调参数见 `python code\<脚本名>.py --help`。
