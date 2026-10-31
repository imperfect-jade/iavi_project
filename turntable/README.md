# 转台程序使用说明

旧版单面 ChArUco 标定板及其配置保存在 `turntable/targets/charuco_7x5_10mm_7mm_*`，生成脚本为 `turntable/generate_charuco_board.py`。它只用于复现早期实验；下面的整周转台轴标定使用四面 ArUco 方柱。

## 四面 ArUco 方柱转台轴标定

当前实物方柱为 45×45×100 mm。用户实测每个黑色标记边长 21 mm；四面贴法相同，每面左右边距各 12 mm、顶边距 26 mm、两标记间距 4 mm、底边距 28 mm。这些数据已写入 `turntable/targets/aruco_cube_4faces_config.json` 的三维角点。首组照片验证实物面序在模型坐标系中为 0、3、2、1；配置中的 `mounted_face_to_model_face` 已据此修正。配置仍假设四面标记平整、朝上；如实物还有偏移，应修正对应 `corners_mm`。

`turntable/targets/aruco_cube_4faces_A4.pdf` 是按上述实测布局重新生成的备用打印图，A4 原尺寸／100% 打印，关闭“适合纸张”缩放。已贴好的方柱不需要重印。四张面板依次标为 FACE 0～3，每面两个标记；裁切线外的标签不要贴到方柱上。方柱底面刚性固定于转盘，旋转时不得晃动。

移开物体，保持相机与转盘位置固定，关闭结构光投影并使用均匀照明。在 0°、30°、…、330°拍摄后返回 0°再拍一次。默认要求每张左右相机**共同识别**至少两个已知 ID。先在 0°、90°、180°、270°试拍，确认四面标记可见；如果某角度不能稳定识别，先调整实物与照明，不建议降低验收条件来凑数据。

在项目目录运行：

```powershell
python turntable\capture_axis.py --port COM6 --session-id axis_cube_01
python turntable\calibrate_axis.py --images raw\turntable_axis\axis_cube_01
```

原始照片和 `capture_manifest.json` 写入 `raw/turntable_axis/<session-id>/`，已存在会话不会覆盖。标定结果默认写入 `calibration/turntable_axis/<session-id>/axis_calibration.json`；可用 `--output` 指定其他结果文件。标定使用左右相机原图和现有双目内外参，对共同可见的标记角点进行三角化，再把其三维位置与同一方柱模型刚性配准，拟合转轴和实际角度。结果输出轴点、轴方向、逐图双目重投影误差、三维配准残差、转轴拟合残差，以及回零图像的角点位移和位姿差。单面近正视时，位姿角度可能仍不稳定；应结合回零图像的角点位移判断。轴点在原始左相机坐标系中，单位毫米。观察结果与实物相符后才能用于点云融合；不同机械臂位姿需分别标定或转换到共同坐标系。

若拍摄后又修正了同一套标记的几何配置，可以保留旧结果并显式指定新文件名重算：

```powershell
python turntable\calibrate_axis.py --images raw\turntable_axis\axis_cube_01 --allow-board-geometry-update --output calibration\turntable_axis\axis_cube_01\axis_calibration_recomputed.json
```

脚本：[turntable.py](turntable.py)。它通过 Arduino 的 USB 串口发送本次实验已试过的 12 个转盘位置指令，不需要 Zide，也不改写控制板固件。角度指盘面相对实验零位的角度；`0500` 定为 0°，`1500` 约为 180°。位置值为首轮实测可用值，尚无精度统计。

## 准备

1. 舵机接原配外部电源，Arduino USB 数据线接电脑；确认设备管理器中的串口号。当前示例为 COM6。
2. 关闭 Zide 和其他占用该串口的软件。安装 Python 3 和 `pyserial`：`python -m pip install pyserial`。如果已有项目虚拟环境，请用该环境的 Python 执行。
3. 若你已取消开机动作，先检查转盘上电后保持静止。打开串口仍可能复位 Arduino，脚本会等 2.5 秒再接受目标命令。

在 `D:\vscode\visual information\project` 下运行：

```powershell
python turntable\turntable.py --list
python turntable\turntable.py --angle 30 --dry-run
python turntable\turntable.py --port COM6
```

最后一条会保持串口连接；看到 `angle>` 后输入 `0`、`30`、`60` … `330`，输入 `q` 退出。首次运行请让转盘周围无障碍并在设备旁观察。一次发送一个目标后脚本等待 2.5 秒。也可以单次运行：

```powershell
python turntable\turntable.py --port COM6 --angle 90
```

每次单次运行都会重新打开串口；连续调试推荐交互模式。当前脚本发送 `#000PxxxxT2000!`，其中 `xxxx` 来自已试的动作表，不接受任意角度或任意 PWM 值。`2333` 到 `0500` 是反向回零，程序不会把它当作向前跨过 360°。串口写入成功仅表示命令已发送；由于位置回读尚未验证，脚本不会报告“已到位”。自动采集前应增加视觉位姿确认和重复性记录。
