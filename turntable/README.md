# 转台控制与转轴标定 `turntable/`

[English](README.en.md) · [实验指导](../实验指导.md)


本模块控制 0°、30°、…、330° 共 12 个预设角度，使用转动的四面 ArUco 方柱与静止 ChArUco 板计算转轴。标定照片在机械臂**下位姿**拍摄，后续物体扫描时固定板相对转台的位置须保持一致。

## 文件

| 文件 | 作用 |
| --- | --- |
| `turntable.py` | 串口控制与角度到已测试指令的映射；支持列出、单次和交互发送 |
| `generate_cube_target.py` | 生成四面方柱的打印稿与三维角点配置 |
| `capture_axis.py` | 依次采集 12 个角度及回零图像，检查左右标记与固定板可见性 |
| `calibrate_axis.py` | 双目三角化方柱角点、拟合转轴/角度与台面高度，并变换到固定板坐标系 |
| `targets/aruco_cube_4faces_config.json` | 实测方柱角点和面序；更换实物后需重新测量 |
| `targets/*.png`, `targets/*.pdf` | 四面标靶打印素材 |
| `tests/` | 转台指令、标靶和轴几何测试 |

## 使用示例

```powershell
python turntable\turntable.py --list
python turntable\turntable.py --angle 90 --dry-run
python turntable\turntable.py --port COM6 --angle 90
python turntable\turntable.py --port COM6
python turntable\generate_cube_target.py
python turntable\capture_axis.py --port COM6 --session-id axis_01
python turntable\calibrate_axis.py --images raw\turntable_axis\axis_01
```

交互模式输入角度或 `q` 退出。轴拍摄默认在 `raw/turntable_axis/axis_01/` 写照片及 `capture_manifest.json`；标定默认输出 `calibration/turntable_axis/axis_01/axis_calibration.json`。`capture_axis.py --settle-s` 控制常规角度等待，`--return-settle-s` 控制回零等待；`calibrate_axis.py --stereo` 可改用其他下位姿 NPZ，`--output` 可指定输出。方柱须刚性固定，底面直接接触转台顶面以供自动估计高度。串口命令不提供可靠到位证明。运行测试：`python -m pytest turntable/tests -q`。
