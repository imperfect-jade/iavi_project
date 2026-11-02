# 机械臂两位姿串口控制

当前 `system/scan_session.py` 每个视角直接从静止 ChArUco 板求相机位姿，并以板为共同坐标系；扫描与融合无需预先标定机械臂两位姿。

`arm.py` 通过 Jibot1 控制板的 USB 串口发送六舵机组合命令，切换已试过的“上”和“下”两个相机姿态。不会修改固件或 Zide 动作组。两组目标取自本次实机测试；下姿态原始文本中的 `##004` 和最后缺失的 `!` 已按协议修正。

## 准备

1. 机械臂使用配套外部电源，USB 接电脑；当前机械臂端口为 **COM7**。转盘使用另一串口，勿混用。
2. 关闭 Zide 及其他占用 COM7 的程序。安装依赖：`python -m pip install pyserial`。
3. 首次发送前确认相机、支架和线缆在上、下姿态间的完整运动路径无干涉。下姿态 002 号舵机曾有保持抖动，须现场观察。

在 `D:\vscode\visual information\project` 目录执行：

```powershell
python arm\arm.py --list
python arm\arm.py --pose upper --dry-run
python arm\arm.py --pose lower --dry-run
python arm\arm.py --port COM7 --pose upper --readback
python arm\arm.py --port COM7 --pose lower --readback
```

持续调姿可只打开一次串口：

```powershell
python arm\arm.py --port COM7 --readback
```

出现 `pose>` 后输入 `upper`、`lower` 或 `q`。程序只接受这两个已试姿态，不接收任意 PWM 值。每次发送六路组合指令，等待 2.5 秒；`--readback` 再逐一发送 `#000PRAD!` 至 `#005PRAD!` 并显示返回值。无回复时显示 `no reply`。

串口写入和舵机读回都**不能证明双目相机的准确位姿**。正式结构光采集前仍需等待振动消失，并用固定标记的定位图确认相机位姿。若出现碰撞、持续抖动、异响或拉线，停止运行并检查机构；不要用读取到的舵机位置替代光学定位。

### 串口连接时出现预设动作组

厂家提供的出厂源码在启动时会执行控制板闪存里的“开机命令”；如果这个命令设置为动作组，主控板复位就可能启动抓取。原驱动打开串口时使用 pySerial 默认的 DTR/RTS 状态，某些 Arduino 串口板会因此复位。现在驱动会**在打开 COM7 前**将 DTR/RTS 设为非激活，降低这种情况的发生概率；具体 USB 串口芯片或驱动仍可能在打开瞬间产生脉冲，因此无法保证绝不复位。

先在机械臂运动范围清空障碍物，然后仅测试连接：

```powershell
python arm\arm.py --port COM7 --connect-only
```

此命令不会发送任何舵机位姿或动作组指令。若仍出现抓取、开机蜂鸣或突然复位，应到 Zide 检查并清除控制板中保存的**开机动作/开机命令**，然后复测；这属于控制板持久配置，Python 脚本没有修改它。若仅在手柄工作或曾启动动作组后出现，关闭手柄并确认当前动作组已结束再测试。厂家源码中位姿 `#...P...T...!` 不会取消正在执行的动作组，后续动作组步骤可能覆盖刚发的位姿。不要在机械臂带相机、靠近标定板时带载尝试不熟悉的动作组。

## 固定 ChArUco 板

[图案](targets/charuco_7x5_10mm_7mm_85x65mm.png)、[A4 打印版](targets/charuco_7x5_10mm_7mm_A4_print.pdf) 和 [配置](targets/charuco_7x5_10mm_7mm_config.json) 使用 10 mm 方格、7 mm 标记。打印时使用 100% 原尺寸，并将纸张平贴在硬质平板上。

把固定板放在转台外侧，确保它不会随转台或机械臂移动，且在转轴标定与物体扫描期间保持同一位置。上下姿态的左右相机都应清晰看到板；避免遮挡、反光和过曝。`system/scan_session.py` 在每个扫描视角拍摄固定板并求出该次相机位姿；`turntable/calibrate_axis.py` 在转轴采集图中同时标定固定板与转轴。无需单独运行机械臂两位姿标定。

`capture_fixed_board.py` 可选用于检查上下姿态的板可见性和机械臂回位，但其照片不参与当前重建流水线。
