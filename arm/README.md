# 机械臂两位姿串口控制

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
