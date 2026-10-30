# 转台程序使用说明

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
