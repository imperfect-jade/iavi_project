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

### 串口连接时出现预设动作组

厂家提供的出厂源码在启动时会执行控制板闪存里的“开机命令”；如果这个命令设置为动作组，主控板复位就可能启动抓取。原驱动打开串口时使用 pySerial 默认的 DTR/RTS 状态，某些 Arduino 串口板会因此复位。现在驱动会**在打开 COM7 前**将 DTR/RTS 设为非激活，降低这种情况的发生概率；具体 USB 串口芯片或驱动仍可能在打开瞬间产生脉冲，因此无法保证绝不复位。

先在机械臂运动范围清空障碍物，然后仅测试连接：

```powershell
python arm\arm.py --port COM7 --connect-only
```

此命令不会发送任何舵机位姿或动作组指令。若仍出现抓取、开机蜂鸣或突然复位，应到 Zide 检查并清除控制板中保存的**开机动作/开机命令**，然后复测；这属于控制板持久配置，Python 脚本没有修改它。若仅在手柄工作或曾启动动作组后出现，关闭手柄并确认当前动作组已结束再测试。厂家源码中位姿 `#...P...T...!` 不会取消正在执行的动作组，后续动作组步骤可能覆盖刚发的位姿。不要在机械臂带相机、靠近标定板时带载尝试不熟悉的动作组。

## 固定 ChArUco 板：两位姿拍摄与标定

使用 [图案](../turntable/targets/charuco_7x5_10mm_7mm_85x65mm.png) 和 [配置](../turntable/targets/charuco_7x5_10mm_7mm_config.json)：7×5 个方格，方格边长 **10 mm**，标记边长 **7 mm**，图案区域 **70×50 mm**，裁切纸张 **85×65 mm**。打印时用 **100% 原尺寸**，关闭“适合页面”，用尺量相邻方格确实为 10 mm；平贴到硬质平板，不要弯折。若实体尺寸不同，应先改配置的实际尺寸再标定，不能只修改文件名。

将板固定在**转台外侧、不会随转台或机械臂移动**的支架上，图案朝两个拍摄位置之间，距离接近物体拍摄深度。先让转台停在 **0°**，移走遮挡板的物体，让上、下位姿的**左相机**都能清晰看到尽量完整的板；右相机同时看到更便于核查。板不能在扫描期间移动。均匀照亮、对焦、避免反光和投影条纹。转台 0° 由操作者确认，本脚本不会控制 COM6。

关闭 Zide 和其他 COM7 占用者；确认机械臂整段运动路径无碰撞、线缆牵拉或持续抖动。安装项目既有相机环境及 `opencv-contrib-python`、`pypylon`、`pyserial`。在项目根目录先做一次往返试拍：

```powershell
python arm\capture_fixed_board.py --session-id board_trial_01 --port COM7 --repeats 1
```

默认顺序是**上→下→上**；每次发送位姿命令后等待 6 秒再拍照（可用 `--settle-s` 调整），左图检测至少 12/24 个 ChArUco 内角点，最多拍 3 次，保存左右原始 PNG、标出左图检测角点的 `left_detected.png` 和 `capture_manifest.json`。原图角点不足时，程序会尝试局部对比度增强，只将增强图用于角点检测，PnP 仍使用原相机内参与原图像素坐标；记录中的 `left_detection_method` 会注明 `raw` 或 `clahe`。若失败，`complete=false`，已拍图片仍留在 `raw/arm_fixed_board/board_trial_01`，可检查遮挡、焦距、曝光和支架，然后用新的 session ID 重拍。串口命令和固定等待时间无法证明机械臂已稳定；尤其要观察曾抖动的 002 舵机。需要保存舵机读回时加 `--readback`，读回仍不能代替照片定位。

`--images` 必须指向拍摄命令的 `--session-id` 对应目录。当前工作目录里的 `board_trial_01` 因下位姿角点不足，`capture_manifest.json` 为 `complete=false`，不能直接标定。调整光照后，请使用一个新的会话名，先拍摄成功，再标定同名目录，例如：

```powershell
python arm\capture_fixed_board.py --session-id fixed_board_01 --port COM7 --repeats 1
python arm\calibrate_fixed_board.py --images raw\arm_fixed_board\fixed_board_01
```

试拍通过后，建议用多次往返检查回位稳定性。以下示例中的 `board_repeat_01` 只有执行第一条命令并成功结束后才会存在：

```powershell
python arm\capture_fixed_board.py --session-id board_repeat_01 --port COM7 --repeats 5
python arm\calibrate_fixed_board.py --images raw\arm_fixed_board\board_repeat_01
```

程序使用已有 `calibration/stereo/stereo_calibration.json` 中**原始左相机**的 `K_left` 和 `D_left`，重新检测每张左图，用 PnP 求 `T_left_from_board`，再以第一张上位姿左相机为参考计算 `T_reference_from_left`。单位为 mm；结果保存在 `calibration/arm_fixed_board/board_repeat_01/two_pose_calibration.json`。程序核对左右相机序列号、图像尺寸、板参数；默认至少 12 个角点且单张重投影 RMS 不超过 3 px，不满足就报错，不能把错误结果悄悄用于融合。输出包含每张图的角点数、重投影误差、变换矩阵和相同姿态首次到位后的平移/旋转回位误差。试拍也可以用同样的标定命令处理。

板的位姿应跨所有拍摄保持不变。若同一姿态回位误差对目标重建精度不可接受，正式扫描时每次到位后都应拍板重新定位。当前 JSON 的键是**标定拍摄**名称，不能直接作为 `fuse_views.py --camera-poses` 的 `view_###/repeat_###` 映射。原有转台轴 JSON 也属于其标定时的左相机坐标系；除非它与新的上位姿相同，否则需要在上位姿重新标定转台轴，或测得两个左相机坐标系间的变换。
