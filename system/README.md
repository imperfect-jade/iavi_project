# 联动扫描调度

`scan_session.py` 是统一调度入口的第一阶段：人工固定机械臂姿态，程序自动控制转台并调用现有 `code/capture.py` 的 `dense_dots_v2` 单视角拍摄。重建、融合和机械臂运动尚未接入。转台协议和角度表仍由 `turntable/turntable.py` 维护。

从项目根目录运行；先关闭 Zide，接好转台外部电源、双目相机和投影仪。Python 环境需要 `numpy`、`opencv-python`、`pypylon`、`pyserial`。

```powershell
python system\scan_session.py --port COM6 --start-angle 0 --end-angle 180 --step-angle 30 --arm-pose-id 1 --object-id object_001 --session-id session_01
```

`--arm-pose-id` 为本次人工固定的姿态编号（0、1、2），本程序不移动机械臂。起止角度包含两端，须为 0～330 的 30 度档位；间隔须是 30 的正整数倍且恰好到达终止角度。启动后先预览第一张投影图与双目画面，按空格开始，按 Q 或 Esc 在运动前取消。开始后保持 COM6、相机、投影窗口连接，逐角度发送位置指令，默认等 2.5 秒，再拍完整的点阵与白光序列；扫描过程中不要移动物体或相机。

输出位于 `raw/<object-id>/<session-id>/view_###/repeat_###/`，其中 `view_id = arm_pose_id*12 + angle/30`。会话根目录保存 `scan_plan.json`，每个视角保存 `pose.json`、逐帧记录和完整 `manifest.json`。串口发送与定时等待不代表真实到位，因此 `verified_pose=false`。任一视角拍摄失败即停止，不继续转动。

已有会话默认拒绝再次运行。只有原计划参数一致时可显式续拍；程序跳过仍保有全部图像的完整视角：

```powershell
python system\scan_session.py --port COM6 --start-angle 0 --end-angle 180 --step-angle 30 --arm-pose-id 1 --object-id object_001 --session-id session_01 --resume
```

如果失败视角已有不完整的 `repeat_001`，在同一命令中增加 `--repeat-id 2`，保留原始失败数据。物体被重新放置或相机配置改变时使用新的 `--session-id`。当前程序没有实现实机位置反馈；正式批量采集前应现场验证等待时间、每视角图像完整性和转台重复定位。
