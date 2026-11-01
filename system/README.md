# 联动扫描调度

`scan_session.py` 使用 COM7 控制机械臂上、下两位姿，COM6 控制转台。默认按上位姿的角度序列、下位姿的角度序列拍摄；`--arm-pose-id 0` 或 `1` 可只拍一个位姿。每个视角先将投影设为黑色，在固定 ChArUco 板仍可见、板未移动且照明足够的条件下拍独立的左右定位图；左图至少识别 12 个角点且重投影 RMS 不超过 3 px 才开始 `dense_dots_v2`。定位图放在 `localization/`，不计入点阵协议帧。

全部视角拍摄成功后，程序先关闭相机、机械臂和转台串口，再按视角顺序调用 `code/reconstruct_view.py`。每个结果写入 `processed/<object-id>/<session-id>/view_###/repeat_###/`，包含 `cloud_left.ply`、`qc.json` 和 `reconstruction_status.json`；会话级结果写入 `processed/<object-id>/<session-id>/reconstruction_summary.json`。重建失败不丢失原始照片，程序继续处理其余视角，最终以非零退出码报告失败；本阶段不做点云融合。

从项目根目录运行：

```powershell
python system\scan_session.py --port COM6 --arm-port COM7 --start-angle 0 --end-angle 330 --step-angle 30 --object-id object_001 --session-id session_01
```

默认参考为 `calibration/arm_fixed_board/board_repeat_01/two_pose_calibration.json` 中第一张上位姿定位图。**固定板必须与该标定时保持同一位置**；若重新安装固定板，先重做两位姿标定，并用 `--reference-calibration` 指向新结果。相机内参使用 `calibration/stereo/stereo_calibration.json`。机械臂默认等待 6 秒，转台默认等待 2.5 秒，可用 `--arm-settle-s`、`--motion-settle-s` 调整；串口命令与等待均不证明实际到位。

上位姿拍完 330° 后切换到下位姿并回到 0°，这一步单独默认等待 12 秒，可用 `--return-settle-s` 调整；其余转台角度仍使用 `--motion-settle-s`。若在定位图取帧时遇到相机 `The buffer was cancelled`，这是相机采集失败，不等同于转台未停稳；应同时检查相机 USB 连接、供电及 pylon 采集状态。已有不完整视角需用新的 `--repeat-id` 配合 `--resume` 续拍。

按空格开始预览后的扫描，Q 或 Esc 在运动前取消。每个完成的 `raw/<object-id>/<session-id>/view_###/repeat_###/` 中保存 `localization/left.png`、`localization/right.png`、`localization.json`、`pose.json` 和原有点阵文件。会话根目录的 `camera_poses.json` 使用 `view_###/repeat_###` 键，值为该次 `T_reference_from_left`，可传给 `fuse_views.py --camera-poses`。定位或点阵拍摄失败即停止；失败目录保留，用新 `--repeat-id` 和 `--resume` 继续。相机相对双目标定发生变化时须重标定。

如果拍摄已完成，使用相同拍摄参数加 `--resume` 会跳过拍摄和硬件连接，只补做失败或缺失的重建。重建默认使用 `--stereo` 对应的 `.npz` 文件，自动估计深度范围；自动估计失败时可同时提供 `--reconstruction-min-depth-mm 550 --reconstruction-max-depth-mm 1100`，再用 `--resume` 重试。重建输出目录可用 `--processed-output` 指定。重建使用的深度范围或标定文件内容改变时，会重新运行相应视角。

```powershell
python system\scan_session.py --port COM6 --arm-port COM7 --start-angle 0 --end-angle 330 --step-angle 30 --object-id object_001 --session-id session_01 --resume --reconstruction-min-depth-mm 550 --reconstruction-max-depth-mm 1100
```

机械臂换位后，原有转台轴标定可能不在此参考相机坐标系中。跨上、下位姿融合前须将转台轴变换到同一参考系，或在参考上位姿重新标定转台轴；仅有 `camera_poses.json` 不足以保证轴坐标一致。
