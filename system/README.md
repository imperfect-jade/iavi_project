# 联动扫描调度

`scan_session.py` 使用 COM7 控制机械臂上、下两位姿，COM6 控制转台。默认按上位姿的角度序列、下位姿的角度序列拍摄；`--arm-pose-id 0` 或 `1` 可只拍一个位姿。每个视角先将投影设为黑色，在固定 ChArUco 板仍可见、板未移动且照明足够的条件下拍独立的左右定位图；左图至少识别 12 个角点且重投影 RMS 不超过 3 px 才开始 `dense_dots_v2`。定位图放在 `localization/`，不计入点阵协议帧。**左右相机都必须看见足够的固定板角点**，供扫描结束后的双目校正使用。

重建前默认运行 `board_stereo_correction.py`：从当前会话每个视角最新的完整拍摄中选取左右固定板对应角点，用原双目标定的左右内参与畸变估计当前会话的相对旋转和基线方向，并保留原标定测得的基线长度。程序按机械臂位姿分组，留出每组约四分之一的图像验证；**需要修正失配时**至少要求 8 对可用定位图。仅当留出图和最终所有图的**每张**极线误差中位数不超过 1 px、p95 不超过 2 px，且拟合基线与原基线相差不超过 5% 时才发布修正标定并继续重建。如果原标定本来满足这些门限，就沿用原标定，少量视角扫描也可继续。报告在 `processed/<object-id>/<session-id>/board_stereo_correction/board_stereo_report.json`，修正后的 NPZ/JSON 放在同目录；原标定文件保持不变。校正失败会停止重建并说明原因，不会悄悄沿用已失配的外参。

固定板的实测方格边长必须与 `arm/targets/charuco_7x5_10mm_7mm_config.json` 一致，且在整次扫描中固定不动。它只有 70×50 mm，通常位于画面一侧，因此这一步主要校正当前会话的双目失配；正式量测仍需用独立标定板覆盖物体工作区验证。确需跳过本步骤时可加 `--no-board-stereo-correction`，此时重建使用 `--stereo` 对应的原 NPZ。

全部视角拍摄成功后，程序向转台再次发送 0° 指令，默认等待 12 秒，再关闭相机、机械臂和转台串口，按视角顺序调用 `code/reconstruct_view.py`。结束回零命令记录在会话根目录的 `turntable_final_home.json`；串口发送和等待不等于实际到位反馈。每个结果写入 `processed/<object-id>/<session-id>/view_###/repeat_###/`，包含 `cloud_left.ply`、`qc.json` 和 `reconstruction_status.json`；会话级结果写入 `processed/<object-id>/<session-id>/reconstruction_summary.json`。重建失败不丢失原始照片，程序继续处理其余视角，最终以非零退出码报告失败；本阶段不做点云融合。

从项目根目录运行：

```powershell
python system\scan_session.py --port COM6 --arm-port COM7 --start-angle 0 --end-angle 330 --step-angle 30 --object-id object_001 --session-id session_01
```

默认参考为 `calibration/arm_fixed_board/board_repeat_01/two_pose_calibration.json` 中第一张下位姿定位图。**固定板必须与该标定时保持同一位置**；若重新安装固定板，先重做两位姿标定，并用 `--reference-calibration` 指向新结果。相机内参使用 `calibration/stereo/stereo_calibration.json`。机械臂默认等待 6 秒，转台默认等待 2.5 秒，可用 `--arm-settle-s`、`--motion-settle-s` 调整；串口命令与等待均不证明实际到位。

上位姿拍完 330° 后切换到下位姿并回到 0°，这一步单独默认等待 12 秒，可用 `--return-settle-s` 调整；其余转台角度仍使用 `--motion-settle-s`。若在定位图取帧时遇到相机 `The buffer was cancelled`，这是相机采集失败，不等同于转台未停稳；应同时检查相机 USB 连接、供电及 pylon 采集状态。已有不完整视角需用新的 `--repeat-id` 配合 `--resume` 续拍。

按空格开始预览后的扫描，Q 或 Esc 在运动前取消。每个完成的 `raw/<object-id>/<session-id>/view_###/repeat_###/` 中保存 `localization/left.png`、`localization/right.png`、`localization.json`、`pose.json` 和原有点阵文件。会话根目录的 `camera_poses.json` 使用 `view_###/repeat_###` 键，值为该次 `T_reference_from_left`，可传给 `fuse_views.py --camera-poses`。定位或点阵拍摄失败即停止；失败目录保留，用新 `--repeat-id` 和 `--resume` 继续。相机相对双目标定发生变化时须重标定。

如果拍摄已完成，使用相同拍摄参数加 `--resume` 会跳过拍摄和硬件连接，先校正双目，再补做失败或标定已变化的重建。也可单独检查已拍会话的板校正：`python system\board_stereo_correction.py --session-root raw\object_007\session_01`。重建自动估计深度范围；仅在双目校正已经通过、但自动视差范围仍失败时，才同时提供 `--reconstruction-min-depth-mm 550 --reconstruction-max-depth-mm 1100` 重试。重建输出目录可用 `--processed-output` 指定。重建使用的深度范围或标定文件内容改变时，会重新运行相应视角。

```powershell
python system\scan_session.py --port COM6 --arm-port COM7 --start-angle 0 --end-angle 330 --step-angle 30 --object-id object_001 --session-id session_01 --resume --reconstruction-min-depth-mm 550 --reconstruction-max-depth-mm 1100
```

本项目统一使用首次下位姿左相机作为定位与融合参考系。转台轴在相同的下位姿相机位置标定；融合时使用 `--axis-already-in-reference`。固定板、机械臂或相机位置改变后须重新核对这一关系。**会话板校正只修正扫描图的双目参数，不会修复旧转台轴结果**；`fuse_views.py` 会拒绝把使用旧标定求得的轴与修正标定点云混用。需要重新标定转台轴并把轴坐标变换到同一参考系，才能做正式融合。
