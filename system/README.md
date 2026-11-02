# 联动扫描调度

`scan_session.py` 使用 COM7 控制机械臂上、下两位姿，COM6 控制转台。默认按上位姿的角度序列、下位姿的角度序列拍摄；`--arm-pose-id 0` 或 `1` 可只拍一个位姿。每个视角按 `black → dense_dots → white` 拍摄：第一张 black 是投影全零的左右固定板定位图，左图至少识别 12 个角点且重投影 RMS 不超过 3 px 才继续拍点阵。原始 black 图保存在 `left/000_black.png`、`right/000_black.png`，同一次曝光复制到 `localization/` 供逐视角定位使用。

双目几何按机械臂位姿分别标定，由同一个 `calibrate_stereo.py` 命令选择 `--pose lower` 或 `--pose upper`。先在**下位姿**拍 `raw/calibration/session_01` 和独立验证组 `validation_01`，拟合共同内参、畸变、基线与下位姿外参；再在**上位姿**拍训练组 `session_02` 和独立验证组 `validation_02`，固定共同内参，仅估计上位姿外参并沿用同一毫米基线。上位姿还从训练组留出每四张中的一张检查稳定性；最终模型必须通过独立验证组的逐图极线检查。每张中位误差上限为 2 px、p95 上限为 3 px，拟合基线与下位姿差异上限为 5%。下位姿文件更新后须重新生成上位姿文件。扫描时固定板用于逐视角相机定位。

```powershell
python code\calibrate_stereo.py --pose lower --train raw\calibration\session_01 --validation raw\calibration\validation_01 --square-mm 12
python code\calibrate_stereo.py --pose upper --train raw\calibration\session_02 --validation raw\calibration\validation_02 --square-mm 12
```

固定板的实测方格边长必须与 `arm/targets/charuco_7x5_10mm_7mm_config.json` 一致，且在整次扫描中固定不动。它提供各视角相机相对转台的位姿；双目外参来自上述大棋盘标定。下位姿的 NPZ、JSON、逐图报告和校正示例全部位于 `calibration/stereo_pose_lower/`；上位姿的 NPZ、JSON 和独立验证逐图结果位于 `calibration/stereo_pose_upper/`。扫描程序可用 `--stereo-upper`、`--stereo-lower` 指定其他结果文件。

全部视角拍摄成功后，程序向转台再次发送 0° 指令，默认等待 12 秒，再关闭相机、机械臂和转台串口，按视角顺序调用 `code/reconstruct_view.py`。结束回零命令记录在会话根目录的 `turntable_final_home.json`；串口发送和等待不等于实际到位反馈。每个结果写入 `processed/<object-id>/<session-id>/view_###/repeat_###/`，包含 `cloud_left.ply`、`qc.json` 和 `reconstruction_status.json`；会话级结果写入 `processed/<object-id>/<session-id>/reconstruction_summary.json`。重建失败不丢失原始照片，程序继续处理其余视角，最终以非零退出码报告失败。**全部视角重建成功后，默认自动运行固定板/转轴联合标定和点云融合。**

从项目根目录运行：

```powershell
python system\scan_session.py --port COM6 --arm-port COM7 --start-angle 0 --end-angle 330 --step-angle 30 --object-id object_001 --session-id session_01 --fusion-axis-images raw\turntable_axis\axis_cube_new
```

扫描直接采用静止 ChArUco 板自身为统一参考系。每个视角都从当次左定位图计算 `T_board_from_left`，无需预先标定机械臂运动。固定板在转轴标定与物体扫描之间必须与转台保持相同的相对位置。上位姿 `view_000–011` 使用上位姿双目标定，下位姿 `view_012–023` 使用下位姿标定，两者共享左右内参和基线长度。机械臂默认等待 6 秒，转台默认等待 2.5 秒，可用 `--arm-settle-s`、`--motion-settle-s` 调整；串口命令与等待均不证明实际到位。

上位姿拍完 330° 后切换到下位姿并回到 0°，这一步单独默认等待 12 秒，可用 `--return-settle-s` 调整；其余转台角度仍使用 `--motion-settle-s`。若在定位图取帧时遇到相机 `The buffer was cancelled`，这是相机采集失败，不等同于转台未停稳；应同时检查相机 USB 连接、供电及 pylon 采集状态。已有不完整视角需用新的 `--repeat-id` 配合 `--resume` 续拍。

按空格开始预览后的扫描，Q 或 Esc 在运动前取消。每个完成的 `raw/<object-id>/<session-id>/view_###/repeat_###/` 中保存 `localization/left.png`、`localization/right.png`、`localization.json`、`pose.json` 和原有点阵文件。会话根目录的 `camera_poses.json` 使用 `view_###/repeat_###` 键，值为该次 `T_reference_from_left`；这里的 reference 就是固定板，可传给 `fuse_views.py --camera-poses`。定位或点阵拍摄失败即停止；失败目录保留，用新 `--repeat-id` 和 `--resume` 继续。相机相对双目标定发生变化时须重标定。

如果拍摄已完成，使用相同拍摄参数加 `--resume` 会跳过拍摄和硬件连接，用当前共同内参从选中视角的 black 图重算 `camera_poses.json`，再补做失败或对应位姿标定已变化的重建，最后重新融合。重建自动估计深度范围；仅在自动视差范围失败时，才同时提供 `--reconstruction-min-depth-mm 550 --reconstruction-max-depth-mm 1100` 重试。重建输出目录可用 `--processed-output` 指定。

```powershell
python system\scan_session.py --port COM6 --arm-port COM7 --start-angle 0 --end-angle 330 --step-angle 30 --object-id object_001 --session-id session_01 --fusion-axis-images raw\turntable_axis\axis_cube_new --resume --reconstruction-min-depth-mm 550 --reconstruction-max-depth-mm 1100
```

## 自动融合

自动融合必须用 `--fusion-axis-images raw/turntable_axis/<session-id>` 显式指定新拍的完整联合标定照片；若只重建而不融合，使用 `--no-fusion`。自动融合**用下位姿双目标定重新解算轴**，同时从轴照片的多张静止固定板图像求出固定板位姿，直接输出板坐标系下的转轴。默认每张可用轴照片至少需要 18 个 ChArUco 角点、定位 RMS 不超过 1 px；至少 6 张板位姿须在 3 mm / 0.5° 内一致。轴拟合 RMS 超过 3 mm 或回零角点位移超过 2 px 时融合停止并报告错误。

`scan_session.py` 自动按每张扫描清单的 `arm_pose_id` 选择双目标定并记录文件哈希。两套标定须使用相同的相机序列号、图像尺寸、左右内参与畸变和基线长度，并提供固定板坐标系下的逐视角相机位姿与转轴；相对旋转和基线方向可以随机械臂位姿不同。`--resume` 会重新计算固定板相机位姿。单独调用 `reconstruct_view.py --view ...` 时也会按清单里的位姿选标定，可用 `--calibration` 显式覆盖。

默认物体裁剪为轴周围半径 65 mm、高度 -95～110 mm，体素 1 mm；可用 `--fusion-radius-mm`、`--fusion-height-mm MIN MAX`、`--fusion-voxel-mm` 按物体尺寸调整。默认融合所有选定视角；疑似异常视角可用 `--fusion-exclude-view 11` 排除，多次传入可排除多个视角。`--resume` 时可以改变这些融合参数而不重新拍摄。确需只重建、不融合时使用 `--no-fusion`。

最终点云在 `processed/fused/<object-id>/<session-id>/all_views_axis_calibrated/fused.ply`，同时保存 `fusion.json`、逐视角 `aligned/*.ply`、会话专属 `fusion_calibration/axis_calibration.json` 和 `fusion_status.json`。默认只选每个视角最新的完整 repeat；融合失败会以非零退出码报告并保留原始照片和已完成的重建。裁剪范围和轴质量门限需按实际物体核对，自动结果是点云而非封闭网格。

相机位姿和转轴均以静止固定板为参考系。扫描与转轴标定期间，固定板相对转台的位置必须一致。
