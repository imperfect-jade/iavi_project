# Core Vision Module `code/`

[中文](README.md) · [Experiment guide](../实验指导.en.md)

Run these commands from the project root. The current projection protocol is `dense_dots`: each view captures black, five dot frames by default, and white, with left/right images. Geometry uses millimeters. Acquisition needs Basler pylon/pypylon and OpenCV windows.

## Files

| File | Role |
| --- | --- |
| `capture.py` | List cameras/displays, test projection, capture checkerboard/color-chart pairs, and save structured-light scans with manifests |
| `generate_stereo_board.py` | Generate an A4 stereo checkerboard print |
| `calibrate_stereo.py` | Calibrate lower stereo geometry and fit upper-pose extrinsics with shared intrinsics/baseline; save NPZ/JSON and validation reports |
| `upper_stereo_fit.py` | Upper-pose extrinsic fitting and error checks called by `calibrate_stereo.py` |
| `reconstruct_view.py` | Rectify images, match dense dots, reproject disparity, and output one view's PLY and quality report |
| `fuse_views.py` | Transform views using camera poses and turntable axis, crop, and voxel-fuse them |
| `estimate_table_height.py` | Read tabletop height from the axis calibration |
| `clean_green.py` | Remove green backdrop using color, radius, and tabletop height |
| `calibrate_color.py` | Optional right-to-left BGR color mapping; not used by geometric matching |
| `tests/` | Calibration, capture, reconstruction, fusion, and upper-pose tests |

## Calibrate and capture

```powershell
python code\generate_stereo_board.py
python code\capture.py list-cameras
python code\capture.py list-displays
python code\capture.py projector-test --display 1
# Lower pose: capture distinct training and validation pairs. Space saves; Q quits.
python code\capture.py calibration --session-id session_01
python code\capture.py calibration --session-id validation_01
python code\calibrate_stereo.py --pose lower --train raw\calibration\session_01 --validation raw\calibration\validation_01 --square-mm 12
# Move to upper pose and capture two new sets.
python code\capture.py calibration --session-id session_02
python code\capture.py calibration --session-id validation_02
python code\calibrate_stereo.py --pose upper --train raw\calibration\session_02 --validation raw\calibration\validation_02 --square-mm 12
```

Use the measured checkerboard square size. Outputs default to `calibration/stereo_pose_lower/` and `calibration/stereo_pose_upper/`. Upper calibration depends on the lower result. Set actual camera serials, display, and exposure as needed.

```powershell
python code\capture.py scan --object-id object_001 --session-id session_01 --view-id 0 --arm-pose-id 0 --turntable-deg 0 --display 1
python code\reconstruct_view.py --view raw\object_001\session_01\view_000\repeat_001
```

Keep object and stereo rig stationary throughout the image sequence. Reconstruction selects the calibration by pose ID in the manifest unless `--calibration` overrides it. `cloud_left.ply` remains in the left-camera frame.

## Fuse and clean

Normally `system/scan_session.py` calls fusion. For manual fusion provide matching camera poses and axis calibration. Use `--axis-already-in-reference` when the axis is already in the stationary-board frame:

```powershell
python code\fuse_views.py --session-root raw\object_001\session_01 --processed-root processed --camera-poses raw\object_001\session_01\camera_poses.json --axis-calibration calibration\turntable_axis\axis_01\axis_calibration.json --axis-already-in-reference --output processed\fused\object_001\session_01\manual
python code\clean_green.py --fusion-dir processed\fused\object_001\session_01\manual
```

Check coordinate frames, pose estimates, crop radius, and reconstruction status before fusion. Optional color fitting: `python code\calibrate_color.py --images raw\color\session_01 --corners configs\color_chart_corners.json`. Use `python code\<script>.py --help` for all parameters.
