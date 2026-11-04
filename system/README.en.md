# Fixed-Order Scan Orchestration `system/`

[中文](README.md) · [Experiment guide](../实验指导.en.md)

`scan_session.py` coordinates the arm, turntable, cameras, projector, reconstruction, and fusion. It scans upper then lower pose by default, visiting the requested angle range in each. A full 0°–330° range in 30° steps produces 24 actions. A black frame first localizes the stationary board; dense dots and a white frame follow. Keep the board fixed relative to the turntable between axis calibration and object scanning.

## Files

| File | Role |
| --- | --- |
| `scan_session.py` | Capture session, preview, resume, reconstruct, summarize status, and fuse |
| `board_stereo_correction.py` | Derive session stereo extrinsic correction from paired board images while retaining intrinsics and baseline length |
| `fusion_pipeline.py` | Check axis images, prepare session geometry, call fusion and green-backdrop cleanup |
| `tests/` | Session recovery, stereo correction, and fusion tests |

## Run

First calibrate both arm poses with independent validation images and capture the full axis set in the lower pose. From the project root:

```powershell
python system\scan_session.py --port COM6 --arm-port COM7 --start-angle 0 --end-angle 330 --step-angle 30 --object-id object_001 --session-id session_01 --fusion-axis-images raw\turntable_axis\axis_01
```

Press Space in preview to start or Q/Esc to cancel. `--arm-pose-id 0` or `1` restricts the run to one pose. `--no-fusion` performs reconstruction only. After an interrupted capture, retain the same session settings and choose a new repeat ID:

```powershell
python system\scan_session.py --port COM6 --arm-port COM7 --start-angle 0 --end-angle 330 --step-angle 30 --object-id object_001 --session-id session_01 --fusion-axis-images raw\turntable_axis\axis_01 --resume --repeat-id 2
```

Resume skips complete views, refreshes board localization, and finishes needed reconstruction/fusion. If automatic depth-range estimation fails, specify both `--reconstruction-min-depth-mm` and `--reconstruction-max-depth-mm`. Adjust cropping with `--fusion-radius-mm`, `--fusion-height-mm MIN MAX`, and `--fusion-voxel-mm`; use `--no-green-filter` to keep the uncleaned output.

Raw images are under `raw/<object>/<session>/view_###/repeat_###/`; `camera_poses.json` sits at session root. Per-view `cloud_left.ply`, `qc.json`, and status files live in corresponding `processed/` paths. Final `fused.ply`, `fusion.json`, `aligned/`, and session axis results live under `processed/fused/<object>/<session>/`. A failure preserves raw images and exits nonzero. Tests: `python -m pytest system/tests -q`.
