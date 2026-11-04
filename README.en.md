# Stereo Structured Light Reconstruction and Active View Planning

[中文](README.md) · [Experiment guide](实验指导.en.md) · [实验指导](实验指导.md)

This repository implements an experimental desktop object scanner. A fixed projector displays dense random dots, while a rigid stereo camera pair moves with a robot arm. Stereo reconstruction produces per-view point clouds; a stationary ChArUco board and a calibrated turntable axis align them. The project also includes a PointNet-style next-view policy, ModelNet geometry simulation, training, and offline replay. See the [experiment guide](实验指导.en.md) for the objective, hardware, workflows, and commands.

> Current acquisition and reconstruction support `dense_dots` only. Simulation coverage and software replay do not establish real-object performance.

## Repository map

| Path | Purpose | Notes |
| --- | --- | --- |
| [`code/`](code/README.en.md) | Acquisition, stereo calibration, reconstruction, fusion, and color cleanup | Core vision commands |
| [`arm/`](arm/README.en.md) | Arm control and stationary ChArUco board | Two preset camera poses |
| [`turntable/`](turntable/README.en.md) | Turntable control, four-face ArUco target, axis calibration | Twelve 30° angles |
| [`system/`](system/README.en.md) | Fixed-order acquisition, reconstruction, and fusion | Full 24-view scan |
| [`server/`](server/README.en.md) | Dataset building, ModelNet simulation, training, evaluation | Linux GPU workflow available |
| [`PointNet/`](PointNet/README.en.md) | Active real scanning, one-shot prediction, historical replay | Requires trained weights |
| `configs/` | Example color-chart corner configuration | `color_chart_corners.json` |
| `tests/` and module `tests/` | Unit and workflow tests | `python -m pytest tests arm/tests turntable/tests system/tests PointNet/tests server/tests -q` |

`raw/`, `processed/`, `calibration/`, `models/`, and training outputs are local data or generated artifacts excluded by `.gitignore`. A fresh clone does not include calibrations, captures, or model weights.

## Python environment and dependencies

Python 3.11 is recommended locally. Install Basler pylon compatible with the cameras before using pypylon. ChArUco/ArUco requires `opencv-contrib-python`; do not install `opencv-python` alongside it in the same environment. From the project root on Windows:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install numpy opencv-contrib-python scipy pypylon pyserial pyyaml pillow open3d pytest
```

Training and PointNet inference additionally require PyTorch. Choose a platform-specific CPU or CUDA command from the [official PyTorch installation page](https://pytorch.org/get-started/locally/). For a CPU environment:

```powershell
.\.venv\Scripts\python.exe -m pip install torch
```

The repository does not include a unified dependency lockfile. Match `runtime.device` in the training configuration to the installed PyTorch device. Commands below assume an activated environment; alternatively use `.\.venv\Scripts\python.exe`.

## Actions and quick start

Run commands from this directory. Example serial ports are `COM6` for the turntable and `COM7` for the arm. Pose ID 0 is upper, ID 1 is lower. `action_id = arm_pose_id × 12 + turntable_deg / 30`, giving 24 actions. Geometry uses millimeters.

```powershell
# Capture separate lower-pose training and validation sets before calibration.
python code\capture.py calibration --session-id session_01
python code\capture.py calibration --session-id validation_01
python code\calibrate_stereo.py --pose lower --train raw\calibration\session_01 --validation raw\calibration\validation_01 --square-mm 12
# Also calibrate the upper pose; see code/README.en.md.
python turntable\capture_axis.py --port COM6 --session-id axis_01
python turntable\calibrate_axis.py --images raw\turntable_axis\axis_01
python system\scan_session.py --port COM6 --arm-port COM7 --start-angle 0 --end-angle 330 --step-angle 30 --object-id object_001 --session-id session_01 --fusion-axis-images raw\turntable_axis\axis_01
```

The full stereo procedure is in [`code/README.en.md`](code/README.en.md); active scanning is in [`PointNet/README.en.md`](PointNet/README.en.md), and training in [`server/README.en.md`](server/README.en.md).

Raw images reside in `raw/<object>/<session>/view_###/repeat_###/`; reconstructed per-view clouds in the corresponding `processed/` path; the usual final cloud is `processed/fused/<object>/<session>/all_views_axis_calibrated/fused.ply`. Keep the stationary board fixed relative to the turntable during axis calibration and scanning. A sent serial command plus a timed wait does not confirm physical arrival.
