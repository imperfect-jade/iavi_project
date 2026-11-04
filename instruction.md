# Experiment Guide: Stereo Structured Light and Active View Planning

[中文版](实验指导.md) · [Repository README](README.en.md)

This guide describes the **current implementation**. Run examples from `project/`. Replace example serial ports, object IDs, and paths for your setup. Module files and additional parameters are documented in [`code/`](code/README.en.md), [`arm/`](arm/README.en.md), [`turntable/`](turntable/README.en.md), [`system/`](system/README.en.md), [`server/`](server/README.en.md), and [`PointNet/`](PointNet/README.en.md). Local `raw/`, `processed/`, `calibration/`, and `models/` artifacts are not supplied by the repository. Python installation commands are in the [README](README.en.md#python-environment-and-dependencies).

## 1. Experiment and objective

**Title: Physical-object 3D scanning through stereo structured-light reconstruction and learning-based next-best-view selection.** A fixed projector illuminates an object on a turntable with dense dots. A rigid stereo pair carried by a robot arm captures multiple views. Stereo calibration, per-view reconstruction, camera-pose estimation, turntable-motion compensation, and fusion produce a 3D point cloud. The experiment compares a **full 24-view scan** with a **PointNet-style active scan** that chooses views one at a time.

The central question is whether the learned policy can approach the observable-surface coverage of a full scan with fewer complete scans, given identical candidate actions, capture protocol, and reconstruction. The full scan provides a dense-acquisition comparison; a complete reference cloud is used only for offline labels/evaluation, never as policy input. The deliverable is a colored fused PLY point cloud, not a watertight mesh. Evaluate scan counts and coverage on held-out physical objects; simulation alone cannot establish real-object results.

## 2. Equipment and software

| Equipment | Role |
| --- | --- |
| Two Basler Dart USB3 color cameras on a rigid stereo mount | Observe projected patterns and recover metric geometry |
| LG PF50KG-GL projector | Fixed projector displaying dense random dots from the computer |
| Robot arm and controller | Move the entire stereo pair between upper/lower preset poses; example port `COM7` |
| 360° turntable and controller | Rotate the object to twelve 30° positions per arm pose; example port `COM6` |
| Stereo checkerboard | Calibrate the two stereo configurations; example: 10×6 inner corners, measured 12 mm squares |
| Stationary ChArUco board | External fixed frame for per-view camera localization and fusion |
| Four-face ArUco cuboid | Rotating target for axis, angle, and tabletop-height calibration |
| Computer and Python | Local acquisition/reconstruction use OpenCV, NumPy, SciPy, pypylon, and pyserial; training uses PyTorch/PyYAML; ModelNet simulation needs Open3D |

Measure printed targets and ensure configuration matches their physical sizes. Install compatible Basler pylon and verify acquisition in Viewer first. Lengths are in millimeters. A serial write followed by a timed wait does not confirm physical arrival.

## 3. Shared geometry and the two workflows

Arm pose ID 0 is **upper**, ID 1 **lower**. Turntable angles are `0, 30, …, 330` degrees. The 24 actions use `action_id = arm_pose_id × 12 + turntable_deg / 30`: IDs 0–11 are upper and 12–23 lower. A scan means the complete projection/image sequence at one mechanical position, not one photo.

The current `code/capture.py` protocol is `dense_dots`: black, five random-dot frames by default, and white. Thus each view has **seven projection states and fourteen left/right images**. The slide's “7 Dense Dot Patterns” corresponds to these seven states; five are actual random-dot frames. Gray code/phase shifting belongs to an earlier design, not the current capture command. Keep `--dot-frames` and other acquisition settings fixed during comparisons.

Lower-pose stereo calibration establishes common intrinsics and metric baseline. Upper-pose calibration preserves them while fitting pose-specific stereo extrinsics. A black localization image detects the stationary board to obtain each left-camera pose. The rotating four-face target plus the same stationary board yields the axis in board coordinates. Each reconstructed cloud moves from left-camera coordinates into the board frame, then turntable rotation is undone before cropping and voxel fusion. The board must remain fixed relative to the turntable from axis calibration through object scanning.

### 3.1 Full 24-view reconstruction

`Stereo/axis calibration → capture twelve upper angles → capture twelve lower angles → board localization and coordinate alignment → stereo dot matching and per-view cloud reconstruction → turntable-motion compensation and fusion → final PLY`.

`system/scan_session.py` runs capture first and reconstructs views afterward. It automatically fuses only after all selected views reconstruct successfully. Per-view outputs include quality diagnostics and `cloud_left.ply`; fusion writes `fused.ply`, aligned view clouds, and reports. A 24-view cloud is a comparison baseline, not a guarantee of complete physical-surface coverage.

### 3.2 PointNet active scan

`Same calibration → initial upper 0° and 180° (actions 0 and 6) → fuse visited views only → adapt observed cloud to model frame → predict gain of unvisited actions → choose arm pose/turntable angle → capture, reconstruct, fuse → repeat until budget → final fusion`.

Use `PointNet/scan_pipeline.py` for hardware and `PointNet/replay.py` for hardware-free replay of a complete historical scan. The default budget is ten successful scans; prediction-score early stopping is **disabled by default**. The slide's “Score > Threshold” decision represents an optional mechanism: enable it explicitly with `--score-stop-enabled --score-threshold <value>`. V2 scores are not calibrated probabilities; validate any threshold before use. Model output suggests an action; the pipeline still performs movement, waiting, localization, and reconstruction.

## 4. Calibration, capture, reconstruction, and fusion

### 4.1 Stereo calibration

Set the arm to lower pose and keep camera identity, optics, focus, and exposure fixed. Capture roughly 25–30 training checkerboard pairs covering positions, distances, and tilts, then 5–10 **independent** validation pairs. In `capture.py calibration`, Space saves a pair and Q exits.

```powershell
python code\generate_stereo_board.py
python code\capture.py list-cameras
python code\capture.py calibration --session-id session_01
python code\capture.py calibration --session-id validation_01
python code\calibrate_stereo.py --pose lower --train raw\calibration\session_01 --validation raw\calibration\validation_01 --square-mm 12
```

Move the arm to upper pose and capture new training/validation pairs. Upper calibration retains the lower-pose intrinsics and baseline scale. Do not duplicate training images into validation.

```powershell
python code\capture.py calibration --session-id session_02
python code\capture.py calibration --session-id validation_02
python code\calibrate_stereo.py --pose upper --train raw\calibration\session_02 --validation raw\calibration\validation_02 --square-mm 12
```

Set `--square-mm` to the measured size. Outputs under `calibration/stereo_pose_lower/` and `calibration/stereo_pose_upper/` include NPZ, JSON, and error reports. Recheck/recompute upper calibration if lower intrinsics, optics, or mounts change. Inspect per-image errors and rectification examples, not only exit status.

### 4.2 Stationary board and turntable axis

Place the stationary ChArUco board outside the turntable and rigidly mount the ArUco cuboid on it. Set the arm to **lower** pose; turn off projection and ensure both targets are visible. The cuboid base must directly touch the turntable for automatic tabletop-height estimation.

```powershell
python turntable\capture_axis.py --port COM6 --session-id axis_01
python turntable\calibrate_axis.py --images raw\turntable_axis\axis_01
```

Capture records twelve angles plus a return-to-zero image under `raw/turntable_axis/axis_01/`. Calibration uses the lower stereo NPZ by default and writes `calibration/turntable_axis/axis_01/axis_calibration.json`. Inspect corner visibility, fit residual, and return error. If the board or turntable moves afterward, recalibrate the session relationship. `--settle-s` and `--return-settle-s` adjust waits.

### 4.3 Capture, reconstruct, and fuse

After both stereo calibrations and the current axis capture, run the full sequence. Space starts at preview; Q/Esc cancels.

```powershell
python system\scan_session.py --port COM6 --arm-port COM7 --start-angle 0 --end-angle 330 --step-angle 30 --object-id object_001 --session-id session_01 --fusion-axis-images raw\turntable_axis\axis_01
```

Raw images are in `raw/object_001/session_01/view_###/repeat_###/`; session `camera_poses.json` holds camera poses. Per-view `cloud_left.ply`, `qc.json`, and reconstruction status are in matching `processed/object_001/session_01/view_###/repeat_###/` paths. The final cloud is `processed/fused/object_001/session_01/all_views_axis_calibrated/fused.ply`, alongside `fusion.json`, `aligned/`, and a session axis copy. Green-backdrop cleanup runs by default; uncleaned results remain in `unfiltered/`. For green objects, `--no-green-filter` avoids color-based removal.

Resume an interrupted capture with unchanged acquisition settings and a fresh repeat ID:

```powershell
python system\scan_session.py --port COM6 --arm-port COM7 --start-angle 0 --end-angle 330 --step-angle 30 --object-id object_001 --session-id session_01 --fusion-axis-images raw\turntable_axis\axis_01 --resume --repeat-id 2
```

To inspect reconstruction of one view or manually fuse a completed session:

```powershell
python code\reconstruct_view.py --view raw\object_001\session_01\view_000\repeat_001
python code\fuse_views.py --session-root raw\object_001\session_01 --processed-root processed --camera-poses raw\object_001\session_01\camera_poses.json --axis-calibration calibration\turntable_axis\axis_01\axis_calibration.json --axis-already-in-reference --output processed\fused\object_001\session_01\manual
```

Use the axis calibration corresponding to that scan; camera poses and axis must share the stationary-board frame. If automatic depth-range estimation fails, give the orchestrator both `--reconstruction-min-depth-mm` and `--reconstruction-max-depth-mm`. Crop/voxel controls are in [`system/README.en.md`](system/README.en.md). Inspect `qc.json` and point-cloud alignment for double surfaces, scale errors, and background residue.

## 5. PointNet training and use

`server/model.py` regresses candidate-action gains from PointNet-style cloud features and action history. Offline reference coverage determines each action's *new* surface coverage for training labels. Split by object, so views of one object never cross train/validation/test. Online inference sees only the observed cloud and visited/available action masks.

### 5.1 ModelNet simulation and training

Download/extract ModelNet40 separately; the directory needs `<category>/train/*.off` and `<category>/test/*.off`. On Linux with a GPU, `server/run_train_200.sh` is the current full experiment: ten categories and **200 CAD objects** (160 train, 20 validation, 20 test), each with 24 virtual views. The slide labels “100 open-source object dataset”; the current script uses 200 objects, so report the configuration actually run.

```bash
cd server
python -m pip install numpy scipy pyyaml open3d pytest
# Install PyTorch separately for the server's CUDA version.
MODELNET_ROOT=/path/to/ModelNet40 bash run_train_200.sh
```

Outputs include `sim_data_diverse_200/`, generated `config_diverse_200_v2.yaml`, and `runs/diverse_200_v2_seed42_24/` with `best_model.pt`, resume data, logs, and simulated test replay. For a pipeline check, run `python make_smoke_data.py` then `python train.py --config config_smoke.yaml`; synthetic smoke data is not an experiment result. The simulation approximates jointly visible CAD geometry, not reflectance, dot-matching errors, or device repeatability. Validate on unseen physical objects.

For measured-data training, first align all 24 per-object view clouds and export millimeter `(P,3)` `.npy` files. `server/build_dataset.py` uses a JSON manifest to produce `train.npz`, `val.npz`, and coverage labels. Camera-frame `cloud_left.ply` cannot be used directly. Set data paths and measured workspace center/scale in `server/config_real.yaml`:

```bash
cd server
python build_dataset.py --manifest dataset_manifest.json --output datasets
python train.py --config config_real.yaml
```

### 5.2 Replay and live deployment

Transfer matching V2 weights and configuration to the local machine; the repository does not contain them. Pass the configuration via `--config`. First replay a complete historical session without connecting devices:

```powershell
python PointNet\replay.py --raw-session raw\object_001\session_01 --processed-root processed --axis-calibration calibration\turntable_axis\axis_01\axis_calibration.json --output PointNet\outputs\replay_object_001 --budget 10 --config models\diverse_200_v2_seed42_24\config_local.yaml
```

For a live active scan, use a new session and complete axis images recorded with the stationary board at its current position:

```powershell
python PointNet\scan_pipeline.py --object-id object_001 --session-id session_02 --axis-images raw\turntable_axis\axis_02 --port COM6 --arm-port COM7 --config models\diverse_200_v2_seed42_24\config_local.yaml --max-scans 10
```

The program scans actions 0 and 6, then selects from unvisited actions using only observed clouds. Each new action must finish acquisition and reconstruction before the next decision. Iteration state/scores and the final fusion are saved. To inspect one prediction without moving hardware, use `PointNet/predict.py` as shown in [`PointNet/README.en.md`](PointNet/README.en.md). Any gain or stopping claim should use held-out physical objects, fixed scan budgets, and identical reconstruction settings.
