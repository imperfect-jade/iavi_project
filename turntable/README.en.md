# Turntable and Axis Calibration `turntable/`

[中文](README.md) · [Experiment guide](../实验指导.en.md)

The turntable exposes twelve tested positions, 0° through 330° in 30° steps. A rotating four-face ArUco cuboid and a stationary ChArUco board establish the rotation axis in the board frame. Capture axis images with the arm in the **lower** pose, and keep the board fixed relative to the turntable for subsequent object scans.

## Files

| File | Role |
| --- | --- |
| `turntable.py` | Map tested angles to serial commands; list, send once, or run interactively |
| `generate_cube_target.py` | Generate the four-face target print and 3D corner coordinates |
| `capture_axis.py` | Capture twelve angles plus a return-to-zero image and check marker visibility |
| `calibrate_axis.py` | Triangulate target corners, fit axis/angles/tabletop, and transform into the stationary-board frame |
| `targets/aruco_cube_4faces_config.json` | Measured cuboid corners and face order; remeasure after changing the physical target |
| `targets/*.png`, `targets/*.pdf` | Printable target artwork |
| `tests/` | Controller, target, and calibration tests |

## Commands

```powershell
python turntable\turntable.py --list
python turntable\turntable.py --angle 90 --dry-run
python turntable\turntable.py --port COM6 --angle 90
python turntable\turntable.py --port COM6
python turntable\generate_cube_target.py
python turntable\capture_axis.py --port COM6 --session-id axis_01
python turntable\calibrate_axis.py --images raw\turntable_axis\axis_01
```

Interactive mode accepts an angle or `q`. Capture writes images and `capture_manifest.json` under `raw/turntable_axis/axis_01/`; calibration defaults to `calibration/turntable_axis/axis_01/axis_calibration.json`. Use `--settle-s` and `--return-settle-s` to adjust motion waits, or `--stereo`/`--output` to choose calibration input/output. Fix the cuboid rigidly with its base directly on the turntable to estimate tabletop height. A sent command is not verified position feedback. Tests: `python -m pytest turntable/tests -q`.
