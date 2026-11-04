# Robot Arm and Stationary Board `arm/`

[中文](README.md) · [Experiment guide](../实验指导.en.md)

The arm carries the rigid stereo pair between two tested poses. The CLI uses `upper` and `lower`; scan manifests encode them as `arm_pose_id` 0 and 1. The stationary ChArUco board provides the camera pose for each scan and must not rotate with the turntable.

## Files

| File | Role |
| --- | --- |
| `arm.py` | Send preset serial commands; list, dry-run, connect, or read controller position feedback |
| `generate_charuco_board.py` | Generate the fixed-board print and geometry configuration |
| `fixed_board.py` | Detect ChArUco corners, estimate board/camera pose, and compare poses |
| `capture_fixed_board.py` | Capture repeated upper/lower board images, overlays, and a manifest |
| `targets/charuco_7x5_10mm_7mm_config.json` | Board dimensions and corner definitions; must match the measured print |
| `targets/*.png`, `targets/*.pdf` | Printable target artwork |
| `tests/` | Controller and board-geometry tests |

## Commands

From the project root, inspect a command before using the real serial port:

```powershell
python arm\arm.py --list
python arm\arm.py --pose upper --dry-run
python arm\arm.py --port COM7 --pose upper
python arm\arm.py --port COM7 --pose lower --readback
python arm\generate_charuco_board.py
python arm\capture_fixed_board.py --session-id board_check_01 --port COM7 --repeats 3
```

Use `--board` for a different board; camera serial, exposure, and minimum corner options can also be set. Capture defaults to `raw/arm_fixed_board/<session-id>/`. A serial write or timed wait alone does not establish the arm's actual pose; inspect the fixed-board detections. Tests: `python -m pytest arm/tests -q`.
