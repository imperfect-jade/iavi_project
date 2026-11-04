# PointNet Active Scanning `PointNet/`

[中文](README.md) · [Experiment guide](../实验指导.en.md)

This module connects the 24-action gain model trained in `server/` to existing devices and reconstruction. A live scan starts with upper-pose 0° and 180° (actions 0 and 6), then fuses **visited views only**, predicts the next action, and repeats. This is separate from the fixed-order [`system/`](../system/README.en.md) workflow.

## Files

| File | Role |
| --- | --- |
| `adapter.py` | Read aligned PLY clouds and adapt visited observations to the model object frame |
| `fusion.py` | Session axis calibration and visited-only fusion |
| `policy.py` | Validate checkpoint/configuration and infer candidate gains |
| `predict.py` | One-shot action prediction or model-input preparation from an existing fusion |
| `replay.py` | Offline stepwise replay of a complete historical scan; no hardware connection |
| `scan_step.py` | Move, localize, acquire, and reconstruct one action |
| `scan_pipeline.py` | Live active-scan entry point, state saving, and resume |
| `tests/` | Adapter, policy, replay, scan-step, and pipeline tests |

`models/` is excluded from the repository. Supply matching trained weights and configuration. The default configuration path is `models/diverse_200_v2_seed42_24/config_local.yaml`; `--config` can override it.

## One-shot prediction

Run from the project root. Input must include one session's `fusion.json`, `aligned/*.ply`, raw `camera_poses.json`, and axis/tabletop information. A camera-frame `cloud_left.ply` cannot be passed directly. The first call needs tabletop height in that axis frame; subsequent calls reuse `object_frame.json` in the same output directory.

```powershell
python PointNet\predict.py --fusion-dir processed\fused\object_001\session_01\all_views_axis_calibrated --raw-session raw\object_001\session_01 --visited 0 6 --table-height-axis-mm -95 --output-dir PointNet\outputs\object_001\session_01 --config models\diverse_200_v2_seed42_24\config_local.yaml
```

`-95` is illustrative: use the height measured for this axis calibration. `--prepare-only` writes `observed.npy` and `object_frame.json` without loading the model; prediction also writes `prediction.json`. Append each newly completed action to `--visited`, keeping the output directory. Use `--available` to restrict executable actions.

## Offline replay and live scan

Replay requires complete raw and reconstructed historical views:

```powershell
python PointNet\replay.py --raw-session raw\object_001\session_01 --processed-root processed --axis-calibration calibration\turntable_axis\axis_01\axis_calibration.json --output PointNet\outputs\replay_object_001 --budget 10 --config models\diverse_200_v2_seed42_24\config_local.yaml
```

`state.json` records actions and stop reason; `steps/` holds each iteration's input/scores; `final_fusion/` holds the final cloud. Use `--resume` to rerun an existing output. For live scanning, first calibrate both stereo poses and provide complete axis images taken with the board at its current position:

```powershell
python PointNet\scan_pipeline.py --object-id object_001 --session-id session_02 --axis-images raw\turntable_axis\axis_02 --port COM6 --arm-port COM7 --config models\diverse_200_v2_seed42_24\config_local.yaml --max-scans 10
```

Press Space to start or Q to cancel at preview. After failure, keep parameters and use `--resume --repeat-id 2`; successful actions are not recaptured. Results are under `PointNet/outputs/<object>/<session>/` and `processed/fused/<object>/<session>/`. Score-based stopping is disabled by default; raw V2 scores are not calibrated probabilities. Green objects may be affected by default backdrop cleanup; compare `--use-unfiltered-model-input` or see [`../code/README.en.md`](../code/README.en.md). Tests: `python -m pytest PointNet/tests -q`.
