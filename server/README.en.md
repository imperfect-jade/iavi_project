# View-Gain Model, Simulation, and Training `server/`

[中文](README.md) · [Experiment guide](../实验指导.en.md)

This module trains a PointNet-style gain model over 24 actions. Pose IDs 0/1 denote upper/lower; `action_id = arm_pose_id * 12 + azimuth_index`, with twelve angles per pose. Policy input contains observed clouds only. Full reference clouds are reserved for offline labels and evaluation. ModelNet simulation approximates geometry visibility, not real structured-light imaging.

## Files

| File | Role |
| --- | --- |
| `model.py` | Base/V2 gain networks, masked loss, and action selection |
| `data.py` | Validate NPZ samples and disjoint object splits |
| `build_dataset.py` | Build supervised states and coverage tables from aligned per-action clouds |
| `train.py` | Train/validate from YAML, save best weights, and resume |
| `predict_next_view.py` | Predict one action from an observed `(N,3)` mm cloud; no hardware control |
| `simulate_modelnet.py` | Generate 24 virtual clouds from ModelNet OFF meshes |
| `evaluate_sim.py` | Compare learned, random, and baseline policies on held-out simulated objects |
| `make_smoke_data.py` | Artificial samples for pipeline checks only |
| `run_train_200.sh` | Linux script for 200 multi-category CAD objects, V2 training, and evaluation |
| `sim_config.json` | Example virtual camera, projector, object, and sampling settings |
| `config_smoke.yaml`, `config_sim.yaml`, `config_sim_v2.yaml`, `config_real.yaml` | Smoke, base simulation, V2 simulation, and real-data templates |
| `tests/` | Data, simulation, model, prediction, and evaluation tests |

## Setup and smoke check

Run inside `server/`. Dependencies include NumPy, SciPy, PyYAML, PyTorch, and pytest; simulation also needs Open3D. Install the PyTorch build appropriate for the host CPU/CUDA and set `runtime.device` accordingly.

```bash
cd server
python make_smoke_data.py
python train.py --config config_smoke.yaml
python -m pytest tests -q
```

Smoke output under `runs/` checks the pipeline only; do not report it as experimental performance.

## ModelNet and measured data

Download/extract ModelNet40 separately, with `<category>/train/*.off` and `<category>/test/*.off`. Small simulation example:

```bash
python simulate_modelnet.py --modelnet-root /path/to/ModelNet40 --output sim_data --categories bottle bowl --max-train 10 --max-val 3 --max-test 3
python train.py --config config_sim.yaml
python evaluate_sim.py --sim-root sim_data --config config_sim.yaml
```

Check the data paths and output directory in `config_sim.yaml` first. The complete Linux GPU run is `MODELNET_ROOT=/path/to/ModelNet40 bash run_train_200.sh`; it checks CUDA and uses 160 train, 20 validation, and 20 test objects. Large simulation requires considerable CPU time and disk space.

For measured-data training, each object needs all 24 nonempty `(P,3)` `.npy` view clouds in millimeters aligned to `object_at_turntable_zero`. The JSON manifest lists 24 relative paths in `objects[].views` and metadata including `unit`, `coordinate_frame`, `action_count`, `initial_views`, and `tolerance_mm`; see `build_dataset.py` validation. Camera-frame `cloud_left.ply` is not directly suitable.

```bash
python build_dataset.py --manifest dataset_manifest.json --output datasets
python train.py --config config_real.yaml
python train.py --config config_real.yaml --resume
python predict_next_view.py --config config_real.yaml --points current_observed.npy --visited 0 6
```

Training writes `best_model.pt`, `last.pt`, configuration, and history. Replace placeholder workspace center/scale in `config_real.yaml` before training. Keep objects disjoint across train/validation/test. Local scanning and replay are in [`../PointNet/README.en.md`](../PointNet/README.en.md).
