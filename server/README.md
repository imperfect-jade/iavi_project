# AutoDL policy training

This folder is independent of the lab computer's `project/code/` acquisition and reconstruction programs. It implements a **candidate gain regressor** and a multi-view dataset builder, but not the final rollout evaluator. The smoke dataset is artificial and must never be used as an experimental result.

## AutoDL quick start

Create `/root/autodl-tmp/nbv_project` on AutoDL (`mkdir -p /root/autodl-tmp/nbv_project`), then copy this `server/` folder into it. From a **local PowerShell** terminal, with the actual AutoDL host and SSH port:

```powershell
cd 'D:\vscode\visual information\project'
$nbvPort = 12345  # replace with your SSH port
$nbvHost = 'your-instance-host'  # replace with your host
scp -P $nbvPort -r .\server "root@${nbvHost}:/root/autodl-tmp/nbv_project/"
```

Use the already working PyTorch 2.12.1+cu130 image; do not reinstall torch. In the **AutoDL Linux** terminal:

```bash
cd /root/autodl-tmp/nbv_project/server
python -m pip install numpy pyyaml
python -m pip check
python -m pip install pytest
python -m pytest tests -q
python make_smoke_data.py
python train.py --config config_smoke.yaml
```

Success means two epochs print and `runs/smoke_24/best_model.pt`, `last.pt`, `model_config.json`, and `history.jsonl` exist. The new output directory keeps 24-action checkpoints separate from earlier 36-action runs. `best_model.pt` contains only `state_dict` for inference. `last.pt` is a trusted local training checkpoint with optimizer and RNG states. Do not load an untrusted checkpoint with pickle.

For real data, build `datasets/train.npz` and `datasets/val.npz` as described below, replace the **measured fixed** workspace center and scale in `config_real.yaml`, then run `python train.py --config config_real.yaml`. Resume an interrupted run with the same command plus `--resume`; the trainer refuses resume if config or dataset SHA-256 changed. Keep input files and outputs on the data disk. Run in `tmux` for long jobs.

## Build the real dataset

Run `python -m pip install scipy` on AutoDL. Prepare one `.npy` cloud per object and action, each a nonempty `(P,3)` XYZ array in millimeters, already registered to `object_at_turntable_zero`. **Do not feed `cloud_left.ply` directly**: it is still in camera coordinates. Remove background and verify alignment before exporting these arrays. The builder refuses missing views; a failed scan is not a zero-gain action.

Create a JSON manifest whose `views` list has exactly 24 relative paths in action order (`view_id = arm_pose_id*12 + azimuth_index`; pose 0 is upper, pose 1 is lower). Required top-level fields are `unit: "mm"`, `coordinate_frame: "object_at_turntable_zero"`, `action_count: 24`, `initial_views: [0,6]` (only if these are the frozen initial actions), `tolerance_mm`, `voxel_mm`, `point_count: 2048`, `reference_max_points: 20000`, `states_per_object: 200`, `seed: 42`, and `objects`. Each object entry contains `object_id`, `split` (`train`, `val`, or `test`), and `views`. Use a separate object ID for each genuinely distinct object; all sessions and variants of one object belong to one split. The chosen voxel size and coverage tolerance must be justified from measured repeatability, not copied from an example. Existing 36-action datasets and checkpoints are incompatible with the 24-action protocol and must be regenerated.

```json
{
  "unit": "mm", "coordinate_frame": "object_at_turntable_zero",
  "action_count": 24, "initial_views": [0, 6],
  "tolerance_mm": 2.0, "voxel_mm": 1.0,
  "point_count": 2048, "reference_max_points": 20000,
  "states_per_object": 200, "seed": 42,
  "objects": [
    {"object_id": "object_001", "split": "train",
     "views": ["clouds/object_001/view_000.npy", "clouds/object_001/view_001.npy"]}
  ]
}
```

The example `views` array is abbreviated for readability and **must contain all 24 paths** before running. The builder generates random and perturbed greedy visited sets, sizes 2–9, then saves `train.npz`, `val.npz`, optional `test.npz`, `metadata.json`, and separate `references/<object_id>.npz` with the reference and Boolean coverage table. Reference files must remain accessible only to label generation and offline evaluation, never to the policy. It refuses to overwrite a nonempty output folder.

```bash
cd /root/autodl-tmp/nbv_project/server
python build_dataset.py --manifest dataset_manifest.json --output datasets
python train.py --config config_real.yaml
```

## NPZ sample contract

Each split is a compressed or uncompressed `.npz` archive containing arrays with first dimension `N` (number of states):

| Key | Shape / dtype | Meaning |
| --- | --- | --- |
| `points` | `(N,2048,3)` float32 | Current observed merged cloud in **object coordinates, millimeters**; pad with zero when needed |
| `point_mask` | `(N,2048)` bool | True for real observed points |
| `visited_mask` | `(N,24)` bool | Scanned action IDs |
| `available_mask` | `(N,24)` bool | Physically executable action IDs |
| `gain_target` | `(N,24)` float32, values 0–1 | Offline marginal coverage gain of each action |
| `current_action_id` | `(N,)` int64 | Current physical action ID; must be visited |
| `used_scans` | `(N,)` float32 | Number of complete scans, including initial scans |
| `object_id` | `(N,)` Unicode string | Split by object, never by state or view |

The loader derives the loss mask as `available_mask & ~visited_mask`. State inputs contain no reference cloud or unvisited view cloud. Labels must be computed offline from a uniform observable reference and per-view coverage table, using the frozen action map and tolerance described in `实验方案.md` §3.9. Keep references and held-out views outside the planner input path. Train and validation object IDs must be disjoint. Save the action map, protocol, pose and reconstruction version with the dataset when it is produced.

Current checkpoint selection uses **validation masked Smooth L1 loss**. Validation rollout AUC, random/fixed/greedy comparisons, and test evaluation require real per-view clouds and a separate evaluator; these are not represented by the synthetic smoke data. Do not treat a low synthetic loss as evidence that view planning works.
