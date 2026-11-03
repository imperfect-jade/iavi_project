#!/usr/bin/env bash
# One command: 200 diverse ModelNet objects -> V2 PointNet -> test rollout.
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export PYTHONUNBUFFERED=1

MODELNET_ROOT="${MODELNET_ROOT:-/root/autodl-tmp/ModelNet40}"
BATCH_SIZE="${BATCH_SIZE:-512}"
CATEGORIES=(bottle bowl cup vase flower_pot lamp stool chair table guitar)
SIM_ROOT="sim_data_diverse_200"
CONFIG="config_diverse_200_v2.yaml"
RUN_DIR="runs/diverse_200_v2_seed42_24"

if ! [[ "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "BATCH_SIZE must be a positive integer" >&2
  exit 2
fi
mkdir -p "$RUN_DIR"
exec > >(tee -a "$RUN_DIR/pipeline.log") 2>&1
trap 'echo "Stopped at line $LINENO; see $RUN_DIR/pipeline.log" >&2' ERR

echo "Starting 200-object experiment: 160 train + 20 val + 20 test"
echo "Categories: ${CATEGORIES[*]}"
echo "ModelNet root: $MODELNET_ROOT"
df -h /root/autodl-tmp

MODELNET_ROOT="$MODELNET_ROOT" NBV_CATEGORIES="${CATEGORIES[*]}" python - <<'PY'
import os
import json
from pathlib import Path

import open3d
import scipy
import torch

from simulate_modelnet import split_meshes

if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable; stop before the long CPU simulation")
root = Path(os.environ["MODELNET_ROOT"])
cfg = json.loads(Path("sim_config.json").read_text(encoding="utf-8"))
categories = os.environ["NBV_CATEGORIES"].split()
groups = split_meshes(root, categories, cfg["val_fraction"], cfg["seed"])
for category in categories:
    counts = {split: sum(p.parent.parent.name == category for p in paths)
              for split, paths in groups.items()}
    print(category, counts, flush=True)
    if any(counts[split] < minimum for split, minimum in
           (("train", 16), ("val", 2), ("test", 2))):
        raise ValueError(f"{category}: not enough ModelNet meshes for the 200-object split")
print("torch", torch.__version__, "CUDA", torch.version.cuda,
      "GPU", torch.cuda.get_device_name(0),
      "Open3D", open3d.__version__, "SciPy", scipy.__version__, flush=True)
PY

if [[ ! -e "$SIM_ROOT" ]]; then
  python simulate_modelnet.py --modelnet-root "$MODELNET_ROOT" \
    --output "$SIM_ROOT" --categories "${CATEGORIES[@]}" \
    --max-train 160 --max-val 20 --max-test 20
fi

NBV_CATEGORIES="${CATEGORIES[*]}" python - <<'PY'
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
from build_dataset import read_manifest

root = Path("sim_data_diverse_200")
manifest = read_manifest(root / "manifest.json")
expected = {"train": 160, "val": 20, "test": 20}
counts = Counter(obj["split"] for obj in manifest["objects"])
if dict(counts) != expected:
    raise ValueError(f"simulation has {dict(counts)} objects, expected {expected}")
categories = os.environ["NBV_CATEGORIES"].split()
by_category = {split: Counter(Path(obj["source_mesh"]).parent.parent.name
                              for obj in manifest["objects"] if obj["split"] == split)
               for split in expected}
for category in categories:
    if any(by_category[split][category] < minimum for split, minimum in
           (("train", 12), ("val", 1), ("test", 1))):
        raise ValueError(f"{category}: too few valid objects in the final split")
print("Objects by category:", by_category, flush=True)
if manifest["action_count"] != 24 or manifest["initial_views"] != [0, 6]:
    raise ValueError("simulation does not use the 2 x 12 action protocol")
for split, count in expected.items():
    path = root / "datasets" / f"{split}.npz"
    with np.load(path, allow_pickle=False) as data:
        states = len(data["object_id"])
    if states != count * 200:
        raise ValueError(f"{split}: found {states} states, expected {count * 200}")
print("Dataset verified:", expected, "objects; 200 states/object", flush=True)
PY

BATCH_SIZE="$BATCH_SIZE" python - <<'PY'
import os
from pathlib import Path

import yaml

template = yaml.safe_load(Path("config_sim_v2.yaml").read_text(encoding="utf-8"))
template["data"]["train"] = "sim_data_diverse_200/datasets/train.npz"
template["data"]["val"] = "sim_data_diverse_200/datasets/val.npz"
template["training"]["batch_size"] = int(os.environ["BATCH_SIZE"])
template["training"]["max_epochs"] = 60
template["training"]["early_stop_patience"] = 10
template["training"]["seed"] = 42
template["runtime"]["num_workers"] = 4
template["runtime"]["output_dir"] = "runs/diverse_200_v2_seed42_24"
if template["model"]["version"] != "v2" or not template["model"]["use_points"]:
    raise ValueError("expected the V2 PointNet template")
path = Path("config_diverse_200_v2.yaml")
content = yaml.safe_dump(template, sort_keys=False, allow_unicode=True)
if path.exists() and path.read_text(encoding="utf-8") != content:
    raise ValueError(f"existing {path} differs; refusing to change a resumable run")
path.write_text(content, encoding="utf-8")
print("Training config:", path, flush=True)
PY

if [[ ! -f "$RUN_DIR/training_done" ]]; then
  if [[ -f "$RUN_DIR/last.pt" ]]; then
    python train.py --config "$CONFIG" --resume
  else
    python train.py --config "$CONFIG"
  fi
  test -s "$RUN_DIR/best_model.pt"
  touch "$RUN_DIR/training_done"
fi

if [[ ! -f "$RUN_DIR/evaluation_done" ]]; then
  python evaluate_sim.py --sim-root "$SIM_ROOT" --config "$CONFIG" \
    --output "$RUN_DIR/simulation_rollout.json"
  test -s "$RUN_DIR/simulation_rollout.json"
  touch "$RUN_DIR/evaluation_done"
fi

echo "Checkpoint: $RUN_DIR/best_model.pt"
echo "Rollout report: $RUN_DIR/simulation_rollout.json"
echo "Log: $RUN_DIR/pipeline.log"
