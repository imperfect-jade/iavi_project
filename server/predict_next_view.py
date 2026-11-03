"""Choose the next 2 x 12 action from the currently observed fused cloud."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from build_dataset import voxel_sample
from model import make_gain_policy, select_action
from simulate_modelnet import action_pose


def predict(config_path, cloud_path, visited, available=None, voxel_mm=1.5):
    config_path = Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model_cfg = config["model"]
    action_count = int(model_cfg["action_count"])
    if action_count != 24 or not model_cfg.get("use_points", True):
        raise ValueError("expected a 24-action point-cloud policy")
    if not visited or len(set(visited)) != len(visited) or any(a < 0 or a >= 24 for a in visited):
        raise ValueError("visited must contain unique action IDs in 0..23")
    if available is None:
        available = list(range(24))
    if len(set(available)) != len(available) or any(a < 0 or a >= 24 for a in available):
        raise ValueError("available must contain unique action IDs in 0..23")
    if not set(available) - set(visited):
        raise ValueError("no available unvisited action")
    if voxel_mm <= 0:
        raise ValueError("voxel_mm must be positive")
    cloud = np.load(cloud_path, allow_pickle=False)
    if (cloud.ndim != 2 or cloud.shape[1] != 3 or len(cloud) == 0 or
            not np.issubdtype(cloud.dtype, np.number) or not np.isfinite(cloud).all()):
        raise ValueError("cloud must be a nonempty finite (N,3) XYZ NPY array in object-frame millimeters")
    point_count = int(model_cfg["point_count"])
    observed = voxel_sample(np.asarray(cloud, dtype=np.float32), voxel_mm,
                            np.random.default_rng(42), limit=point_count)
    points = np.zeros((1, point_count, 3), np.float32)
    mask = np.zeros((1, point_count), bool)
    points[0, :len(observed)] = observed
    mask[0, :len(observed)] = True
    visited_mask = np.zeros((1, action_count), bool)
    available_mask = np.zeros((1, action_count), bool)
    visited_mask[0, visited] = True
    available_mask[0, available] = True
    device = torch.device(config["runtime"]["device"])
    weights = (config_path.parent / config["runtime"]["output_dir"] / "best_model.pt").resolve()
    model = make_gain_policy(model_cfg).to(device)
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    model.eval()
    with torch.inference_mode():
        scores = model(torch.from_numpy(points).to(device),
                       torch.from_numpy(mask).to(device),
                       torch.from_numpy(visited_mask).to(device),
                       torch.from_numpy(available_mask).to(device),
                       torch.tensor([visited[-1]], device=device),
                       torch.tensor([len(visited)], dtype=torch.float32, device=device))
        choice = int(select_action(scores, torch.from_numpy(visited_mask).to(device),
                                   torch.from_numpy(available_mask).to(device))[0])
    pose, angle = action_pose(choice)
    return {"action_id": choice, "arm_pose_id": pose, "turntable_deg": angle,
            "observed_points_used": int(len(observed)), "visited": visited}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--points", type=Path, required=True,
                        help="current fused object-frame XYZ NPY cloud in millimeters")
    parser.add_argument("--visited", type=int, nargs="+", required=True,
                        help="already scanned action IDs, in acquisition order")
    parser.add_argument("--available", type=int, nargs="+",
                        help="physically allowed actions; default all 24")
    parser.add_argument("--voxel-mm", type=float, default=1.5)
    args = parser.parse_args()
    print(json.dumps(predict(args.config, args.points, args.visited,
                             args.available, args.voxel_mm), ensure_ascii=False))


if __name__ == "__main__":
    main()
