"""Load the trained gain model and return a structured executable action."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server"))
from build_dataset import voxel_sample
from model import make_gain_policy, select_action
from simulate_modelnet import action_pose


def validate_checkpoint(config_path) -> Path:
    """Fail before opening devices when the policy or its runtime is unusable."""
    config_path = Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model_cfg = config["model"]
    if (model_cfg.get("version") != "v2" or model_cfg.get("action_count") != 24 or
            model_cfg.get("point_count") != 2048 or not model_cfg.get("use_points", True)):
        raise ValueError("active scan requires a V2 24-action 2048-point cloud policy")
    device = torch.device(config["runtime"]["device"])
    torch.empty(1).to(device)
    weights = (config_path.parent / config["runtime"]["output_dir"] / "best_model.pt").resolve()
    model = make_gain_policy(model_cfg)
    try:
        model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    except Exception as exc:
        raise RuntimeError(f"cannot load trained policy checkpoint: {weights}") from exc
    return weights


def predict_scores(config_path, cloud_path, visited, available=None, voxel_mm=1.5) -> dict:
    config_path = Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model_cfg = config["model"]
    count = int(model_cfg["action_count"])
    if count != 24 or not model_cfg.get("use_points", True):
        raise ValueError("expected a 24-action point-cloud policy")
    if not visited or len(set(visited)) != len(visited) or any(not 0 <= a < count for a in visited):
        raise ValueError("visited must contain unique action IDs in 0..23")
    available = list(range(count)) if available is None else list(available)
    if len(set(available)) != len(available) or any(not 0 <= a < count for a in available):
        raise ValueError("available must contain unique action IDs in 0..23")
    eligible = sorted(set(available) - set(visited))
    if not eligible:
        raise ValueError("no available unvisited action")
    if voxel_mm <= 0:
        raise ValueError("voxel_mm must be positive")
    cloud = np.load(cloud_path, allow_pickle=False)
    if (cloud.ndim != 2 or cloud.shape[1] != 3 or not len(cloud) or
            not np.issubdtype(cloud.dtype, np.number) or not np.isfinite(cloud).all()):
        raise ValueError("cloud must be a nonempty finite (N,3) XYZ NPY array in object-frame millimeters")
    point_count = int(model_cfg["point_count"])
    observed = voxel_sample(np.asarray(cloud, dtype=np.float32), voxel_mm,
                            np.random.default_rng(42), limit=point_count)
    points = np.zeros((1, point_count, 3), np.float32)
    mask = np.zeros((1, point_count), bool)
    points[0, :len(observed)] = observed
    mask[0, :len(observed)] = True
    visited_mask = np.zeros((1, count), bool)
    available_mask = np.zeros((1, count), bool)
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
        action = int(select_action(scores, torch.from_numpy(visited_mask).to(device),
                                   torch.from_numpy(available_mask).to(device))[0])
    values = scores[0].detach().cpu().numpy()
    if not np.isfinite(values[eligible]).all():
        raise ValueError("model produced nonfinite action scores")
    pose, angle = action_pose(action)
    return {"action_id": action, "arm_pose_id": pose, "turntable_deg": angle,
            "observed_points_used": int(len(observed)), "visited": list(visited),
            "scores_by_action": {str(a): float(values[a]) for a in eligible},
            "best_score": float(values[action]), "model_version": model_cfg.get("version", "v1")}
