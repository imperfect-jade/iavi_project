"""Offline simulated coverage rollouts; never use reference clouds as policy input."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from build_dataset import voxel_sample
from model import GainPolicy, select_action


FIXED_ORDER = [0, 6, 12, 18, 3, 9, 15, 21, 1, 7, 13, 19,
               4, 10, 16, 22, 2, 8, 14, 20, 5, 11, 17, 23]


def coverage_score(table, visited):
    return float(table[np.asarray(visited, dtype=int)].any(axis=0).mean())


def policy_choice(model, clouds, visited, point_count, voxel_mm, rng, device):
    observed = voxel_sample(np.concatenate([clouds[i] for i in visited]), voxel_mm,
                            rng, limit=point_count)
    points = np.zeros((point_count, 3), np.float32)
    mask = np.zeros(point_count, bool)
    points[:len(observed)], mask[:len(observed)] = observed, True
    visited_mask = np.zeros(24, bool)
    visited_mask[visited] = True
    with torch.inference_mode():
        score = model(torch.from_numpy(points[None]).to(device),
                      torch.from_numpy(mask[None]).to(device),
                      torch.from_numpy(visited_mask[None]).to(device),
                      torch.ones((1, 24), dtype=torch.bool, device=device),
                      torch.tensor([visited[-1]], device=device),
                      torch.tensor([len(visited)], dtype=torch.float32, device=device))
        return int(select_action(score, torch.from_numpy(visited_mask[None]).to(device),
                                 torch.ones((1, 24), dtype=torch.bool, device=device))[0])


def rollout(strategy, table, clouds, model, initial, budget, voxel_mm, rng, device):
    visited = list(initial)
    curve = [coverage_score(table, visited)]
    while len(visited) < budget:
        eligible = [i for i in range(24) if i not in visited]
        if strategy == "learned":
            choice = policy_choice(model, clouds, visited, 2048, voxel_mm, rng, device)
        elif strategy == "oracle":
            covered = table[visited].any(axis=0)
            choice = max(eligible, key=lambda i: int((table[i] & ~covered).sum()))
        elif strategy == "fixed":
            choice = next(i for i in FIXED_ORDER if i in eligible)
        else:
            choice = int(rng.choice(eligible))
        visited.append(choice)
        curve.append(coverage_score(table, visited))
    return {"actions": visited, "coverage": curve, "auc": float(np.mean(curve))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config_sim.yaml"))
    parser.add_argument("--budget", type=int, default=10)
    parser.add_argument("--random-repeats", type=int, default=20)
    args = parser.parse_args()
    if not 2 <= args.budget <= 24 or args.random_repeats < 1:
        raise ValueError("budget must be 2..24 and random-repeats positive")
    sim_root = args.sim_root.resolve()
    manifest = json.loads((sim_root / "manifest.json").read_text(encoding="utf-8"))
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    device = torch.device(config["runtime"]["device"])
    mc = config["model"]
    if mc["action_count"] != 24 or manifest["action_count"] != 24:
        raise ValueError("evaluation requires the 24-action protocol")
    model = GainPolicy(24, mc["center_mm"], mc["scale_mm"]).to(device)
    weights = (args.config.parent / config["runtime"]["output_dir"] / "best_model.pt").resolve()
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    model.eval()
    output = {}
    for obj in manifest["objects"]:
        if obj["split"] != "test":
            continue
        oid = obj["object_id"]
        with np.load(sim_root / "datasets" / "references" / f"{oid}.npz", allow_pickle=False) as archive:
            table = archive["coverage"]
        clouds = [np.load(sim_root / relative, allow_pickle=False) for relative in obj["views"]]
        result = {}
        for strategy in ("learned", "fixed", "oracle"):
            result[strategy] = rollout(strategy, table, clouds, model, manifest["initial_views"],
                                       args.budget, manifest["voxel_mm"], np.random.default_rng(42), device)
        random_runs = [rollout("random", table, clouds, model, manifest["initial_views"],
                               args.budget, manifest["voxel_mm"], np.random.default_rng(seed), device)
                       for seed in range(args.random_repeats)]
        result["random_auc_mean"] = float(np.mean([r["auc"] for r in random_runs]))
        result["random_auc_std"] = float(np.std([r["auc"] for r in random_runs]))
        output[oid] = result
        print(oid, {k: round(v["auc"], 4) for k, v in result.items() if isinstance(v, dict)})
    if not output:
        raise ValueError("manifest has no test objects")
    report = weights.parent / "simulation_rollout.json"
    report.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"wrote {report}")


if __name__ == "__main__":
    main()
