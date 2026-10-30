"""Train and resume the NBV policy on validated NPZ splits."""

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from data import load_split, validate_disjoint_objects
from model import GainPolicy, masked_smooth_l1


class Samples(Dataset):
    def __init__(self, arrays):
        self.arrays = {k: torch.from_numpy(v) for k, v in arrays.items()
                       if k != "object_id"}
        self.length = len(arrays["object_id"])

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return {key: value[index] for key, value in self.arrays.items()}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def epoch_loss(model, loader, device, optimizer=None):
    model.train(optimizer is not None)
    total, count = 0.0, 0
    context = torch.enable_grad() if optimizer is not None else torch.no_grad()
    with context:
        for batch in loader:
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            prediction = model(*(batch[key] for key in (
                "points", "point_mask", "visited_mask", "available_mask",
                "current_action_id", "used_scans")))
            loss = masked_smooth_l1(prediction, batch["gain_target"], batch["gain_loss_mask"])
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total += float(loss.detach()) * len(batch["points"])
            count += len(batch["points"])
    return total / count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="resume runs/last.pt")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root = config_path.parent
    data_cfg, train_cfg, model_cfg, run_cfg = (config[k] for k in
                                              ("data", "training", "model", "runtime"))
    point_count, action_count = model_cfg["point_count"], model_cfg["action_count"]
    paths = {split: (root / data_cfg[split]).resolve() for split in ("train", "val")}
    arrays = {split: load_split(path, point_count, action_count) for split, path in paths.items()}
    validate_disjoint_objects(arrays["train"], arrays["val"])
    fingerprints = {split: sha256(path) for split, path in paths.items()}
    fingerprints["config"] = sha256(config_path)
    seed = int(train_cfg["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(run_cfg["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    output = (root / run_cfg["output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    model = GainPolicy(action_count, model_cfg["center_mm"], model_cfg["scale_mm"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(train_cfg["learning_rate"]),
                                 weight_decay=float(train_cfg["weight_decay"]))
    train_loader = DataLoader(Samples(arrays["train"]), batch_size=int(train_cfg["batch_size"]),
                              shuffle=True, num_workers=int(run_cfg["num_workers"]),
                              pin_memory=device.type == "cuda")
    val_loader = DataLoader(Samples(arrays["val"]), batch_size=int(train_cfg["batch_size"]),
                            num_workers=int(run_cfg["num_workers"]),
                            pin_memory=device.type == "cuda")
    start, best, stale = 0, float("inf"), 0
    if args.resume:
        checkpoint = torch.load(output / "last.pt", map_location=device, weights_only=False)
        if checkpoint["fingerprints"] != fingerprints:
            raise ValueError("config or dataset changed; refusing to resume")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start, best, stale = checkpoint["epoch"] + 1, checkpoint["best"], checkpoint["stale"]
        random.setstate(checkpoint["python_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        torch.set_rng_state(checkpoint["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
    metadata = {"model": "compact_pointnet_gain_v1", "model_config": model_cfg,
                "data_fingerprints": fingerprints, "selection_metric": "validation_masked_smooth_l1",
                "note": "Validation rollout AUC requires measured per-view clouds."}
    (output / "model_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    for epoch in range(start, int(train_cfg["max_epochs"])):
        train_loss = epoch_loss(model, train_loader, device, optimizer)
        val_loss = epoch_loss(model, val_loader, device)
        improved = val_loss < best - 1e-8
        if improved:
            best, stale = val_loss, 0
            torch.save(model.state_dict(), output / "best_model.pt")
        else:
            stale += 1
        record = {"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss,
                  "best_val_loss": best, "peak_cuda_mib":
                  torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0}
        with (output / "history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "best": best, "stale": stale, "fingerprints": fingerprints,
                    "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None},
                   output / "last.pt")
        if stale >= int(train_cfg["early_stop_patience"]):
            print("Early stopping", flush=True)
            break


if __name__ == "__main__":
    main()
