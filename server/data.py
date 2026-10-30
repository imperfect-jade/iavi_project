"""Validated, non-pickle training sample format for the NBV policy."""

from pathlib import Path
import numpy as np


REQUIRED = ("points", "point_mask", "visited_mask", "available_mask",
            "gain_target", "current_action_id", "used_scans", "object_id")


def load_split(path: str | Path, point_count: int, action_count: int) -> dict:
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        missing = set(REQUIRED) - set(archive.files)
        if missing:
            raise ValueError(f"{path}: missing fields {sorted(missing)}")
        data = {key: archive[key] for key in REQUIRED}
    n = len(data["points"])
    shapes = {"points": (n, point_count, 3), "point_mask": (n, point_count),
              "visited_mask": (n, action_count), "available_mask": (n, action_count),
              "gain_target": (n, action_count), "current_action_id": (n,),
              "used_scans": (n,), "object_id": (n,)}
    if n == 0:
        raise ValueError(f"{path}: empty split")
    for key, shape in shapes.items():
        if data[key].shape != shape:
            raise ValueError(f"{path}: {key} shape {data[key].shape}, expected {shape}")
    if data["object_id"].dtype.kind not in "US":
        raise ValueError(f"{path}: object_id must be a string array")
    for key in ("points", "gain_target", "used_scans"):
        if not np.isfinite(data[key]).all():
            raise ValueError(f"{path}: {key} must be finite")
    if not np.all((data["gain_target"] >= 0) & (data["gain_target"] <= 1)):
        raise ValueError(f"{path}: gain_target must be in [0, 1]")
    for key in ("point_mask", "visited_mask", "available_mask"):
        if data[key].dtype != np.bool_:
            raise ValueError(f"{path}: {key} must be bool")
    if not data["point_mask"].any(axis=1).all():
        raise ValueError(f"{path}: every state needs at least one observed point")
    if (data["current_action_id"] < 0).any() or (data["current_action_id"] >= action_count).any():
        raise ValueError(f"{path}: current_action_id out of range")
    if not data["visited_mask"][np.arange(n), data["current_action_id"]].all():
        raise ValueError(f"{path}: current_action_id must be visited")
    if (data["used_scans"] < 1).any():
        raise ValueError(f"{path}: used_scans must be positive")
    data["gain_loss_mask"] = data["available_mask"] & ~data["visited_mask"]
    if not data["gain_loss_mask"].any(axis=1).all():
        raise ValueError(f"{path}: every state needs an eligible action")
    for key in ("points", "gain_target", "used_scans"):
        data[key] = np.asarray(data[key], dtype=np.float32)
    data["current_action_id"] = np.asarray(data["current_action_id"], dtype=np.int64)
    return data


def validate_disjoint_objects(*splits: dict) -> None:
    seen: set[str] = set()
    for split in splits:
        ids = set(map(str, split["object_id"]))
        overlap = seen & ids
        if overlap:
            raise ValueError(f"object_id overlap across splits: {sorted(overlap)}")
        seen |= ids
