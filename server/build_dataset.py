"""Convert verified object-coordinate view clouds to supervised NBV states."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
try:
    from scipy.spatial import cKDTree
except ImportError:  # small local tests only; real datasets require SciPy
    cKDTree = None

from data import load_split, validate_disjoint_objects


FIELDS = ("points", "point_mask", "visited_mask", "available_mask",
          "gain_target", "current_action_id", "used_scans", "object_id")


def voxel_sample(points, voxel_mm, rng, limit=None):
    cells = np.floor(points / voxel_mm).astype(np.int64)
    _, indices = np.unique(cells, axis=0, return_index=True)
    result = points[np.sort(indices)]
    if limit is not None and len(result) > limit:
        result = result[np.sort(rng.choice(len(result), limit, replace=False))]
    return result


def coverage_table(reference, views, tolerance_mm):
    if tolerance_mm <= 0 or len(reference) == 0:
        raise ValueError("reference must be nonempty and tolerance_mm positive")
    rows = []
    for view in views:
        if cKDTree is not None:
            rows.append(cKDTree(view).query(reference, k=1)[0] < tolerance_mm)
        else:
            if len(reference) * len(view) > 1_000_000:
                raise RuntimeError("SciPy is required for real-size coverage tables")
            squared = ((reference[:, None, :] - view[None, :, :]) ** 2).sum(axis=2)
            rows.append(squared.min(axis=1) < tolerance_mm ** 2)
    return np.stack(rows)


def marginal_gains(table, visited):
    covered = table[visited].any(axis=0) if visited.any() else np.zeros(table.shape[1], bool)
    gains = (table & ~covered[None, :]).mean(axis=1).astype(np.float32)
    gains[visited] = 0
    return gains


def read_manifest(path):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("unit") != "mm" or manifest.get("coordinate_frame") != "object_at_turntable_zero":
        raise ValueError("manifest must declare unit=mm and coordinate_frame=object_at_turntable_zero")
    count = int(manifest["action_count"])
    initial = list(manifest["initial_views"])
    if count < 3 or len(initial) < 1 or len(initial) >= count or len(set(initial)) != len(initial):
        raise ValueError("invalid action_count or initial_views")
    if any(not isinstance(x, int) or x < 0 or x >= count for x in initial):
        raise ValueError("initial_views out of range")
    for name in ("tolerance_mm", "voxel_mm", "point_count", "reference_max_points", "states_per_object"):
        if float(manifest[name]) <= 0:
            raise ValueError(f"{name} must be positive")
    if not manifest["objects"]:
        raise ValueError("objects must be nonempty")
    ids = set()
    for obj in manifest["objects"]:
        oid = obj["object_id"]
        if not isinstance(oid, str) or not oid or "/" in oid or "\\" in oid or oid in ids:
            raise ValueError("object_id must be unique and safe as a filename")
        ids.add(oid)
        if obj["split"] not in ("train", "val", "test"):
            raise ValueError("split must be train, val, or test")
        if len(obj["views"]) != count:
            raise ValueError(f"{oid}: views must have exactly {count} entries in action order")
        for relative in obj["views"]:
            cloud_path = (path.parent / relative).resolve()
            if not cloud_path.is_file() or cloud_path.suffix.lower() != ".npy":
                raise ValueError(f"{oid}: missing or non-NPY view: {cloud_path}")
    return manifest


def load_views(obj, base):
    views = []
    for action, relative in enumerate(obj["views"]):
        cloud = np.load(base / relative, allow_pickle=False)
        if cloud.ndim != 2 or cloud.shape[1] != 3 or len(cloud) == 0:
            raise ValueError(f"{obj['object_id']} action {action}: expected nonempty (P,3) cloud")
        if not np.issubdtype(cloud.dtype, np.number) or not np.isfinite(cloud).all():
            raise ValueError(f"{obj['object_id']} action {action}: nonfinite/non-numeric cloud")
        views.append(np.asarray(cloud, np.float32))
    return views


def make_states(obj, views, table, cfg, rng):
    count = int(cfg["action_count"])
    initial = list(cfg["initial_views"])
    target_count = int(cfg["states_per_object"])
    max_visits = min(9, count - 1)
    if len(initial) > max_visits:
        raise ValueError("initial_views leave no eligible action at max state size")
    seen, rows = set(), []
    attempts = 0
    while len(rows) < target_count:
        attempts += 1
        if attempts > 200 * target_count:
            raise ValueError(f"{obj['object_id']}: cannot sample {target_count} unique states")
        size = int(rng.integers(len(initial), max_visits + 1))
        visited = np.zeros(count, bool)
        visited[initial] = True
        current = initial[-1]
        if attempts % 2:
            extra = rng.choice([x for x in range(count) if x not in initial],
                               size=size - len(initial), replace=False)
            visited[extra] = True
            if len(extra):
                current = int(extra[-1])
        else:
            for _ in range(size - len(initial)):
                gain = marginal_gains(table, visited)
                choices = np.flatnonzero(~visited)
                ranked = choices[np.argsort(-gain[choices], kind="stable")]
                current = int(rng.choice(ranked[:min(3, len(ranked))]))
                visited[current] = True
        key = tuple(np.flatnonzero(visited))
        if key in seen:
            continue
        seen.add(key)
        observed = voxel_sample(np.concatenate([views[i] for i in np.flatnonzero(visited)]),
                                float(cfg["voxel_mm"]), rng,
                                limit=int(cfg["point_count"]))
        points = np.zeros((int(cfg["point_count"]), 3), np.float32)
        point_mask = np.zeros(int(cfg["point_count"]), bool)
        points[:len(observed)] = observed
        point_mask[:len(observed)] = True
        rows.append((points, point_mask, visited, np.ones(count, bool),
                     marginal_gains(table, visited), current, float(size), obj["object_id"]))
    return rows


def build_dataset(manifest_path, output_dir):
    manifest_path = Path(manifest_path).resolve()
    output_dir = Path(output_dir).resolve()
    cfg = read_manifest(manifest_path)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output_dir}")
    rng = np.random.default_rng(int(cfg["seed"]))
    splits = {"train": [], "val": [], "test": []}
    references = {}
    for obj in cfg["objects"]:
        views = load_views(obj, manifest_path.parent)
        state_views = [voxel_sample(view, float(cfg["voxel_mm"]), rng) for view in views]
        reference = voxel_sample(np.concatenate(state_views), float(cfg["voxel_mm"]), rng,
                                 limit=int(cfg["reference_max_points"]))
        table = coverage_table(reference, views, float(cfg["tolerance_mm"]))
        references[obj["object_id"]] = (reference, table)
        splits[obj["split"]].extend(make_states(obj, state_views, table, cfg, rng))
    if not splits["train"] or not splits["val"]:
        raise ValueError("both train and val require at least one object")
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        if not rows:
            continue
        columns = zip(*rows)
        arrays = {key: np.asarray(value) for key, value in zip(FIELDS, columns)}
        np.savez_compressed(output_dir / f"{split}.npz", **arrays)
    ref_dir = output_dir / "references"
    ref_dir.mkdir()
    for oid, (reference, table) in references.items():
        np.savez_compressed(ref_dir / f"{oid}.npz", reference=reference, coverage=table)
    meta = {key: value for key, value in cfg.items() if key != "objects"}
    meta["source_manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    meta["objects"] = [{"object_id": obj["object_id"], "split": obj["split"]} for obj in cfg["objects"]]
    (output_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    loaded = [load_split(output_dir / f"{split}.npz", int(cfg["point_count"]), int(cfg["action_count"]))
              for split in splits if splits[split]]
    validate_disjoint_objects(*loaded)
    print({split: len(rows) for split, rows in splits.items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build_dataset(args.manifest, args.output)
