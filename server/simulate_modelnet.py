"""ModelNet mesh -> 24 virtual structured-light view clouds (geometry proxy)."""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

from build_dataset import build_dataset


def action_pose(action_id: int) -> tuple[int, int]:
    if not 0 <= action_id < 24:
        raise ValueError("action_id must be 0..23")
    return action_id // 12, (action_id % 12) * 30


def normalize_mesh_vertices(vertices: np.ndarray, max_dimension_mm: float) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=np.float64)
    extent = np.ptp(vertices, axis=0)
    if max_dimension_mm <= 0 or max(extent) <= 0:
        raise ValueError("mesh extent and max_dimension_mm must be positive")
    result = vertices * (max_dimension_mm / max(extent))
    result[:, :2] -= (result[:, :2].min(axis=0) + result[:, :2].max(axis=0)) / 2
    result[:, 2] -= result[:, 2].min()
    return result.astype(np.float32)


def split_meshes(root: Path, categories: list[str], val_fraction: float, seed: int):
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    rng = np.random.default_rng(seed)
    result = {"train": [], "val": [], "test": []}
    for category in categories:
        source_train = sorted((root / category / "train").glob("*.off"))
        source_test = sorted((root / category / "test").glob("*.off"))
        if not source_train or not source_test:
            raise ValueError(f"missing ModelNet train/test OFF files for category {category}")
        order = rng.permutation(len(source_train))
        val_count = max(1, round(len(source_train) * val_fraction))
        result["val"].extend(source_train[int(i)] for i in order[:val_count])
        result["train"].extend(source_train[int(i)] for i in order[val_count:])
        result["test"].extend(source_test)
    return result


def diverse_order(paths: list[Path]) -> list[Path]:
    """Round-robin categories so a small cap does not select only the first one."""
    groups = {}
    for path in paths:
        groups.setdefault(path.parent.parent.name, []).append(path)
    result = []
    while any(groups.values()):
        for group in groups.values():
            if group:
                result.append(group.pop(0))
    return result


def read_modelnet_mesh(mesh_path: Path):
    """Read ModelNet OFF files, including the common `OFF308 228 0` header."""
    import open3d as o3d

    mesh_path = Path(mesh_path)
    with mesh_path.open("rb") as source:
        first_line = source.readline()
        if first_line.startswith(b"OFF") and first_line[3:4].isdigit():
            with tempfile.TemporaryDirectory(prefix="nbv_off_") as temporary:
                normalized = Path(temporary) / "mesh.off"
                with normalized.open("wb") as target:
                    target.write(b"OFF\n")
                    target.write(first_line[3:])
                    shutil.copyfileobj(source, target)
                mesh = o3d.io.read_triangle_mesh(str(normalized))
            print(f"normalized glued OFF header: {mesh_path.name}", flush=True)
            return mesh
    return o3d.io.read_triangle_mesh(str(mesh_path))


def _joint_visibility(scene, points, eye, target, fov):
    import open3d as o3d

    direction = points - np.asarray(eye, np.float32)
    distance = np.linalg.norm(direction, axis=1)
    optical_axis = np.asarray(target, np.float32) - np.asarray(eye, np.float32)
    optical_axis /= np.linalg.norm(optical_axis)
    in_cone = (direction @ optical_axis) / distance >= np.cos(np.deg2rad(fov / 2))
    rays = np.column_stack((np.broadcast_to(eye, points.shape), direction))
    first = scene.cast_rays(o3d.core.Tensor(rays.astype(np.float32)))["t_hit"].numpy()
    return in_cone & np.isfinite(first) & (first >= 1 - 1e-3)


def _raycast_view(scene, left_eye, right_eye, target, projector, width, height, fov, noise_mm, dropout, rng):
    import open3d as o3d

    rays = scene.create_rays_pinhole(fov_deg=fov, center=target, eye=left_eye,
                                     up=[0, 0, 1], width_px=width, height_px=height)
    rays_np = rays.numpy().reshape(-1, 6)
    hit = scene.cast_rays(rays)["t_hit"].numpy().reshape(-1)
    good = np.isfinite(hit)
    points = rays_np[good, :3] + hit[good, None] * rays_np[good, 3:]
    if len(points) == 0:
        raise ValueError("camera rays missed the mesh")
    # Keep only surfaces jointly visible to both cameras and the fixed projector.
    visible = (_joint_visibility(scene, points, right_eye, target, fov) &
               _joint_visibility(scene, points, projector, target, fov))
    points = points[visible]
    if len(points) == 0:
        raise ValueError("no surface jointly visible to camera and projector")
    points = points[rng.random(len(points)) >= dropout]
    if len(points) == 0:
        raise ValueError("all points dropped")
    if noise_mm:
        points += rng.normal(0, noise_mm, points.shape).astype(np.float32)
    return points.astype(np.float32)


def simulate_mesh(mesh_path, output_dir, cfg, rng):
    import open3d as o3d

    source = read_modelnet_mesh(mesh_path)
    if not source.has_triangles():
        raise ValueError(f"no triangles in {mesh_path}")
    base = normalize_mesh_vertices(np.asarray(source.vertices), cfg["max_dimension_mm"])
    triangles = np.asarray(source.triangles)
    if not np.isfinite(base).all():
        raise ValueError(f"nonfinite vertices in {mesh_path}")
    target = [0, 0, cfg["target_height_mm"]]
    projector = cfg["projector_xyz_mm"]
    clouds = []
    for action in range(24):
        pose_id, angle = action_pose(action)
        radians = np.deg2rad(angle)
        c, s = np.cos(radians), np.sin(radians)
        rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float32)
        world_vertices = base @ rotation.T
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(world_vertices.astype(np.float64))
        mesh.triangles = o3d.utility.Vector3iVector(triangles.astype(np.int32))
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        camera_center = np.asarray(cfg["camera_xyz_mm"][pose_id], np.float32)
        baseline = float(cfg["stereo_baseline_mm"])
        left_eye = (camera_center + [0, -baseline / 2, 0]).tolist()
        right_eye = (camera_center + [0, baseline / 2, 0]).tolist()
        try:
            world_points = _raycast_view(scene, left_eye, right_eye, target,
                                         projector, cfg["image_width"], cfg["image_height"],
                                         cfg["fov_deg"], cfg["noise_mm"], cfg["dropout"], rng)
        except ValueError as exc:
            raise ValueError(f"action {action} (pose {pose_id}, turntable {angle}°): {exc}") from exc
        object_points = world_points @ rotation
        clouds.append(object_points)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for action, points in enumerate(clouds):
        path = output_dir / f"view_{action:03d}.npy"
        np.save(path, points)
        paths.append(path)
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelnet-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--categories", nargs="+", required=True)
    parser.add_argument("--max-train", type=int, default=10)
    parser.add_argument("--max-val", type=int, default=3)
    parser.add_argument("--max-test", type=int, default=3)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("sim_config.json"))
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"output is not empty: {args.output}")
    if not 0 <= cfg["dropout"] < 1 or cfg["noise_mm"] < 0:
        raise ValueError("invalid dropout or noise_mm")
    groups = split_meshes(args.modelnet_root, args.categories, cfg["val_fraction"], cfg["seed"])
    caps = {"train": args.max_train, "val": args.max_val, "test": args.max_test}
    if any(cap < 1 for cap in caps.values()):
        raise ValueError("each split needs at least one requested object")
    rng = np.random.default_rng(cfg["seed"])
    objects = []
    rejected = []
    for split, paths in groups.items():
        accepted = 0
        for mesh_path in diverse_order(paths):
            if accepted >= caps[split]:
                break
            oid = f"{mesh_path.parent.parent.name}_{mesh_path.stem}"
            cloud_dir = args.output / "clouds" / oid
            try:
                views = simulate_mesh(mesh_path, cloud_dir, cfg, rng)
            except ValueError as exc:
                rejected.append({"split": split, "mesh": str(mesh_path.resolve()), "reason": str(exc)})
                print(f"skipped {split} {mesh_path}: {exc}", flush=True)
                if sum(item["split"] == split for item in rejected) > max(5, caps[split]):
                    raise RuntimeError(f"too many invalid {split} meshes; review simulation geometry") from exc
                continue
            objects.append({"object_id": oid, "split": split,
                            "source_mesh": str(mesh_path.resolve()),
                            "views": [str(path.relative_to(args.output)) for path in views]})
            accepted += 1
            print(f"simulated {split} {oid}", flush=True)
        if accepted < caps[split]:
            raise RuntimeError(f"only {accepted} valid {split} meshes; requested {caps[split]}")
    manifest = {"unit": "mm", "coordinate_frame": "object_at_turntable_zero",
                "action_count": 24, "initial_views": [0, 6],
                "tolerance_mm": cfg["coverage_tolerance_mm"],
                "voxel_mm": cfg["voxel_mm"], "point_count": 2048,
                "reference_max_points": 20000, "states_per_object": cfg["states_per_object"],
                "seed": cfg["seed"], "simulation": cfg, "objects": objects}
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (args.output / "rejected_meshes.json").write_text(json.dumps(rejected, indent=2), encoding="utf-8")
    build_dataset(manifest_path, args.output / "datasets")


if __name__ == "__main__":
    main()
