"""Adapt already aligned scan views to the ModelNet policy's object frame."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


VERTEX = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                   ("red", "u1"), ("green", "u1"), ("blue", "u1")])
HEADER_PROPERTIES = ["property float x", "property float y", "property float z",
                     "property uchar red", "property uchar green", "property uchar blue"]


def read_aligned_ply(path: Path) -> np.ndarray:
    """Read the binary PLY format emitted by code/fuse_views.py."""
    with path.open("rb") as stream:
        header = []
        for _ in range(64):
            line = stream.readline()
            if not line:
                raise ValueError(f"incomplete PLY header: {path}")
            header.append(line.decode("ascii").strip())
            if header[-1] == "end_header":
                break
        else:
            raise ValueError(f"PLY header too long: {path}")
        if ("format binary_little_endian 1.0" not in header or
                "comment coordinates turntable_zero_calibrated; unit mm" not in header):
            raise ValueError(f"expected calibrated binary PLY in mm: {path}")
        properties = [line for line in header if line.startswith("property ")]
        if properties != HEADER_PROPERTIES:
            raise ValueError(f"unsupported PLY vertex fields: {path}")
        counts = [line for line in header if line.startswith("element vertex ")]
        if len(counts) != 1:
            raise ValueError(f"missing PLY vertex count: {path}")
        count = int(counts[0].split()[2])
        data = np.fromfile(stream, dtype=VERTEX, count=count)
    if count <= 0 or len(data) != count:
        raise ValueError(f"empty or truncated PLY: {path}")
    xyz = np.stack([data[name] for name in ("x", "y", "z")], axis=1)
    if not np.isfinite(xyz).all():
        raise ValueError(f"nonfinite PLY coordinates: {path}")
    return xyz


def prepare_observation(fusion_dir: Path, raw_session_dir: Path, visited: list[int],
                        *, table_height_axis_mm: float | None = None,
                        frame: dict | None = None,
                        crop_radius_mm: float = 55.0,
                        crop_height_mm: tuple[float, float] = (3.0, 150.0)
                        ) -> tuple[np.ndarray, dict]:
    """Build (N,3) model-frame millimetres from visited views only.

    Pass the returned frame back on later calls so the object centre and
    scale remain fixed as more surface becomes visible.
    """
    fusion_dir = Path(fusion_dir)
    raw_session_dir = Path(raw_session_dir)
    if (len(visited) < 2 or visited[:2] != [0, 6] or
            len(visited) != len(set(visited)) or any(not 0 <= v < 24 for v in visited)):
        raise ValueError("visited must start with unique initial views [0, 6] and use IDs 0..23")
    if crop_radius_mm <= 0 or crop_height_mm[0] >= crop_height_mm[1]:
        raise ValueError("invalid object crop")
    report = json.loads((fusion_dir / "fusion.json").read_text(encoding="utf-8"))
    if report.get("coordinate_frame") != "turntable_zero_calibrated" or report.get("unit") != "mm":
        raise ValueError("calibrated turntable-zero fusion in mm is required")
    axis_point = np.asarray(report["axis_point_mm"], dtype=np.float64)
    axis = np.asarray(report["axis_direction"], dtype=np.float64)
    if (axis_point.shape != (3,) or axis.shape != (3,) or
            not np.isfinite(axis_point).all() or not np.isfinite(axis).all() or
            np.linalg.norm(axis) < 1e-8):
        raise ValueError("invalid fusion axis")
    axis /= np.linalg.norm(axis)
    repeats = {}
    for item in report["selected_repeats"]:
        parts = Path(item.replace("\\", "/")).parts
        if len(parts) < 2 or not parts[-2].startswith("view_") or not parts[-1].startswith("repeat_"):
            raise ValueError(f"invalid selected repeat: {item}")
        view_name, repeat_name = parts[-2:]
        action = int(view_name[5:])
        if action in repeats:
            raise ValueError(f"duplicate selected action {action}")
        repeats[action] = (f"{view_name}_{repeat_name}.ply", f"{view_name}/{repeat_name}")
    clouds = []
    for action in visited:
        if action not in repeats:
            raise FileNotFoundError(f"view_{action:03d} absent from fusion report")
        path = fusion_dir / "aligned" / repeats[action][0]
        if not path.is_file():
            raise FileNotFoundError(path)
        clouds.append(read_aligned_ply(path))
    points = np.concatenate(clouds).astype(np.float64)

    if frame is None:
        if table_height_axis_mm is None or not np.isfinite(table_height_axis_mm):
            raise ValueError("table_height_axis_mm is required for the first call")
        poses = json.loads((raw_session_dir / "camera_poses.json").read_text(encoding="utf-8"))
        zero_name = repeats[0][1]
        camera = np.asarray(poses[zero_name], dtype=np.float64)
        if camera.shape != (4, 4) or not np.isfinite(camera).all():
            raise ValueError("invalid camera pose for view 0")
        x_axis = camera[:3, 3] - axis_point
        x_axis -= np.dot(x_axis, axis) * axis
        norm = np.linalg.norm(x_axis)
        if norm < 1e-8:
            raise ValueError("view 0 camera lies on the turntable axis")
        x_axis /= norm
        y_axis = np.cross(axis, x_axis)
        frame = {"axis_point_mm": axis_point.tolist(), "z_axis": axis.tolist(),
                 "x_axis": x_axis.tolist(), "y_axis": y_axis.tolist(),
                 "table_height_axis_mm": float(table_height_axis_mm),
                 "crop_radius_mm": float(crop_radius_mm),
                 "crop_height_mm": list(map(float, crop_height_mm)),
                 "initial_views": [0, 6]}
    else:
        if frame.get("initial_views") != [0, 6]:
            raise ValueError("frame initial views do not match the model")
        if (not np.allclose(frame["axis_point_mm"], axis_point, atol=1e-3) or
                not np.allclose(frame["z_axis"], axis, atol=1e-5)):
            raise ValueError("frame axis differs from this fusion session")
        if table_height_axis_mm is not None and not np.isclose(
                table_height_axis_mm, frame["table_height_axis_mm"]):
            raise ValueError("table height differs from frozen frame")
        crop_radius_mm = frame["crop_radius_mm"]
        crop_height_mm = tuple(frame["crop_height_mm"])

    offset = points - axis_point
    axial = offset @ axis
    radial = offset - axial[:, None] * axis
    height = axial - frame["table_height_axis_mm"]
    keep = ((np.linalg.norm(radial, axis=1) <= crop_radius_mm) &
            (height >= crop_height_mm[0]) & (height <= crop_height_mm[1]))
    if not keep.any():
        raise ValueError("object crop contains no observed points; check table height and axis")
    radial, height = radial[keep], height[keep]
    local = np.column_stack((radial @ np.asarray(frame["x_axis"]),
                             radial @ np.asarray(frame["y_axis"]), height))
    if "xy_center_mm" not in frame:
        xy_min, xy_max = local[:, :2].min(axis=0), local[:, :2].max(axis=0)
        xy_extent = np.ptp(local[:, :2], axis=0)
        scale = 100.0 / max(xy_extent.max(), local[:, 2].max(), 1e-8)
        frame["xy_center_mm"] = ((xy_min + xy_max) / 2).tolist()
        frame["model_scale"] = float(scale)
    local[:, :2] -= np.asarray(frame["xy_center_mm"])
    local *= frame["model_scale"]
    return local.astype(np.float32), frame
