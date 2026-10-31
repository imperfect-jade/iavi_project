"""Fuse turntable views into the zero-angle camera frame (millimetres).

This first-stage fusion requires an object crop. For different camera poses,
provide measured camera-to-reference transforms with --camera-poses.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from reconstruct_view import write_ply

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AXIS_CALIBRATION = ROOT / "calibration/turntable_axis/axis_cube_01/axis_calibration.json"
VERTEX = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                   ("red", "u1"), ("green", "u1"), ("blue", "u1")])


def read_ply(path):
    with Path(path).open("rb") as stream:
        header = []
        while True:
            line = stream.readline().decode("ascii")
            if not line:
                raise ValueError(f"Incomplete PLY header: {path}")
            header.append(line.strip())
            if line == "end_header\n":
                break
        if "format binary_little_endian 1.0" not in header:
            raise ValueError(f"Expected binary little-endian PLY: {path}")
        count = int(next(row.split()[2] for row in header if row.startswith("element vertex ")))
        data = np.fromfile(stream, VERTEX, count)
    if len(data) != count:
        raise ValueError(f"Truncated PLY: {path}")
    xyz = np.stack([data[name] for name in ("x", "y", "z")], axis=1)
    bgr = np.stack([data[name] for name in ("blue", "green", "red")], axis=1)
    return xyz, bgr


def rotate_about_axis(points, center, direction, degrees):
    axis = np.asarray(direction, np.float64)
    axis /= np.linalg.norm(axis)
    p = np.asarray(points, np.float64) - center
    angle = np.deg2rad(degrees)
    c, s = np.cos(angle), np.sin(angle)
    return (p*c + np.cross(axis, p)*s + np.outer(p @ axis, axis)*(1-c) + center).astype(np.float32)


def voxel_average(points, colors, voxel_mm):
    keys = np.floor(points / voxel_mm).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    counts = np.bincount(inverse)
    xyz = np.column_stack([np.bincount(inverse, weights=points[:, k])/counts for k in range(3)])
    bgr = np.column_stack([np.bincount(inverse, weights=colors[:, k])/counts for k in range(3)])
    return xyz.astype(np.float32), np.rint(bgr).astype(np.uint8)


def crop_cylinder(points, center, direction, radius_mm, height_mm):
    offset = points - center
    height = offset @ direction
    radial = offset - height[:, None] * direction
    return ((height >= height_mm[0]) & (height <= height_mm[1]) &
            (np.einsum("ij,ij->i", radial, radial) <= radius_mm ** 2))


def load_axis_calibration(path):
    """Read the measured axis in the original left-camera coordinate frame."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("unit") != "mm" or data.get("coordinate_frame") != "unrectified_left_camera":
        raise ValueError("Axis calibration must use mm in unrectified_left_camera")
    point = np.asarray(data["axis_point_mm"], np.float64)
    direction = np.asarray(data["axis_direction"], np.float64)
    if point.shape != (3,) or direction.shape != (3,) or not np.isfinite(point).all() or \
       not np.isfinite(direction).all() or np.linalg.norm(direction) == 0:
        raise ValueError("Invalid axis point or direction")
    angles = {0.0: 0.0}
    for item in data["angles"]:
        command, measured = float(item["command_deg"]), float(item["measured_deg"])
        if not np.isfinite([command, measured]).all() or command in angles:
            raise ValueError("Invalid or duplicate measured turntable angle")
        angles[command] = measured
    if not data.get("stereo_calibration"):
        raise ValueError("Axis calibration does not identify its stereo calibration")
    return {"point": point, "direction": direction / np.linalg.norm(direction),
            "angles_by_command": angles, "frame": "original_left_camera",
            "stereo_calibration": Path(data["stereo_calibration"]).resolve(),
            "point_rms_mm": data.get("point_rms_mm"),
            "return_to_zero": data.get("return_to_zero")}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--session-root", type=Path,
                        help="discover every view_*/repeat_* in one object session")
    source.add_argument("--view-root", type=Path, nargs="+",
                   help="one or more raw/object/session/view_NNN directories")
    p.add_argument("--processed-root", type=Path, default=ROOT / "processed")
    p.add_argument("--output", type=Path, help="defaults to processed/fused/object/session/view_NNN")
    p.add_argument("--bounds-mm", type=float, nargs=6,
                   metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
                   help="optional XYZ crop in the common reference frame, after camera-pose transform")
    p.add_argument("--radius-mm", type=float, default=60,
                   help="cylinder crop radius around calibrated axis (default 60 mm)")
    p.add_argument("--height-mm", type=float, nargs=2, default=(-20, 110),
                   metavar=("MIN", "MAX"), help="axis-coordinate crop in mm (default -20 110)")
    p.add_argument("--axis-point-mm", type=float, nargs=3,
                   help="measured turntable axis point in reference camera frame; default estimates center from crops")
    p.add_argument("--axis-calibration", type=Path,
                   help=f"turntable/calibrate_axis.py JSON (default: {DEFAULT_AXIS_CALIBRATION})")
    p.add_argument("--axis-direction", type=float, nargs=3, default=(0, 1, 0),
                   help="measured axis direction; default camera Y is provisional")
    p.add_argument("--angle-sign", type=int, choices=(-1, 1), default=1,
                   help="physical rotation sign about axis; inverse applied for alignment")
    p.add_argument("--camera-poses", type=Path,
                   help="JSON mapping repeat names to 4x4 T_reference_from_left; required for differing arm poses")
    p.add_argument("--voxel-mm", type=float, default=1.0)
    args = p.parse_args(argv)
    axis_path = args.axis_calibration
    if axis_path is None and args.axis_point_mm is None and args.angle_sign == 1 and \
       tuple(args.axis_direction) == (0, 1, 0):
        axis_path = DEFAULT_AXIS_CALIBRATION
    if axis_path and (args.axis_point_mm is not None or args.angle_sign != 1 or
                      tuple(args.axis_direction) != (0, 1, 0)):
        p.error("--axis-calibration supplies the axis and angle sign; omit --axis-point-mm and --angle-sign")
    axis_cal = load_axis_calibration(axis_path) if axis_path else None
    if not axis_cal and args.bounds_mm is None:
        p.error("Manual-axis fusion requires --bounds-mm to remove static background")
    bounds = np.asarray(args.bounds_mm, np.float64).reshape(3, 2) if args.bounds_mm else None
    if (bounds is not None and (np.any(bounds[:, 0] >= bounds[:, 1]) or not np.isfinite(bounds).all())) or \
       args.voxel_mm <= 0 or args.radius_mm <= 0 or args.height_mm[0] >= args.height_mm[1]:
        p.error("Invalid object bounds or voxel size")
    axis = axis_cal["direction"] if axis_cal else np.asarray(args.axis_direction, np.float64)
    if not np.isfinite(axis).all() or np.linalg.norm(axis) == 0:
        p.error("Axis direction must be finite and nonzero")
    poses = json.loads(args.camera_poses.read_text(encoding="utf-8")) if args.camera_poses else {}
    roots = (sorted(args.session_root.resolve().glob("view_*")) if args.session_root
             else [path.resolve() for path in args.view_root])
    roots = [path for path in roots if path.is_dir()]
    if not roots:
        raise ValueError("No view directories found")
    if len({(root.parent.parent, root.parent) for root in roots}) != 1:
        raise ValueError("All view roots must belong to the same object and session")
    default_name = "all_views_axis_calibrated" if args.session_root and axis_cal else (
        "all_views" if args.session_root else roots[0].name + ("_axis_calibrated" if axis_cal else ""))
    output = args.output or ROOT / "processed" / "fused" / roots[0].parent.parent.name / roots[0].parent.name / default_name
    repeats = [(root, path) for root in roots for path in sorted(root.glob("repeat_*/pose.json"))]
    if len(repeats) < 2:
        raise ValueError("Need at least two captured angles")
    scans = []
    for root, pose_path in repeats:
        name = f"{root.name}/{pose_path.parent.name}"
        pose = json.loads(pose_path.read_text(encoding="utf-8"))
        manifest = json.loads((pose_path.parent / "manifest.json").read_text(encoding="utf-8"))
        if not manifest.get("complete") or not manifest["protocol"].startswith("dense_dots_"):
            raise ValueError(f"Incomplete or unsupported scan: {name}")
        if pose["arm_pose_id"] != manifest["arm_pose_id"] or pose["turntable_deg_commanded"] != manifest["turntable_deg_commanded"]:
            raise ValueError(f"Pose and scan manifest disagree: {name}")
        result = args.processed_root / root.relative_to(ROOT / "raw") / pose_path.parent.name
        qc = json.loads((result / "qc.json").read_text(encoding="utf-8"))
        if qc["point_count"] < 1 or qc["coordinate_frame"] != "rectified_left_camera":
            raise ValueError(f"Invalid reconstruction: {name}")
        cloud_path = result / "cloud_left.ply"
        with cloud_path.open("rb") as stream:
            header = stream.read(256)
        if b"comment coordinates original_left_camera; unit mm" not in header:
            raise ValueError(f"Expected original-left-camera cloud: {cloud_path}")
        xyz, bgr = read_ply(cloud_path)
        command_angle = float(pose["turntable_deg_commanded"])
        if axis_cal and command_angle not in axis_cal["angles_by_command"]:
            raise ValueError(f"No calibrated angle for {command_angle:g} deg: {name}")
        if axis_cal and Path(qc["calibration"]).resolve() != axis_cal["stereo_calibration"]:
            raise ValueError(f"Axis and cloud use different stereo calibrations: {name}")
        scans.append({"name": name, "command_angle": command_angle,
                      "angle": axis_cal["angles_by_command"][command_angle] if axis_cal else command_angle,
                      "arm": pose["arm_pose_id"], "calibration": qc["calibration"],
                      "xyz": xyz, "bgr": bgr})
    actions = [(row["arm"], row["angle"] % 360) for row in scans]
    if len(set(actions)) != len(actions):
        raise ValueError("Duplicate arm pose and turntable angle")
    if len({row["arm"] for row in scans}) > 1 and not args.camera_poses:
        raise ValueError("Different arm poses require --camera-poses")
    if len({row["calibration"] for row in scans}) > 1:
        raise ValueError("Reconstructed views use different stereo calibrations")
    for row in scans:
        transform = np.asarray(poses.get(row["name"], np.eye(4)), np.float64)
        if transform.shape != (4, 4) or not np.isfinite(transform).all() or not np.allclose(transform[3], [0, 0, 0, 1]):
            raise ValueError(f"Invalid camera transform: {row['name']}")
        if args.camera_poses and row["name"] not in poses:
            raise ValueError(f"Missing camera transform: {row['name']}")
        row["xyz"] = (row["xyz"] @ transform[:3, :3].T + transform[:3, 3]).astype(np.float32)
    if axis_cal:
        center = axis_cal["point"]
        axis_source = "axis_calibration_json"
    elif args.axis_point_mm is None:
        centers = [np.median(row["xyz"], axis=0) for row in scans]
        center = np.median(centers, axis=0)
        axis_source = "estimated_from_cropped_clouds"
    else:
        center = np.asarray(args.axis_point_mm, np.float64)
        axis_source = "user_supplied"
    if not np.isfinite(center).all():
        raise ValueError("Invalid axis point")
    for row in scans:
        xyz = row["xyz"]
        keep = np.isfinite(xyz).all(axis=1)
        if bounds is not None:
            keep &= np.all((xyz >= bounds[:, 0]) & (xyz <= bounds[:, 1]), axis=1)
        if axis_cal:
            keep &= crop_cylinder(xyz, center, axis, args.radius_mm, args.height_mm)
        row["xyz"], row["bgr"] = xyz[keep], row["bgr"][keep]
        if not len(row["xyz"]):
            raise ValueError(f"Object crop is empty: {row['name']}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "aligned").mkdir(exist_ok=True)
    output_frame = "turntable_zero_calibrated" if axis_cal else "turntable_zero_estimated"
    all_xyz, all_bgr, summary = [], [], []
    for row in scans:
        aligned = rotate_about_axis(row["xyz"], center, axis, -args.angle_sign * row["angle"])
        filename = row["name"].split("/")[-1] if len(roots) == 1 else row["name"].replace("/", "_")
        write_ply(output / "aligned" / f"{filename}.ply", aligned, row["bgr"], output_frame)
        all_xyz.append(aligned)
        all_bgr.append(row["bgr"])
        summary.append({"repeat": row["name"], "command_angle_deg": row["command_angle"],
                        "angle_deg": row["angle"], "arm_pose_id": row["arm"],
                        "cropped_points": len(aligned)})
        print(f"{row['name']}: {row['angle']:g} deg, {len(aligned):,} object points", flush=True)
    points, colors = voxel_average(np.concatenate(all_xyz), np.concatenate(all_bgr), args.voxel_mm)
    write_ply(output / "fused.ply", points, colors, output_frame)
    report = {"coordinate_frame": output_frame, "unit": "mm", "sources": [str(root) for root in roots],
              "axis_point_mm": center.tolist(), "axis_direction": (axis/np.linalg.norm(axis)).tolist(),
              "axis_source": axis_source, "axis_calibration": str(axis_path.resolve()) if axis_cal else None,
              "angle_source": "calibrated_command_lookup" if axis_cal else "commanded",
              "axis_fit_rms_mm": axis_cal["point_rms_mm"] if axis_cal else None,
              "axis_return_to_zero": axis_cal["return_to_zero"] if axis_cal else None,
              "angle_sign": args.angle_sign, "bounds_mm": args.bounds_mm,
              "radius_mm": args.radius_mm if axis_cal else None,
              "height_mm": args.height_mm if axis_cal else None,
              "voxel_mm": args.voxel_mm, "input_points": int(sum(len(x) for x in all_xyz)),
              "fused_points": len(points), "views": summary,
              "warning": ("Calibrated command-angle lookup is from a separate session; object-scan arrival and rig stability remain unverified."
                          if axis_cal else "Axis and commanded angles are unverified; inspect overlap before using as ground truth.")}
    (output / "fusion.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if axis_source == "estimated_from_cropped_clouds":
        print("[WARNING] Turntable axis was estimated from object clouds; verify alignment before using as ground truth.", flush=True)
    elif axis_cal:
        print("[WARNING] Measured angles come from a separate calibration session; verify object-scan arrival and rig stability.", flush=True)
    print(f"Fused {len(points):,} points from {len(scans)} views -> {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
