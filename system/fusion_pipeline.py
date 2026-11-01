"""Prepare session-specific turntable geometry and fuse reconstructed views."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for package in ("arm", "turntable", "code"):
    sys.path.insert(0, str(ROOT / package))

from fixed_board import board_from_config, detect_corners_with_contrast_fallback, estimate_pose
from fuse_views import validate_rigid_transform


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def validate_axis_capture(axis_images: Path) -> None:
    manifest_path = Path(axis_images) / "capture_manifest.json"
    if not manifest_path.is_file() or not json.loads(
            manifest_path.read_text(encoding="utf-8")).get("complete"):
        raise ValueError(f"Automatic fusion requires a complete axis capture: {axis_images}; "
                         "set --fusion-axis-images or use --no-fusion")


def axis_to_reference_transform(T_axis_from_board, T_reference_from_scan,
                                T_scan_from_board) -> np.ndarray:
    """Convert points from the axis capture camera to the scan reference."""
    axis_board = validate_rigid_transform(T_axis_from_board, "axis board pose")
    reference_scan = validate_rigid_transform(T_reference_from_scan, "scan reference pose")
    scan_board = validate_rigid_transform(T_scan_from_board, "scan board pose")
    return validate_rigid_transform(reference_scan @ scan_board @ np.linalg.inv(axis_board),
                                    "axis-to-reference")


def find_axis_board_pose(axis_images: Path, board, K, D,
                         min_corners: int = 6, max_rms_px: float = 2.0):
    """Use the best well-spread, accurately fitted left-camera board image."""
    candidates = []
    for path in sorted(Path(axis_images).glob("angle_*/left.png")):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        corners, ids, method = detect_corners_with_contrast_fallback(image, board, min_corners)
        if len(ids) < min_corners:
            continue
        try:
            pose, rms = estimate_pose(board, corners, ids, K, D)
        except (ValueError, cv2.error):
            continue
        if np.isfinite(rms) and rms <= max_rms_px:
            candidates.append((len(ids), rms, path, pose, method))
    if not candidates:
        raise ValueError(f"No axis image gives a valid fixed board pose: need at least "
                         f"{min_corners} non-collinear corners and RMS <= {max_rms_px:g} px")
    count, rms, path, pose, method = min(candidates, key=lambda row: (-row[0], row[1]))
    return pose, {"image": str(path.resolve()), "corners": count,
                  "reprojection_rms_px": rms, "detection_method": method}


def register_axis_camera(axis_images: Path, sources: list[Path], board_path: Path,
                         calibration: Path, output: Path) -> Path:
    """Locate the axis camera using the same stationary board as the scan."""
    config = json.loads(board_path.read_text(encoding="utf-8"))
    board = board_from_config(config)
    with np.load(calibration) as data:
        K, D = data["K_left"], data["D_left"]
    axis_board, observation = find_axis_board_pose(axis_images, board, K, D)
    scan_sources = sorted(sources, key=lambda source: (
        json.loads((source / "pose.json").read_text(encoding="utf-8"))["arm_pose_id"] != 1,
        str(source)))
    source = scan_sources[0]
    localization = json.loads((source / "localization.json").read_text(encoding="utf-8"))
    if localization.get("reprojection_rms_px", float("inf")) > 3.0:
        raise ValueError(f"Scan fixed board pose is unreliable: {source}")
    transform = axis_to_reference_transform(
        axis_board, localization["T_reference_from_left"],
        localization["T_left_from_board"])
    _write_json(output, {"status": "calibrated_fixed_board",
                         "T_reference_from_axis_camera": transform.tolist(),
                         "axis_board_observation": observation,
                         "scan_board_localization": str((source / "localization.json").resolve()),
                         "scan_board_reprojection_rms_px": localization["reprojection_rms_px"],
                         "stereo_calibration": str(calibration.resolve())})
    return output


def fuse_reconstructed_session(session_root: Path, selected_sources: list[Path],
                               calibration: Path, axis_images: Path, board_path: Path,
                               raw_root: Path, processed_root: Path,
                               radius_mm: float = 65.0,
                               height_mm: tuple[float, float] = (-95.0, 110.0),
                               voxel_mm: float = 1.0,
                               exclude_view_ids: set[int] | None = None,
                               calibrate=None, fuse=None) -> Path:
    """Run axis fit, fixed-board registration and fusion as one reported stage."""
    if calibrate is None:
        from calibrate_axis import main as calibrate
    if fuse is None:
        from fuse_views import main as fuse
    session_root = Path(session_root)
    calibration = Path(calibration)
    axis_images = Path(axis_images)
    board_path = Path(board_path)
    raw_root = Path(raw_root)
    processed_root = Path(processed_root)
    output_root = processed_root / "fused" / session_root.parent.name / session_root.name
    output = output_root / "all_views_axis_calibrated"
    status_path = output_root / "fusion_status.json"
    excluded = set(exclude_view_ids or ())
    try:
        if radius_mm <= 0 or voxel_mm <= 0 or height_mm[0] >= height_mm[1]:
            raise ValueError("Invalid fusion radius, height or voxel size")
        validate_axis_capture(axis_images)
        plan_path = session_root / "scan_plan.json"
        if plan_path.is_file() and json.loads(plan_path.read_text(encoding="utf-8"))["board"] != \
                json.loads(board_path.read_text(encoding="utf-8")):
            raise ValueError("Scan fixed board differs from fusion board")
        selected = [source for source in selected_sources
                    if int(source.parent.name.removeprefix("view_")) not in excluded]
        available = {int(source.parent.name.removeprefix("view_")) for source in selected_sources}
        if excluded - available:
            raise ValueError(f"Excluded views are not in this scan: {sorted(excluded - available)}")
        if len(selected) < 2:
            raise ValueError("Fusion requires at least two reconstructed views after exclusion")
        poses_path = session_root / "camera_poses.json"
        if not poses_path.is_file():
            raise FileNotFoundError(f"Camera pose map missing: {poses_path}")
        _write_json(status_path, {"state": "running", "selected_repeats": [str(p) for p in selected],
                                  "excluded_view_ids": sorted(excluded)})
        axis_output = output_root / "fusion_calibration" / "axis_calibration.json"
        result = calibrate(["--images", str(axis_images), "--stereo", str(calibration),
                            "--output", str(axis_output)])
        if result != 0 or not axis_output.is_file():
            raise RuntimeError("Turntable axis calibration did not produce a result")
        axis_data = json.loads(axis_output.read_text(encoding="utf-8"))
        if axis_data["point_rms_mm"] > 3.0 or \
           axis_data["return_to_zero"]["left_corner_shift_rms_px"] > 2.0:
            raise ValueError("Turntable axis fit or return-to-zero exceeds fusion quality limits")
        registration = register_axis_camera(axis_images, selected, board_path, calibration,
                                            output_root / "fusion_calibration" / "axis_to_reference.json")
        command = ["--repeat-root", *map(str, selected),
                   "--raw-root", str(raw_root), "--processed-root", str(processed_root),
                   "--output", str(output), "--camera-poses", str(poses_path),
                   "--axis-calibration", str(axis_output), "--axis-to-reference", str(registration),
                   "--radius-mm", str(radius_mm), "--height-mm", *map(str, height_mm),
                   "--voxel-mm", str(voxel_mm)]
        if fuse(command) != 0:
            raise RuntimeError("Point cloud fusion failed")
        report_path = output / "fusion.json"
        if not (output / "fused.ply").is_file() or not report_path.is_file():
            raise RuntimeError("Point cloud fusion did not produce fused.ply and fusion.json")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if len(report["views"]) != len(selected) or report["fused_points"] < 1:
            raise RuntimeError("Point cloud fusion report does not match selected views")
        _write_json(status_path, {"state": "complete", "output": str(output.resolve()),
                                  "selected_repeats": [str(p) for p in selected],
                                  "excluded_view_ids": sorted(excluded),
                                  "fused_points": report["fused_points"],
                                  "axis_calibration": str(axis_output.resolve()),
                                  "axis_to_reference": str(registration.resolve())})
        print(f"Automatic fusion complete: {report['fused_points']:,} points -> {output.resolve()}",
              flush=True)
        return output
    except Exception as exc:
        _write_json(status_path, {"state": "failed", "error": str(exc),
                                  "excluded_view_ids": sorted(excluded)})
        raise
