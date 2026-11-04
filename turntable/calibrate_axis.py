"""Calibrate the rotating axis and stationary board in one coordinate frame."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BOARD = ROOT / "turntable/targets/aruco_cube_4faces_config.json"
DEFAULT_FIXED_BOARD = ROOT / "arm/targets/charuco_7x5_10mm_7mm_config.json"
sys.path.insert(0, str(ROOT / "arm"))
from fixed_board import board_from_config, detect_corners_with_contrast_fallback, estimate_pose  # noqa: E402


def rotation_difference_deg(a, b):
    relative = a[:3, :3] @ b[:3, :3].T
    return float(np.rad2deg(np.arccos(np.clip((np.trace(relative) - 1) / 2, -1, 1))))


def fit_stationary_board(image_rows, board, K, D, min_corners=18,
                         max_rms_px=1.0, min_inliers=6,
                         max_translation_mm=3.0, max_rotation_deg=0.5):
    """Estimate one board pose from multiple stationary-board photographs."""
    candidates, observations = [], []
    for name, image in image_rows:
        corners, ids, method = detect_corners_with_contrast_fallback(image, board, min_corners)
        record = {"image": name, "corners": len(ids), "detection_method": method}
        if len(ids) < min_corners:
            record["status"] = "too_few_corners"
        else:
            try:
                pose, rms = estimate_pose(board, corners, ids, K, D)
                record["reprojection_rms_px"] = rms
                if np.isfinite(rms) and rms <= max_rms_px:
                    candidates.append((name, pose))
                    record["status"] = "candidate"
                else:
                    record["status"] = "high_reprojection_error"
            except (ValueError, cv2.error) as exc:
                record["status"] = f"pose_failed: {exc}"
        observations.append(record)
    if len(candidates) < min_inliers:
        raise ValueError(f"Stationary fixed board has only {len(candidates)} usable images; "
                         f"need {min_inliers} with at least {min_corners} corners and "
                         f"RMS <= {max_rms_px:g} px. Adjust board exposure and recapture.")
    poses = [row[1] for row in candidates]
    # A medoid rejects one accurate-looking but geometrically inconsistent PnP result.
    distances = np.array([[np.linalg.norm(a[:3, 3] - b[:3, 3]) +
                           100 * np.deg2rad(rotation_difference_deg(a, b))
                           for b in poses] for a in poses])
    medoid = poses[int(np.argmin(np.median(distances, axis=1)))]
    inliers = [(name, pose) for name, pose in candidates
               if np.linalg.norm(pose[:3, 3] - medoid[:3, 3]) <= max_translation_mm
               and rotation_difference_deg(pose, medoid) <= max_rotation_deg]
    if len(inliers) < min_inliers:
        raise ValueError(f"Stationary board pose is inconsistent across axis images: "
                         f"{len(inliers)}/{len(candidates)} agree within "
                         f"{max_translation_mm:g} mm and {max_rotation_deg:g} deg")
    mean = np.mean([pose[:3, :3] for _, pose in inliers], axis=0)
    U, _, Vt = np.linalg.svd(mean)
    rotation = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
    consensus = np.eye(4)
    consensus[:3, :3] = rotation
    consensus[:3, 3] = np.mean([pose[:3, 3] for _, pose in inliers], axis=0)
    selected = {name for name, _ in inliers}
    for row in observations:
        if row["status"] == "candidate":
            row["status"] = "inlier" if row["image"] in selected else "pose_outlier"
    return consensus, {"observations": observations,
                       "inlier_count": len(inliers),
                       "max_inlier_translation_mm": float(max(
                           np.linalg.norm(pose[:3, 3] - consensus[:3, 3])
                           for _, pose in inliers)),
                       "max_inlier_rotation_deg": float(max(
                           rotation_difference_deg(pose, consensus)
                           for _, pose in inliers))}


def axis_in_board_frame(point, direction, T_left_from_board):
    T_board_from_left = np.linalg.inv(T_left_from_board)
    axis = T_board_from_left[:3, :3] @ np.asarray(direction, float)
    axis /= np.linalg.norm(axis)
    point = T_board_from_left[:3, :3] @ np.asarray(point, float) + T_board_from_left[:3, 3]
    # Preserve the along-axis origin used by existing height crop settings.
    return point, axis


def default_output_path(images):
    return ROOT / "calibration" / "turntable_axis" / images.name / "axis_calibration.json"


def marker_pose(corners, ids, config, K, D, min_markers=2):
    """Map visible marker corners into one cuboid coordinate frame."""
    if ids is None:
        raise ValueError("No ArUco markers detected")
    obj_rows, img_rows = [], []
    for marker_corners, marker_id in zip(corners, ids.ravel()):
        marker = config["markers"].get(str(int(marker_id)))
        if marker is None:
            continue
        obj_rows.append(np.asarray(marker["corners_mm"], np.float32))
        img_rows.append(np.asarray(marker_corners, np.float32).reshape(4, 2))
    count = len(obj_rows)
    if count < min_markers:
        raise ValueError(f"Only {count} known markers; need {min_markers}")
    obj = np.concatenate(obj_rows)
    img = np.concatenate(img_rows)
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise ValueError("solvePnP failed")
    projected, _ = cv2.projectPoints(obj, rvec, tvec, K, D)
    rms = float(np.sqrt(np.mean(np.sum((projected.reshape(-1, 2) - img) ** 2, axis=1))))
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec.reshape(3), count, rms


def detect_pose(image, config, K, D, min_markers=2):
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, config["dictionary"]))
    corners, ids, _ = cv2.aruco.ArucoDetector(dictionary).detectMarkers(
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    return marker_pose(corners, ids, config, K, D, min_markers)


def stereo_marker_pose(left_corners, left_ids, right_corners, right_ids, config,
                       K_left, D_left, K_right, D_right, R_right_from_left,
                       t_right_from_left, min_markers=2):
    """Triangulate shared marker corners and rigidly align the measured cuboid."""
    if left_ids is None or right_ids is None:
        raise ValueError("Markers are missing from one camera")
    left = {int(i): c.reshape(4, 2) for c, i in zip(left_corners, left_ids.ravel())}
    right = {int(i): c.reshape(4, 2) for c, i in zip(right_corners, right_ids.ravel())}
    common = sorted(set(left) & set(right) & {int(i) for i in config["markers"]})
    if len(common) < min_markers:
        raise ValueError(f"Only {len(common)} markers visible in both cameras; need {min_markers}")
    obj = np.concatenate([config["markers"][str(i)]["corners_mm"] for i in common]).astype(np.float64)
    img_left = np.concatenate([left[i] for i in common]).astype(np.float64)
    img_right = np.concatenate([right[i] for i in common]).astype(np.float64)
    ray_left = cv2.undistortPoints(img_left.reshape(-1, 1, 2), K_left, D_left).reshape(-1, 2).T
    ray_right = cv2.undistortPoints(img_right.reshape(-1, 1, 2), K_right, D_right).reshape(-1, 2).T
    R_lr = np.asarray(R_right_from_left, dtype=np.float64)
    t_lr = np.asarray(t_right_from_left, dtype=np.float64).reshape(3)
    P_left = np.column_stack((np.eye(3), np.zeros(3)))
    P_right = np.column_stack((R_lr, t_lr))
    homogeneous = cv2.triangulatePoints(P_left, P_right, ray_left, ray_right)
    xyz = (homogeneous[:3] / homogeneous[3]).T
    if not np.isfinite(xyz).all() or np.any(xyz[:, 2] <= 0):
        raise ValueError("Stereo triangulation produced invalid 3D points")
    obj_center, xyz_center = obj.mean(axis=0), xyz.mean(axis=0)
    U, _, Vt = np.linalg.svd((obj - obj_center).T @ (xyz - xyz_center))
    correction = np.diag([1., 1., np.linalg.det(Vt.T @ U.T)])
    R = Vt.T @ correction @ U.T
    t = xyz_center - R @ obj_center
    fitted = (R @ obj.T).T + t
    fit_rms_mm = float(np.sqrt(np.mean(np.sum((fitted - xyz) ** 2, axis=1))))
    rvec_left, _ = cv2.Rodrigues(R)
    proj_left, _ = cv2.projectPoints(obj, rvec_left, t, K_left, D_left)
    rvec_right, _ = cv2.Rodrigues(R_lr @ R)
    proj_right, _ = cv2.projectPoints(obj, rvec_right, R_lr @ t + t_lr, K_right, D_right)
    rms_left = float(np.sqrt(np.mean(np.sum((proj_left.reshape(-1, 2) - img_left) ** 2, axis=1))))
    rms_right = float(np.sqrt(np.mean(np.sum((proj_right.reshape(-1, 2) - img_right) ** 2, axis=1))))
    return R, t, len(common), fit_rms_mm, rms_left, rms_right


def detect_stereo_pose(left_image, right_image, config, stereo, min_markers=2):
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, config["dictionary"]))
    detector = cv2.aruco.ArucoDetector(dictionary)
    left_corners, left_ids, _ = detector.detectMarkers(cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY))
    right_corners, right_ids, _ = detector.detectMarkers(cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY))
    return stereo_marker_pose(left_corners, left_ids, right_corners, right_ids, config,
                              *stereo, min_markers)


def marker_corner_shift(first_image, second_image, config):
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, config["dictionary"]))
    detector = cv2.aruco.ArucoDetector(dictionary)
    views = []
    for image in (first_image, second_image):
        corners, ids, _ = detector.detectMarkers(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        views.append({int(i): c.reshape(4, 2) for c, i in zip(corners, ids.ravel())}
                     if ids is not None else {})
    common = sorted(set(views[0]) & set(views[1]) & {int(i) for i in config["markers"]})
    if not common:
        raise ValueError("No common markers in initial and return-to-zero images")
    difference = np.concatenate([views[1][i] - views[0][i] for i in common])
    return len(common), float(np.sqrt(np.mean(np.sum(difference ** 2, axis=1))))


def fit_axis_from_poses(poses):
    """Use observed rigid rotations; axis point is the point nearest camera origin."""
    if len(poses) < 8 or poses[0][0] != 0:
        raise ValueError("Need zero pose and at least 8 distinct angles")
    R0, t0 = poses[0][1:]
    relatives = []
    axis_samples = []
    for command, Rb, tb in poses[1:]:
        R = Rb @ R0.T
        t = tb - R @ t0
        relatives.append((command, R, t))
        if 0 < command <= 150:
            rv, _ = cv2.Rodrigues(R)
            axis_samples.append(rv.ravel())
    if len(axis_samples) < 3:
        raise ValueError("Need at least three views between 30 and 150 degrees")
    _, _, vh = np.linalg.svd(np.asarray(axis_samples), full_matrices=False)
    axis = vh[0]
    if np.dot(axis, axis_samples[0]) < 0:
        axis = -axis
    # The rotation axis has one free coordinate along itself. c·n=0 fixes it.
    A = np.vstack([np.eye(3) - R for _, R, _ in relatives] + [axis[None, :]])
    b = np.concatenate([t for _, _, t in relatives] + [np.array([0.])])
    center = np.linalg.lstsq(A, b, rcond=None)[0]
    errors = []
    actual = []
    for command, R, t in relatives:
        measured = np.rad2deg(np.arctan2(
            np.dot(axis, [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / 2,
            (np.trace(R) - 1) / 2)) % 360
        actual.append({"command_deg": command, "measured_deg": float(measured),
                       "error_deg": float((measured - command + 180) % 360 - 180)})
        errors.append(np.linalg.norm((np.eye(3) - R) @ center - t))
    return {"axis_direction": axis, "axis_point_mm": center,
            "point_rms_mm": float(np.sqrt(np.mean(np.square(errors)))),
            "angles": actual}


def turntable_top_from_target_poses(poses, fitted_axis, max_spread_mm=5.0):
    """Project the bottom-centred cuboid origins onto the calibrated axis."""
    point = np.asarray(fitted_axis["axis_point_mm"], np.float64)
    direction = np.asarray(fitted_axis["axis_direction"], np.float64).copy()
    if (point.shape != (3,) or direction.shape != (3,) or
            not np.isfinite(point).all() or not np.isfinite(direction).all() or
            np.linalg.norm(direction) < 1e-8):
        raise ValueError("invalid axis for target bottom height")
    direction /= np.linalg.norm(direction)
    for _, rotation, _ in poses:
        rotation = np.asarray(rotation, np.float64)
        if (rotation.shape != (3, 3) or not np.isfinite(rotation).all() or
                np.dot(rotation[:, 2], direction) < 0.95):
            raise ValueError("target up direction disagrees with fitted turntable axis")
    heights = np.asarray([np.dot(np.asarray(t, np.float64) - point, direction)
                          for _, _, t in poses], np.float64)
    if len(heights) < 8 or not np.isfinite(heights).all():
        raise ValueError("target bottom height requires eight finite poses")
    spread = float(np.ptp(heights))
    if spread > max_spread_mm:
        raise ValueError(f"target bottom height varies by {spread:.1f} mm across rotation")
    return {"turntable_top_axis_mm": float(np.median(heights)),
            "turntable_top_height_spread_mm": spread,
            "turntable_top_source": "bottom_centred_axis_cuboid_on_turntable"}


def optional_turntable_top(config, poses, fitted_axis):
    """Keep axis calibration usable when the optional height check fails."""
    if not str(config.get("frame", "")).startswith("cuboid_center_at_bottom"):
        return {}
    try:
        return turntable_top_from_target_poses(poses, fitted_axis)
    except ValueError as exc:
        return {"turntable_top_error": str(exc)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images", type=Path, required=True, help="capture_axis.py session directory")
    p.add_argument("--stereo", type=Path,
                   default=ROOT / "calibration/stereo_pose_lower/stereo_calibration.npz")
    p.add_argument("--board", type=Path, default=DEFAULT_BOARD)
    p.add_argument("--fixed-board", type=Path, default=DEFAULT_FIXED_BOARD,
                   help="stationary ChArUco board visible beside the rotating target")
    p.add_argument("--min-fixed-board-corners", type=int, default=18)
    p.add_argument("--max-fixed-board-rms-px", type=float, default=1.0)
    p.add_argument("--output", type=Path,
                   help="Default: calibration/turntable_axis/<session-id>/axis_calibration.json")
    p.add_argument("--min-markers", type=int, default=2)
    p.add_argument("--allow-board-geometry-update", action="store_true",
                   help="Reprocess captured images with a corrected geometry for the same marker IDs")
    args = p.parse_args(argv)
    config = json.loads(args.board.read_text(encoding="utf-8"))
    fixed_config = json.loads(args.fixed_board.read_text(encoding="utf-8"))
    fixed_board = board_from_config(fixed_config)
    manifest_path = args.images / "capture_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("complete") or len(manifest.get("angles", [])) != 13:
        raise ValueError("Axis capture session is incomplete")
    board_updated = manifest["board"] != config
    if board_updated:
        old = manifest["board"]
        if not args.allow_board_geometry_update or \
           old["target_type"] != config["target_type"] or \
           old["dictionary"] != config["dictionary"] or \
           set(old["markers"]) != set(config["markers"]):
            raise ValueError("Capture board differs from calibration board; use --allow-board-geometry-update only for corrected geometry of the same marker IDs")
    if args.min_markers < 1:
        p.error("--min-markers must be positive")
    if not 4 <= args.min_fixed_board_corners <= 24 or args.max_fixed_board_rms_px <= 0:
        p.error("Invalid stationary fixed-board quality limits")
    if manifest.get("fixed_board") != fixed_config:
        raise ValueError("Axis capture stationary board differs from --fixed-board")
    with np.load(args.stereo) as cal:
        stereo = tuple(cal[key] for key in
                       ("K_left", "D_left", "K_right", "D_right",
                        "R_right_from_left", "t_right_from_left_mm"))
        size = tuple(int(v) for v in cal["image_size"])
    stereo_metadata = json.loads(args.stereo.with_suffix(".json").read_text(encoding="utf-8"))
    if tuple(stereo_metadata["image_size"]) != size or \
       manifest["left_serial"] != stereo_metadata["left_serial"] or \
       manifest["right_serial"] != stereo_metadata["right_serial"] or \
       tuple(manifest["camera_size_px"]) != size:
        raise ValueError("Capture camera identity or size is inconsistent with calibration")
    poses, observations, fixed_board_images = [], [], []
    for angle in range(0, 360, 30):
        folder = args.images / f"angle_{angle:03d}"
        images = [cv2.imread(str(folder / f"{side}.png"), cv2.IMREAD_COLOR)
                  for side in ("left", "right")]
        if any(image is None or (image.shape[1], image.shape[0]) != size for image in images):
            raise ValueError(f"Missing image or size differs from stereo calibration: {folder}")
        R, t, count, fit_rms, left_rms, right_rms = detect_stereo_pose(
            *images, config, stereo, args.min_markers)
        fixed_board_images.append((f"angle_{angle:03d}/left.png", images[0]))
        poses.append((angle, R, t))
        observations.append({"command_deg": angle, "common_markers": count,
                             "triangulation_fit_rms_mm": fit_rms,
                             "reprojection_rms_left_px": left_rms,
                             "reprojection_rms_right_px": right_rms})
        print(f"{angle:3d} deg: {count} common markers, 3D fit {fit_rms:.3f} mm, "
              f"reprojection L/R {left_rms:.2f}/{right_rms:.2f} px")
    return_folder = args.images / "angle_000_return"
    return_images = [cv2.imread(str(return_folder / f"{side}.png"), cv2.IMREAD_COLOR)
                     for side in ("left", "right")]
    if any(image is None or (image.shape[1], image.shape[0]) != size for image in return_images):
        raise ValueError(f"Missing or invalid return-to-zero images: {return_folder}")
    fixed_board_images.append(("angle_000_return/left.png", return_images[0]))
    R_return, t_return, return_count, return_fit, return_left, return_right = detect_stereo_pose(
        *return_images, config, stereo, args.min_markers)
    first_image = cv2.imread(str(args.images / "angle_000" / "left.png"), cv2.IMREAD_COLOR)
    shift_markers, shift_rms = marker_corner_shift(first_image, return_images[0], config)
    R_zero, t_zero = poses[0][1:]
    closure_R = R_return @ R_zero.T
    closure_rot = float(np.rad2deg(np.arccos(np.clip((np.trace(closure_R) - 1) / 2, -1, 1))))
    closure_mm = float(np.linalg.norm(t_return - t_zero))
    fitted = fit_axis_from_poses(poses)
    table_top = optional_turntable_top(config, poses, fitted)
    T_left_from_board, fixed_board_fit = fit_stationary_board(
        fixed_board_images, fixed_board, stereo[0], stereo[1],
        min_corners=args.min_fixed_board_corners,
        max_rms_px=args.max_fixed_board_rms_px)
    board_point, board_direction = axis_in_board_frame(
        fitted["axis_point_mm"], fitted["axis_direction"], T_left_from_board)
    result = {"coordinate_frame": "fixed_ChArUco_board", "unit": "mm",
              "pose_method": "stereo_triangulated_marker_corners_and_rigid_fit",
              "board_geometry_updated_after_capture": board_updated,
              "axis_point_mm": board_point.tolist(),
              "axis_direction": board_direction.tolist(),
              "axis_point_definition": "camera-nearest axis point transformed into fixed-board coordinates",
              "axis_point_original_left_camera_mm": fitted["axis_point_mm"].tolist(),
              "axis_direction_original_left_camera": fitted["axis_direction"].tolist(),
              "point_rms_mm": fitted["point_rms_mm"],
              **table_top,
              "angles": fitted["angles"], "observations": observations,
              "return_to_zero": {"markers": return_count,
                                 "triangulation_fit_rms_mm": return_fit,
                                 "reprojection_rms_left_px": return_left,
                                 "reprojection_rms_right_px": return_right,
                                 "common_left_markers": shift_markers,
                                 "left_corner_shift_rms_px": shift_rms,
                                 "rotation_error_deg": closure_rot,
                                 "translation_error_mm": closure_mm},
              "board": config, "fixed_board": fixed_config,
              "fixed_board_calibration": {**fixed_board_fit,
                                          "T_left_from_board": T_left_from_board.tolist()},
              "stereo_calibration": str(args.stereo.resolve()),
              "warning": "Single-face orientation can remain noisy even with stereo; compare return image corner shift as well as pose errors before fusion."}
    output = args.output or default_output_path(args.images)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Axis RMS {result['point_rms_mm']:.3f} mm; stationary board "
          f"{fixed_board_fit['inlier_count']}/{len(fixed_board_images)} pose inliers, "
          f"max translation {fixed_board_fit['max_inlier_translation_mm']:.2f} mm / "
          f"rotation {fixed_board_fit['max_inlier_rotation_deg']:.2f} deg -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
