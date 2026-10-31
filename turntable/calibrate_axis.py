"""Estimate a turntable axis in the left-camera frame from four-face ArUco photos."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BOARD = ROOT / "turntable/targets/aruco_cube_4faces_config.json"


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


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images", type=Path, required=True, help="capture_axis.py session directory")
    p.add_argument("--stereo", type=Path, default=ROOT / "calibration/stereo/stereo_calibration.npz")
    p.add_argument("--board", type=Path, default=DEFAULT_BOARD)
    p.add_argument("--output", type=Path,
                   help="Default: calibration/turntable_axis/<session-id>/axis_calibration.json")
    p.add_argument("--min-markers", type=int, default=2)
    p.add_argument("--allow-board-geometry-update", action="store_true",
                   help="Reprocess captured images with a corrected geometry for the same marker IDs")
    args = p.parse_args(argv)
    config = json.loads(args.board.read_text(encoding="utf-8"))
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
    poses, observations = [], []
    for angle in range(0, 360, 30):
        folder = args.images / f"angle_{angle:03d}"
        images = [cv2.imread(str(folder / f"{side}.png"), cv2.IMREAD_COLOR)
                  for side in ("left", "right")]
        if any(image is None or (image.shape[1], image.shape[0]) != size for image in images):
            raise ValueError(f"Missing image or size differs from stereo calibration: {folder}")
        R, t, count, fit_rms, left_rms, right_rms = detect_stereo_pose(
            *images, config, stereo, args.min_markers)
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
    R_return, t_return, return_count, return_fit, return_left, return_right = detect_stereo_pose(
        *return_images, config, stereo, args.min_markers)
    first_image = cv2.imread(str(args.images / "angle_000" / "left.png"), cv2.IMREAD_COLOR)
    shift_markers, shift_rms = marker_corner_shift(first_image, return_images[0], config)
    R_zero, t_zero = poses[0][1:]
    closure_R = R_return @ R_zero.T
    closure_rot = float(np.rad2deg(np.arccos(np.clip((np.trace(closure_R) - 1) / 2, -1, 1))))
    closure_mm = float(np.linalg.norm(t_return - t_zero))
    fitted = fit_axis_from_poses(poses)
    result = {"coordinate_frame": "unrectified_left_camera", "unit": "mm",
              "pose_method": "stereo_triangulated_marker_corners_and_rigid_fit",
              "board_geometry_updated_after_capture": board_updated,
              "axis_point_mm": fitted["axis_point_mm"].tolist(),
              "axis_direction": fitted["axis_direction"].tolist(),
              "axis_point_definition": "point on axis closest to left camera origin",
              "point_rms_mm": fitted["point_rms_mm"],
              "angles": fitted["angles"], "observations": observations,
              "return_to_zero": {"markers": return_count,
                                 "triangulation_fit_rms_mm": return_fit,
                                 "reprojection_rms_left_px": return_left,
                                 "reprojection_rms_right_px": return_right,
                                 "common_left_markers": shift_markers,
                                 "left_corner_shift_rms_px": shift_rms,
                                 "rotation_error_deg": closure_rot,
                                 "translation_error_mm": closure_mm},
              "board": config, "stereo_calibration": str(args.stereo.resolve()),
              "warning": "Single-face orientation can remain noisy even with stereo; compare return image corner shift as well as pose errors before fusion."}
    output = args.output or default_output_path(args.images)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Axis RMS {result['point_rms_mm']:.3f} mm -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
