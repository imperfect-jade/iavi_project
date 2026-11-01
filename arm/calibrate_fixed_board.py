"""Estimate fixed-board camera poses from capture_fixed_board.py images."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

from fixed_board import (board_from_config, detect_corners_with_contrast_fallback, estimate_pose,
                         relative_transform, transform_delta)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BOARD = ROOT / "arm/targets/charuco_7x5_10mm_7mm_config.json"
DEFAULT_STEREO = ROOT / "calibration/stereo/stereo_calibration.json"


def output_path(images):
    return ROOT / "calibration/arm_fixed_board" / images.name / "two_pose_calibration.json"


def read_image(path):
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def calibrate(images, stereo_path=DEFAULT_STEREO, board_path=DEFAULT_BOARD,
              min_corners=12, max_rms_px=3.0):
    manifest = json.loads((images / "capture_manifest.json").read_text(encoding="utf-8"))
    if not manifest.get("complete"):
        raise ValueError("Capture manifest is incomplete")
    config = json.loads(board_path.read_text(encoding="utf-8"))
    board_keys = ("board_type", "dictionary", "squares_x", "squares_y",
                  "square_length_mm", "marker_length_mm", "opencv_legacy_pattern")
    if any(manifest["board"].get(k) != config.get(k) for k in board_keys):
        raise ValueError("Capture board differs from the supplied board config")
    board = board_from_config(config)
    stereo = json.loads(stereo_path.read_text(encoding="utf-8"))
    for key in ("left_serial", "right_serial"):
        if manifest[key] != stereo[key]:
            raise ValueError(f"{key} differs from stereo calibration")
    if list(manifest["camera_size_px"]) != list(stereo["image_size"]):
        raise ValueError("Image size differs from stereo calibration")
    if manifest.get("camera_frame") != "original_left_camera":
        raise ValueError("Expected original, unrectified left-camera images")
    K = np.asarray(stereo["K_left"], np.float64)
    D = np.asarray(stereo["D_left"], np.float64)
    observations = []
    pose_matrices = {}
    for row in manifest["observations"]:
        image_path = images / row["left_image"]
        image = read_image(image_path)
        if [image.shape[1], image.shape[0]] != list(stereo["image_size"]):
            raise ValueError(f"Image size mismatch: {image_path}")
        corners, ids, detection_method = detect_corners_with_contrast_fallback(
            image, board, min_corners)
        if len(ids) < min_corners:
            raise ValueError(f"{row['name']}: only {len(ids)} ChArUco corners; need {min_corners}")
        T, rms = estimate_pose(board, corners, ids, K, D)
        if rms > max_rms_px:
            raise ValueError(f"{row['name']}: reprojection RMS {rms:.2f} px exceeds {max_rms_px:.2f} px")
        observations.append({"name": row["name"], "pose": row["pose"],
                             "left_image": row["left_image"], "corners": len(ids),
                             "detection_method": detection_method,
                             "reprojection_rms_px": rms, "T_left_from_board": T.tolist()})
        pose_matrices[row["name"]] = T
    if len(observations) < 2 or not {"upper", "lower"}.issubset({r["pose"] for r in observations}):
        raise ValueError("Need images from both arm poses")
    reference = next((r for r in observations if r["pose"] == "upper"), None)
    if reference is None:
        raise ValueError("Missing upper reference pose")
    T_ref = pose_matrices[reference["name"]]
    first_by_pose = {}
    for row in observations:
        name, pose = row["name"], row["pose"]
        T = pose_matrices[name]
        row["T_reference_from_left"] = relative_transform(T_ref, T).tolist()
        if pose not in first_by_pose:
            first_by_pose[pose] = name
        translation, rotation = transform_delta(T, pose_matrices[first_by_pose[pose]])
        row["return_error_mm_from_first_same_pose"] = translation
        row["return_error_deg_from_first_same_pose"] = rotation
    summary = {}
    for pose in ("upper", "lower"):
        rows = [r for r in observations if r["pose"] == pose]
        summary[pose] = {"count": len(rows),
                         "first_capture": first_by_pose[pose],
                         "max_return_translation_mm": max(r["return_error_mm_from_first_same_pose"] for r in rows),
                         "max_return_rotation_deg": max(r["return_error_deg_from_first_same_pose"] for r in rows),
                         "max_reprojection_rms_px": max(r["reprojection_rms_px"] for r in rows)}
    return {"coordinate_unit": "mm", "board_frame": "fixed_ChArUco_board",
            "camera_frame": "original_left_camera", "reference_capture": reference["name"],
            "transform_convention": "T_reference_from_left = T_reference_from_board @ inverse(T_left_from_board)",
            "board": config, "stereo_calibration": str(stereo_path.resolve()),
            "source_capture_manifest": str((images / "capture_manifest.json").resolve()),
            "summary": summary, "observations": observations,
            "axis_note": "Existing turntable axis may belong to another left-camera frame; recalibrate axis at upper pose or measure the frame transform before fusion.",
            "fusion_note": "These matrices are keyed by board capture names, not scan view_###/repeat_### names. Localize every scan arrival or verify repeatability before building fuse_views --camera-poses."}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images", type=Path, required=True, help="Capture session folder")
    p.add_argument("--stereo", type=Path, default=DEFAULT_STEREO)
    p.add_argument("--board", type=Path, default=DEFAULT_BOARD)
    p.add_argument("--output", type=Path, help="Default: calibration/arm_fixed_board/<session>/two_pose_calibration.json")
    p.add_argument("--min-corners", type=int, default=12)
    p.add_argument("--max-rms-px", type=float, default=3.0)
    args = p.parse_args(argv)
    if not 4 <= args.min_corners <= 24 or args.max_rms_px <= 0:
        p.error("min-corners must be 4..24 and max-rms-px positive")
    try:
        result = calibrate(args.images, args.stereo, args.board, args.min_corners, args.max_rms_px)
        output = args.output or output_path(args.images)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Saved: {output}")
        for pose, values in result["summary"].items():
            print(f"{pose}: n={values['count']}, max return {values['max_return_translation_mm']:.3f} mm / "
                  f"{values['max_return_rotation_deg']:.3f} deg, max reprojection "
                  f"{values['max_reprojection_rms_px']:.3f} px")
        return 0
    except (OSError, ValueError, cv2.error) as exc:
        print(f"Calibration failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
