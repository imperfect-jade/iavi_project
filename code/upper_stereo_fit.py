"""Upper-arm stereo estimator used by calibrate_stereo.py --pose upper."""
from __future__ import annotations

import hashlib
import json

import cv2
import numpy as np

from calibrate_stereo import board_points, detect


def _fit(records, lower, object_points):
    rms, _, _, _, _, rotation, translation, _, _ = cv2.stereoCalibrate(
        [object_points for _ in records],
        [record[1] for record in records], [record[2] for record in records],
        lower["K_left"], lower["D_left"], lower["K_right"], lower["D_right"],
        tuple(int(v) for v in lower["image_size"]), flags=cv2.CALIB_FIX_INTRINSIC,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-10))
    if not np.isfinite(rms) or not np.isfinite(rotation).all() or \
            not np.isfinite(translation).all():
        raise ValueError("Upper stereo fit produced nonfinite geometry")
    return float(rms), rotation, translation


def _errors(records, lower, rotation, translation):
    r1, r2, p1, p2, *_ = cv2.stereoRectify(
        lower["K_left"], lower["D_left"], lower["K_right"], lower["D_right"],
        tuple(int(v) for v in lower["image_size"]), rotation, translation,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    result = []
    for name, left, right, _ in records:
        ly = cv2.undistortPoints(left, lower["K_left"], lower["D_left"],
                                  R=r1, P=p1)[:, 0, 1]
        ry = cv2.undistortPoints(right, lower["K_right"], lower["D_right"],
                                  R=r2, P=p2)[:, 0, 1]
        dy = np.abs(ly - ry)
        result.append({"image": name, "median_abs_px": float(np.median(dy)),
                       "p95_abs_px": float(np.percentile(dy, 95))})
    return result


def _check(errors, label, max_median_px, max_p95_px):
    bad = [row for row in errors if row["median_abs_px"] > max_median_px or
           row["p95_abs_px"] > max_p95_px]
    if bad:
        worst = max(bad, key=lambda row: row["p95_abs_px"])
        raise ValueError(f"Upper {label} epipolar check failed at {worst['image']}: "
                         f"median {worst['median_abs_px']:.2f} px, "
                         f"p95 {worst['p95_abs_px']:.2f} px")


def fit_upper_pose(records, lower, square_mm, validation_records,
                   max_median_px=2.0, max_p95_px=3.0, square_y_mm=None):
    """Fit upper extrinsics and check held-out plus independent pairs."""
    if len(records) < 8 or len({record[0] for record in records}) != len(records):
        raise ValueError("Need at least 8 distinct upper checkerboard pairs")
    if not validation_records:
        raise ValueError("Need at least one independent upper validation pair")
    object_points = board_points(square_mm, square_y_mm)
    train = [record for i, record in enumerate(records) if i % 4 != 3]
    held_out = [record for i, record in enumerate(records) if i % 4 == 3]
    if len(held_out) < 2:
        raise ValueError("Need at least 2 held-out upper checkerboard pairs")
    _, train_rotation, train_translation = _fit(train, lower, object_points)
    holdout_errors = _errors(held_out, lower, train_rotation, train_translation)
    _check(holdout_errors, "held-out", max_median_px, max_p95_px)
    rms, rotation, fitted_translation = _fit(records, lower, object_points)
    original_baseline = float(np.linalg.norm(lower["t_right_from_left_mm"]))
    fitted_baseline = float(np.linalg.norm(fitted_translation))
    if fitted_baseline <= 0 or abs(fitted_baseline/original_baseline - 1) > .05:
        raise ValueError(f"Upper fitted baseline {fitted_baseline:.2f} mm differs "
                         f"by over 5% from lower {original_baseline:.2f} mm")
    translation = fitted_translation * (original_baseline / fitted_baseline)
    final_errors = _errors(records, lower, rotation, translation)
    _check(final_errors, "final", max_median_px, max_p95_px)
    validation_errors = _errors(validation_records, lower, rotation, translation)
    _check(validation_errors, "independent validation", max_median_px, max_p95_px)
    corrected = {key: np.asarray(value).copy() for key, value in lower.items()}
    corrected.update({"R_right_from_left": rotation,
                      "t_right_from_left_mm": translation})
    r1, r2, p1, p2, q, roi1, roi2 = cv2.stereoRectify(
        lower["K_left"], lower["D_left"], lower["K_right"], lower["D_right"],
        tuple(int(v) for v in lower["image_size"]), rotation, translation,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    tx = np.array([[0., -translation[2, 0], translation[1, 0]],
                   [translation[2, 0], 0., -translation[0, 0]],
                   [-translation[1, 0], translation[0, 0], 0.]])
    essential = tx @ rotation
    corrected.update({"rectify_left": r1, "rectify_right": r2,
                      "P_left": p1, "P_right": p2, "Q": q,
                      "valid_roi_left": np.asarray(roi1),
                      "valid_roi_right": np.asarray(roi2),
                      "essential": essential,
                      "fundamental": np.linalg.inv(lower["K_right"]).T @
                                     essential @ np.linalg.inv(lower["K_left"])})
    rotation_delta = float(np.degrees(np.linalg.norm(cv2.Rodrigues(
        rotation @ lower["R_right_from_left"].T)[0])))
    report = {"method": "upper_fixed_intrinsics_pose_stereo",
              "upper_pair_count": len(records), "heldout_pair_count": len(held_out),
              "heldout_per_view": holdout_errors, "final_per_view": final_errors,
              "validation_pair_count": len(validation_records),
              "validation_per_view": validation_errors,
              "validation_max_median_px": max(row["median_abs_px"] for row in validation_errors),
              "validation_max_p95_px": max(row["p95_abs_px"] for row in validation_errors),
              "holdout_max_median_px": max(row["median_abs_px"] for row in holdout_errors),
              "holdout_max_p95_px": max(row["p95_abs_px"] for row in holdout_errors),
              "fit_rms_px": rms, "lower_baseline_mm": original_baseline,
              "fitted_baseline_mm": fitted_baseline,
              "used_baseline_mm": float(np.linalg.norm(translation)),
              "rotation_delta_from_lower_deg": rotation_delta,
              "square_mm": square_mm}
    return corrected, report


def calibrate_upper(args):
    """Fit and publish an upper-pose model using the lower model's intrinsics."""
    lower_path = args.lower_calibration.with_suffix(".npz").resolve()
    lower_meta = json.loads(lower_path.with_suffix(".json").read_text(encoding="utf-8"))
    with np.load(lower_path) as archive:
        lower = {key: archive[key].copy() for key in archive.files}
    records, rejected, size = detect(args.train)
    if rejected or tuple(int(v) for v in lower["image_size"]) != size:
        raise ValueError(f"Upper checkerboard images unusable or camera size differs: {rejected}")
    validation, rejected_validation, _ = detect(args.validation, size)
    if not validation:
        raise ValueError("Need at least one usable independent upper validation pair")
    train_hashes = {hashlib.sha256(path.read_bytes()).hexdigest()
                    for side in ("left", "right")
                    for path in (args.train / side).glob("calib_*.png")}
    validation_hashes = {hashlib.sha256(path.read_bytes()).hexdigest()
                         for side in ("left", "right")
                         for path in (args.validation / side).glob("calib_*.png")}
    if train_hashes & validation_hashes:
        raise ValueError("Upper independent validation contains copied training images")
    corrected, report = fit_upper_pose(records, lower, args.square_mm, validation,
                                      square_y_mm=args.square_y_mm)
    report.update({"left_serial": lower_meta["left_serial"],
                   "right_serial": lower_meta["right_serial"],
                   "image_size": list(size),
                   "lower_calibration": str(lower_path),
                   "lower_sha256": hashlib.sha256(lower_path.read_bytes()).hexdigest(),
                   "upper_images": str(args.train.resolve()),
                   "validation_dir": str(args.validation.resolve()),
                   "rejected_validation": rejected_validation})
    args.output.mkdir(parents=True, exist_ok=True)
    npz_path = args.output / "stereo_calibration.npz"
    with (args.output / "stereo_calibration.npz.tmp").open("wb") as stream:
        np.savez_compressed(stream, **corrected)
    (args.output / "stereo_calibration.npz.tmp").replace(npz_path)
    metadata_path = npz_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Upper stereo: {len(records)} train / {len(validation)} independent validation pairs, "
          f"held-out max median "
          f"{report['holdout_max_median_px']:.3f} px / p95 "
          f"{report['holdout_max_p95_px']:.3f} px; independent max median "
          f"{report['validation_max_median_px']:.3f} px / p95 "
          f"{report['validation_max_p95_px']:.3f} px, rotation change "
          f"{report['rotation_delta_from_lower_deg']:.3f} deg -> {npz_path.resolve()}")
    return 0
