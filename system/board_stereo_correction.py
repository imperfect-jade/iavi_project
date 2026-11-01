"""Correct session stereo extrinsics from paired fixed ChArUco localization images.

The original camera intrinsics and baseline length remain the metric reference.
Only the cameras' relative rotation and baseline direction are fitted here.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "arm"))
from fixed_board import board_from_config, detect_corners_with_contrast_fallback  # noqa: E402


def _rectify(model, R, t):
    return cv2.stereoRectify(
        model["K_left"], model["D_left"], model["K_right"], model["D_right"],
        tuple(int(v) for v in model["image_size"]), R, t,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)


def _errors(rows, model, R, t):
    R1, R2, P1, P2, *_ = _rectify(model, R, t)
    result = []
    for row in rows:
        left = cv2.undistortPoints(row["left"], model["K_left"], model["D_left"],
                                   R=R1, P=P1)[:, 0, 1]
        right = cv2.undistortPoints(row["right"], model["K_right"], model["D_right"],
                                    R=R2, P=P2)[:, 0, 1]
        residual = np.abs(left - right)
        result.append({"view": row["name"], "pose_id": int(row["pose_id"]),
                       "corners": len(residual),
                       "median_abs_px": float(np.median(residual)),
                       "p95_abs_px": float(np.percentile(residual, 95))})
    return result


def _passes(rows, max_median_px, max_p95_px):
    return (all(row["median_abs_px"] <= max_median_px for row in rows) and
            all(row["p95_abs_px"] <= max_p95_px for row in rows))


def _fit(rows, model):
    rms, _, _, _, _, R, t, _, _ = cv2.stereoCalibrate(
        [row["object"] for row in rows],
        [row["left"] for row in rows], [row["right"] for row in rows],
        model["K_left"], model["D_left"], model["K_right"], model["D_right"],
        tuple(int(v) for v in model["image_size"]),
        flags=cv2.CALIB_FIX_INTRINSIC,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-10))
    if not np.isfinite(rms) or not np.isfinite(R).all() or not np.isfinite(t).all():
        raise ValueError("Stereo correction fit produced nonfinite parameters")
    return float(rms), R, t


def _split(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[int(row["pose_id"])].append(row)
    held_out = {row["name"] for group in grouped.values()
                for index, row in enumerate(group) if index % 4 == 3}
    train = [row for row in rows if row["name"] not in held_out]
    validation = [row for row in rows if row["name"] in held_out]
    if len(train) < 6 or len(validation) < 2:
        raise ValueError("Need at least 6 training and 2 held-out board pairs")
    return train, validation


def estimate_session_geometry(rows, original, max_median_px=1.0,
                              max_p95_px=2.0):
    """Return a validated session model and report from matched board corners.

    Each row contains name, pose_id, matching object/left/right point arrays.
    A fourth of each arm-pose group is held out before the final all-view fit.
    """
    if not rows:
        raise ValueError("No paired fixed-board views")
    if len({row["name"] for row in rows}) != len(rows):
        raise ValueError("Duplicate fixed-board view names")
    for row in rows:
        count = len(row["object"])
        if (count < 12 or len(row["left"]) != count or
                len(row["right"]) != count or
                not all(np.isfinite(row[key]).all() for key in ("object", "left", "right"))):
            raise ValueError(f"Invalid paired fixed-board corners: {row['name']}")
    original_R = np.asarray(original["R_right_from_left"], np.float64)
    original_t = np.asarray(original["t_right_from_left_mm"], np.float64).reshape(3, 1)
    old_errors = _errors(rows, original, original_R, original_t)
    report = {"source_view_count": len(rows), "original_per_view": old_errors,
              "original_median_of_view_medians_px": float(np.median(
                  [row["median_abs_px"] for row in old_errors])),
              "limits_px": {"max_view_median": max_median_px,
                            "max_view_p95": max_p95_px}}
    if _passes(old_errors, max_median_px, max_p95_px):
        report.update({"status": "unchanged", "holdout_views": [],
                       "corrected_per_view": old_errors})
        return original, report

    if len(rows) < 8:
        raise ValueError("Need at least 8 paired fixed-board views to correct invalid stereo geometry")

    train, validation = _split(rows)
    _, train_R, train_t = _fit(train, original)
    held_out = _errors(validation, original, train_R, train_t)
    report.update({"holdout_views": [row["view"] for row in held_out],
                   "holdout_per_view": held_out,
                   "holdout_max_median_abs_px": max(row["median_abs_px"] for row in held_out),
                   "holdout_max_p95_abs_px": max(row["p95_abs_px"] for row in held_out)})
    if not _passes(held_out, max_median_px, max_p95_px):
        raise ValueError("Fixed-board held-out epipolar check failed: "
                         f"max median {report['holdout_max_median_abs_px']:.2f} px, "
                         f"max p95 {report['holdout_max_p95_abs_px']:.2f} px")

    rms, R, fitted_t = _fit(rows, original)
    original_baseline = float(np.linalg.norm(original_t))
    fitted_baseline = float(np.linalg.norm(fitted_t))
    if fitted_baseline <= 0 or abs(fitted_baseline / original_baseline - 1) > .05:
        raise ValueError(f"Fixed-board fitted baseline {fitted_baseline:.2f} mm "
                         f"differs by over 5% from original {original_baseline:.2f} mm")
    t = fitted_t * (original_baseline / fitted_baseline)
    corrected = {key: np.asarray(value).copy() for key, value in original.items()}
    corrected.update({"R_right_from_left": R, "t_right_from_left_mm": t})
    R1, R2, P1, P2, Q, roi1, roi2 = _rectify(corrected, R, t)
    tx = np.array([[0., -t[2, 0], t[1, 0]],
                   [t[2, 0], 0., -t[0, 0]],
                   [-t[1, 0], t[0, 0], 0.]])
    essential = tx @ R
    fundamental = np.linalg.inv(corrected["K_right"]).T @ essential @ np.linalg.inv(corrected["K_left"])
    corrected.update({"rectify_left": R1, "rectify_right": R2,
                      "P_left": P1, "P_right": P2, "Q": Q,
                      "valid_roi_left": np.asarray(roi1),
                      "valid_roi_right": np.asarray(roi2),
                      "essential": essential, "fundamental": fundamental})
    final_errors = _errors(rows, corrected, R, t)
    if not _passes(final_errors, max_median_px, max_p95_px):
        worst = max(final_errors, key=lambda row: row["p95_abs_px"])
        raise ValueError(f"Final fixed-board epipolar check failed at {worst['view']}: "
                         f"median {worst['median_abs_px']:.2f} px, "
                         f"p95 {worst['p95_abs_px']:.2f} px")
    report.update({"status": "corrected", "fit_rms_px": rms,
                   "original_baseline_mm": original_baseline,
                   "fitted_baseline_mm": fitted_baseline,
                   "used_baseline_mm": float(np.linalg.norm(t)),
                   "corrected_per_view": final_errors,
                   "corrected_median_of_view_medians_px": float(np.median(
                       [row["median_abs_px"] for row in final_errors]))})
    return corrected, report


def _read_image(path):
    image = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read fixed-board image: {path}")
    return image


def collect_observations(sources, board_config, original, camera_serials,
                         min_common_corners=12):
    """Match ChArUco IDs in the two localization images of each complete view."""
    board = board_from_config(board_config)
    object_corners = board.getChessboardCorners()
    rows, rejected = [], {}
    for source in sources:
        manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
        name = f"{source.parent.name}/{source.name}"
        if (not manifest.get("complete") or
                list(manifest["camera_size_px"]) != list(original["image_size"]) or
                (manifest.get("left_serial"), manifest.get("right_serial")) != camera_serials):
            raise ValueError(f"Incomplete capture or camera identity/size differs: {name}")
        detections = []
        for side in ("left", "right"):
            image = _read_image(source / "localization" / f"{side}.png")
            corners, ids, _ = detect_corners_with_contrast_fallback(image, board,
                                                                     min_common_corners)
            detections.append({int(i): xy.reshape(2) for i, xy in zip(ids.reshape(-1), corners)})
        common = sorted(detections[0].keys() & detections[1].keys())
        if len(common) < min_common_corners:
            rejected[name] = f"only {len(common)} common ChArUco corners"
            continue
        rows.append({"name": name, "pose_id": int(manifest["arm_pose_id"]),
                     "object": np.asarray(object_corners[common], np.float32).reshape(-1, 1, 3),
                     "left": np.asarray([detections[0][i] for i in common], np.float32).reshape(-1, 1, 2),
                     "right": np.asarray([detections[1][i] for i in common], np.float32).reshape(-1, 1, 2)})
    if len(rejected) > max(1, len(sources) // 5):
        raise ValueError(f"Too many unusable paired board views: {rejected}")
    return rows, rejected


def derive_session_calibration(sources, base_calibration, board_path, output_dir):
    """Return the original NPZ or a validated, cached session-specific NPZ."""
    sources = sorted(Path(source).resolve() for source in sources)
    base_calibration = Path(base_calibration).with_suffix(".npz").resolve()
    board_path, output_dir = Path(board_path).resolve(), Path(output_dir).resolve()
    if not sources:
        raise ValueError("No complete scan views for fixed-board stereo correction")
    base_metadata = json.loads(base_calibration.with_suffix(".json").read_text(encoding="utf-8"))
    with np.load(base_calibration) as archive:
        original = {key: archive[key].copy() for key in archive.files}
    board_config = json.loads(board_path.read_text(encoding="utf-8"))
    plan_path = sources[0].parent.parent / "scan_plan.json"
    if plan_path.is_file():
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if plan.get("board") != board_config or not plan.get("stereo") or \
                Path(plan["stereo"]).resolve() != \
                base_calibration.with_suffix(".json"):
            raise ValueError("Scan plan board or base stereo calibration differs")
    digest = hashlib.sha256()
    digest.update(Path(__file__).read_bytes())
    for path in (base_calibration, base_calibration.with_suffix(".json"), board_path):
        digest.update(path.read_bytes())
    for source in sources:
        digest.update(str(source).encode("utf-8"))
        digest.update((source / "manifest.json").read_bytes())
        for side in ("left", "right"):
            digest.update((source / "localization" / f"{side}.png").read_bytes())
    input_hash = digest.hexdigest()
    report_path = output_dir / "board_stereo_report.json"
    corrected_npz = output_dir / "stereo_calibration.npz"
    if report_path.is_file():
        cached = json.loads(report_path.read_text(encoding="utf-8"))
        if cached.get("input_sha256") == input_hash:
            if cached["status"] == "unchanged":
                print("Fixed-board stereo: cached original calibration is valid", flush=True)
                return base_calibration
            if cached["status"] == "corrected" and corrected_npz.is_file() and \
               corrected_npz.with_suffix(".json").is_file():
                print(f"Fixed-board stereo: using cached session correction {corrected_npz}", flush=True)
                return corrected_npz
    rows, rejected = collect_observations(
        sources, board_config, original,
        (base_metadata["left_serial"], base_metadata["right_serial"]))
    corrected, report = estimate_session_geometry(rows, original)
    report.update({"input_sha256": input_hash, "source_calibration": str(base_calibration),
                   "board_config": str(board_path), "rejected_views": rejected,
                   "selected_views": [str(source) for source in sources]})
    output_dir.mkdir(parents=True, exist_ok=True)
    if report["status"] == "corrected":
        metadata = {"method": "session_fixed_board_stereo_correction",
                    "source_calibration": str(base_calibration),
                    "board_config": board_config, "input_sha256": input_hash,
                    "left_serial": base_metadata["left_serial"],
                    "right_serial": base_metadata["right_serial"],
                    "image_size": [int(v) for v in corrected["image_size"]],
                    "original_baseline_mm": report["original_baseline_mm"],
                    "fitted_baseline_mm": report["fitted_baseline_mm"],
                    "baseline_mm": report["used_baseline_mm"],
                    "validation": {"holdout_views": report["holdout_views"],
                                   "max_median_abs_px": report["holdout_max_median_abs_px"],
                                   "max_p95_abs_px": report["holdout_max_p95_abs_px"]}}
        for key in ("K_left", "D_left", "K_right", "D_right", "R_right_from_left",
                    "t_right_from_left_mm", "rectify_left", "rectify_right",
                    "P_left", "P_right", "Q"):
            metadata[key] = corrected[key].tolist()
        tmp_npz = output_dir / "stereo_calibration.npz.tmp"
        with tmp_npz.open("wb") as stream:
            np.savez_compressed(stream, **corrected)
        tmp_npz.replace(corrected_npz)
        tmp_json = output_dir / "stereo_calibration.json.tmp"
        tmp_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        tmp_json.replace(corrected_npz.with_suffix(".json"))
    tmp_report = output_dir / "board_stereo_report.json.tmp"
    tmp_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    tmp_report.replace(report_path)
    print(f"Fixed-board stereo: {report['status']}; median epipolar "
          f"{report['original_median_of_view_medians_px']:.2f} -> "
          f"{report.get('corrected_median_of_view_medians_px', report['original_median_of_view_medians_px']):.2f} px; "
          f"{len(rows)} board pairs, {len(rejected)} rejected", flush=True)
    return corrected_npz if report["status"] == "corrected" else base_calibration


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-root", type=Path, required=True)
    parser.add_argument("--stereo", type=Path,
                        default=ROOT / "calibration/stereo/stereo_calibration.npz")
    parser.add_argument("--board", type=Path,
                        default=ROOT / "arm/targets/charuco_7x5_10mm_7mm_config.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    sources = []
    for view in sorted(args.session_root.glob("view_*")):
        candidates = [folder for folder in sorted(view.glob("repeat_*"))
                      if (folder / "manifest.json").is_file() and
                      json.loads((folder / "manifest.json").read_text(encoding="utf-8")).get("complete")]
        if candidates:
            sources.append(candidates[-1])
    output = args.output or ROOT / "processed" / args.session_root.parent.name / \
        args.session_root.name / "board_stereo_correction"
    result = derive_session_calibration(sources, args.stereo, args.board, output)
    print(f"Calibration for reconstruction: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
