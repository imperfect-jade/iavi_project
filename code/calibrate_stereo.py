"""Calibrate and independently validate the rigid Basler stereo pair."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BOARD = (10, 6)


def pairs(folder):
    folder = Path(folder)
    left = {p.name: p for p in (folder / "left").glob("calib_*.png")}
    right = {p.name: p for p in (folder / "right").glob("calib_*.png")}
    for name in sorted(left.keys() ^ right.keys()):
        print(f"[UNUSABLE] {name}: missing {'left' if name not in left else 'right'} image", flush=True)
    if not left or left.keys() != right.keys():
        raise ValueError(f"Missing or unmatched calibration pairs in {folder}")
    return [(left[n], right[n]) for n in sorted(left)]


def read(path):
    image = cv2.imdecode(np.frombuffer(Path(path).read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unreadable image: {path}")
    return image


def board_points(square_mm, square_y_mm=None):
    square_y_mm = square_mm if square_y_mm is None else square_y_mm
    if square_mm <= 0 or square_y_mm <= 0:
        raise ValueError("Square sizes must be positive")
    xy = np.mgrid[0:BOARD[0], 0:BOARD[1]].T.reshape(-1, 2)
    return np.column_stack((xy * [square_mm, square_y_mm], np.zeros(len(xy)))).astype(np.float32)


def calibration_flags(model):
    if model == "radial2":
        return cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3
    if model == "standard5":
        return 0
    raise ValueError(f"Unsupported distortion model: {model}")


def find_corners(image):
    flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    found, corners = cv2.findChessboardCornersSB(
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), BOARD, flags=flags)
    return corners.astype(np.float32) if found else None


def board_sharpness(image, corners):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    xy = corners.reshape(-1, 2)
    lower = np.maximum(np.floor(xy.min(0)).astype(int), 0)
    upper = np.minimum(np.ceil(xy.max(0)).astype(int) + 1, [gray.shape[1], gray.shape[0]])
    return float(cv2.Laplacian(gray[lower[1]:upper[1], lower[0]:upper[0]], cv2.CV_64F).var())


def view_geometry(corners, size, obj, K, D):
    xy = corners.reshape(-1, 2)
    ok, rvec, tvec = cv2.solvePnP(obj, corners, K, D)
    if not ok:
        raise RuntimeError("solvePnP failed while measuring board geometry")
    rotation, _ = cv2.Rodrigues(rvec)
    tilt = np.degrees(np.arccos(np.clip(abs(rotation[2, 2]), 0, 1)))
    center = rotation @ obj.mean(0) + tvec.reshape(3)
    extent = (xy.max(0) - xy.min(0)) / size
    return {"tilt_deg": float(tilt), "board_center_z_mm": float(center[2]),
            "center_x_fraction": float(xy[:, 0].mean() / size[0]),
            "center_y_fraction": float(xy[:, 1].mean() / size[1]),
            "area_fraction": float(np.prod(extent))}


def save_image(path, image):
    ok, encoded = cv2.imencode(Path(path).suffix, image)
    if not ok:
        raise RuntimeError(f"Could not encode image: {path}")
    Path(path).write_bytes(encoded.tobytes())


def corner_coverage(records, size):
    result = {}
    for side, column in (("left", 1), ("right", 2)):
        xy = np.concatenate([record[column].reshape(-1, 2) for record in records]) / size
        result[side] = {"minimum_fraction": xy.min(0).tolist(),
                        "maximum_fraction": xy.max(0).tolist()}
    return result


def write_coverage_image(records, size, path):
    scale = min(1.0, 1200 / size[0])
    width, height = [int(x * scale) for x in size]
    panels = []
    for side, column, color in (("left", 1, (50, 170, 50)),
                                ("right", 2, (220, 120, 40))):
        canvas = np.full((height, width, 3), 245, np.uint8)
        for record in records:
            for point in np.rint(record[column].reshape(-1, 2) * scale).astype(int):
                cv2.circle(canvas, tuple(point), 2, color, -1, cv2.LINE_AA)
        cv2.putText(canvas, side.upper(), (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (20, 20, 20), 2)
        panels.append(canvas)
    save_image(path, np.hstack(panels))


def write_rectified_examples(records, rows, model, size, output):
    maps = []
    for side in ("left", "right"):
        maps.append(cv2.initUndistortRectifyMap(
            model[f"K_{side}"], model[f"D_{side}"], model[f"rectify_{side}"],
            model[f"P_{side}"], size, cv2.CV_32FC1))
    order = sorted(range(len(rows)), key=lambda i: rows[i]["rectified_vertical_rms_px"])
    for index in sorted({0, order[len(order) // 2], order[-1]}):
        record = records[index]
        left = cv2.remap(read(record[3]["left_path"]), *maps[0], cv2.INTER_LINEAR)
        right = cv2.remap(read(record[3]["right_path"]), *maps[1], cv2.INTER_LINEAR)
        combined = np.hstack((left, right))
        for y in range(50, combined.shape[0], 100):
            cv2.line(combined, (0, y), (combined.shape[1] - 1, y),
                     (0, 255, 255), 2, cv2.LINE_AA)
        cv2.line(combined, (size[0], 0), (size[0], size[1]), (255, 255, 255), 3)
        save_image(output / f"{Path(record[0]).stem}_rectified_epilines.jpg", combined)


def detect(folder, size=None):
    records, rejected = [], []
    for lp, rp in pairs(folder):
        try:
            images = [read(lp), read(rp)]
        except (OSError, ValueError) as exc:
            reason = str(exc)
            rejected.append({"image": lp.name, "reason": reason})
            print(f"[UNUSABLE] {lp.name}: {reason}", flush=True)
            continue
        current = (images[0].shape[1], images[0].shape[0])
        if images[1].shape[:2] != images[0].shape[:2] or (size and size != current):
            reason = f"image size mismatch (left={current}, right={(images[1].shape[1], images[1].shape[0])}, expected={size})"
            rejected.append({"image": lp.name, "reason": reason})
            print(f"[UNUSABLE] {lp.name}: {reason}", flush=True)
            continue
        size = current
        found = [find_corners(im) for im in images]
        if any(corners is None for corners in found):
            reason = f"checkerboard missing (left={found[0] is not None}, right={found[1] is not None})"
            rejected.append({"image": lp.name, "reason": reason})
            print(f"[UNUSABLE] {lp.name}: {reason}", flush=True)
            continue
        lc, rc = found
        # A symmetric checkerboard can have opposite 180-degree indexing.
        a = lc.reshape(-1, 2) - lc.reshape(-1, 2).mean(0)
        normal = rc.reshape(-1, 2) - rc.reshape(-1, 2).mean(0)
        reversed_ = rc[::-1].reshape(-1, 2) - rc[::-1].reshape(-1, 2).mean(0)
        if np.mean(np.linalg.norm(a-reversed_, axis=1)) < np.mean(np.linalg.norm(a-normal, axis=1)):
            rc = rc[::-1].copy()
        records.append((lp.name, lc, rc, {"left_path": lp, "right_path": rp,
            "left_board_sharpness": board_sharpness(images[0], lc),
            "right_board_sharpness": board_sharpness(images[1], rc)}))
        print(f"[USABLE] {lp.name}: {len(lc)} corners in each image, size={current}", flush=True)
    return records, rejected, size


def reprojection(object_points, corners, K, D, rvec, tvec):
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, K, D)
    return float(np.sqrt(np.mean(np.sum((projected.reshape(-1, 2)-corners.reshape(-1, 2))**2, axis=1))))


def evaluate(records, model, obj):
    K1, D1, K2, D2, R1, R2, P1, P2 = (model[k] for k in
        ("K_left", "D_left", "K_right", "D_right", "rectify_left", "rectify_right", "P_left", "P_right"))
    rows = []
    for record in records:
        name, lc, rc = record[:3]
        details = record[3] if len(record) > 3 else {}
        ul = cv2.undistortPoints(lc, K1, D1, R=R1, P=P1).reshape(-1, 2)
        ur = cv2.undistortPoints(rc, K2, D2, R=R2, P=P2).reshape(-1, 2)
        vertical = np.abs(ul[:, 1]-ur[:, 1])
        one = {"image": name, "epipolar_median_px": float(np.median(vertical)),
               "epipolar_p95_px": float(np.percentile(vertical, 95)),
               "rectified_vertical_mean_abs_px": float(np.mean(vertical)),
               "rectified_vertical_rms_px": float(np.sqrt(np.mean(vertical**2))),
               "rectified_vertical_max_abs_px": float(np.max(vertical))}
        for label, corners, K, D in (("left", lc, K1, D1), ("right", rc, K2, D2)):
            ok, rvec, tvec = cv2.solvePnP(obj, corners, K, D)
            if not ok:
                raise RuntimeError(f"PnP failed: {name} {label}")
            one[f"{label}_reprojection_rms_px"] = reprojection(obj, corners, K, D, rvec, tvec)
            one.update({f"{label}_{key}": value for key, value in
                        view_geometry(corners, tuple(int(x) for x in model["image_size"]), obj, K, D).items()})
            if f"{label}_board_sharpness" in details:
                one[f"{label}_board_sharpness"] = details[f"{label}_board_sharpness"]
        rows.append(one)
    return rows


def check_validation_error(rows, max_median_px):
    """Reject a calibration before publication when held-out geometry fails."""
    if max_median_px <= 0:
        raise ValueError("Validation threshold must be positive")
    bad = [row for row in rows if row["epipolar_median_px"] > max_median_px or
           row["epipolar_p95_px"] > 1.5 * max_median_px]
    if bad:
        names = ", ".join(row["image"] for row in bad)
        raise ValueError(f"Independent validation failed for {len(bad)}/{len(rows)} pairs "
                         f"({names}); check board motion and stereo rig stability. "
                         "No calibration files were published.")


def select_outliers(rows):
    """Flag conspicuous training outliers from the initial fit."""
    flagged = {}
    for key, label, floor in (("epipolar_median_px", "epipolar", .75),
                              ("left_reprojection_rms_px", "left RMS", .8),
                              ("right_reprojection_rms_px", "right RMS", .8)):
        values = np.array([row[key] for row in rows], float)
        center = np.median(values)
        cutoff = max(floor, center + 3.5 * 1.4826 * np.median(np.abs(values-center)))
        for row in rows:
            if row[key] > cutoff:
                flagged.setdefault(row["image"], []).append(f"{label}={row[key]:.3f}px > {cutoff:.3f}px")
    return {name: "; ".join(reasons) for name, reasons in flagged.items()}


def choose_training_exclusions(records, suggestions, manual_numbers, auto_enabled=True):
    """Exclude flagged training pairs, with a cap against removing a whole pose group."""
    names = {record[0] for record in records}
    numbered = {int(name.removeprefix("calib_").removesuffix(".png")): name for name in names}
    unknown = manual_numbers - numbered.keys()
    if unknown:
        raise ValueError(f"--exclude contains unknown training numbers: {sorted(unknown)}")
    if suggestions.keys() - names:
        raise ValueError("Outlier suggestions contain unknown training pairs")
    manual_names = {numbered[number] for number in manual_numbers}
    automatic = ({name: reason for name, reason in suggestions.items()
                  if name not in manual_names} if auto_enabled else {})
    max_automatic = max(1, len(records) // 5)
    if len(automatic) > max_automatic:
        raise ValueError(f"Too many automatic exclusions ({len(automatic)}/{len(records)}): "
                         f"{', '.join(automatic)}. Review the images or use --no-auto-exclude.")
    excluded_names = manual_names | automatic.keys()
    excluded = [record for record in records if record[0] in excluded_names]
    remaining = [record for record in records if record[0] not in excluded_names]
    if len(remaining) < 8:
        raise ValueError("At least eight training pairs must remain after exclusions")
    return excluded, remaining, automatic


def fit_stereo(records, size, obj, distortion_model="standard5", alpha=0.0):
    objects = [obj.copy() for _ in records]
    lc, rc = [[record[i] for record in records] for i in (1, 2)]
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-9)
    flags = calibration_flags(distortion_model)
    l_rms, K1, D1, _, _ = cv2.calibrateCamera(objects, lc, size, None, None,
                                                flags=flags, criteria=criteria)
    r_rms, K2, D2, _, _ = cv2.calibrateCamera(objects, rc, size, None, None,
                                                flags=flags, criteria=criteria)
    stereo_rms, K1, D1, K2, D2, R, t, E, F = cv2.stereoCalibrate(
        objects, lc, rc, K1, D1, K2, D2, size,
        criteria=criteria, flags=cv2.CALIB_FIX_INTRINSIC)
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K1, D1, K2, D2, size, R, t, flags=cv2.CALIB_ZERO_DISPARITY, alpha=alpha)
    model = {"image_size": np.array(size), "K_left": K1, "D_left": D1,
             "K_right": K2, "D_right": D2, "R_right_from_left": R,
             "t_right_from_left_mm": t, "rectify_left": R1, "rectify_right": R2,
             "P_left": P1, "P_right": P2, "Q": Q,
             "essential": E, "fundamental": F,
             "valid_roi_left": np.array(roi1), "valid_roi_right": np.array(roi2)}
    return model, (l_rms, r_rms, stereo_rms)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", type=Path, default=ROOT / "raw/calibration/session_01")
    p.add_argument("--validation", type=Path, required=True,
                   help="independent board pairs; must be a separate directory")
    p.add_argument("--output", type=Path, default=ROOT / "calibration/stereo")
    p.add_argument("--square-mm", type=float, required=True,
                   help="measured horizontal square size in millimetres; required for correct 3D scale")
    p.add_argument("--square-y-mm", type=float, help="measured vertical square size; defaults to --square-mm")
    p.add_argument("--distortion-model", choices=("radial2", "standard5"), default="standard5",
                   help="lens distortion model; standard5 estimates k1, k2, p1, p2, k3 (default)")
    p.add_argument("--alpha", type=float, default=0.0,
                   help="Lab5 stereoRectify crop parameter: 0=crop, 1=retain all")
    p.add_argument("--left-serial", default="40031431")
    p.add_argument("--right-serial", default="40031435")
    p.add_argument("--max-validation-epipolar-px", type=float, default=2.0,
                   help="maximum held-out per-image median |dy|; p95 must be <= 1.5 times this")
    p.add_argument("--exclude", default="", help="comma-separated TRAIN pair numbers, e.g. 3,17; applied after all-pair report")
    p.add_argument("--no-auto-exclude", action="store_false", dest="auto_exclude",
                   help="report suspicious training pairs without automatically excluding them")
    return p


def main(argv=None):
    p = build_parser()
    args = p.parse_args(argv)
    if args.train.resolve() == args.validation.resolve():
        p.error("Train and validation directories must differ")
    if not 0 <= args.alpha <= 1:
        p.error("--alpha must be between 0 and 1")
    obj = board_points(args.square_mm, args.square_y_mm)
    train, rejected_train, size = detect(args.train)
    print(f"Training images: usable={len(train)}, unusable={len(rejected_train)}", flush=True)
    print(f"Checking independent validation: {args.validation}", flush=True)
    valid, rejected_valid, _ = detect(args.validation, size)
    print(f"Validation images: usable={len(valid)}, unusable={len(rejected_valid)}", flush=True)
    if len(train) < 8 or len(valid) < 1:
        raise ValueError(f"Need at least 8 train and 1 independent validation pairs; got {len(train)}, {len(valid)}")
    print("Initial fit: all detected training pairs", flush=True)
    initial_model, initial_rms = fit_stereo(train, size, obj, args.distortion_model, args.alpha)
    initial_rows = evaluate(train, initial_model, obj)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "initial_per_view.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(initial_rows[0]))
        writer.writeheader(); writer.writerows(initial_rows)
    suggestions = select_outliers(initial_rows)
    for row in initial_rows:
        print(f"[INITIAL] {row['image']}: L={row['left_reprojection_rms_px']:.3f}, "
              f"R={row['right_reprojection_rms_px']:.3f}, "
              f"epipolar={row['epipolar_median_px']:.3f}px"
              + (f" [REVIEW: {suggestions[row['image']]}]" if row['image'] in suggestions else ""), flush=True)
    print(f"Initial RMS: left={initial_rms[0]:.3f}, right={initial_rms[1]:.3f}, stereo={initial_rms[2]:.3f}px", flush=True)
    excluded_numbers = {int(x.strip()) for x in args.exclude.split(",") if x.strip()}
    excluded, final_train, auto_excluded = choose_training_exclusions(
        train, suggestions, excluded_numbers, args.auto_exclude)
    for name, reason in auto_excluded.items():
        print(f"[AUTO-EXCLUDE] {name}: {reason}", flush=True)
    print(f"Final fit: {len(final_train)} training pairs; excluded={[r[0] for r in excluded]}", flush=True)
    model, (l_rms, r_rms, stereo_rms) = fit_stereo(final_train, size, obj,
                                                   args.distortion_model, args.alpha)
    R, t = model["R_right_from_left"], model["t_right_from_left_mm"]
    train_rows, valid_rows = evaluate(final_train, model, obj), evaluate(valid, model, obj)
    for group, rows in (("TRAIN", train_rows), ("VALIDATION", valid_rows)):
        for row in rows:
            print(f"[{group}] {row['image']}: left RMS={row['left_reprojection_rms_px']:.3f}px, "
                  f"right RMS={row['right_reprojection_rms_px']:.3f}px, "
                  f"epipolar median={row['epipolar_median_px']:.3f}px, "
                  f"p95={row['epipolar_p95_px']:.3f}px", flush=True)
    check_validation_error(valid_rows, args.max_validation_epipolar_px)
    np.savez_compressed(args.output / "stereo_calibration.npz", **model)
    report = {k: v.tolist() for k, v in model.items()}
    report.update({"board_inner_corners": list(BOARD), "square_mm": args.square_mm,
        "square_size_mm": [args.square_mm, args.square_y_mm or args.square_mm],
        "distortion_model": args.distortion_model, "rectification_alpha": args.alpha,
        "left_serial": args.left_serial, "right_serial": args.right_serial,
        "train_dir": str(args.train.resolve()), "validation_dir": str(args.validation.resolve()),
        "train_image_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in ((f"{side}/{p.name}", p) for side in ("left", "right")
                for p in (args.train / side).glob("calib_*.png"))},
        "train_pair_count": len(final_train), "initial_train_pair_count": len(train),
        "excluded_train_pairs": [r[0] for r in excluded],
        "auto_excluded_train_pairs": auto_excluded,
        "manual_excluded_train_pairs": [r[0] for r in excluded if r[0] not in auto_excluded],
        "suggested_train_outliers": suggestions,
        "initial_rms_px": {"left": initial_rms[0], "right": initial_rms[1], "stereo": initial_rms[2]},
        "validation_pair_count": len(valid),
        "rejected_train": rejected_train, "rejected_validation": rejected_valid,
        "rms_px": {"left": l_rms, "right": r_rms, "stereo": stereo_rms},
        "baseline_mm": float(np.linalg.norm(t)),
        "corner_coverage": corner_coverage(final_train, size),
        "rectified_vertical_error": {
            "mean_abs_px": float(np.mean([r["rectified_vertical_mean_abs_px"] for r in train_rows])),
            "rms_px": float(np.sqrt(np.mean([r["rectified_vertical_rms_px"]**2 for r in train_rows]))),
            "max_abs_px": float(np.max([r["rectified_vertical_max_abs_px"] for r in train_rows]))},
        "validation_epipolar_median_px": float(np.median([r["epipolar_median_px"] for r in valid_rows])),
        "validation_epipolar_p95_px": float(np.percentile([r["epipolar_p95_px"] for r in valid_rows], 95))})
    (args.output / "stereo_calibration.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name, rows in (("train", train_rows), ("validation", valid_rows)):
        with (args.output / f"{name}_per_view.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    with (args.output / "per_view.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(train_rows[0]))
        writer.writeheader(); writer.writerows(train_rows)
    write_coverage_image(final_train, size, args.output / "corner_coverage.png")
    write_rectified_examples(final_train, train_rows, model, size, args.output)
    print(f"RESULT: train {len(final_train)}/{len(train)+len(rejected_train)}, "
          f"validation {len(valid)}/{len(valid)+len(rejected_valid)}; "
          f"left RMS={l_rms:.3f}px, right RMS={r_rms:.3f}px, stereo RMS={stereo_rms:.3f}px; "
          f"baseline={report['baseline_mm']:.2f}mm; validation epipolar "
          f"median={report['validation_epipolar_median_px']:.3f}px, "
          f"p95={report['validation_epipolar_p95_px']:.3f}px", flush=True)
    print(f"Saved calibration and per-image reports: {args.output.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
