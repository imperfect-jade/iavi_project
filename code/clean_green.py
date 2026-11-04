"""Remove a green backdrop from aligned fusion clouds or a display-only PLY."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from fuse_views import read_ply, voxel_average
from reconstruct_view import write_ply


def _axis_from_report(report: dict) -> tuple[np.ndarray, np.ndarray]:
    if report.get("coordinate_frame") != "turntable_zero_calibrated" or report.get("unit") != "mm":
        raise ValueError("expected calibrated turntable-zero fusion in millimetres")
    center = np.asarray(report["axis_point_mm"], dtype=np.float64)
    axis = np.asarray(report["axis_direction"], dtype=np.float64)
    if (center.shape != (3,) or axis.shape != (3,) or
            not np.isfinite(center).all() or not np.isfinite(axis).all() or
            np.linalg.norm(axis) < 1e-8):
        raise ValueError("invalid calibrated rotation axis")
    return center, axis / np.linalg.norm(axis)


def _masks(xyz, bgr, center, axis, radius_mm, table_height_axis_mm,
           min_object_height_mm, hue_min, hue_max, min_saturation, min_value):
    hsv = cv2.cvtColor(bgr[:, None, :], cv2.COLOR_BGR2HSV)[:, 0, :]
    green = ((hsv[:, 0] >= hue_min) & (hsv[:, 0] <= hue_max) &
             (hsv[:, 1] >= min_saturation) & (hsv[:, 2] >= min_value))
    offset = xyz.astype(np.float64) - center
    axial = offset @ axis
    radial = offset - axial[:, None] * axis
    outside_radius = (np.einsum("ij,ij->i", radial, radial) > radius_mm ** 2
                      if radius_mm is not None else np.zeros(len(xyz), dtype=bool))
    below_height = (axial - table_height_axis_mm < min_object_height_mm
                    if table_height_axis_mm is not None else np.zeros(len(xyz), dtype=bool))
    return green, outside_radius, below_height, hsv, np.linalg.norm(radial, axis=1), axial


def estimate_object_radius(fusion_dir, *, max_radius_mm=65.0,
                           table_height_axis_mm=None, min_object_height_mm=3.0,
                           margin_mm=5.0) -> tuple[float, dict]:
    """Estimate a conservative footprint from bright non-green object candidates."""
    fusion_dir = Path(fusion_dir)
    report = json.loads((fusion_dir / "fusion.json").read_text(encoding="utf-8"))
    center, axis = _axis_from_report(report)
    xyz, bgr = read_ply(fusion_dir / "fused.ply")
    green, _, _, hsv, radial, axial = _masks(
        xyz, bgr, center, axis, None, None, 0, 35, 95, 60, 20)
    candidates = ~green & (hsv[:, 1] >= 60) & (hsv[:, 2] >= 80)
    if table_height_axis_mm is not None:
        candidates &= axial - table_height_axis_mm >= min_object_height_mm
    count = int(candidates.sum())
    if count < 500 or count < len(xyz) * 0.05:
        return float(max_radius_mm), {"method": "fallback", "reason": "too_few_foreground_candidates",
                                       "candidate_points": count}
    percentile = float(np.percentile(radial[candidates], 95))
    if not np.isfinite(percentile) or percentile > max_radius_mm - margin_mm - 2:
        return float(max_radius_mm), {"method": "fallback", "reason": "foreground_reaches_crop_boundary",
                                       "candidate_points": count, "percentile_95_mm": percentile}
    radius = float(np.clip(percentile + margin_mm, 10, max_radius_mm))
    return radius, {"method": "bright_non_green_95th_percentile_plus_margin",
                    "candidate_points": count, "percentile_95_mm": percentile,
                    "margin_mm": float(margin_mm)}


def clean_aligned_fusion(source_dir, output_dir, *, radius_mm=None,
                         max_radius_mm=65.0, table_height_axis_mm=None,
                         min_object_height_mm=3.0) -> dict:
    """Filter aligned views before voxel averaging into a complete fusion directory."""
    source_dir, output_dir = Path(source_dir).resolve(), Path(output_dir).resolve()
    if source_dir == output_dir or not (source_dir / "aligned").is_dir():
        raise ValueError("source and cleaned output must be distinct fusion directories")
    if not np.isfinite(max_radius_mm) or max_radius_mm <= 0:
        raise ValueError("max_radius_mm must be positive and finite")
    if radius_mm is not None and (not np.isfinite(radius_mm) or radius_mm <= 0 or
                                  radius_mm > max_radius_mm):
        raise ValueError("radius_mm must be within the broad fusion crop")
    if table_height_axis_mm is not None and not np.isfinite(table_height_axis_mm):
        raise ValueError("table_height_axis_mm must be finite")
    report = json.loads((source_dir / "fusion.json").read_text(encoding="utf-8"))
    center, axis = _axis_from_report(report)
    if radius_mm is None:
        radius_mm, estimate = estimate_object_radius(
            source_dir, max_radius_mm=max_radius_mm,
            table_height_axis_mm=table_height_axis_mm,
            min_object_height_mm=min_object_height_mm)
    else:
        estimate = {"method": "fixed_session_radius"}
    voxel_mm = float(report["voxel_mm"])
    if not np.isfinite(voxel_mm) or voxel_mm <= 0:
        raise ValueError("invalid fusion voxel size")
    repeats = report.get("selected_repeats", [])
    if not repeats or len(report.get("views", [])) != len(repeats):
        raise ValueError("fusion report has inconsistent selected repeats and views")
    output_dir.mkdir(parents=True, exist_ok=True)
    aligned_output = output_dir / "aligned"
    aligned_output.mkdir(exist_ok=True)
    all_xyz, all_bgr, per_view, names = [], [], [], set()
    for repeat in repeats:
        path = Path(repeat.replace("\\", "/"))
        name = f"{path.parent.name}_{path.name}.ply"
        xyz, bgr = read_ply(source_dir / "aligned" / name)
        green, outside, below, _, _, _ = _masks(
            xyz, bgr, center, axis, radius_mm, table_height_axis_mm,
            min_object_height_mm, 35, 95, 60, 20)
        keep = ~(green | outside | below)
        if not keep.any():
            raise ValueError(f"green/background filter removed all points from {name}")
        temporary = aligned_output / (name + ".tmp")
        write_ply(temporary, xyz[keep], bgr[keep], "turntable_zero_calibrated")
        temporary.replace(aligned_output / name)
        names.add(name)
        all_xyz.append(xyz[keep])
        all_bgr.append(bgr[keep])
        per_view.append({"repeat": repeat, "source_points": int(len(xyz)),
                         "retained_points": int(keep.sum()), "green_points": int(green.sum()),
                         "outside_radius_points": int(outside.sum()),
                         "below_height_points": int(below.sum())})
    for stale in aligned_output.glob("*.ply"):
        if stale.name not in names:
            stale.unlink()
    fused_xyz, fused_bgr = voxel_average(np.concatenate(all_xyz), np.concatenate(all_bgr), voxel_mm)
    temporary = output_dir / "fused.ply.tmp"
    write_ply(temporary, fused_xyz, fused_bgr, "turntable_zero_calibrated")
    temporary.replace(output_dir / "fused.ply")
    view_rows = []
    for original, filtered in zip(report.get("views", []), per_view):
        row = dict(original)
        row["cropped_points_unfiltered"] = row.get("cropped_points")
        row["cropped_points"] = filtered["retained_points"]
        view_rows.append(row)
    clean_report = {**report, "input_points": int(sum(len(x) for x in all_xyz)),
                    "fused_points": int(len(fused_xyz)),
                    "views": view_rows,
                    "green_filter": {"source_fusion": str(source_dir),
                                     "radius_mm": radius_mm,
                                     "radius_estimate": estimate,
                                     "hue_range_opencv": [35, 95],
                                     "min_saturation": 60, "min_value": 20,
                                     "table_height_axis_mm": table_height_axis_mm,
                                     "min_object_height_mm": min_object_height_mm,
                                     "views": per_view}}
    temporary_report = output_dir / "fusion.json.tmp"
    temporary_report.write_text(json.dumps(clean_report, indent=2, ensure_ascii=False) + "\n",
                                encoding="utf-8")
    temporary_report.replace(output_dir / "fusion.json")
    return clean_report


def clean_fusion(fusion_dir, output, *, radius_mm=None,
                 table_height_axis_mm=None, min_object_height_mm=3.0,
                 hue_min=35, hue_max=95, min_saturation=60, min_value=20) -> dict:
    """Filter a fused PLY for viewing; retain the original fusion and aligned views."""
    fusion_dir, output = Path(fusion_dir).resolve(), Path(output).resolve()
    source = fusion_dir / "fused.ply"
    report_path = output.with_suffix(".json")
    if output == source or output.exists() or report_path.exists():
        raise FileExistsError("clean output already exists; choose a new output path")
    if not (0 <= hue_min <= hue_max <= 179 and
            0 <= min_saturation <= 255 and 0 <= min_value <= 255):
        raise ValueError("invalid OpenCV HSV green range")
    if radius_mm is not None and (not np.isfinite(radius_mm) or radius_mm <= 0):
        raise ValueError("radius_mm must be positive and finite")
    if table_height_axis_mm is not None and not np.isfinite(table_height_axis_mm):
        raise ValueError("table_height_axis_mm must be finite")
    if not np.isfinite(min_object_height_mm):
        raise ValueError("min_object_height_mm must be finite")
    fusion = json.loads((fusion_dir / "fusion.json").read_text(encoding="utf-8"))
    if fusion.get("coordinate_frame") != "turntable_zero_calibrated" or fusion.get("unit") != "mm":
        raise ValueError("expected calibrated turntable-zero fusion in millimetres")
    center = np.asarray(fusion["axis_point_mm"], dtype=np.float64)
    axis = np.asarray(fusion["axis_direction"], dtype=np.float64)
    if (center.shape != (3,) or axis.shape != (3,) or
            not np.isfinite(center).all() or not np.isfinite(axis).all() or
            np.linalg.norm(axis) < 1e-8):
        raise ValueError("invalid calibrated rotation axis")
    axis /= np.linalg.norm(axis)
    xyz, bgr = read_ply(source)
    if not len(xyz) or not np.isfinite(xyz).all():
        raise ValueError("source cloud is empty or has nonfinite points")
    hsv = cv2.cvtColor(bgr[:, None, :], cv2.COLOR_BGR2HSV)[:, 0, :]
    green = ((hsv[:, 0] >= hue_min) & (hsv[:, 0] <= hue_max) &
             (hsv[:, 1] >= min_saturation) & (hsv[:, 2] >= min_value))
    offset = xyz.astype(np.float64) - center
    axial = offset @ axis
    outside_radius = np.zeros(len(xyz), dtype=bool)
    if radius_mm is not None:
        radial = offset - axial[:, None] * axis
        outside_radius = np.einsum("ij,ij->i", radial, radial) > radius_mm ** 2
    below_height = np.zeros(len(xyz), dtype=bool)
    if table_height_axis_mm is not None:
        below_height = axial - table_height_axis_mm < min_object_height_mm
    keep = ~(green | outside_radius | below_height)
    if not keep.any():
        raise ValueError("filter removed every point; loosen the green or geometric limits")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    write_ply(temporary, xyz[keep], bgr[keep], "turntable_zero_calibrated")
    temporary.replace(output)
    report = {"source": str(source), "output": str(output),
              "source_points": int(len(xyz)), "retained_points": int(keep.sum()),
              "removed_green": int(green.sum()),
              "removed_outside_radius": int(outside_radius.sum()),
              "removed_below_height": int(below_height.sum()),
              "hue_range_opencv": [hue_min, hue_max],
              "min_saturation": min_saturation, "min_value": min_value,
              "radius_mm": radius_mm, "table_height_axis_mm": table_height_axis_mm,
              "min_object_height_mm": min_object_height_mm if table_height_axis_mm is not None else None,
              "note": "Display-only filter; source fusion, aligned views and model input are unchanged."}
    temporary_report = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary_report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary_report.replace(report_path)
    return report


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fusion-dir", type=Path, required=True)
    p.add_argument("--output", type=Path,
                   help="new PLY path; defaults to <fusion-dir>/fused_no_green.ply")
    p.add_argument("--radius-mm", type=float,
                   help="optional maximum distance from the turntable axis")
    p.add_argument("--table-height-axis-mm", type=float,
                   help="optional tabletop coordinate along the calibrated axis")
    p.add_argument("--min-object-height-mm", type=float, default=3,
                   help="minimum height above the tabletop when tabletop coordinate is given")
    p.add_argument("--hue-min", type=int, default=35)
    p.add_argument("--hue-max", type=int, default=95)
    p.add_argument("--min-saturation", type=int, default=60)
    p.add_argument("--min-value", type=int, default=20)
    args = p.parse_args(argv)
    output = args.output or args.fusion_dir / "fused_no_green.ply"
    report = clean_fusion(args.fusion_dir, output, radius_mm=args.radius_mm,
                          table_height_axis_mm=args.table_height_axis_mm,
                          min_object_height_mm=args.min_object_height_mm,
                          hue_min=args.hue_min, hue_max=args.hue_max,
                          min_saturation=args.min_saturation, min_value=args.min_value)
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
