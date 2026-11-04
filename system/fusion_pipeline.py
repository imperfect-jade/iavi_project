"""Prepare session-specific turntable geometry and fuse reconstructed views."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for package in ("arm", "turntable", "code"):
    sys.path.insert(0, str(ROOT / package))

def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def validate_axis_capture(axis_images: Path) -> None:
    manifest_path = Path(axis_images) / "capture_manifest.json"
    if not manifest_path.is_file() or not json.loads(
            manifest_path.read_text(encoding="utf-8")).get("complete"):
        raise ValueError(f"Automatic fusion requires a complete axis capture: {axis_images}; "
                         "set --fusion-axis-images or use --no-fusion")


def fuse_reconstructed_session(session_root: Path, selected_sources: list[Path],
                               calibration: Path, axis_images: Path, board_path: Path,
                               raw_root: Path, processed_root: Path,
                               radius_mm: float = 65.0,
                               height_mm: tuple[float, float] = (-95.0, 110.0),
                               voxel_mm: float = 1.0,
                               exclude_view_ids: set[int] | None = None,
                               table_height_axis_mm: float | None = None,
                               clean_green_background: bool = True,
                               calibrate=None, fuse=None, cleaner=None,
                               height_estimator=None) -> Path:
    """Run joint board/axis calibration and fuse in the fixed-board frame."""
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
        if table_height_axis_mm is not None:
            import math
            if not math.isfinite(table_height_axis_mm):
                raise ValueError("Table height must be finite")
        validate_axis_capture(axis_images)
        plan_path = session_root / "scan_plan.json"
        if not plan_path.is_file():
            raise ValueError("Scan plan is required to verify the camera-pose frame")
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        board_config = json.loads(board_path.read_text(encoding="utf-8"))
        if plan.get("board") != board_config or \
           plan.get("reference_frame") != "fixed_ChArUco_board":
            raise ValueError("Scan board or camera-pose frame differs from fixed-board fusion")
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
                            "--fixed-board", str(board_path), "--output", str(axis_output)])
        if result != 0 or not axis_output.is_file():
            raise RuntimeError("Turntable axis calibration did not produce a result")
        axis_data = json.loads(axis_output.read_text(encoding="utf-8"))
        if axis_data.get("coordinate_frame") != "fixed_ChArUco_board" or \
           axis_data.get("fixed_board") != board_config:
            raise ValueError("Axis calibration must use this session's stationary fixed board")
        if "up direction" in axis_data.get("turntable_top_error", ""):
            raise ValueError("Axis direction points below the turntable; check calibration and turntable rotation")
        if axis_data["point_rms_mm"] > 3.0 or \
           axis_data["return_to_zero"]["left_corner_shift_rms_px"] > 2.0:
            raise ValueError("Turntable axis fit or return-to-zero exceeds fusion quality limits")
        height_report = None
        from estimate_table_height import height_crop_from_table
        if table_height_axis_mm is None:
            if height_estimator is None:
                from estimate_table_height import table_height_from_axis as height_estimator
            height_report = height_estimator(axis_output)
            table_height_axis_mm = float(height_report["table_height_axis_mm"])
            _write_json(output_root / "table_height_estimate.json", height_report)
        effective_height = (height_crop_from_table(height_mm, table_height_axis_mm)
                            if table_height_axis_mm is not None else height_mm)
        fusion_output = output / "unfiltered" if clean_green_background else output
        command = ["--repeat-root", *map(str, selected),
                   "--raw-root", str(raw_root), "--processed-root", str(processed_root),
                   "--output", str(fusion_output), "--camera-poses", str(poses_path),
                   "--axis-calibration", str(axis_output), "--axis-already-in-reference",
                   "--radius-mm", str(radius_mm), "--height-mm", *map(str, effective_height),
                   "--voxel-mm", str(voxel_mm)]
        if fuse(command) != 0:
            raise RuntimeError("Point cloud fusion failed")
        report_path = fusion_output / "fusion.json"
        if not (fusion_output / "fused.ply").is_file() or not report_path.is_file():
            raise RuntimeError("Point cloud fusion did not produce fused.ply and fusion.json")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if len(report["views"]) != len(selected) or report["fused_points"] < 1:
            raise RuntimeError("Point cloud fusion report does not match selected views")
        if clean_green_background:
            if cleaner is None:
                from clean_green import clean_aligned_fusion as cleaner
            report = cleaner(fusion_output, output, max_radius_mm=radius_mm,
                             table_height_axis_mm=table_height_axis_mm)
            if len(report.get("selected_repeats", [])) != len(selected) or report.get("fused_points", 0) < 1:
                raise RuntimeError("Green filter report does not match selected views")
        _write_json(status_path, {"state": "complete", "output": str(output.resolve()),
                                  "selected_repeats": [str(p) for p in selected],
                                  "excluded_view_ids": sorted(excluded),
                                  "fused_points": report["fused_points"],
                                  "green_filter": report.get("green_filter"),
                                  "table_height_axis_mm": table_height_axis_mm,
                                  "table_height_estimate": height_report,
                                  "axis_calibration": str(axis_output.resolve()),
                                  "axis_frame": "fixed_ChArUco_board"})
        print(f"Automatic fusion complete: {report['fused_points']:,} points -> {output.resolve()}",
              flush=True)
        return output
    except Exception as exc:
        _write_json(status_path, {"state": "failed", "error": str(exc),
                                  "excluded_view_ids": sorted(excluded)})
        raise
