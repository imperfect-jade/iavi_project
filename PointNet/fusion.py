"""Per-session axis calibration and visited-only fusion for active scanning."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "turntable"))
sys.path.insert(0, str(ROOT / "code"))


def _axis_data(path: Path, board: dict | None = None) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("coordinate_frame") != "fixed_ChArUco_board" or data.get("unit") != "mm":
        raise ValueError("axis must use fixed ChArUco board coordinates in mm")
    if board is not None and data.get("fixed_board") != board:
        raise ValueError("axis fixed board differs from scan board")
    if "up direction" in data.get("turntable_top_error", ""):
        raise ValueError("axis direction points below the turntable; check calibration and turntable rotation")
    rms = data.get("point_rms_mm")
    shift = data.get("return_to_zero", {}).get("left_corner_shift_rms_px")
    if (not isinstance(rms, (int, float)) or not isinstance(shift, (int, float)) or
            not math.isfinite(rms) or not math.isfinite(shift) or
            rms > 3.0 or shift > 2.0):
        raise ValueError("axis quality exceeds fusion limits")
    return data


def calibrate_session_axis(axis_images, lower_stereo, board_config, output,
                           *, calibrate=None) -> Path:
    images, board, output = map(Path, (axis_images, board_config, output))
    manifest = json.loads((images / "capture_manifest.json").read_text(encoding="utf-8"))
    if not manifest.get("complete"):
        raise ValueError("axis capture is incomplete")
    if calibrate is None:
        from calibrate_axis import main as calibrate
    output.parent.mkdir(parents=True, exist_ok=True)
    argv = ["--images", str(images), "--stereo", str(lower_stereo),
            "--fixed-board", str(board), "--output", str(output)]
    if calibrate(argv) != 0 or not output.is_file():
        raise RuntimeError("axis calibration did not produce a result")
    _axis_data(output, json.loads(board.read_text(encoding="utf-8")))
    return output


def fuse_selected(repeats, axis_calibration, camera_poses, raw_root,
                  processed_root, output, crop=None, *, fuse=None,
                  clean_green_background=False, table_height_axis_mm=None,
                  clean_radius_mm=None, cleaner=None) -> Path:
    repeats = [Path(path).resolve() for path in repeats]
    if len(repeats) < 2:
        raise ValueError("fusion requires at least two repeats")
    if len(set(repeats)) != len(repeats) or any(not p.is_dir() for p in repeats):
        raise ValueError("repeats must be unique existing directories")
    if len({p.parent.parent for p in repeats}) != 1:
        raise ValueError("repeats must belong to one session")
    axis_calibration = Path(axis_calibration).resolve()
    camera_poses = Path(camera_poses).resolve()
    output = Path(output).resolve()
    plan_path = repeats[0].parent.parent / "scan_plan.json"
    if plan_path.is_file():
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if plan.get("reference_frame") != "fixed_ChArUco_board":
            raise ValueError("scan plan must use the fixed board reference frame")
        _axis_data(axis_calibration, plan.get("board"))
    else:
        _axis_data(axis_calibration)
    if not camera_poses.is_file():
        raise FileNotFoundError(camera_poses)
    if fuse is None:
        from fuse_views import main as fuse
    crop = crop or {}
    radius = float(crop.get("radius_mm", 65.0))
    height = tuple(crop.get("height_mm", (-95.0, 110.0)))
    voxel = float(crop.get("voxel_mm", 1.0))
    if radius <= 0 or voxel <= 0 or len(height) != 2 or height[0] >= height[1]:
        raise ValueError("invalid fusion crop")
    fusion_output = output / "unfiltered" if clean_green_background else output
    argv = ["--repeat-root", *map(str, repeats), "--raw-root", str(Path(raw_root).resolve()),
            "--processed-root", str(Path(processed_root).resolve()),
            "--output", str(fusion_output), "--camera-poses", str(camera_poses),
            "--axis-calibration", str(axis_calibration), "--axis-already-in-reference",
            "--radius-mm", str(radius), "--height-mm", *map(str, height),
            "--voxel-mm", str(voxel)]
    if fuse(argv) != 0:
        raise RuntimeError("visited view fusion failed")
    report_path = fusion_output / "fusion.json"
    if not report_path.is_file() or not (fusion_output / "fused.ply").is_file():
        raise RuntimeError("fusion did not produce report and cloud")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    selected = [Path(path).resolve() for path in report.get("selected_repeats", [])]
    if (selected != repeats or report.get("coordinate_frame") != "turntable_zero_calibrated" or
            report.get("unit") != "mm" or report.get("fused_points", 0) < 1):
        raise RuntimeError("fusion report does not match visited repeat selection")
    if clean_green_background:
        if cleaner is None:
            from clean_green import clean_aligned_fusion as cleaner
        cleaned = cleaner(fusion_output, output, radius_mm=clean_radius_mm,
                          max_radius_mm=radius,
                          table_height_axis_mm=table_height_axis_mm)
        if cleaned.get("selected_repeats") != report["selected_repeats"] or cleaned.get("fused_points", 0) < 1:
            raise RuntimeError("green filter report does not match selected views")
    return output
