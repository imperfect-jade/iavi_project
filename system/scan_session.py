"""Capture an arm/turntable scan, then reconstruct its views one by one."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "turntable"))
sys.path.insert(0, str(ROOT / "arm"))
from turntable import ANGLE_TO_PWM, SETTLE_WAIT_S, open_controller
from arm import open_controller as open_arm_controller
from fixed_board import (board_from_config, detect_corners_with_contrast_fallback,
                         estimate_pose)


def angles_for_range(start: int, end: int, step: int) -> list[int]:
    if step < 30 or step % 30 or start not in ANGLE_TO_PWM or end not in ANGLE_TO_PWM or start > end:
        raise ValueError("Use ascending tested angles 0..330 and a positive multiple-of-30 step")
    values = list(range(start, end + 1, step))
    if values[-1] != end:
        raise ValueError("Step must land exactly on end-angle")
    return values


def return_turntable_home(controller, session_root: Path, settle_s: float) -> None:
    """Return to the tested zero command after all requested captures."""
    command = controller.move_to(0)
    sent_at = datetime.now().astimezone().isoformat()
    print(f"Final turntable home: {command.decode('ascii')}; waiting {settle_s:g} s", flush=True)
    time.sleep(settle_s)
    record = {"command_deg": 0, "serial_command": command.decode("ascii"),
              "sent_at": sent_at, "settle_s": settle_s, "arrival_verified": False}
    (session_root / "turntable_final_home.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def pending_views(session_root: Path, angles: list[int], arm_pose_id: int,
                  repeat_id: int, resume: bool) -> list[int]:
    """Preflight every output before movement; never overwrite an incomplete view."""
    pending = []
    for angle in angles:
        view_id = arm_pose_id * 12 + angle // 30
        view_root = session_root / f"view_{view_id:03d}"
        repeats = sorted(view_root.glob("repeat_*"))
        if repeats and not resume:
            raise FileExistsError(f"{view_root} exists; use --resume or a new session-id")
        if any(is_complete_view(folder, view_id, arm_pose_id, angle) for folder in repeats):
            continue
        destination = view_root / f"repeat_{repeat_id:03d}"
        if destination.exists():
            raise FileExistsError(f"{destination} is incomplete; choose a new --repeat-id")
        pending.append(angle)
    return pending


def is_complete_view(folder: Path, view_id: int, arm_pose_id: int, angle: int) -> bool:
    manifest_path = folder / "manifest.json"
    if not manifest_path.is_file():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frames = manifest.get("frames", [])
    files_present = bool(frames) and all(
        (folder / row[side]).is_file() for row in frames for side in ("left", "right"))
    return bool(files_present and manifest.get("frame_count") == len(frames) and
                manifest.get("complete") and manifest.get("view_id") == view_id and
                manifest.get("arm_pose_id") == arm_pose_id and
                manifest.get("turntable_deg_commanded") == angle and
                manifest.get("protocol") in ("dense_dots_v2", "dense_dots_v3") and
                (folder / "localization" / "left.png").is_file() and
                (folder / "localization" / "right.png").is_file() and
                (folder / "localization.json").is_file())


def selected_complete_sources(session_root: Path, pose_ids: list[int],
                              angles: list[int]) -> list[Path]:
    """Choose the latest complete repeat of each requested view."""
    selected = []
    for pose_id in pose_ids:
        for angle in angles:
            view_id = pose_id * 12 + angle // 30
            view_root = session_root / f"view_{view_id:03d}"
            candidates = [folder for folder in sorted(view_root.glob("repeat_*"))
                          if is_complete_view(folder, view_id, pose_id, angle)]
            if not candidates:
                raise ValueError(f"No complete capture for {view_root}")
            selected.append(candidates[-1])
    return selected


def load_pose_calibrations(upper_json: Path, lower_json: Path,
                           camera_serials: tuple[str, str]):
    """Load the two compatible pose models and their shared left intrinsics."""
    from fuse_views import validate_stereo_compatibility

    calibrations = {0: Path(upper_json).with_suffix(".npz").resolve(),
                    1: Path(lower_json).with_suffix(".npz").resolve()}
    if calibrations[0] == calibrations[1]:
        raise ValueError("Upper and lower poses require separate stereo calibrations")
    validate_stereo_compatibility(calibrations.values(),
        {"frame": "fixed_ChArUco_board", "stereo_calibration": calibrations[1]}, True)
    upper_metadata = json.loads(calibrations[0].with_suffix(".json").read_text(encoding="utf-8"))
    lower_hash = hashlib.sha256(calibrations[1].read_bytes()).hexdigest()
    if upper_metadata.get("lower_sha256") != lower_hash:
        raise ValueError("Lower calibration changed; rerun upper pose calibration")
    metadata = json.loads(calibrations[1].with_suffix(".json").read_text(encoding="utf-8"))
    if camera_serials != (metadata["left_serial"], metadata["right_serial"]):
        raise ValueError("Camera serials differ from pose stereo calibrations")
    with np.load(calibrations[1]) as archive:
        K = archive["K_left"].copy()
        D = archive["D_left"].copy()
        size = tuple(int(v) for v in archive["image_size"])
    return calibrations, K, D, size


def reconstruct_session(session_root: Path, processed_root: Path, pose_ids: list[int],
                        angles: list[int], calibrations: dict[int, Path], reconstruct,
                        min_depth_mm: float | None = None,
                        max_depth_mm: float | None = None) -> int:
    """Run reconstruction after capture; preserve each view's result for retry."""
    from capture import _write_json

    processed_root.mkdir(parents=True, exist_ok=True)
    calibrations = {pose: path.resolve() for pose, path in calibrations.items()}
    calibration_hashes = {pose: hashlib.sha256(path.read_bytes()).hexdigest()
                          for pose, path in calibrations.items()}
    summary = {"complete": 0, "skipped": 0, "failed": 0,
               "calibrations_by_pose": {str(pose): {"path": str(path),
                   "sha256": calibration_hashes[pose]}
                   for pose, path in calibrations.items()}, "views": {}}
    for source in selected_complete_sources(session_root, pose_ids, angles):
        manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
        pose_id = int(manifest["arm_pose_id"])
        calibration = calibrations[pose_id]
        view_root = source.parent
        name = f"{view_root.name}/{source.name}"
        target = processed_root / view_root.name / source.name
        target.mkdir(parents=True, exist_ok=True)
        status_path = target / "reconstruction_status.json"
        fingerprint = {"source_manifest_sha256": hashlib.sha256(
            (source / "manifest.json").read_bytes()).hexdigest(),
            "calibration_sha256": calibration_hashes[pose_id],
            "min_depth_mm": min_depth_mm, "max_depth_mm": max_depth_mm}
        if status_path.is_file():
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if (status.get("state") == "complete" and
                all(status.get(key) == value for key, value in fingerprint.items()) and
                (target / "cloud_left.ply").is_file() and (target / "qc.json").is_file()):
                summary["complete"] += 1
                summary["skipped"] += 1
                summary["views"][name] = "skipped"
                print(f"Reconstruction already complete: {name}", flush=True)
                continue
        _write_json(status_path, {**fingerprint, "state": "running"})
        command = ["--view", str(source), "--output", str(target),
                   "--calibration", str(calibration)]
        if min_depth_mm is not None:
            command += ["--min-depth-mm", str(min_depth_mm),
                        "--max-depth-mm", str(max_depth_mm)]
        print(f"Reconstructing {name}", flush=True)
        try:
            result = reconstruct(command)
            if result != 0 or not (target / "qc.json").is_file() or not (target / "cloud_left.ply").is_file():
                raise RuntimeError(f"Reconstruction returned {result} or did not produce required outputs")
            qc = json.loads((target / "qc.json").read_text(encoding="utf-8"))
            _write_json(status_path, {**fingerprint, "state": "complete",
                                      "point_count": qc.get("point_count"),
                                      "warnings": qc.get("warnings", [])})
            summary["complete"] += 1
            summary["views"][name] = "complete"
        except Exception as exc:
            _write_json(status_path, {**fingerprint, "state": "failed", "error": str(exc)})
            summary["failed"] += 1
            summary["views"][name] = f"failed: {exc}"
            print(f"[ERROR] Reconstruction {name}: {exc}", flush=True)
    _write_json(processed_root / "reconstruction_summary.json", summary)
    return 1 if summary["failed"] else 0


def finish_session(args, session_root: Path, processed_root: Path,
                   pose_ids: list[int], angles: list[int], calibrations: dict[int, Path],
                   reconstruct) -> int:
    """Reconstruct first, then fuse the selected repeats if every view passed."""
    result = reconstruct_session(session_root, processed_root, pose_ids, angles,
                                 calibrations, reconstruct,
                                 args.reconstruction_min_depth_mm,
                                 args.reconstruction_max_depth_mm)
    if result or args.no_fusion:
        return result
    from fusion_pipeline import fuse_reconstructed_session
    try:
        sources = selected_complete_sources(session_root, pose_ids, angles)
        fuse_reconstructed_session(
            session_root, sources, calibrations[1], args.fusion_axis_images, args.board,
            args.output, args.processed_output, radius_mm=args.fusion_radius_mm,
            height_mm=tuple(args.fusion_height_mm), voxel_mm=args.fusion_voxel_mm,
            exclude_view_ids=set(args.fusion_exclude_view or ()),
            table_height_axis_mm=getattr(args, "fusion_table_height_axis_mm", None),
            clean_green_background=not getattr(args, "no_green_filter", False))
        return 0
    except Exception as exc:
        print(f"[ERROR] Automatic fusion: {exc}", flush=True)
        return 1


def localize(left, board, K, D, min_corners=12, max_rms_px=3.0):
    corners, ids, method = detect_corners_with_contrast_fallback(left, board, min_corners)
    if len(ids) < min_corners:
        raise ValueError(f"Only {len(ids)} fixed-board corners; need {min_corners}")
    current, rms = estimate_pose(board, corners, ids, K, D)
    if rms > max_rms_px:
        raise ValueError(f"Fixed-board reprojection RMS {rms:.2f} px exceeds {max_rms_px:.2f} px")
    return {"coordinate_frame": "fixed_ChArUco_board",
            "T_reference_from_left": np.linalg.inv(current).tolist(),
            "T_left_from_board": current.tolist(), "corners": len(ids),
            "detection_method": method, "reprojection_rms_px": rms}


def refresh_camera_poses(session_root: Path, processed_root: Path,
                         sources: list[Path], board, K, D,
                         min_corners: int, max_rms_px: float) -> None:
    """Re-localize complete captures with the intrinsics used for reconstruction."""
    from capture import _write_json

    poses, per_view = {}, {}
    for source in sources:
        image_path = source / "left" / "000_black.png"
        image = cv2.imdecode(np.frombuffer(image_path.read_bytes(), np.uint8),
                             cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot read fixed-board black frame: {image_path}")
        result = localize(image, board, K, D, min_corners, max_rms_px)
        name = f"{source.parent.name}/{source.name}"
        poses[name] = result["T_reference_from_left"]
        per_view[name] = {"corners": result["corners"],
                          "reprojection_rms_px": result["reprojection_rms_px"],
                          "black_frame": str(image_path.resolve())}
    _write_json(session_root / "camera_poses.json", poses)
    processed_root.mkdir(parents=True, exist_ok=True)
    _write_json(processed_root / "camera_pose_refresh.json", {
        "intrinsics_sha256": hashlib.sha256(np.asarray(K).tobytes() +
                                             np.asarray(D).tobytes()).hexdigest(),
        "views": per_view})


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", default="COM6")
    p.add_argument("--arm-port", default="COM7")
    p.add_argument("--start-angle", type=int, required=True)
    p.add_argument("--end-angle", type=int, required=True)
    p.add_argument("--step-angle", type=int, default=30)
    p.add_argument("--object-id", required=True)
    p.add_argument("--session-id", required=True)
    p.add_argument("--arm-pose-id", type=int, choices=(0, 1),
                   help="Capture one pose; default captures upper and lower")
    p.add_argument("--arm-settle-s", type=float, default=6.0)
    p.add_argument("--board", type=Path,
                   default=ROOT / "arm/targets/charuco_7x5_10mm_7mm_config.json")
    p.add_argument("--stereo-upper", type=Path,
                   default=ROOT / "calibration/stereo_pose_upper/stereo_calibration.json")
    p.add_argument("--stereo-lower", type=Path,
                   default=ROOT / "calibration/stereo_pose_lower/stereo_calibration.json")
    p.add_argument("--min-board-corners", type=int, default=12)
    p.add_argument("--max-board-rms-px", type=float, default=3.0)
    p.add_argument("--repeat-id", type=int, default=1)
    p.add_argument("--resume", action="store_true", help="Skip complete views in the same session")
    p.add_argument("--output", type=Path, default=ROOT / "raw")
    p.add_argument("--processed-output", type=Path, default=ROOT / "processed")
    p.add_argument("--reconstruction-min-depth-mm", type=float)
    p.add_argument("--reconstruction-max-depth-mm", type=float)
    p.add_argument("--no-fusion", action="store_true",
                   help="Reconstruct views but skip automatic point cloud fusion")
    p.add_argument("--fusion-axis-images", type=Path,
                   help="required for automatic fusion: new joint fixed-board/axis capture session")
    p.add_argument("--fusion-radius-mm", type=float, default=65.0)
    p.add_argument("--fusion-height-mm", type=float, nargs=2, default=(-95.0, 110.0),
                   metavar=("MIN", "MAX"))
    p.add_argument("--fusion-voxel-mm", type=float, default=1.0)
    p.add_argument("--fusion-table-height-axis-mm", type=float,
                   help="manual turntable-top coordinate; omit to use the axis target's measured bottom")
    p.add_argument("--no-green-filter", action="store_true",
                   help="keep the original unfiltered fusion output")
    p.add_argument("--fusion-exclude-view", type=int, choices=range(24), action="append",
                   metavar="VIEW_ID", help="exclude a suspect view ID; may be repeated")
    p.add_argument("--left-serial", default="40031431")
    p.add_argument("--right-serial", default="40031435")
    p.add_argument("--display", help="Exact projector display name if its location changed")
    p.add_argument("--exposure-us", type=float, default=10000)
    p.add_argument("--gain-db", type=float, default=0)
    p.add_argument("--timeout-ms", type=int, default=5000)
    p.add_argument("--settle-ms", type=int, default=250, help="Pattern stabilization time")
    p.add_argument("--motion-settle-s", type=float, default=SETTLE_WAIT_S)
    p.add_argument("--return-settle-s", type=float, default=12.0,
                   help="Wait for 0-degree return between poses and after final capture")
    p.add_argument("--dot-cell-px", type=int, default=8)
    p.add_argument("--dot-seed", type=int, default=5)
    p.add_argument("--dot-frames", type=int, default=5)
    return p


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    from capture import (Pattern, PREVIEW, PROJECTOR, SAFE, _write_json, capture_once,
                         close_cameras, displays, grab_pair, make_dense_dots, open_cameras,
                         preview_pair, reserve_view, select_projector_display, show_pattern)
    angles = angles_for_range(args.start_angle, args.end_angle, args.step_angle)
    pose_ids = [args.arm_pose_id] if args.arm_pose_id is not None else [0, 1]
    if args.port == args.arm_port or args.arm_settle_s < 0 or \
       not 4 <= args.min_board_corners <= 24 or args.max_board_rms_px <= 0:
        raise ValueError("Invalid arm port, settle time, or fixed-board quality limits")
    board_config = json.loads(args.board.read_text(encoding="utf-8"))
    board = board_from_config(board_config)
    calibrations, K, D, camera_size = load_pose_calibrations(
        args.stereo_upper, args.stereo_lower,
        (args.left_serial, args.right_serial))
    if not SAFE.fullmatch(args.object_id) or not SAFE.fullmatch(args.session_id):
        raise ValueError("object-id and session-id require letters, digits, _ or -")
    if args.repeat_id < 1 or args.motion_settle_s < 0 or args.return_settle_s < 0 or args.settle_ms < 1 or \
       args.timeout_ms < 1 or args.exposure_us <= 0 or not 1 <= args.dot_frames <= 32:
        raise ValueError("Invalid repeat, timing, exposure, or dot-frame value")
    if (args.reconstruction_min_depth_mm is None) != (args.reconstruction_max_depth_mm is None):
        raise ValueError("Set both reconstruction depth bounds or neither")
    if args.reconstruction_min_depth_mm is not None and not (0 < args.reconstruction_min_depth_mm < args.reconstruction_max_depth_mm):
        raise ValueError("Invalid reconstruction depth range")
    if not args.no_fusion and (args.fusion_radius_mm <= 0 or args.fusion_voxel_mm <= 0 or
                               args.fusion_height_mm[0] >= args.fusion_height_mm[1]):
        raise ValueError("Invalid fusion radius, height or voxel size")
    if args.fusion_table_height_axis_mm is not None and (
            not np.isfinite(args.fusion_table_height_axis_mm) or
            (not args.no_fusion and args.fusion_table_height_axis_mm >= args.fusion_height_mm[1])):
        raise ValueError("Fusion tabletop height must be finite and below the fusion upper height")
    if not args.no_fusion:
        if args.fusion_axis_images is None:
            raise ValueError("Automatic fusion requires --fusion-axis-images from a joint "
                             "fixed-board/turntable-axis capture; or use --no-fusion")
        from fusion_pipeline import validate_axis_capture
        validate_axis_capture(args.fusion_axis_images)
    args.trigger = "software"
    session_root = args.output / args.object_id / args.session_id
    plan = {"angles_deg": angles, "arm_pose_ids": pose_ids,
            "protocol": "dense_dots_v3", "dot_cell_px": args.dot_cell_px,
            "dot_seed": args.dot_seed, "dot_frames": args.dot_frames,
            "left_serial": args.left_serial, "right_serial": args.right_serial,
            "exposure_us": args.exposure_us, "gain_db": args.gain_db,
            "port": args.port, "arm_port": args.arm_port,
            "arm_settle_s": args.arm_settle_s,
            "reference_frame": "fixed_ChArUco_board",
            "board": board_config,
            "stereo_by_pose": {str(pose): str(calibrations[pose].with_suffix(".json"))
                               for pose in (0, 1)},
            "motion_settle_s": args.motion_settle_s,
            "pattern_settle_ms": args.settle_ms, "timeout_ms": args.timeout_ms,
            "display_name": args.display}
    plan_path = session_root / "scan_plan.json"
    if args.resume:
        if not plan_path.exists() or json.loads(plan_path.read_text(encoding="utf-8")) != plan:
            raise ValueError("--resume requires a matching scan_plan.json; use a new session-id")
    elif session_root.exists():
        raise FileExistsError(f"{session_root} exists; use --resume or a new session-id")
    pending = [(pose, angle) for pose in pose_ids for angle in
               pending_views(session_root, angles, pose, args.repeat_id, args.resume)]
    processed_root = args.processed_output / args.object_id / args.session_id
    if not pending:
        print("All requested views already captured; checking reconstruction.")
        from reconstruct_view import main as reconstruct_view
        refresh_camera_poses(session_root, processed_root,
            selected_complete_sources(session_root, pose_ids, angles), board, K, D,
            args.min_board_corners, args.max_board_rms_px)
        return finish_session(args, session_root, processed_root, pose_ids, angles,
                              calibrations, reconstruct_view)

    display = select_projector_display(displays(), args.display)
    patterns = make_dense_dots(dot_cell=args.dot_cell_px, seed=args.dot_seed,
                               count=args.dot_frames)
    cameras = controller = arm_controller = None
    try:
        cameras, converter, pylon, size = open_cameras(args)
        if tuple(size) != camera_size:
            raise ValueError("Camera size differs from stereo calibration")
        controller = open_controller(args.port)
        arm_controller = open_arm_controller(args.arm_port)
        cv2.namedWindow(PREVIEW, cv2.WINDOW_NORMAL)
        cv2.namedWindow(PROJECTOR, cv2.WINDOW_NORMAL)
        show_pattern(patterns[0], display)
        cv2.waitKey(args.settle_ms)
        while True:
            left, right, _ = grab_pair(cameras, converter, pylon, args.timeout_ms)
            cv2.imshow(PREVIEW, preview_pair(left, right, "Space start session / Q cancel"))
            key = cv2.waitKey(1) & 255
            if key in (27, ord("q")):
                print("Cancelled before movement.")
                return 0
            if key == 32:
                break
        show_pattern(Pattern("black", np.zeros_like(patterns[0].image)), display)
        cv2.waitKey(args.settle_ms)
        if not args.resume:
            session_root.mkdir(parents=True, exist_ok=False)
            _write_json(plan_path, plan)

        poses_path = session_root / "camera_poses.json"
        camera_poses = json.loads(poses_path.read_text(encoding="utf-8")) if poses_path.exists() else {}
        last_pose = None
        for arm_pose_id, angle in pending:
            pose_name = "upper" if arm_pose_id == 0 else "lower"
            if arm_pose_id != last_pose:
                arm_command = arm_controller.move_to(pose_name)
                print(f"Arm -> {pose_name}: {arm_command.decode('ascii')}", flush=True)
                time.sleep(args.arm_settle_s)
                last_pose = arm_pose_id
            view_id = arm_pose_id * 12 + angle // 30
            command = controller.move_to(angle)
            sent_at = datetime.now().astimezone().isoformat()
            settle_s = args.return_settle_s if arm_pose_id == 1 and angle == 0 else args.motion_settle_s
            print(f"Moving to {angle} deg: {command.decode('ascii')}; waiting {settle_s:g} s", flush=True)
            time.sleep(settle_s)
            directory = reserve_view(args.output, args.object_id, args.session_id,
                                     view_id, args.repeat_id)
            metadata = {"created_at": datetime.now().astimezone().isoformat(),
                "protocol": "dense_dots_v3", "dot_cell_px": args.dot_cell_px,
                "dot_seed": args.dot_seed, "dot_frames": args.dot_frames,
                "pattern_order": [p.name for p in patterns],
                "pattern_sha256": {p.name: hashlib.sha256(p.image.tobytes()).hexdigest()
                                   for p in patterns},
                "projector_size_px": [1920, 1080], "display": display,
                "camera_size_px": list(size), "left_serial": args.left_serial,
                "right_serial": args.right_serial, "exposure_us": args.exposure_us,
                "gain_db": args.gain_db, "settle_ms": args.settle_ms,
                "view_id": view_id, "arm_pose_id": arm_pose_id,
                "arm_serial_command": arm_command.decode("ascii"),
                "localization_image": "localization/left.png",
                "turntable_deg_commanded": angle,
                "turntable_pwm_commanded": ANGLE_TO_PWM[angle],
                "turntable_serial_command": command.decode("ascii"),
                "turntable_command_sent_at": sent_at,
                "motion_settle_s": settle_s,
                "timestamp_note": "Camera ticks are from independent clocks; do not subtract"}
            localization_result = {}

            def on_black(left, right, stamps, row):
                localization = localize(left, board, K, D,
                                        args.min_board_corners, args.max_board_rms_px)
                for side, image in (("left", left), ("right", right)):
                    target = directory / "localization" / f"{side}.png"
                    target.parent.mkdir(exist_ok=True)
                    ok, encoded = cv2.imencode(".png", image)
                    if not ok:
                        raise RuntimeError(f"Cannot encode localization image: {target}")
                    target.write_bytes(encoded.tobytes())
                localization.update({"left_image": "localization/left.png",
                                     "right_image": "localization/right.png",
                                     "black_frame_left": row["left"],
                                     "black_frame_right": row["right"],
                                     "projection_pattern": "black", "device_ticks": stamps,
                                     "board_config": str(args.board.resolve())})
                _write_json(directory / "localization.json", localization)
                localization_result.update(localization)
                metadata["localization_rms_px"] = localization["reprojection_rms_px"]
                _write_json(directory / "pose.json", {"arm_pose_id": arm_pose_id,
                    "turntable_deg_commanded": angle, "turntable_pwm_commanded": ANGLE_TO_PWM[angle],
                    "turntable_command_sent_at": sent_at, "verified_pose": False,
                    "camera_pose_verified": True,
                    "localization_file": "localization.json",
                    "note": "Camera pose optically localized; turntable arrival still unverified"})
            try:
                result = capture_once({"destination": directory, "metadata": metadata,
                                       "on_black": on_black},
                    (cameras, converter, pylon, display, args.settle_ms, args.timeout_ms), patterns)
                camera_poses[f"view_{view_id:03d}/repeat_{args.repeat_id:03d}"] = \
                    localization_result["T_reference_from_left"]
                _write_json(poses_path, camera_poses)
                print(f"Complete {angle} deg: {result}", flush=True)
            except BaseException as exc:
                _write_json(directory / "failure.json", {"error": str(exc),
                    "time": datetime.now().astimezone().isoformat()})
                raise
        return_turntable_home(controller, session_root, args.return_settle_s)
    finally:
        if controller is not None:
            controller.connection.close()
        if arm_controller is not None:
            arm_controller.connection.close()
        if cameras is not None:
            close_cameras(cameras)
        cv2.destroyAllWindows()
    from reconstruct_view import main as reconstruct_view
    refresh_camera_poses(session_root, processed_root,
        selected_complete_sources(session_root, pose_ids, angles), board, K, D,
        args.min_board_corners, args.max_board_rms_px)
    return finish_session(args, session_root, processed_root, pose_ids, angles,
                          calibrations, reconstruct_view)


if __name__ == "__main__":
    raise SystemExit(main())
