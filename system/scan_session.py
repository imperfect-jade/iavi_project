"""Capture an arm/turntable scan, then reconstruct its views one by one."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "turntable"))
sys.path.insert(0, str(ROOT / "arm"))
from turntable import ANGLE_TO_PWM, SETTLE_WAIT_S, open_controller
from arm import open_controller as open_arm_controller
from fixed_board import (board_from_config, detect_corners_with_contrast_fallback,
                         estimate_pose, relative_transform)


def angles_for_range(start: int, end: int, step: int) -> list[int]:
    if step < 30 or step % 30 or start not in ANGLE_TO_PWM or end not in ANGLE_TO_PWM or start > end:
        raise ValueError("Use ascending tested angles 0..330 and a positive multiple-of-30 step")
    values = list(range(start, end + 1, step))
    if values[-1] != end:
        raise ValueError("Step must land exactly on end-angle")
    return values


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
                manifest.get("protocol") == "dense_dots_v2" and
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


def prepare_reconstruction_calibration(session_root: Path, processed_root: Path,
                                       pose_ids: list[int], angles: list[int],
                                       base_calibration: Path, board: Path,
                                       enabled: bool = True) -> Path:
    """Validate a session board fit before allowing its views to reconstruct."""
    if not enabled:
        return base_calibration
    from board_stereo_correction import derive_session_calibration
    sources = selected_complete_sources(session_root, pose_ids, angles)
    return derive_session_calibration(sources, base_calibration, board,
                                      processed_root / "board_stereo_correction")


def reconstruct_session(session_root: Path, processed_root: Path, pose_ids: list[int],
                        angles: list[int], calibration: Path, reconstruct,
                        min_depth_mm: float | None = None,
                        max_depth_mm: float | None = None) -> int:
    """Run reconstruction after capture; preserve each view's result for retry."""
    from capture import _write_json

    processed_root.mkdir(parents=True, exist_ok=True)
    calibration = calibration.resolve()
    calibration_hash = hashlib.sha256(calibration.read_bytes()).hexdigest()
    summary = {"complete": 0, "skipped": 0, "failed": 0,
               "calibration": str(calibration),
               "calibration_sha256": calibration_hash, "views": {}}
    for source in selected_complete_sources(session_root, pose_ids, angles):
        view_root = source.parent
        name = f"{view_root.name}/{source.name}"
        target = processed_root / view_root.name / source.name
        target.mkdir(parents=True, exist_ok=True)
        status_path = target / "reconstruction_status.json"
        fingerprint = {"source_manifest_sha256": hashlib.sha256(
            (source / "manifest.json").read_bytes()).hexdigest(),
            "calibration_sha256": calibration_hash,
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


def load_reference(path: Path, board_config: dict, stereo_path: Path):
    import numpy as np
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("board") != board_config or \
       Path(data["stereo_calibration"]).resolve() != stereo_path.resolve():
        raise ValueError("Fixed-board reference differs from board or stereo calibration")
    name = data["reference_capture"]
    row = next(item for item in data["observations"] if item["name"] == name)
    if row["pose"] != "lower":
        raise ValueError("Reference capture must be lower pose")
    return np.asarray(row["T_left_from_board"], dtype=np.float64)


def localize(left, board, K, D, reference, min_corners=12, max_rms_px=3.0):
    corners, ids, method = detect_corners_with_contrast_fallback(left, board, min_corners)
    if len(ids) < min_corners:
        raise ValueError(f"Only {len(ids)} fixed-board corners; need {min_corners}")
    current, rms = estimate_pose(board, corners, ids, K, D)
    if rms > max_rms_px:
        raise ValueError(f"Fixed-board reprojection RMS {rms:.2f} px exceeds {max_rms_px:.2f} px")
    return {"T_reference_from_left": relative_transform(reference, current).tolist(),
            "T_left_from_board": current.tolist(), "corners": len(ids),
            "detection_method": method, "reprojection_rms_px": rms}


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
    p.add_argument("--reference-calibration", type=Path,
                   default=ROOT / "calibration/arm_fixed_board/board_repeat_01/two_pose_calibration.json")
    p.add_argument("--board", type=Path,
                   default=ROOT / "arm/targets/charuco_7x5_10mm_7mm_config.json")
    p.add_argument("--stereo", type=Path,
                   default=ROOT / "calibration/stereo/stereo_calibration.json")
    p.add_argument("--min-board-corners", type=int, default=12)
    p.add_argument("--max-board-rms-px", type=float, default=3.0)
    p.add_argument("--repeat-id", type=int, default=1)
    p.add_argument("--resume", action="store_true", help="Skip complete views in the same session")
    p.add_argument("--output", type=Path, default=ROOT / "raw")
    p.add_argument("--processed-output", type=Path, default=ROOT / "processed")
    p.add_argument("--reconstruction-min-depth-mm", type=float)
    p.add_argument("--reconstruction-max-depth-mm", type=float)
    p.add_argument("--no-board-stereo-correction", action="store_true",
                   help="Use the original stereo calibration without fitting the scan's paired fixed-board images")
    p.add_argument("--left-serial", default="40031431")
    p.add_argument("--right-serial", default="40031435")
    p.add_argument("--display", help="Exact projector display name if its location changed")
    p.add_argument("--exposure-us", type=float, default=10000)
    p.add_argument("--gain-db", type=float, default=0)
    p.add_argument("--timeout-ms", type=int, default=5000)
    p.add_argument("--settle-ms", type=int, default=250, help="Pattern stabilization time")
    p.add_argument("--motion-settle-s", type=float, default=SETTLE_WAIT_S)
    p.add_argument("--return-settle-s", type=float, default=12.0,
                   help="Wait after 330-to-0 motion between arm poses")
    p.add_argument("--dot-cell-px", type=int, default=8)
    p.add_argument("--dot-seed", type=int, default=5)
    p.add_argument("--dot-frames", type=int, default=5)
    return p


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    from capture import (Pattern, PREVIEW, PROJECTOR, SAFE, _write_json, capture_once,
                         close_cameras, displays, grab_pair, make_dense_dots, open_cameras,
                         preview_pair, reserve_view, select_projector_display, show_pattern)
    import cv2
    import numpy as np

    angles = angles_for_range(args.start_angle, args.end_angle, args.step_angle)
    pose_ids = [args.arm_pose_id] if args.arm_pose_id is not None else [0, 1]
    if args.port == args.arm_port or args.arm_settle_s < 0 or \
       not 4 <= args.min_board_corners <= 24 or args.max_board_rms_px <= 0:
        raise ValueError("Invalid arm port, settle time, or fixed-board quality limits")
    board_config = json.loads(args.board.read_text(encoding="utf-8"))
    board = board_from_config(board_config)
    stereo = json.loads(args.stereo.read_text(encoding="utf-8"))
    if (args.left_serial, args.right_serial) != (stereo["left_serial"], stereo["right_serial"]):
        raise ValueError("Camera serials differ from stereo calibration")
    reference = load_reference(args.reference_calibration, board_config, args.stereo)
    K = np.asarray(stereo["K_left"], dtype=np.float64)
    D = np.asarray(stereo["D_left"], dtype=np.float64)
    if not SAFE.fullmatch(args.object_id) or not SAFE.fullmatch(args.session_id):
        raise ValueError("object-id and session-id require letters, digits, _ or -")
    if args.repeat_id < 1 or args.motion_settle_s < 0 or args.return_settle_s < 0 or args.settle_ms < 1 or \
       args.timeout_ms < 1 or args.exposure_us <= 0 or not 1 <= args.dot_frames <= 32:
        raise ValueError("Invalid repeat, timing, exposure, or dot-frame value")
    if (args.reconstruction_min_depth_mm is None) != (args.reconstruction_max_depth_mm is None):
        raise ValueError("Set both reconstruction depth bounds or neither")
    if args.reconstruction_min_depth_mm is not None and not (0 < args.reconstruction_min_depth_mm < args.reconstruction_max_depth_mm):
        raise ValueError("Invalid reconstruction depth range")
    calibration_npz = args.stereo.with_suffix(".npz")
    if not calibration_npz.is_file():
        raise FileNotFoundError(f"Stereo reconstruction calibration missing: {calibration_npz}")
    args.trigger = "software"
    session_root = args.output / args.object_id / args.session_id
    plan = {"angles_deg": angles, "arm_pose_ids": pose_ids,
            "protocol": "dense_dots_v2", "dot_cell_px": args.dot_cell_px,
            "dot_seed": args.dot_seed, "dot_frames": args.dot_frames,
            "left_serial": args.left_serial, "right_serial": args.right_serial,
            "exposure_us": args.exposure_us, "gain_db": args.gain_db,
            "port": args.port, "arm_port": args.arm_port,
            "arm_settle_s": args.arm_settle_s,
            "reference_calibration": str(args.reference_calibration.resolve()),
            "board": board_config, "stereo": str(args.stereo.resolve()),
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
        reconstruction_calibration = prepare_reconstruction_calibration(
            session_root, processed_root, pose_ids, angles, calibration_npz,
            args.board, enabled=not args.no_board_stereo_correction)
        return reconstruct_session(session_root, processed_root, pose_ids, angles,
                                   reconstruction_calibration, reconstruct_view,
                                   args.reconstruction_min_depth_mm,
                                   args.reconstruction_max_depth_mm)

    display = select_projector_display(displays(), args.display)
    patterns = make_dense_dots(dot_cell=args.dot_cell_px, seed=args.dot_seed,
                               count=args.dot_frames)
    cameras = controller = arm_controller = None
    try:
        cameras, converter, pylon, size = open_cameras(args)
        if list(size) != list(stereo["image_size"]):
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
            try:
                show_pattern(Pattern("black", np.zeros_like(patterns[0].image)), display)
                cv2.waitKey(args.settle_ms)
                left, right, stamps = grab_pair(cameras, converter, pylon, args.timeout_ms)
                for side, image in (("left", left), ("right", right)):
                    target = directory / "localization" / f"{side}.png"
                    target.parent.mkdir(exist_ok=True)
                    ok, encoded = cv2.imencode(".png", image)
                    if not ok:
                        raise RuntimeError(f"Cannot encode localization image: {target}")
                    target.write_bytes(encoded.tobytes())
                localization = localize(left, board, K, D, reference,
                                        args.min_board_corners, args.max_board_rms_px)
                localization.update({"left_image": "localization/left.png",
                                     "right_image": "localization/right.png", "device_ticks": stamps,
                                     "reference_calibration": str(args.reference_calibration.resolve())})
                _write_json(directory / "localization.json", localization)
            except BaseException as exc:
                _write_json(directory / "failure.json", {"error": str(exc),
                    "time": datetime.now().astimezone().isoformat()})
                raise
            metadata = {"created_at": datetime.now().astimezone().isoformat(),
                "protocol": "dense_dots_v2", "dot_cell_px": args.dot_cell_px,
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
                "localization_rms_px": localization["reprojection_rms_px"],
                "turntable_deg_commanded": angle,
                "turntable_pwm_commanded": ANGLE_TO_PWM[angle],
                "turntable_serial_command": command.decode("ascii"),
                "turntable_command_sent_at": sent_at,
                "motion_settle_s": settle_s,
                "timestamp_note": "Camera ticks are from independent clocks; do not subtract"}
            _write_json(directory / "pose.json", {"arm_pose_id": arm_pose_id,
                "turntable_deg_commanded": angle, "turntable_pwm_commanded": ANGLE_TO_PWM[angle],
                "turntable_command_sent_at": sent_at, "verified_pose": False,
                "camera_pose_verified": True,
                "localization_file": "localization.json",
                "note": "Camera pose optically localized; turntable arrival still unverified"})
            try:
                result = capture_once({"destination": directory, "metadata": metadata},
                    (cameras, converter, pylon, display, args.settle_ms, args.timeout_ms), patterns)
                camera_poses[f"view_{view_id:03d}/repeat_{args.repeat_id:03d}"] = localization["T_reference_from_left"]
                _write_json(poses_path, camera_poses)
                print(f"Complete {angle} deg: {result}", flush=True)
            except BaseException as exc:
                _write_json(directory / "failure.json", {"error": str(exc),
                    "time": datetime.now().astimezone().isoformat()})
                raise
    finally:
        if controller is not None:
            controller.connection.close()
        if arm_controller is not None:
            arm_controller.connection.close()
        if cameras is not None:
            close_cameras(cameras)
        cv2.destroyAllWindows()
    from reconstruct_view import main as reconstruct_view
    reconstruction_calibration = prepare_reconstruction_calibration(
        session_root, processed_root, pose_ids, angles, calibration_npz,
        args.board, enabled=not args.no_board_stereo_correction)
    return reconstruct_session(session_root, processed_root, pose_ids, angles,
                               reconstruction_calibration, reconstruct_view,
                               args.reconstruction_min_depth_mm,
                               args.reconstruction_max_depth_mm)


if __name__ == "__main__":
    raise SystemExit(main())
