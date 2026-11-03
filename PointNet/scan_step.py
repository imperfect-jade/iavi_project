"""Acquire and reconstruct one action using the existing device modules."""
from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "arm"))
sys.path.insert(0, str(ROOT / "turntable"))
from capture import capture_once, reserve_view
from fixed_board import detect_corners_with_contrast_fallback, estimate_pose
from turntable import ANGLE_TO_PWM


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def localize_frame(left, board, K, D, min_corners=12, max_rms_px=3.0) -> dict:
    corners, ids, method = detect_corners_with_contrast_fallback(left, board, min_corners)
    if len(ids) < min_corners:
        raise ValueError(f"Only {len(ids)} fixed-board corners; need {min_corners}")
    transform, rms = estimate_pose(board, corners, ids, K, D)
    if not np.isfinite(rms) or rms > max_rms_px:
        raise ValueError(f"Fixed-board reprojection RMS {rms:.2f} px exceeds {max_rms_px:.2f} px")
    if np.asarray(transform).shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("Fixed-board pose is not a finite 4x4 transform")
    return {"coordinate_frame": "fixed_ChArUco_board",
            "T_reference_from_left": np.linalg.inv(transform).tolist(),
            "T_left_from_board": transform.tolist(), "corners": len(ids),
            "detection_method": method, "reprojection_rms_px": rms}


def turntable_settle_seconds(action_id: int, previous_action: int | None,
                             normal_s: float, return_s: float) -> float:
    if action_id % 12 == 0 and previous_action is not None and previous_action % 12 != 0:
        return float(return_s)
    return float(normal_s)


def capture_and_reconstruct(action_id, session_context, devices, reconstruct=None) -> Path:
    if not 0 <= action_id < 24:
        raise ValueError("action_id must be 0..23")
    from reconstruct_view import main as default_reconstruct
    reconstruct = reconstruct or default_reconstruct
    c = session_context
    pose_id, angle = action_id // 12, (action_id % 12) * 30
    pose_name = "upper" if pose_id == 0 else "lower"
    raw_root = Path(c["raw_root"])
    processed_root = Path(c["processed_root"])
    session_root = raw_root / c["object_id"] / c["session_id"]
    repeat_id = int(c["repeat_id"])
    destination = session_root / f"view_{action_id:03d}" / f"repeat_{repeat_id:03d}"
    target = processed_root / c["object_id"] / c["session_id"] / destination.parent.name / destination.name
    if destination.exists() or target.exists():
        raise FileExistsError(destination if destination.exists() else target)
    arm_command = devices["arm"].move_to(pose_name)
    devices.get("sleep", time.sleep)(float(c["arm_settle_s"]))
    table_command = devices["turntable"].move_to(angle)
    sent_at = datetime.now().astimezone().isoformat()
    settle_s = turntable_settle_seconds(action_id, c.get("previous_action"),
                                       c["motion_settle_s"], c.get("return_settle_s", 12.0))
    devices.get("sleep", time.sleep)(settle_s)
    destination = reserve_view(raw_root, c["object_id"], c["session_id"], action_id, repeat_id)
    patterns = c["patterns"]
    calibration = Path(c["calibrations"][pose_id]).resolve()
    metadata = {"created_at": datetime.now().astimezone().isoformat(),
                "protocol": "dense_dots_v3", "dot_cell_px": c["dot_cell_px"],
                "dot_seed": c["dot_seed"], "dot_frames": c["dot_frames"],
                "pattern_order": [getattr(p, "name", "pattern") for p in patterns],
                "pattern_sha256": {p.name: hashlib.sha256(p.image.tobytes()).hexdigest()
                                   for p in patterns if hasattr(p, "name") and hasattr(p, "image")},
                "projector_size_px": [1920, 1080], "display": c.get("display"),
                "camera_size_px": list(c.get("camera_size", ())),
                "left_serial": c["left_serial"], "right_serial": c["right_serial"],
                "exposure_us": c["exposure_us"], "gain_db": c["gain_db"],
                "settle_ms": c["settle_ms"], "view_id": action_id,
                "arm_pose_id": pose_id, "arm_serial_command": arm_command.decode("ascii"),
                "localization_image": "localization/left.png",
                "turntable_deg_commanded": angle,
                "turntable_pwm_commanded": ANGLE_TO_PWM[angle],
                "turntable_serial_command": table_command.decode("ascii"),
                "turntable_command_sent_at": sent_at,
                "motion_settle_s": settle_s,
                "timestamp_note": "Camera ticks are from independent clocks; do not subtract"}
    localization_result = {}

    def on_black(left, right, stamps, row):
        import cv2
        result = devices.get("localize", localize_frame)(
            left, c["board"], c["K"], c["D"],
            c["min_board_corners"], c["max_board_rms_px"])
        for side, image in (("left", left), ("right", right)):
            target = destination / "localization" / f"{side}.png"
            target.parent.mkdir(exist_ok=True)
            ok, encoded = cv2.imencode(".png", image)
            if not ok:
                raise RuntimeError(f"Cannot encode localization image: {target}")
            target.write_bytes(encoded.tobytes())
        result.update({"left_image": "localization/left.png",
                       "right_image": "localization/right.png",
                       "black_frame_left": row["left"], "black_frame_right": row["right"],
                       "projection_pattern": "black", "device_ticks": stamps,
                       "board_config": str(Path(c["board_path"]).resolve())})
        _write_json(destination / "localization.json", result)
        localization_result.update(result)
        metadata["localization_rms_px"] = result["reprojection_rms_px"]
        _write_json(destination / "pose.json", {
            "arm_pose_id": pose_id, "turntable_deg_commanded": angle,
            "turntable_pwm_commanded": ANGLE_TO_PWM[angle],
            "turntable_command_sent_at": sent_at, "verified_pose": False,
            "camera_pose_verified": True, "localization_file": "localization.json"})

    try:
        devices.get("capture", capture_once)(
            {"destination": destination, "metadata": metadata, "on_black": on_black},
            devices["capture_devices"], patterns)
        manifest_path = destination / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not manifest.get("complete") or not localization_result:
            raise RuntimeError("capture did not complete with valid localization")
        poses_path = session_root / "camera_poses.json"
        poses = json.loads(poses_path.read_text(encoding="utf-8")) if poses_path.is_file() else {}
        poses[f"{destination.parent.name}/{destination.name}"] = localization_result["T_reference_from_left"]
        _write_json(poses_path, poses)
        target.mkdir(parents=True, exist_ok=True)
        fingerprint = {"source_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                       "calibration_sha256": hashlib.sha256(calibration.read_bytes()).hexdigest(),
                       "min_depth_mm": None, "max_depth_mm": None}
        _write_json(target / "reconstruction_status.json", {**fingerprint, "state": "running"})
        argv = ["--view", str(destination), "--output", str(target),
                "--calibration", str(calibration)]
        if reconstruct(argv) != 0 or not (target / "cloud_left.ply").is_file() or not (target / "qc.json").is_file():
            raise RuntimeError("reconstruction did not produce cloud_left.ply and qc.json")
        qc = json.loads((target / "qc.json").read_text(encoding="utf-8"))
        if qc.get("point_count", 0) < 1:
            raise ValueError("reconstruction has no points")
        _write_json(target / "reconstruction_status.json", {
            **fingerprint, "state": "complete", "point_count": qc["point_count"],
            "warnings": qc.get("warnings", [])})
        return destination
    except BaseException as exc:
        _write_json(destination / "failure.json", {"error": str(exc),
                                                   "time": datetime.now().astimezone().isoformat()})
        if target.exists():
            _write_json(target / "reconstruction_status.json", {"state": "failed", "error": str(exc)})
        raise
