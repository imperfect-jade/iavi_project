"""Capture one fixed ChArUco board at repeated upper/lower arm poses."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from capture import (DEFAULT_LEFT_SERIAL, DEFAULT_RIGHT_SERIAL, close_cameras,
                     grab_pair, open_cameras)
from arm import POSE_TARGETS, open_controller
from fixed_board import board_from_config, detect_corners_with_contrast_fallback

DEFAULT_BOARD = ROOT / "arm/targets/charuco_7x5_10mm_7mm_config.json"


def save_png(path, image):
    ok, data = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"PNG encoding failed: {path}")
    path.write_bytes(data.tobytes())


def annotate_corners(image, corners, ids):
    """Draw ChArUco IDs using OpenCV's expected two-channel point layout."""
    overlay = image.copy()
    if len(ids):
        points = np.asarray(corners, np.float32).reshape(-1, 1, 2)
        labels = np.asarray(ids, np.int32).reshape(-1, 1)
        cv2.aruco.drawDetectedCornersCharuco(overlay, points, labels)
    return overlay


def save_manifest(path, manifest):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--session-id", required=True)
    p.add_argument("--output-root", type=Path, default=ROOT / "raw/arm_fixed_board")
    p.add_argument("--board", type=Path, default=DEFAULT_BOARD)
    p.add_argument("--port", default="COM7", help="Arm controller; close Zide first")
    p.add_argument("--repeats", type=int, default=1, help="upper/lower cycles; final upper is also captured")
    p.add_argument("--left-serial", default=DEFAULT_LEFT_SERIAL)
    p.add_argument("--right-serial", default=DEFAULT_RIGHT_SERIAL)
    p.add_argument("--exposure-us", type=float, default=10000)
    p.add_argument("--gain-db", type=float, default=0)
    p.add_argument("--trigger", choices=("software", "hardware"), default="software")
    p.add_argument("--timeout-ms", type=int, default=5000)
    p.add_argument("--settle-s", type=float, default=6.0)
    p.add_argument("--min-corners", type=int, default=12)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--readback", action="store_true", help="Record six servo replies; optical check remains necessary")
    args = p.parse_args(argv)
    if not args.session_id or not all(c.isalnum() or c in "_-" for c in args.session_id):
        p.error("session-id must contain only letters, digits, _ or -")
    if args.repeats < 1 or args.retries < 1 or args.settle_s < 0 or args.timeout_ms < 1:
        p.error("Invalid repeats, retries, settle, or timeout")
    if not 4 <= args.min_corners <= 24:
        p.error("min-corners must be between 4 and 24")
    config = json.loads(args.board.read_text(encoding="utf-8"))
    board = board_from_config(config)
    session = args.output_root / args.session_id
    session.mkdir(parents=True, exist_ok=False)
    manifest_path = session / "capture_manifest.json"
    manifest = {"complete": False, "board": config, "board_config_path": str(args.board.resolve()),
                "camera_frame": "original_left_camera", "left_serial": args.left_serial,
                "right_serial": args.right_serial, "arm_port": args.port,
                "pose_targets": {key: list(value) for key, value in POSE_TARGETS.items()},
                "capture_settings": {"exposure_us": args.exposure_us, "gain_db": args.gain_db,
                                     "trigger": args.trigger, "settle_s": args.settle_s},
                "turntable_angle_deg": 0, "turntable_angle_verified": False,
                "observations": []}
    save_manifest(manifest_path, manifest)
    cameras = controller = None
    try:
        cameras, converter, pylon, size = open_cameras(args)
        manifest["camera_size_px"] = list(size)
        save_manifest(manifest_path, manifest)
        controller = open_controller(args.port)
        sequence = [pose for _ in range(args.repeats) for pose in ("upper", "lower")] + ["upper"]
        for index, pose in enumerate(sequence):
            command = controller.move_to(pose)
            print(f"{index:03d} {pose}: {command.decode('ascii')}; settle {args.settle_s:g} s", flush=True)
            time.sleep(args.settle_s)
            readback = [controller.read_position(i) for i in range(6)] if args.readback else None
            folder = session / f"capture_{index:03d}_{pose}"
            folder.mkdir()
            for attempt in range(1, args.retries + 1):
                left, right, stamps = grab_pair(cameras, converter, pylon,
                                                args.timeout_ms, args.trigger)
                left_corners, left_ids, left_method = detect_corners_with_contrast_fallback(
                    left, board, args.min_corners)
                right_corners, _, right_method = detect_corners_with_contrast_fallback(
                    right, board, args.min_corners)
                if len(left_corners) >= args.min_corners:
                    break
                print(f"  attempt {attempt}: left {len(left_corners)} / {args.min_corners} corners", flush=True)
            save_png(folder / "left.png", left)
            save_png(folder / "right.png", right)
            save_png(folder / "left_detected.png", annotate_corners(left, left_corners, left_ids))
            row = {"name": folder.name, "pose": pose, "left_image": f"{folder.name}/left.png",
                   "right_image": f"{folder.name}/right.png", "serial_command": command.decode("ascii"),
                   "readback_pwm": readback, "left_corners": len(left_corners),
                   "right_corners": len(right_corners),
                   "left_detection_method": left_method, "right_detection_method": right_method,
                   "device_ticks": stamps,
                   "captured_at": datetime.now().astimezone().isoformat()}
            manifest["observations"].append(row)
            save_manifest(manifest_path, manifest)
            print(f"  saved {folder.name}: left/right corners {len(left_corners)}/{len(right_corners)}", flush=True)
            if len(left_corners) < args.min_corners:
                raise RuntimeError(f"Board not sufficiently visible at {pose}; inspect {folder / 'left.png'}")
        manifest["complete"] = True
        save_manifest(manifest_path, manifest)
        print(f"Capture complete: {session}")
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        print(f"Capture stopped: {exc}; partial data: {session}", file=sys.stderr)
        return 1
    finally:
        if cameras is not None:
            close_cameras(cameras)
        if controller is not None:
            controller.connection.close()


if __name__ == "__main__":
    raise SystemExit(main())
