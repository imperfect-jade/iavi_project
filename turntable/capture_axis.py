"""Capture the four-face ArUco cuboid at twelve angles and return to zero."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from capture import (DEFAULT_LEFT_SERIAL, DEFAULT_RIGHT_SERIAL, close_cameras,
                     grab_pair, open_cameras)
from calibrate_axis import DEFAULT_BOARD
from turntable import ANGLE_TO_PWM, SETTLE_WAIT_S, open_controller


def detect_ids(image, config):
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, config["dictionary"]))
    _, ids, _ = cv2.aruco.ArucoDetector(dictionary).detectMarkers(
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    return set() if ids is None else {int(i) for i in ids.ravel()
                                      if str(int(i)) in config["markers"]}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", default="COM6")
    p.add_argument("--session-id", required=True)
    p.add_argument("--output-root", type=Path, default=ROOT / "raw/turntable_axis")
    p.add_argument("--board", type=Path, default=DEFAULT_BOARD)
    p.add_argument("--left-serial", default=DEFAULT_LEFT_SERIAL)
    p.add_argument("--right-serial", default=DEFAULT_RIGHT_SERIAL)
    p.add_argument("--exposure-us", type=float, default=10000)
    p.add_argument("--gain-db", type=float, default=0)
    p.add_argument("--trigger", choices=("software", "hardware"), default="software")
    p.add_argument("--timeout-ms", type=int, default=5000)
    p.add_argument("--settle-s", type=float, default=SETTLE_WAIT_S)
    p.add_argument("--min-markers", type=int, default=2)
    p.add_argument("--retries", type=int, default=3)
    args = p.parse_args(argv)
    if not args.session_id or not all(c.isalnum() or c in "_-" for c in args.session_id):
        p.error("session-id must contain only letters, digits, _ or -")
    if args.settle_s < 0 or args.retries < 1 or args.min_markers < 1:
        p.error("Invalid settle, retries, or minimum markers")
    config = json.loads(args.board.read_text(encoding="utf-8"))
    if config.get("target_type") != "four_face_aruco_cuboid":
        p.error("--board must be a measured four-face cuboid config")
    target = args.output_root / args.session_id
    target.mkdir(parents=True, exist_ok=False)
    cameras = controller = None
    rows = []
    try:
        cameras, converter, pylon, size = open_cameras(args)
        controller = open_controller(args.port)
        for index, angle in enumerate([*ANGLE_TO_PWM, 0]):
            returned = index == len(ANGLE_TO_PWM)
            command = controller.move_to(angle)
            print(f"{angle:3d} deg: sent {command.decode('ascii')}; waiting {args.settle_s:g} s", flush=True)
            time.sleep(args.settle_s)
            for attempt in range(1, args.retries + 1):
                left, right, stamps = grab_pair(cameras, converter, pylon,
                                                args.timeout_ms, args.trigger)
                left_ids, right_ids = detect_ids(left, config), detect_ids(right, config)
                common_ids = left_ids & right_ids
                if len(common_ids) >= args.min_markers:
                    break
                print(f"  attempt {attempt}: markers left={len(left_ids)}, "
                      f"right={len(right_ids)}, common={len(common_ids)}", flush=True)
                time.sleep(0.3)
            else:
                raise RuntimeError(f"Board not visible at {angle} deg; adjust lighting/focus and use a new session")
            folder = target / ("angle_000_return" if returned else f"angle_{angle:03d}")
            folder.mkdir()
            for name, image in (("left", left), ("right", right)):
                ok, encoded = cv2.imencode(".png", image)
                if not ok:
                    raise RuntimeError(f"PNG encoding failed: {angle} {name}")
                (folder / f"{name}.png").write_bytes(encoded.tobytes())
            rows.append({"command_deg": angle, "return_to_zero": returned,
                         "pwm": ANGLE_TO_PWM[angle],
                         "serial_command": command.decode("ascii"),
                         "left_markers": len(left_ids), "right_markers": len(right_ids),
                         "common_markers": len(common_ids),
                         "left_ids": sorted(left_ids), "right_ids": sorted(right_ids),
                         "device_ticks": stamps, "captured_at": datetime.now().astimezone().isoformat()})
            print(f"  saved left/right PNG; markers left/right/common "
                  f"{len(left_ids)}/{len(right_ids)}/{len(common_ids)}", flush=True)
        manifest = {"complete": True, "camera_frame": "original_left_camera",
                    "camera_size_px": size, "left_serial": args.left_serial,
                    "right_serial": args.right_serial, "port": args.port,
                    "board": config, "angles": rows,
                    "arrival_verified": False,
                    "note": "Serial write and settle delay do not verify physical arrival."
                    }
        (target / "capture_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Captured 12 angles plus return to zero -> {target}")
        return 0
    finally:
        if cameras is not None:
            close_cameras(cameras)
        if controller is not None:
            controller.connection.close()


if __name__ == "__main__":
    raise SystemExit(main())
