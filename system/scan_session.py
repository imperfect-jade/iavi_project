"""Coordinate turntable motion and one complete stereo capture per view."""
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
from turntable import ANGLE_TO_PWM, SETTLE_WAIT_S, open_controller


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
        complete = False
        for folder in repeats:
            manifest_path = folder / "manifest.json"
            if not manifest_path.exists():
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            frames = manifest.get("frames", [])
            files_present = bool(frames) and all(
                (folder / row[side]).is_file() for row in frames for side in ("left", "right"))
            if files_present and manifest.get("frame_count") == len(frames) and \
               manifest.get("complete") and manifest.get("view_id") == view_id and \
               manifest.get("arm_pose_id") == arm_pose_id and \
               manifest.get("turntable_deg_commanded") == angle and \
               manifest.get("protocol") == "dense_dots_v2":
                complete = True
                break
        if complete:
            continue
        destination = view_root / f"repeat_{repeat_id:03d}"
        if destination.exists():
            raise FileExistsError(f"{destination} is incomplete; choose a new --repeat-id")
        pending.append(angle)
    return pending


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", default="COM6")
    p.add_argument("--start-angle", type=int, required=True)
    p.add_argument("--end-angle", type=int, required=True)
    p.add_argument("--step-angle", type=int, default=30)
    p.add_argument("--object-id", required=True)
    p.add_argument("--session-id", required=True)
    p.add_argument("--arm-pose-id", type=int, choices=(0, 1, 2), required=True)
    p.add_argument("--repeat-id", type=int, default=1)
    p.add_argument("--resume", action="store_true", help="Skip complete views in the same session")
    p.add_argument("--output", type=Path, default=ROOT / "raw")
    p.add_argument("--left-serial", default="40031431")
    p.add_argument("--right-serial", default="40031435")
    p.add_argument("--display", help="Exact projector display name if its location changed")
    p.add_argument("--exposure-us", type=float, default=10000)
    p.add_argument("--gain-db", type=float, default=0)
    p.add_argument("--timeout-ms", type=int, default=5000)
    p.add_argument("--settle-ms", type=int, default=250, help="Pattern stabilization time")
    p.add_argument("--motion-settle-s", type=float, default=SETTLE_WAIT_S)
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
    if not SAFE.fullmatch(args.object_id) or not SAFE.fullmatch(args.session_id):
        raise ValueError("object-id and session-id require letters, digits, _ or -")
    if args.repeat_id < 1 or args.motion_settle_s < 0 or args.settle_ms < 1 or \
       args.timeout_ms < 1 or args.exposure_us <= 0 or not 1 <= args.dot_frames <= 32:
        raise ValueError("Invalid repeat, timing, exposure, or dot-frame value")
    args.trigger = "software"
    session_root = args.output / args.object_id / args.session_id
    plan = {"angles_deg": angles, "arm_pose_id": args.arm_pose_id,
            "protocol": "dense_dots_v2", "dot_cell_px": args.dot_cell_px,
            "dot_seed": args.dot_seed, "dot_frames": args.dot_frames,
            "left_serial": args.left_serial, "right_serial": args.right_serial,
            "exposure_us": args.exposure_us, "gain_db": args.gain_db,
            "port": args.port, "motion_settle_s": args.motion_settle_s,
            "pattern_settle_ms": args.settle_ms, "timeout_ms": args.timeout_ms,
            "display_name": args.display}
    plan_path = session_root / "scan_plan.json"
    if args.resume:
        if not plan_path.exists() or json.loads(plan_path.read_text(encoding="utf-8")) != plan:
            raise ValueError("--resume requires a matching scan_plan.json; use a new session-id")
    elif session_root.exists():
        raise FileExistsError(f"{session_root} exists; use --resume or a new session-id")
    pending = pending_views(session_root, angles, args.arm_pose_id, args.repeat_id, args.resume)
    if not pending:
        print("All requested views already complete.")
        return 0

    display = select_projector_display(displays(), args.display)
    patterns = make_dense_dots(dot_cell=args.dot_cell_px, seed=args.dot_seed,
                               count=args.dot_frames)
    cameras = controller = None
    try:
        cameras, converter, pylon, size = open_cameras(args)
        controller = open_controller(args.port)
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

        for angle in pending:
            view_id = args.arm_pose_id * 12 + angle // 30
            command = controller.move_to(angle)
            sent_at = datetime.now().astimezone().isoformat()
            print(f"Moving to {angle} deg: {command.decode('ascii')}", flush=True)
            time.sleep(args.motion_settle_s)
            directory = reserve_view(args.output, args.object_id, args.session_id,
                                     view_id, args.repeat_id)
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
                "view_id": view_id, "arm_pose_id": args.arm_pose_id,
                "turntable_deg_commanded": angle,
                "turntable_pwm_commanded": ANGLE_TO_PWM[angle],
                "turntable_serial_command": command.decode("ascii"),
                "turntable_command_sent_at": sent_at,
                "motion_settle_s": args.motion_settle_s,
                "timestamp_note": "Camera ticks are from independent clocks; do not subtract"}
            _write_json(directory / "pose.json", {"arm_pose_id": args.arm_pose_id,
                "turntable_deg_commanded": angle, "turntable_pwm_commanded": ANGLE_TO_PWM[angle],
                "turntable_command_sent_at": sent_at, "verified_pose": False,
                "note": "Serial write and wait do not verify physical arrival"})
            try:
                result = capture_once({"destination": directory, "metadata": metadata},
                    (cameras, converter, pylon, display, args.settle_ms, args.timeout_ms), patterns)
                print(f"Complete {angle} deg: {result}", flush=True)
            except BaseException as exc:
                _write_json(directory / "failure.json", {"error": str(exc),
                    "time": datetime.now().astimezone().isoformat()})
                raise
        return 0
    finally:
        if controller is not None:
            controller.connection.close()
        if cameras is not None:
            close_cameras(cameras)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
