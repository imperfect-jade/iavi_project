"""PointNet-guided real scan: acquire, reconstruct, fuse visited views, repeat."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_session(config: dict, services: dict) -> dict:
    """Run the policy state machine. Services supply real or replayed I/O."""
    session_root = Path(config["session_root"]).resolve()
    output_root = Path(config["output_root"]).resolve()
    state_path = output_root / "state.json"
    max_scans = int(config.get("max_scans", 10))
    available = list(config.get("available", range(24)))
    if (not 2 <= max_scans <= 24 or len(set(available)) != len(available) or
            any(not 0 <= a < 24 for a in available) or not {0, 6}.issubset(available)):
        raise ValueError("invalid budget or available actions; initial actions 0 and 6 are required")
    enabled = bool(config.get("score_stop_enabled", False))
    threshold = config.get("score_threshold")
    if enabled and (threshold is None or not np.isfinite(threshold)):
        raise ValueError("enabled score stop requires a finite threshold")
    fingerprint = {"session_root": str(session_root), "max_scans": max_scans,
                   "available": available, "score_stop_enabled": enabled,
                   "score_threshold": threshold if enabled else None,
                   "source_fingerprint": config.get("source_fingerprint", {})}
    if config.get("resume"):
        if state_path.is_file():
            state = validate_resume_state(state_path, fingerprint)
            if state.get("phase") == "complete":
                return state
            state.pop("error", None)
        elif (session_root / "scan_plan.json").is_file():
            state = {"config": fingerprint, "visited": [], "repeats": [],
                     "decisions": [], "phase": "running"}
            _write_json(state_path, state)
        else:
            raise FileNotFoundError("resume requires state.json or an existing scan_plan.json")
    else:
        if state_path.exists():
            raise FileExistsError("state.json exists; use --resume or a new session")
        state = {"config": fingerprint, "visited": [], "repeats": [],
                 "decisions": [], "phase": "running"}
        _write_json(state_path, state)
    try:
        while True:
            visited = state["visited"]
            if len(visited) < 2:
                action = [0, 6][len(visited)]
            else:
                step_dir = output_root / "steps" / f"step_{len(visited):02d}"
                fusion_dir = services["fuse"]([Path(p) for p in state["repeats"]],
                                               step_dir / "fusion")
                if len(visited) >= max_scans:
                    state["stop_reason"] = "budget"
                    break
                eligible = set(available) - set(visited)
                if not eligible:
                    state["stop_reason"] = "no_available_action"
                    break
                prediction = services["predict"](fusion_dir, list(visited), step_dir)
                action = int(prediction["action_id"])
                if action not in eligible:
                    raise ValueError(f"policy selected unavailable or visited action {action}")
                score = float(prediction["best_score"])
                if not np.isfinite(score):
                    raise ValueError("policy selected nonfinite score")
                decision = {"visited_before": list(visited), "prediction": prediction,
                            "fused_dir": str(fusion_dir)}
                state["decisions"].append(decision)
                _write_json(state_path, state)
                if enabled and score < float(threshold):
                    state["stop_reason"] = "low_predicted_gain"
                    break
            recovered = services["recover"](action) if "recover" in services else None
            repeat = Path(recovered if recovered is not None else services["capture"](action)).resolve()
            if not repeat.is_dir() or repeat.parent.name != f"view_{action:03d}":
                raise ValueError(f"capture did not return a complete repeat for action {action}")
            state["visited"].append(action)
            state["repeats"].append(str(repeat))
            _write_json(state_path, state)
        final_dir = Path(config.get("final_fusion_dir", output_root / "final_fusion"))
        state["final_fusion_dir"] = str(services["fuse"](
            [Path(p) for p in state["repeats"]], final_dir))
        if "home" in services:
            services["home"]()
        state["phase"] = "complete"
        _write_json(state_path, state)
        return state
    except BaseException as exc:
        state["phase"] = "failed"
        state["error"] = str(exc)
        _write_json(state_path, state)
        raise


def _load_calibrations(upper: Path, lower: Path, serials: tuple[str, str]):
    sys.path.insert(0, str(ROOT / "code"))
    from fuse_views import validate_stereo_compatibility
    paths = {0: upper.with_suffix(".npz").resolve(), 1: lower.with_suffix(".npz").resolve()}
    validate_stereo_compatibility(paths.values(),
        {"frame": "fixed_ChArUco_board", "stereo_calibration": paths[1]}, True)
    upper_meta = json.loads(paths[0].with_suffix(".json").read_text(encoding="utf-8"))
    lower_meta = json.loads(paths[1].with_suffix(".json").read_text(encoding="utf-8"))
    if upper_meta.get("lower_sha256") != hashlib.sha256(paths[1].read_bytes()).hexdigest():
        raise ValueError("upper stereo calibration does not match lower stereo calibration")
    if serials != (lower_meta["left_serial"], lower_meta["right_serial"]):
        raise ValueError("camera serials differ from stereo calibration")
    with np.load(paths[1]) as archive:
        return paths, archive["K_left"].copy(), archive["D_left"].copy(), tuple(archive["image_size"])


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_resume_state(state_path: Path, expected_config: dict) -> dict:
    state = json.loads(Path(state_path).read_text(encoding="utf-8"))
    if state.get("config") != expected_config:
        raise ValueError("resume configuration differs from saved session")
    session_root = Path(expected_config["session_root"])
    visited, repeats = state.get("visited", []), state.get("repeats", [])
    if (len(visited) != len(repeats) or len(visited) != len(set(visited)) or
            visited[:min(len(visited), 2)] != [0, 6][:min(len(visited), 2)] or
            len(visited) > expected_config["max_scans"] or
            not set(visited).issubset(expected_config["available"]) or
            any(not Path(p).is_dir() or Path(p).parent.name != f"view_{a:03d}" or
                Path(p).resolve().parent.parent != session_root
                for a, p in zip(visited, repeats))):
        raise ValueError("saved action and repeat history is inconsistent")
    return state


def validate_live_settings(args) -> None:
    if (not 4 <= args.min_board_corners <= 24 or
            not np.isfinite(args.max_board_rms_px) or args.max_board_rms_px <= 0):
        raise ValueError("invalid fixed-board localization quality limits")
    height = args.fusion_height_mm
    if (not np.isfinite(args.fusion_radius_mm) or args.fusion_radius_mm <= 0 or
            not np.isfinite(args.fusion_voxel_mm) or args.fusion_voxel_mm <= 0 or
            len(height) != 2 or not np.isfinite(height).all() or height[0] >= height[1]):
        raise ValueError("invalid fusion crop or voxel size")
    manual_height = getattr(args, "table_height_axis_mm", None)
    if manual_height is not None and (not np.isfinite(manual_height) or
                                      manual_height >= height[1]):
        raise ValueError("manual table height must be below fusion upper crop")


def live_input_fingerprint(paths: dict[str, Path], crop: dict,
                           table_height_axis_mm: float | None, processed_root: Path) -> dict:
    return {**{f"{name}_sha256": _sha256(path) for name, path in paths.items()},
            "fusion_crop": json.loads(json.dumps(crop)),
            "table_height_axis_mm": (None if table_height_axis_mm is None
                                     else float(table_height_axis_mm)),
            "processed_root": str(Path(processed_root).resolve())}


def requested_fusion_crop(radius_mm: float, height_mm, voxel_mm: float) -> dict:
    """Keep every requested crop bound in the resume fingerprint."""
    return {"radius_mm": float(radius_mm), "height_mm": list(map(float, height_mm)),
            "voxel_mm": float(voxel_mm)}


def resolve_table_height(manual_height, report_path, axis_calibration,
                         estimator=None, *, persist=True, check_existing=True) -> float:
    """Use a manual height or persist and verify this session's auto estimate."""
    if manual_height is not None:
        return float(manual_height)
    if estimator is None:
        sys.path.insert(0, str(ROOT / "code"))
        from estimate_table_height import table_height_from_axis as estimator
    report_path = Path(report_path)
    current = estimator(axis_calibration)
    if check_existing and report_path.is_file():
        saved = json.loads(report_path.read_text(encoding="utf-8"))
        if saved != current:
            raise ValueError("saved table height differs from current axis calibration")
    if persist and (not report_path.is_file() or not check_existing):
        _write_json(report_path, current)
    return float(current["table_height_axis_mm"])


def prepare_scan_plan(plan_path: Path, plan: dict, resume: bool):
    """Validate now; create a new raw session only after preview acceptance."""
    plan_path = Path(plan_path)
    if resume:
        if not plan_path.is_file() or json.loads(plan_path.read_text(encoding="utf-8")) != plan:
            raise ValueError("resume scan plan differs from saved session")
        return lambda: None
    if plan_path.parent.exists():
        raise FileExistsError(plan_path.parent)
    def commit():
        plan_path.parent.mkdir(parents=True, exist_ok=False)
        _write_json(plan_path, plan)
    return commit


def completed_session(state_path: Path, resume: bool) -> dict | None:
    if not resume or not Path(state_path).is_file():
        return None
    state = json.loads(Path(state_path).read_text(encoding="utf-8"))
    return state if state.get("phase") == "complete" else None


def legacy_completed_session(state_path: Path, resume: bool, args, raw_session: Path) -> dict | None:
    """Read an archived result without applying today's processing to it."""
    previous = completed_session(state_path, resume)
    if previous is None or "green_filter" in previous.get("config", {}).get("source_fingerprint", {}):
        return None
    saved = previous["config"]
    source = saved["source_fingerprint"]
    if (source.get("table_height_axis_mm") != args.table_height_axis_mm or
            source.get("fusion_crop") != requested_fusion_crop(
                args.fusion_radius_mm, args.fusion_height_mm, args.fusion_voxel_mm) or
            source.get("processed_root") != str(args.processed_root.resolve()) or
            saved.get("session_root") != str(raw_session.resolve()) or
            saved.get("max_scans") != args.max_scans or
            saved.get("available") != args.available or
            saved.get("score_stop_enabled") != args.score_stop_enabled or
            saved.get("score_threshold") != (
                args.score_threshold if args.score_stop_enabled else None)):
        raise ValueError("legacy completed session settings differ from saved session")
    return previous


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--object-id", required=True)
    p.add_argument("--session-id", required=True)
    p.add_argument("--port", default="COM6")
    p.add_argument("--arm-port", default="COM7")
    p.add_argument("--axis-images", type=Path, required=True)
    p.add_argument("--table-height-axis-mm", type=float,
                   help="manual turntable-top coordinate; omit to use the axis target's measured bottom")
    p.add_argument("--stereo-upper", type=Path, default=ROOT / "calibration/stereo_pose_upper/stereo_calibration.json")
    p.add_argument("--stereo-lower", type=Path, default=ROOT / "calibration/stereo_pose_lower/stereo_calibration.json")
    p.add_argument("--board", type=Path, default=ROOT / "arm/targets/charuco_7x5_10mm_7mm_config.json")
    p.add_argument("--raw-root", type=Path, default=ROOT / "raw")
    p.add_argument("--processed-root", type=Path, default=ROOT / "processed")
    p.add_argument("--output-root", type=Path, default=ROOT / "PointNet/outputs")
    p.add_argument("--config", type=Path, default=ROOT / "models/diverse_200_v2_seed42_24/config_local.yaml")
    p.add_argument("--max-scans", type=int, default=10)
    p.add_argument("--available", type=int, nargs="+", default=list(range(24)))
    p.add_argument("--score-stop-enabled", action="store_true")
    p.add_argument("--score-threshold", type=float)
    p.add_argument("--use-unfiltered-model-input", action="store_true",
                   help="use original aligned clouds for PointNet decisions")
    p.add_argument("--repeat-id", type=int, default=1)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--left-serial", default="40031431")
    p.add_argument("--right-serial", default="40031435")
    p.add_argument("--display")
    p.add_argument("--exposure-us", type=float, default=10000)
    p.add_argument("--gain-db", type=float, default=0)
    p.add_argument("--settle-ms", type=int, default=250)
    p.add_argument("--timeout-ms", type=int, default=5000)
    p.add_argument("--dot-cell-px", type=int, default=8)
    p.add_argument("--dot-seed", type=int, default=5)
    p.add_argument("--dot-frames", type=int, default=5)
    p.add_argument("--arm-settle-s", type=float, default=6)
    p.add_argument("--motion-settle-s", type=float, default=6)
    p.add_argument("--return-settle-s", type=float, default=12)
    p.add_argument("--min-board-corners", type=int, default=12)
    p.add_argument("--max-board-rms-px", type=float, default=3)
    p.add_argument("--fusion-radius-mm", type=float, default=65)
    p.add_argument("--fusion-height-mm", type=float, nargs=2, default=(-95, 110))
    p.add_argument("--fusion-voxel-mm", type=float, default=1)
    return p


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    from fusion import calibrate_session_axis, fuse_selected, _axis_data
    from policy import predict_scores, validate_checkpoint
    from adapter import prepare_observation
    from scan_step import capture_and_reconstruct
    sys.path.insert(0, str(ROOT / "code"))
    sys.path.insert(0, str(ROOT / "arm"))
    sys.path.insert(0, str(ROOT / "turntable"))
    from capture import (Pattern, PREVIEW, PROJECTOR, SAFE, close_cameras, displays,
                         grab_pair, make_dense_dots, open_cameras, preview_pair,
                         select_projector_display, show_pattern)
    from arm import open_controller as open_arm_controller
    from turntable import open_controller as open_table_controller
    from fixed_board import board_from_config
    from estimate_table_height import height_crop_from_table
    import cv2
    if (args.port == args.arm_port or not SAFE.fullmatch(args.object_id) or
            not SAFE.fullmatch(args.session_id) or args.repeat_id < 1 or
            (args.table_height_axis_mm is not None and
             not np.isfinite(args.table_height_axis_mm)) or
            not 2 <= args.max_scans <= 24 or len(set(args.available)) != len(args.available) or
            any(not 0 <= a < 24 for a in args.available) or
            not {0, 6}.issubset(args.available) or
            (args.score_stop_enabled and
             (args.score_threshold is None or not np.isfinite(args.score_threshold))) or
            args.arm_settle_s < 0 or args.motion_settle_s < 0 or
            args.return_settle_s < 0 or args.settle_ms < 1 or
            args.timeout_ms < 1 or not 1 <= args.dot_frames <= 32):
        raise ValueError("invalid scanner configuration")
    validate_live_settings(args)
    board_config = json.loads(args.board.read_text(encoding="utf-8"))
    board = board_from_config(board_config)
    calibrations, K, D, camera_size = _load_calibrations(
        args.stereo_upper, args.stereo_lower, (args.left_serial, args.right_serial))
    checkpoint = validate_checkpoint(args.config)
    raw_session = args.raw_root / args.object_id / args.session_id
    output = args.output_root / args.object_id / args.session_id
    state_path = output / "state.json"
    previous = legacy_completed_session(state_path, args.resume, args, raw_session)
    if previous is not None:
        print(json.dumps({"visited": previous["visited"],
                          "stop_reason": previous["stop_reason"],
                          "final_fusion_dir": previous["final_fusion_dir"]}, ensure_ascii=False))
        return 0
    if args.resume and state_path.is_file():
        saved_state = json.loads(state_path.read_text(encoding="utf-8"))
        if "green_filter" not in saved_state.get("config", {}).get("source_fingerprint", {}):
            raise ValueError("unfinished legacy session uses different fusion processing; start a new session")
    if state_path.exists() and not args.resume:
        raise ValueError("existing session requires --resume or a new ID")
    if raw_session.exists() and not args.resume:
        raise FileExistsError(raw_session)
    axis_output = args.processed_root / "fused" / args.object_id / args.session_id / "fusion_calibration" / "axis_calibration.json"
    if args.resume and axis_output.is_file():
        _axis_data(axis_output, board_config)
    else:
        calibrate_session_axis(args.axis_images, calibrations[1], args.board, axis_output)
    table_height_path = output / "table_height_estimate.json"
    resolved_table_height = resolve_table_height(
        args.table_height_axis_mm, table_height_path, axis_output,
        persist=False, check_existing=args.resume)
    height_crop_from_table(args.fusion_height_mm, resolved_table_height)
    crop = requested_fusion_crop(args.fusion_radius_mm, args.fusion_height_mm,
                                 args.fusion_voxel_mm)
    source_fingerprint = live_input_fingerprint(
        {"stereo_upper": calibrations[0], "stereo_lower": calibrations[1],
         "board": args.board, "axis": axis_output,
         "axis_manifest": args.axis_images / "capture_manifest.json",
         "model_config": args.config, "model_checkpoint": checkpoint},
        crop, args.table_height_axis_mm, args.processed_root)
    source_fingerprint["green_filter"] = {"version": 1, "enabled": True,
                                          "radius_mode": "auto_from_initial_views",
                                          "min_object_height_mm": 3.0,
                                          "model_input": "unfiltered" if args.use_unfiltered_model_input else "cleaned"}
    plan = {"protocol": "dense_dots_v3", "reference_frame": "fixed_ChArUco_board",
            "board": board_config, "initial_views": [0, 6],
            "stereo_by_pose": {str(k): str(v.with_suffix('.json')) for k, v in calibrations.items()},
            "left_serial": args.left_serial, "right_serial": args.right_serial,
            "dot_cell_px": args.dot_cell_px, "dot_seed": args.dot_seed,
            "dot_frames": args.dot_frames, "exposure_us": args.exposure_us,
            "gain_db": args.gain_db, "table_height_axis_mm": args.table_height_axis_mm,
            "axis_calibration": str(axis_output.resolve()),
            "source_fingerprint": source_fingerprint}
    plan_path = raw_session / "scan_plan.json"
    commit_scan_plan = prepare_scan_plan(plan_path, plan, args.resume)
    run_config = {"session_root": str(raw_session.resolve()), "max_scans": args.max_scans,
                  "available": args.available, "score_stop_enabled": args.score_stop_enabled,
                  "score_threshold": args.score_threshold if args.score_stop_enabled else None,
                  "source_fingerprint": source_fingerprint}
    if args.resume and state_path.is_file():
        validate_resume_state(state_path, run_config)
    previous = completed_session(state_path, args.resume)
    if previous is not None:
        if previous.get("config", {}).get("source_fingerprint") != source_fingerprint:
            raise ValueError("completed session fingerprint differs from current inputs")
        print(json.dumps({"visited": previous["visited"],
                          "stop_reason": previous["stop_reason"],
                          "final_fusion_dir": previous["final_fusion_dir"]}, ensure_ascii=False))
        return 0
    patterns = make_dense_dots(dot_cell=args.dot_cell_px, seed=args.dot_seed, count=args.dot_frames)
    display = select_projector_display(displays(), args.display)
    args.trigger = "software"
    cameras = arm_controller = table_controller = None
    try:
        cameras, converter, pylon, size = open_cameras(args)
        if tuple(size) != tuple(camera_size):
            raise ValueError("camera size differs from stereo calibration")
        table_controller = open_table_controller(args.port)
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
                return 0
            if key == 32:
                break
        commit_scan_plan()
        resolve_table_height(args.table_height_axis_mm, table_height_path, axis_output,
                             check_existing=args.resume)
        show_pattern(Pattern("black", np.zeros_like(patterns[0].image)), display)
        cv2.waitKey(args.settle_ms)
        context = vars(args).copy()
        context.update({"board": board, "board_path": args.board,
                        "calibrations": calibrations, "K": K, "D": D,
                        "patterns": patterns, "camera_size": size,
                        "display": display})
        devices = {"arm": arm_controller, "turntable": table_controller,
                   "capture_devices": (cameras, converter, pylon, display,
                                       args.settle_ms, args.timeout_ms)}
        def capture(action):
            state = json.loads(state_path.read_text(encoding="utf-8"))
            context["previous_action"] = state["visited"][-1] if state["visited"] else None
            return capture_and_reconstruct(action, context, devices)
        def recover(action):
            from replay import _selected_repeat
            try:
                return _selected_repeat(raw_session, args.processed_root, action)
            except ValueError:
                return None
        filter_radius_path = output / "green_filter_radius.json"
        def fuse(repeats, destination):
            effective_crop = dict(crop)
            effective_crop["height_mm"] = list(height_crop_from_table(
                args.fusion_height_mm, resolved_table_height))
            saved = (json.loads(filter_radius_path.read_text(encoding="utf-8"))
                     if filter_radius_path.is_file() else None)
            result = fuse_selected(repeats, axis_output, raw_session / "camera_poses.json",
                                   args.raw_root, args.processed_root, destination, effective_crop,
                                   clean_green_background=True,
                                   table_height_axis_mm=resolved_table_height,
                                   clean_radius_mm=saved["radius_mm"] if saved else None)
            if saved is None:
                cleaned = json.loads((result / "fusion.json").read_text(encoding="utf-8"))
                _write_json(filter_radius_path, {
                    "radius_mm": cleaned["green_filter"]["radius_mm"],
                    "estimate": cleaned["green_filter"]["radius_estimate"],
                    "source_views": [0, 6]})
            return result
        frame_path = output / "object_frame.json"
        def predict(fusion_dir, visited, step_dir):
            frame = json.loads(frame_path.read_text(encoding="utf-8")) if frame_path.is_file() else None
            model_fusion_dir = (fusion_dir / "unfiltered" if args.use_unfiltered_model_input
                                else fusion_dir)
            points, frame = prepare_observation(
                model_fusion_dir, raw_session, visited, table_height_axis_mm=resolved_table_height,
                frame=frame)
            _write_json(frame_path, frame)
            step_dir.mkdir(parents=True, exist_ok=True)
            cloud_path = step_dir / "observed.npy"
            np.save(cloud_path, points)
            result = predict_scores(args.config, cloud_path, visited, args.available)
            result["input_fusion_dir"] = str(model_fusion_dir.resolve())
            _write_json(step_dir / "prediction.json", result)
            return result
        def home():
            from datetime import datetime
            import time
            command = table_controller.move_to(0)
            time.sleep(args.return_settle_s)
            _write_json(raw_session / "turntable_final_home.json", {
                "command_deg": 0, "serial_command": command.decode("ascii"),
                "sent_at": datetime.now().astimezone().isoformat(),
                "settle_s": args.return_settle_s, "arrival_verified": False})
        result = run_session({"session_root": raw_session, "output_root": output,
                              "final_fusion_dir": args.processed_root / "fused" / args.object_id /
                                                  args.session_id / "all_views_axis_calibrated",
                              "max_scans": args.max_scans, "available": args.available,
                              "score_stop_enabled": args.score_stop_enabled,
                              "score_threshold": args.score_threshold,
                              "source_fingerprint": source_fingerprint,
                              "resume": args.resume},
                             {"capture": capture, "recover": recover,
                              "fuse": fuse, "predict": predict,
                              "home": home})
        print(json.dumps({"visited": result["visited"], "stop_reason": result["stop_reason"],
                          "final_fusion_dir": result["final_fusion_dir"]}, ensure_ascii=False))
        return 0
    finally:
        if table_controller is not None:
            table_controller.connection.close()
        if arm_controller is not None:
            arm_controller.connection.close()
        if cameras is not None:
            close_cameras(cameras)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
