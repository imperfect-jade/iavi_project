"""Offline active-scan replay over previously captured and reconstructed views."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from adapter import prepare_observation
from fusion import fuse_selected
from policy import predict_scores
from scan_pipeline import _write_json, run_session

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "models/diverse_200_v2_seed42_24/config_local.yaml"


def _selected_repeat(raw_session: Path, processed_root: Path, action: int) -> Path:
    candidates = sorted((raw_session / f"view_{action:03d}").glob("repeat_*"), reverse=True)
    for repeat in candidates:
        manifest_path = repeat / "manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        result = processed_root / raw_session.parent.name / raw_session.name / repeat.parent.name / repeat.name
        status_path = result / "reconstruction_status.json"
        if (manifest.get("complete") and manifest.get("view_id") == action and
                status_path.is_file() and
                json.loads(status_path.read_text(encoding="utf-8")).get("state") == "complete" and
                (result / "cloud_left.ply").is_file() and (result / "qc.json").is_file()):
            return repeat
    raise ValueError(f"no complete reconstructed historical view for action {action}")


def _complete_actions(raw_session: Path, processed_root: Path) -> list[int]:
    result = []
    for action in range(24):
        try:
            _selected_repeat(raw_session, processed_root, action)
        except ValueError:
            continue
        result.append(action)
    return result


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replay_session(raw_session, processed_root, axis_calibration, table_height_axis_mm,
                   output, budget, *, config_path=DEFAULT_CONFIG, available=None,
                   crop=None, fuse=None, predict=None, resume=False) -> dict:
    raw_session, processed_root = Path(raw_session).resolve(), Path(processed_root).resolve()
    axis_calibration, output = Path(axis_calibration).resolve(), Path(output).resolve()
    if not np.isfinite(table_height_axis_mm):
        raise ValueError("table_height_axis_mm must be finite")
    if not raw_session.is_dir():
        raise FileNotFoundError(raw_session)
    complete = _complete_actions(raw_session, processed_root)
    if not {0, 6}.issubset(complete):
        raise ValueError("historical initial views 0 and 6 must be complete")
    available = complete if available is None else list(available)
    if not set(available).issubset(complete):
        raise ValueError("requested available action lacks a complete historical view")
    config_path = Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    checkpoint = (config_path.parent / config["runtime"]["output_dir"] / "best_model.pt").resolve()
    source_fingerprint = {"axis_path": str(axis_calibration), "axis_sha256": _sha256(axis_calibration),
                          "model_config_path": str(config_path), "model_config_sha256": _sha256(config_path),
                          "model_checkpoint_sha256": _sha256(checkpoint),
                          "processed_root": str(processed_root),
                          "table_height_axis_mm": float(table_height_axis_mm),
                          "crop": json.loads(json.dumps(crop or {}))}
    frame_path = output / "object_frame.json"
    def capture_historical(action):
        return _selected_repeat(raw_session, processed_root, action)
    def fuse_historical(repeats, destination):
        return fuse_selected(repeats, axis_calibration, raw_session / "camera_poses.json",
                             raw_session.parent.parent, processed_root, destination, crop)
    def predict_historical(fusion_dir, visited, step_dir):
        frame = json.loads(frame_path.read_text(encoding="utf-8")) if frame_path.is_file() else None
        points, frame = prepare_observation(
            fusion_dir, raw_session, visited, table_height_axis_mm=table_height_axis_mm,
            frame=frame)
        _write_json(frame_path, frame)
        cloud_path = step_dir / "observed.npy"
        cloud_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cloud_path, points)
        result = predict_scores(config_path, cloud_path, visited, available)
        _write_json(step_dir / "prediction.json", result)
        return result
    return run_session({"session_root": raw_session, "output_root": output,
                        "max_scans": budget, "available": available,
                        "score_stop_enabled": False, "resume": resume,
                        "source_fingerprint": source_fingerprint},
                       {"capture": capture_historical,
                        "fuse": fuse or fuse_historical,
                        "predict": predict or predict_historical})


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-session", type=Path, required=True)
    p.add_argument("--processed-root", type=Path, default=ROOT / "processed")
    p.add_argument("--axis-calibration", type=Path, required=True)
    p.add_argument("--table-height-axis-mm", type=float, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--budget", type=int, default=10)
    p.add_argument("--available", type=int, nargs="+")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args(argv)
    result = replay_session(args.raw_session, args.processed_root, args.axis_calibration,
                            args.table_height_axis_mm, args.output, args.budget,
                            config_path=args.config, available=args.available,
                            resume=args.resume)
    print(json.dumps({"visited": result["visited"], "stop_reason": result["stop_reason"],
                      "final_fusion_dir": result["final_fusion_dir"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
