"""Prepare real scan clouds and request the next PointNet view."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from adapter import prepare_observation


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "models" / "diverse_200_v2_seed42_24" / "config_local.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fusion-dir", type=Path, required=True,
                        help="session all_views_axis_calibrated directory")
    parser.add_argument("--raw-session", type=Path, required=True,
                        help="matching raw object/session directory containing camera_poses.json")
    parser.add_argument("--visited", type=int, nargs="+", required=True,
                        help="acquisition-order action IDs; starts with 0 6")
    parser.add_argument("--table-height-axis-mm", type=float,
                        help="measured tabletop coordinate along calibrated axis; required on first call")
    parser.add_argument("--available", type=int, nargs="+",
                        help="physically allowed action IDs; defaults to all 24")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--prepare-only", action="store_true",
                        help="write model input without loading PyTorch")
    args = parser.parse_args()

    fusion_dir = args.fusion_dir.resolve()
    raw_session = args.raw_session.resolve()
    output_dir = args.output_dir.resolve()
    frame_path = output_dir / "object_frame.json"
    frame = json.loads(frame_path.read_text(encoding="utf-8")) if frame_path.is_file() else None
    if frame is not None and (frame.get("source_fusion_dir") != str(fusion_dir) or
                              frame.get("source_raw_session") != str(raw_session)):
        parser.error("saved object frame belongs to a different scan session")
    points, frame = prepare_observation(
        fusion_dir, raw_session, args.visited,
        table_height_axis_mm=args.table_height_axis_mm, frame=frame)
    frame["source_fusion_dir"] = str(fusion_dir)
    frame["source_raw_session"] = str(raw_session)
    output_dir.mkdir(parents=True, exist_ok=True)
    points_path = output_dir / "observed.npy"
    np.save(points_path, points)
    frame_path.write_text(json.dumps(frame, ensure_ascii=False, indent=2), encoding="utf-8")
    result = {"observed_points": int(len(points)), "visited": args.visited,
              "points_path": str(points_path), "frame_path": str(frame_path)}
    if not args.prepare_only:
        from policy import predict_scores
        result.update(predict_scores(args.config, points_path, args.visited, available=args.available))
        (output_dir / "prediction.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
