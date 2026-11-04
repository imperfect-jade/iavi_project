import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import replay
from replay import replay_session


def test_replay_selects_only_complete_historical_views(tmp_path):
    raw = tmp_path / "raw" / "item" / "run"
    processed = tmp_path / "processed"
    for action in (0, 6, 15):
        repeat = raw / f"view_{action:03d}" / "repeat_001"
        repeat.mkdir(parents=True)
        (repeat / "manifest.json").write_text(json.dumps({"complete": True,
                                                              "view_id": action}), encoding="utf-8")
        rebuilt = processed / "item" / "run" / f"view_{action:03d}" / "repeat_001"
        rebuilt.mkdir(parents=True)
        (rebuilt / "cloud_left.ply").write_bytes(b"ply")
        (rebuilt / "qc.json").write_text('{"point_count":3}', encoding="utf-8")
        (rebuilt / "reconstruction_status.json").write_text('{"state":"complete"}', encoding="utf-8")
    (raw / "camera_poses.json").write_text("{}", encoding="utf-8")
    axis = tmp_path / "axis.json"
    axis.write_text("{}", encoding="utf-8")
    config = tmp_path / "model.yaml"
    config.write_text("runtime:\n  output_dir: model_run\n", encoding="utf-8")
    (tmp_path / "model_run").mkdir()
    (tmp_path / "model_run" / "best_model.pt").write_bytes(b"model")
    fused = []
    def fuse(repeats, output):
        fused.append([int(p.parent.name[5:]) for p in repeats])
        output.mkdir(parents=True)
        return output
    def predict(_fusion, visited, _step):
        return {"action_id": 15, "best_score": 0.2}

    result = replay_session(raw, processed, axis, -95,
                            tmp_path / "out", 3, config_path=config,
                            crop={"height_mm": (-95, 110)}, fuse=fuse, predict=predict)

    assert result["visited"] == [0, 6, 15]
    assert fused == [[0, 6], [0, 6, 15], [0, 6, 15]]
    assert result["config"]["available"] == [0, 6, 15]
    assert replay_session(raw, processed, axis, -95, tmp_path / "out", 3,
                          config_path=config, crop={"height_mm": (-95, 110)},
                          fuse=fuse, predict=predict, resume=True)["visited"] == [0, 6, 15]
    with pytest.raises(ValueError, match="resume configuration"):
        replay_session(raw, processed, axis, -90, tmp_path / "out", 3,
                       config_path=config, crop={"height_mm": (-95, 110)},
                       fuse=fuse, predict=predict, resume=True)


def test_replay_rejects_incomplete_view(tmp_path):
    raw = tmp_path / "raw" / "item" / "run"
    for action in (0, 6):
        repeat = raw / f"view_{action:03d}" / "repeat_001"
        repeat.mkdir(parents=True)
        (repeat / "manifest.json").write_text('{"complete":false}', encoding="utf-8")
    with pytest.raises(ValueError, match="complete"):
        replay_session(raw, tmp_path / "processed", tmp_path / "axis.json", -95,
                       tmp_path / "out", 2, available=[0, 6],
                       fuse=lambda *_: None, predict=lambda *_: None)


def test_replay_auto_height_expands_default_fusion_crop(tmp_path, monkeypatch):
    raw = tmp_path / "raw" / "item" / "run"
    processed = tmp_path / "processed"
    for action in (0, 6):
        repeat = raw / f"view_{action:03d}" / "repeat_001"
        repeat.mkdir(parents=True)
        (repeat / "manifest.json").write_text(
            json.dumps({"complete": True, "view_id": action}), encoding="utf-8")
        rebuilt = processed / "item" / "run" / f"view_{action:03d}" / "repeat_001"
        rebuilt.mkdir(parents=True)
        (rebuilt / "cloud_left.ply").write_bytes(b"ply")
        (rebuilt / "qc.json").write_text("{}", encoding="utf-8")
        (rebuilt / "reconstruction_status.json").write_text(
            '{"state":"complete"}', encoding="utf-8")
    (raw / "camera_poses.json").write_text("{}", encoding="utf-8")
    axis = tmp_path / "axis.json"
    axis.write_text(json.dumps({"coordinate_frame": "fixed_ChArUco_board", "unit": "mm",
                                "turntable_top_axis_mm": -180.0,
                                "turntable_top_height_spread_mm": 1.0,
                                "turntable_top_source": "bottom_centred_axis_cuboid_on_turntable"}),
                    encoding="utf-8")
    config = tmp_path / "model.yaml"
    config.write_text("runtime:\n  output_dir: model_run\n", encoding="utf-8")
    (tmp_path / "model_run").mkdir()
    (tmp_path / "model_run" / "best_model.pt").write_bytes(b"model")
    crops = []
    def fake_fuse(_repeats, _axis, _poses, _raw_root, _processed, output, crop, **kwargs):
        crops.append(crop)
        output.mkdir(parents=True)
        (output / "fusion.json").write_text(
            json.dumps({"green_filter": {"radius_mm": 45,
                                           "radius_estimate": {"method": "test"}}}), encoding="utf-8")
        return output
    monkeypatch.setattr(replay, "fuse_selected", fake_fuse)
    result = replay_session(raw, processed, axis, None, tmp_path / "out", 2,
                            config_path=config)
    assert result["visited"] == [0, 6]
    assert [crop["height_mm"] for crop in crops] == [[-185.0, 110.0]] * 2
