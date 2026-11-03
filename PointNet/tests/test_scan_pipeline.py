import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scan_pipeline import (completed_session, live_input_fingerprint,
                           prepare_scan_plan, run_session, validate_live_settings)


def _services(tmp_path, actions, score=-1.0, fail=None):
    captures, fused = [], []
    def capture(action):
        captures.append(action)
        if action == fail:
            raise RuntimeError("camera failure")
        repeat = tmp_path / "raw" / "item" / "run" / f"view_{action:03d}" / "repeat_001"
        repeat.mkdir(parents=True, exist_ok=True)
        return repeat
    def fuse(repeats, output):
        fused.append([int(p.parent.name[5:]) for p in repeats])
        output.mkdir(parents=True, exist_ok=True)
        return output
    def predict(_fusion, visited, _step):
        return {"action_id": next(a for a in actions if a not in visited),
                "best_score": score, "scores_by_action": {str(a): score for a in actions}}
    return {"capture": capture, "fuse": fuse, "predict": predict}, captures, fused


def test_run_session_uses_initial_views_then_budget_and_final_fusion(tmp_path):
    services, captures, fused = _services(tmp_path, [15, 12])
    config = {"session_root": tmp_path / "raw" / "item" / "run",
              "output_root": tmp_path / "out", "max_scans": 3,
              "available": [0, 6, 15, 12], "score_stop_enabled": False}

    result = run_session(config, services)

    assert result["visited"] == [0, 6, 15]
    assert result["stop_reason"] == "budget"
    assert captures == [0, 6, 15]
    assert fused == [[0, 6], [0, 6, 15], [0, 6, 15]]
    assert json.loads((tmp_path / "out" / "state.json").read_text(encoding="utf-8"))["visited"] == [0, 6, 15]


def test_score_stop_requires_explicit_enablement(tmp_path):
    services, captures, _ = _services(tmp_path, [15], score=-2)
    base = {"session_root": tmp_path / "raw" / "item" / "run",
            "output_root": tmp_path / "out", "max_scans": 4,
            "available": [0, 6, 15], "score_stop_enabled": True,
            "score_threshold": -1.0}
    result = run_session(base, services)
    assert result["visited"] == [0, 6]
    assert result["stop_reason"] == "low_predicted_gain"
    assert captures == [0, 6]


def test_failed_action_can_resume_without_repeating_successful_views(tmp_path):
    services, captures, _ = _services(tmp_path, [15], fail=15)
    config = {"session_root": tmp_path / "raw" / "item" / "run",
              "output_root": tmp_path / "out", "max_scans": 3,
              "available": [0, 6, 15], "score_stop_enabled": False}
    with pytest.raises(RuntimeError, match="camera failure"):
        run_session(config, services)
    assert captures == [0, 6, 15]
    assert json.loads((tmp_path / "out" / "state.json").read_text(encoding="utf-8"))["visited"] == [0, 6]
    resumed, later, _ = _services(tmp_path, [15])
    result = run_session({**config, "resume": True}, resumed)
    assert result["visited"] == [0, 6, 15]
    assert later == [15]


def test_resume_rejects_changed_model_or_calibration_fingerprint(tmp_path):
    services, _, _ = _services(tmp_path, [15], fail=15)
    config = {"session_root": tmp_path / "raw" / "item" / "run",
              "output_root": tmp_path / "out", "max_scans": 3,
              "available": [0, 6, 15], "score_stop_enabled": False,
              "source_fingerprint": {"model_sha256": "first"}}
    with pytest.raises(RuntimeError, match="camera failure"):
        run_session(config, services)
    resumed, captures, _ = _services(tmp_path, [15])
    with pytest.raises(ValueError, match="resume configuration"):
        run_session({**config, "source_fingerprint": {"model_sha256": "changed"},
                     "resume": True}, resumed)
    assert captures == []


def test_resume_rejects_mismatched_saved_action_and_repeat(tmp_path):
    services, _, _ = _services(tmp_path, [15], fail=15)
    config = {"session_root": tmp_path / "raw" / "item" / "run",
              "output_root": tmp_path / "out", "max_scans": 3,
              "available": [0, 6, 15], "score_stop_enabled": False}
    with pytest.raises(RuntimeError, match="camera failure"):
        run_session(config, services)
    state_path = tmp_path / "out" / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["repeats"] = list(reversed(state["repeats"]))
    state_path.write_text(json.dumps(state), encoding="utf-8")
    resumed, captures, _ = _services(tmp_path, [15])
    with pytest.raises(ValueError, match="saved action"):
        run_session({**config, "resume": True}, resumed)
    assert captures == []


def test_new_scan_plan_is_committed_only_after_preview_acceptance(tmp_path):
    plan_path = tmp_path / "raw" / "item" / "run" / "scan_plan.json"
    commit = prepare_scan_plan(plan_path, {"protocol": "dense_dots_v3"}, resume=False)
    assert not plan_path.exists()
    commit()
    assert json.loads(plan_path.read_text(encoding="utf-8"))["protocol"] == "dense_dots_v3"


def test_recovery_reuses_completed_repeat_missing_from_state(tmp_path):
    services, captures, _ = _services(tmp_path, [15])
    recovered = tmp_path / "raw" / "item" / "run" / "view_015" / "repeat_001"
    recovered.mkdir(parents=True)
    services["recover"] = lambda action: recovered if action == 15 else None
    config = {"session_root": tmp_path / "raw" / "item" / "run",
              "output_root": tmp_path / "out", "max_scans": 3,
              "available": [0, 6, 15], "score_stop_enabled": False}
    result = run_session(config, services)
    assert result["visited"] == [0, 6, 15]
    assert captures == [0, 6]


def test_completed_resume_is_detected_before_hardware_start(tmp_path):
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"phase": "complete", "visited": [0, 6],
                                 "stop_reason": "budget", "final_fusion_dir": "done"}),
                     encoding="utf-8")
    assert completed_session(state, resume=False) is None
    assert completed_session(state, resume=True)["final_fusion_dir"] == "done"


def test_resume_recovers_plan_written_before_first_state(tmp_path):
    session = tmp_path / "raw" / "item" / "run"
    session.mkdir(parents=True)
    (session / "scan_plan.json").write_text("{}", encoding="utf-8")
    services, captures, _ = _services(tmp_path, [])
    result = run_session({"session_root": session, "output_root": tmp_path / "out",
                          "max_scans": 2, "available": [0, 6],
                          "score_stop_enabled": False, "resume": True}, services)
    assert result["visited"] == [0, 6]
    assert captures == [0, 6]


def test_live_preflight_rejects_nonfinite_localization_and_invalid_crop():
    valid = {"min_board_corners": 12, "max_board_rms_px": 3.0,
             "fusion_radius_mm": 65.0, "fusion_height_mm": (-95.0, 110.0),
             "fusion_voxel_mm": 1.0}
    with pytest.raises(ValueError, match="localization"):
        validate_live_settings(SimpleNamespace(**{**valid, "max_board_rms_px": float("nan")}))
    with pytest.raises(ValueError, match="fusion"):
        validate_live_settings(SimpleNamespace(**{**valid, "fusion_radius_mm": -1}))


def test_live_fingerprint_changes_with_fusion_crop(tmp_path):
    paths = {}
    for name in ("stereo_upper", "stereo_lower", "board", "axis", "axis_manifest",
                 "model_config", "model_checkpoint"):
        paths[name] = tmp_path / name
        paths[name].write_bytes(name.encode())
    first = live_input_fingerprint(paths, {"radius_mm": 65, "height_mm": (-95, 110),
                                           "voxel_mm": 1}, -95, tmp_path / "processed")
    second = live_input_fingerprint(paths, {"radius_mm": 60, "height_mm": (-95, 110),
                                            "voxel_mm": 1}, -95, tmp_path / "processed")
    assert first != second
    assert first["fusion_crop"]["height_mm"] == [-95, 110]
