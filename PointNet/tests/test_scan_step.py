import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scan_step
from scan_step import _write_json, capture_and_reconstruct, turntable_settle_seconds


def test_capture_step_moves_localizes_and_reconstructs_before_completion(tmp_path):
    events = []
    class Arm:
        def move_to(self, pose):
            events.append(("arm", pose))
            return b"arm-command"
    class Table:
        def move_to(self, angle):
            events.append(("table", angle))
            return b"table-command"
    def localize(*_):
        events.append(("localize",))
        return {"T_reference_from_left": np.eye(4).tolist(),
                "reprojection_rms_px": 0.5, "corners": 20}
    def capture(view, _devices, _patterns):
        events.append(("capture",))
        view["on_black"](np.zeros((2, 2, 3), np.uint8),
                         np.zeros((2, 2, 3), np.uint8), {},
                         {"left": "left/000_black.png", "right": "right/000_black.png"})
        path = view["destination"]
        (path / "manifest.json").write_text(json.dumps({**view["metadata"], "complete": True,
                                                          "frame_count": 1, "frames": []}),
                                             encoding="utf-8")
    def reconstruct(argv):
        events.append(("reconstruct",))
        target = Path(argv[argv.index("--output") + 1])
        target.mkdir(parents=True, exist_ok=True)
        (target / "cloud_left.ply").write_bytes(b"ply")
        (target / "qc.json").write_text('{"point_count":3}', encoding="utf-8")
        return 0
    board = tmp_path / "board.json"
    board.write_text("{}", encoding="utf-8")
    stereo = tmp_path / "lower.npz"
    stereo.write_bytes(b"stereo")
    context = {"raw_root": tmp_path / "raw", "processed_root": tmp_path / "processed",
               "object_id": "item", "session_id": "run", "repeat_id": 1,
               "calibrations": {0: stereo, 1: stereo}, "board_path": board,
               "board": object(), "K": np.eye(3), "D": np.zeros(5),
               "patterns": [object()], "arm_settle_s": 0, "motion_settle_s": 0,
               "left_serial": "left", "right_serial": "right", "exposure_us": 10000,
               "gain_db": 0, "settle_ms": 250, "dot_cell_px": 8, "dot_seed": 5,
               "dot_frames": 5, "min_board_corners": 12, "max_board_rms_px": 3}
    devices = {"arm": Arm(), "turntable": Table(), "capture_devices": (),
               "capture": capture, "localize": localize,
               "sleep": lambda _: events.append(("sleep",))}

    repeat = capture_and_reconstruct(15, context, devices, reconstruct)

    assert repeat.name == "repeat_001"
    assert events.index(("arm", "lower")) < events.index(("table", 90))
    assert events.index(("table", 90)) < events.index(("localize",))
    assert events.index(("localize",)) < events.index(("reconstruct",))
    poses = json.loads((repeat.parent.parent / "camera_poses.json").read_text(encoding="utf-8"))
    assert "view_015/repeat_001" in poses
    assert json.loads((repeat / "pose.json").read_text(encoding="utf-8"))["arm_pose_id"] == 1
    with pytest.raises(FileExistsError):
        capture_and_reconstruct(15, context, devices, reconstruct)


def test_capture_step_rejects_stale_processed_repeat_before_motion(tmp_path):
    processed = tmp_path / "processed" / "item" / "run" / "view_015" / "repeat_001"
    processed.mkdir(parents=True)
    context = {"raw_root": tmp_path / "raw", "processed_root": tmp_path / "processed",
               "object_id": "item", "session_id": "run", "repeat_id": 1}
    class MustNotMove:
        def move_to(self, _):
            raise AssertionError("motion occurred before stale output check")
    with pytest.raises(FileExistsError):
        capture_and_reconstruct(15, context, {"arm": MustNotMove(),
                                               "turntable": MustNotMove()},
                                reconstruct=lambda _: 0)


def test_return_to_zero_uses_long_settle_after_other_angles():
    assert turntable_settle_seconds(12, 11, 2.5, 12.0) == 12.0
    assert turntable_settle_seconds(0, None, 2.5, 12.0) == 2.5
    assert turntable_settle_seconds(6, 0, 2.5, 12.0) == 2.5


def test_localization_rejects_nonfinite_pose_quality(monkeypatch):
    monkeypatch.setattr(scan_step, "detect_corners_with_contrast_fallback",
                        lambda *_: (np.zeros((12, 2)), np.arange(12), "test"))
    monkeypatch.setattr(scan_step, "estimate_pose",
                        lambda *_: (np.eye(4), float("nan")))
    with pytest.raises(ValueError, match="reprojection RMS"):
        scan_step.localize_frame(np.zeros((2, 2), np.uint8), object(),
                                 np.eye(3), np.zeros(5))


def test_pose_map_write_keeps_previous_file_if_temp_write_fails(tmp_path, monkeypatch):
    path = tmp_path / "camera_poses.json"
    path.write_text('{"old":1}', encoding="utf-8")
    original = Path.write_text
    def failing_temp_write(target, data, *args, **kwargs):
        if target.name.endswith(".tmp"):
            raise OSError("disk full")
        return original(target, data, *args, **kwargs)
    monkeypatch.setattr(Path, "write_text", failing_temp_write)
    with pytest.raises(OSError, match="disk full"):
        _write_json(path, {"new": 2})
    assert path.read_text(encoding="utf-8") == '{"old":1}'
