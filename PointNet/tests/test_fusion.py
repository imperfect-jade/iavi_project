import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fusion import _axis_data, calibrate_session_axis, fuse_selected


def test_calibrate_session_axis_rejects_bad_axis_quality(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    (images / "capture_manifest.json").write_text('{"complete":true}', encoding="utf-8")
    board = tmp_path / "board.json"
    board.write_text('{"board_type":"ChArUco"}', encoding="utf-8")
    output = tmp_path / "axis.json"

    def calibrate(argv):
        output.write_text(json.dumps({"coordinate_frame": "fixed_ChArUco_board",
                                      "unit": "mm", "fixed_board": {"board_type": "ChArUco"},
                                      "point_rms_mm": 3.1,
                                      "return_to_zero": {"left_corner_shift_rms_px": 0.2}}),
                          encoding="utf-8")
        return 0

    with pytest.raises(ValueError, match="quality"):
        calibrate_session_axis(images, tmp_path / "lower.npz", board, output,
                               calibrate=calibrate)


def test_axis_with_reversed_up_direction_is_rejected_even_with_manual_height(tmp_path):
    axis = tmp_path / "axis.json"
    axis.write_text(json.dumps({"coordinate_frame": "fixed_ChArUco_board", "unit": "mm",
                                "point_rms_mm": 1.0,
                                "return_to_zero": {"left_corner_shift_rms_px": 0.5},
                                "turntable_top_error": "target up direction disagrees with fitted turntable axis"}),
                    encoding="utf-8")
    with pytest.raises(ValueError, match="below the turntable"):
        _axis_data(axis)


def test_fuse_selected_passes_only_requested_repeats_and_axis(tmp_path):
    repeats = [tmp_path / "raw" / "item" / "run" / f"view_{a:03d}" / "repeat_001"
               for a in (0, 6)]
    for repeat in repeats:
        repeat.mkdir(parents=True)
    axis = tmp_path / "axis.json"
    axis.write_text('{"coordinate_frame":"fixed_ChArUco_board","unit":"mm",'
                    '"fixed_board":{},"point_rms_mm":1.2,'
                    '"return_to_zero":{"left_corner_shift_rms_px":0.4}}', encoding="utf-8")
    poses = tmp_path / "poses.json"
    poses.write_text("{}", encoding="utf-8")
    output = tmp_path / "step"
    seen = {}

    def fuse(argv):
        seen["argv"] = argv
        output.mkdir()
        (output / "fused.ply").write_bytes(b"ply")
        (output / "fusion.json").write_text(json.dumps({"coordinate_frame": "turntable_zero_calibrated",
                                                          "unit": "mm", "fused_points": 3,
                                                          "selected_repeats": list(map(str, repeats))}),
                                             encoding="utf-8")
        return 0

    result = fuse_selected(repeats, axis, poses, tmp_path / "raw", tmp_path / "processed",
                           output, fuse=fuse)

    assert result == output
    assert seen["argv"][seen["argv"].index("--axis-calibration") + 1] == str(axis)
    assert seen["argv"][1:3] == list(map(str, repeats))
    with pytest.raises(ValueError, match="at least two"):
        fuse_selected(repeats[:1], axis, poses, tmp_path / "raw", tmp_path / "processed",
                      tmp_path / "one", fuse=fuse)


def test_fuse_selected_rejects_axis_from_different_fixed_board(tmp_path):
    session = tmp_path / "raw" / "item" / "run"
    repeats = [session / f"view_{a:03d}" / "repeat_001" for a in (0, 6)]
    for repeat in repeats:
        repeat.mkdir(parents=True)
    (session / "scan_plan.json").write_text(json.dumps({"reference_frame": "fixed_ChArUco_board",
                                                         "board": {"board_type": "different"}}), encoding="utf-8")
    axis = tmp_path / "axis.json"
    axis.write_text(json.dumps({"coordinate_frame": "fixed_ChArUco_board", "unit": "mm",
                                "fixed_board": {"board_type": "ChArUco"},
                                "point_rms_mm": 1.0,
                                "return_to_zero": {"left_corner_shift_rms_px": 0.5}}), encoding="utf-8")
    poses = session / "camera_poses.json"
    poses.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="fixed board"):
        fuse_selected(repeats, axis, poses, tmp_path / "raw", tmp_path / "processed",
                      tmp_path / "output", fuse=lambda _: 0)


def test_fuse_selected_routes_cleaned_views_to_policy_fusion(tmp_path):
    session = tmp_path / "raw" / "item" / "run"
    repeats = [session / f"view_{a:03d}" / "repeat_001" for a in (0, 6)]
    for repeat in repeats:
        repeat.mkdir(parents=True)
    axis = tmp_path / "axis.json"
    axis.write_text(json.dumps({"coordinate_frame": "fixed_ChArUco_board", "unit": "mm",
                                "point_rms_mm": 1.0,
                                "return_to_zero": {"left_corner_shift_rms_px": 0.5}}),
                    encoding="utf-8")
    poses = session / "camera_poses.json"
    poses.write_text("{}", encoding="utf-8")
    output = tmp_path / "fusion"
    seen = {}
    def fake_fuse(argv):
        source = Path(argv[argv.index("--output") + 1])
        seen["source"] = source
        source.mkdir(parents=True)
        (source / "fused.ply").write_bytes(b"ply")
        (source / "fusion.json").write_text(json.dumps({
            "coordinate_frame": "turntable_zero_calibrated", "unit": "mm",
            "fused_points": 3, "selected_repeats": list(map(str, repeats))}), encoding="utf-8")
        return 0
    def fake_clean(source, destination, **kwargs):
        seen["clean"] = (source, destination, kwargs)
        return {"selected_repeats": list(map(str, repeats)), "fused_points": 2}

    result = fuse_selected(repeats, axis, poses, tmp_path / "raw", tmp_path / "processed",
                           output, crop={"radius_mm": 65}, fuse=fake_fuse,
                           clean_green_background=True, table_height_axis_mm=-101,
                           cleaner=fake_clean)

    assert result == output
    assert seen["source"] == output / "unfiltered"
    assert seen["clean"][0:2] == (output / "unfiltered", output)
    assert seen["clean"][2]["table_height_axis_mm"] == -101
