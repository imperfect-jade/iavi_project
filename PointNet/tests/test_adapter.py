import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adapter import prepare_observation


VERTEX = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                   ("red", "u1"), ("green", "u1"), ("blue", "u1")])


def write_ply(path, points):
    data = np.zeros(len(points), dtype=VERTEX)
    for index, name in enumerate(("x", "y", "z")):
        data[name] = np.asarray(points)[:, index]
    with path.open("wb") as stream:
        stream.write(("ply\nformat binary_little_endian 1.0\n"
                      "comment coordinates turntable_zero_calibrated; unit mm\n"
                      f"element vertex {len(points)}\n"
                      "property float x\nproperty float y\nproperty float z\n"
                      "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                      "end_header\n").encode("ascii"))
        data.tofile(stream)


def fixture(tmp_path):
    fusion = tmp_path / "fusion"
    aligned = fusion / "aligned"
    aligned.mkdir(parents=True)
    raw = tmp_path / "raw"
    raw.mkdir()
    (fusion / "fusion.json").write_text(json.dumps({
        "coordinate_frame": "turntable_zero_calibrated", "unit": "mm",
        "axis_point_mm": [0, 0, 0], "axis_direction": [0, 0, 1],
        "selected_repeats": [str(raw / "view_000" / "repeat_001"),
                             str(raw / "view_006" / "repeat_001"),
                             str(raw / "view_015" / "repeat_001")]}), encoding="utf-8")
    pose = np.eye(4).tolist()
    pose[0][3] = 100
    (raw / "camera_poses.json").write_text(json.dumps({
        "view_000/repeat_001": pose}), encoding="utf-8")
    write_ply(aligned / "view_000_repeat_001.ply", [[10, 0, 10], [20, 0, 20]])
    write_ply(aligned / "view_006_repeat_001.ply", [[0, 10, 30]])
    write_ply(aligned / "view_015_repeat_001.ply", [[30, 0, 15]])
    return fusion, raw


def test_uses_only_visited_views_and_normalizes_to_initial_object_frame(tmp_path):
    fusion, raw = fixture(tmp_path)
    points, frame = prepare_observation(fusion, raw, [0, 6], table_height_axis_mm=0)
    assert points.shape == (3, 3)
    np.testing.assert_allclose(points, [[0, -100/6, 100/3],
                                        [100/3, -100/6, 200/3],
                                        [-100/3, 100/6, 100]], atol=1e-4)
    assert frame["initial_views"] == [0, 6]


def test_reuses_frozen_frame_when_next_view_is_added(tmp_path):
    fusion, raw = fixture(tmp_path)
    initial, frame = prepare_observation(fusion, raw, [0, 6], table_height_axis_mm=0)
    updated, same_frame = prepare_observation(fusion, raw, [0, 6, 15], frame=frame)
    np.testing.assert_allclose(updated[:3], initial)
    np.testing.assert_allclose(updated[3], [200/3, -100/6, 50], atol=1e-4)
    assert same_frame == frame


def test_rejects_uncalibrated_fusion_and_missing_visited_view(tmp_path):
    fusion, raw = fixture(tmp_path)
    report = json.loads((fusion / "fusion.json").read_text())
    report["coordinate_frame"] = "turntable_zero_estimated"
    (fusion / "fusion.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="calibrated"):
        prepare_observation(fusion, raw, [0, 6], table_height_axis_mm=0)
    report["coordinate_frame"] = "turntable_zero_calibrated"
    (fusion / "fusion.json").write_text(json.dumps(report))
    (fusion / "aligned" / "view_006_repeat_001.ply").unlink()
    with pytest.raises(FileNotFoundError, match="view_006"):
        prepare_observation(fusion, raw, [0, 6], table_height_axis_mm=0)


def test_requires_measured_table_height_on_first_call(tmp_path):
    fusion, raw = fixture(tmp_path)
    with pytest.raises(ValueError, match="table_height_axis_mm"):
        prepare_observation(fusion, raw, [0, 6])
