import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "PointNet"))
sys.path.insert(0, str(ROOT / "code"))
from clean_green import clean_aligned_fusion, clean_fusion
from fuse_views import read_ply
from reconstruct_view import write_ply


def test_clean_fusion_preserves_source_and_keeps_object_colors(tmp_path):
    fusion = tmp_path / "fusion"
    fusion.mkdir()
    points = np.array([[1, 0, 0], [2, 0, 0], [3, 0, 0],
                       [60, 0, 0], [0, 0, -4]], dtype=np.float32)
    bgr = np.array([[0, 200, 255], [60, 160, 20], [255, 0, 0],
                    [0, 200, 255], [0, 200, 255]], dtype=np.uint8)
    write_ply(fusion / "fused.ply", points, bgr, "turntable_zero_calibrated")
    (fusion / "fusion.json").write_text(json.dumps({
        "coordinate_frame": "turntable_zero_calibrated", "unit": "mm",
        "axis_point_mm": [0, 0, 0], "axis_direction": [0, 0, 1]}), encoding="utf-8")
    output = fusion / "fused_clean.ply"

    report = clean_fusion(fusion, output, radius_mm=45,
                          table_height_axis_mm=-5, min_object_height_mm=3)

    kept, colors = read_ply(output)
    original, _ = read_ply(fusion / "fused.ply")
    assert len(original) == 5
    assert kept.tolist() == [[1, 0, 0], [3, 0, 0]]
    assert colors.tolist() == [[0, 200, 255], [255, 0, 0]]
    assert report["removed_green"] == 1
    assert report["removed_outside_radius"] == 1
    assert report["removed_below_height"] == 1
    assert report["retained_points"] == 2
    with pytest.raises(FileExistsError):
        clean_fusion(fusion, output)


def test_clean_aligned_fusion_estimates_radius_and_keeps_selected_views(tmp_path):
    source = tmp_path / "unfiltered"
    (source / "aligned").mkdir(parents=True)
    repeats = [str(tmp_path / f"view_{action:03d}" / "repeat_001") for action in (0, 6)]
    theta = np.linspace(0, 2 * np.pi, 1200, endpoint=False)
    object_points = np.column_stack([20 * np.cos(theta), 20 * np.sin(theta),
                                     np.full(len(theta), 15)]).astype(np.float32)
    green_points = np.column_stack([55 * np.cos(theta[:100]),
                                    55 * np.sin(theta[:100]), np.full(100, 15)]).astype(np.float32)
    outliers = np.column_stack([60 * np.cos(theta[:30]),
                                60 * np.sin(theta[:30]), np.full(30, 15)]).astype(np.float32)
    points = np.concatenate([object_points, green_points, outliers])
    colors = np.concatenate([np.tile([0, 200, 255], (1200, 1)),
                             np.tile([60, 160, 20], (100, 1)),
                             np.tile([0, 200, 255], (30, 1))]).astype(np.uint8)
    for repeat in repeats:
        path = Path(repeat)
        write_ply(source / "aligned" / f"{path.parent.name}_{path.name}.ply",
                  points, colors, "turntable_zero_calibrated")
    write_ply(source / "fused.ply", points, colors, "turntable_zero_calibrated")
    (source / "fusion.json").write_text(json.dumps({
        "coordinate_frame": "turntable_zero_calibrated", "unit": "mm",
        "axis_point_mm": [0, 0, 0], "axis_direction": [0, 0, 1],
        "selected_repeats": repeats, "voxel_mm": 1, "fused_points": len(points),
        "views": [{"repeat": repeat} for repeat in repeats]}), encoding="utf-8")

    report = clean_aligned_fusion(source, tmp_path / "cleaned", max_radius_mm=65,
                                  table_height_axis_mm=0)

    assert report["selected_repeats"] == repeats
    assert 23 < report["green_filter"]["radius_mm"] < 30
    assert report["green_filter"]["radius_estimate"]["method"] != "fallback"
    cleaned, _ = read_ply(tmp_path / "cleaned" / "fused.ply")
    assert np.linalg.norm(cleaned[:, :2], axis=1).max() < 30
    assert len(read_ply(source / "fused.ply")[0]) == len(points)
