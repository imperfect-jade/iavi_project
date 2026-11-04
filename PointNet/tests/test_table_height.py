import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from estimate_table_height import height_crop_from_table, table_height_from_axis


def test_table_height_comes_from_calibration_target_bottom(tmp_path):
    path = tmp_path / "axis.json"
    path.write_text(json.dumps({"coordinate_frame": "fixed_ChArUco_board", "unit": "mm",
                                "turntable_top_axis_mm": -179.6,
                                "turntable_top_height_spread_mm": 1.7,
                                "turntable_top_source": "bottom_centred_axis_cuboid_on_turntable"}),
                    encoding="utf-8")
    result = table_height_from_axis(path)
    assert result["table_height_axis_mm"] == -179.6
    assert result["axis_calibration_sha256"]


def test_table_height_rejects_axis_without_target_bottom(tmp_path):
    path = tmp_path / "axis.json"
    path.write_text(json.dumps({"coordinate_frame": "fixed_ChArUco_board", "unit": "mm"}),
                    encoding="utf-8")
    with pytest.raises(ValueError, match="turntable top"):
        table_height_from_axis(path)


def test_table_height_expands_fusion_lower_bound_without_narrowing_it():
    assert height_crop_from_table((-95, 110), -181) == (-186.0, 110.0)
    assert height_crop_from_table((-200, 110), -181) == (-200.0, 110.0)
