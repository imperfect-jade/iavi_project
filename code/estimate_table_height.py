"""Read the turntable height measured by the axis calibration target."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def height_crop_from_table(height_mm, table_height_axis_mm: float,
                           margin_mm: float = 5.0) -> tuple[float, float]:
    """Keep the turntable top inside a broad fusion cylinder."""
    lower, upper = map(float, height_mm)
    table = float(table_height_axis_mm)
    if (not np.isfinite([lower, upper, table, margin_mm]).all() or
            lower >= upper or margin_mm < 0 or table >= upper):
        raise ValueError("invalid table height or fusion height range")
    return min(lower, table - margin_mm), upper


def table_height_from_axis(axis_calibration) -> dict:
    """Use the bottom-centred cuboid that sat directly on the turntable."""
    path = Path(axis_calibration).resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("coordinate_frame") != "fixed_ChArUco_board" or data.get("unit") != "mm":
        raise ValueError("turntable top requires a calibrated fixed-board axis in mm")
    if data.get("turntable_top_source") != "bottom_centred_axis_cuboid_on_turntable":
        detail = data.get("turntable_top_error", "target bottom height was not measured")
        raise ValueError(f"axis calibration does not include a measured turntable top: {detail}")
    height = data.get("turntable_top_axis_mm")
    spread = data.get("turntable_top_height_spread_mm")
    if (height is None or spread is None or not np.isfinite([height, spread]).all() or
            spread < 0 or spread > 5):
        raise ValueError("invalid measured turntable top")
    return {"table_height_axis_mm": float(height),
            "method": "axis_target_bottom", "height_spread_mm": float(spread),
            "axis_calibration_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
