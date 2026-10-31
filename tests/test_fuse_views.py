import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from fuse_views import DEFAULT_AXIS_CALIBRATION, read_ply, rotate_about_axis, voxel_average, load_axis_calibration, crop_cylinder
from reconstruct_view import write_ply


class FuseViewsTests(unittest.TestCase):
    def test_cylinder_crop_excludes_background_and_platform(self):
        points = np.array([[0, 0, 0], [40, 0, 0], [61, 0, 0], [0, -30, 0]])
        keep = crop_cylinder(points, np.zeros(3), np.array([0, 1, 0]), 60, (-20, 110))
        np.testing.assert_array_equal(keep, [True, True, False, False])

    def test_default_axis_path_is_the_calibration_directory(self):
        self.assertEqual(DEFAULT_AXIS_CALIBRATION,
            Path(__file__).resolve().parents[1] / "calibration/turntable_axis/axis_cube_01/axis_calibration.json")

    def test_axis_calibration_uses_measured_angles_and_original_left_frame(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "axis.json"
            path.write_text(json.dumps({"coordinate_frame": "unrectified_left_camera", "unit": "mm",
                "stereo_calibration": "C:/calibration.npz", "axis_point_mm": [1, 2, 3],
                "axis_direction": [0, -1, 0], "angles": [{"command_deg": 30, "measured_deg": 27}]}))
            axis = load_axis_calibration(path)
            self.assertEqual(axis["angles_by_command"][30.0], 27.0)
            np.testing.assert_array_equal(axis["point"], [1, 2, 3])
            self.assertEqual(axis["frame"], "original_left_camera")

    def test_axis_calibration_rejects_rectified_frame(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "axis.json"
            path.write_text(json.dumps({"coordinate_frame": "rectified_left_camera", "unit": "mm"}))
            with self.assertRaisesRegex(ValueError, "unrectified_left_camera"):
                load_axis_calibration(path)

    def test_inverse_turntable_rotation_restores_zero_angle_point(self):
        center = np.array([10., 0., 20.])
        original = np.array([[15., 3., 20.]], np.float32)
        moved = rotate_about_axis(original, center, [0, 1, 0], 30)
        recovered = rotate_about_axis(moved, center, [0, 1, 0], -30)
        np.testing.assert_allclose(recovered, original, atol=1e-5)

    def test_voxel_average_merges_overlapping_views_and_colors(self):
        xyz, bgr = voxel_average(np.array([[1.1, 2, 3], [1.2, 2, 3], [3, 4, 5]], np.float32),
                                 np.array([[10, 20, 30], [20, 30, 40], [50, 60, 70]], np.uint8), 1)
        self.assertEqual(len(xyz), 2)
        np.testing.assert_allclose(xyz[0], [1.15, 2, 3])
        np.testing.assert_array_equal(bgr[0], [15, 25, 35])

    def test_reads_single_view_ply_colors(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cloud.ply"
            write_ply(path, np.array([[1, 2, 3]], np.float32), np.array([[4, 5, 6]], np.uint8), "left")
            xyz, bgr = read_ply(path)
            np.testing.assert_array_equal(xyz, [[1, 2, 3]])
            np.testing.assert_array_equal(bgr, [[4, 5, 6]])
