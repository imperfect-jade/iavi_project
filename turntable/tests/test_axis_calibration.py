import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calibrate_axis import ROOT, default_output_path, fit_axis_from_poses


class AxisFitTests(unittest.TestCase):
    def test_axis_result_defaults_to_calibration_session_directory(self):
        images = ROOT / "raw" / "turntable_axis" / "axis_cube_01"
        self.assertEqual(default_output_path(images),
                         ROOT / "calibration" / "turntable_axis" / "axis_cube_01" /
                         "axis_calibration.json")

    def test_recovers_tilted_axis_and_point(self):
        axis = np.array([0.2, -0.3, 0.932])
        axis /= np.linalg.norm(axis)
        center = np.array([32., -24., 780.])
        board_rotation, _ = cv2.Rodrigues(np.array([0.3, 0.1, -0.2]))
        board_translation = np.array([55., 17., 765.])
        poses = []
        for angle in range(0, 360, 30):
            R, _ = cv2.Rodrigues(axis * np.deg2rad(angle))
            t = center - R @ center
            poses.append((angle, R @ board_rotation, R @ board_translation + t))
        result = fit_axis_from_poses(poses)
        self.assertGreater(np.dot(result['axis_direction'], axis), 0.999)
        self.assertLess(np.linalg.norm(np.cross(result['axis_point_mm'] - center, axis)), 0.01)
        self.assertLess(result['point_rms_mm'], 0.01)


if __name__ == '__main__':
    unittest.main()
