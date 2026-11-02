import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calibrate_axis import (ROOT, axis_in_board_frame, default_output_path,
                            fit_axis_from_poses, fit_stationary_board)


class AxisFitTests(unittest.TestCase):
    def test_stationary_board_fit_uses_multiple_images_and_rejects_pose_outlier(self):
        images = [(f"angle_{i:03d}/left.png", i) for i in range(9)]

        def detect(image, board, min_corners):
            corners = np.full((20, 1, 2), image, np.float32)
            return corners, np.arange(20), "raw"

        def pose(board, corners, ids, K, D):
            index = int(corners[0, 0, 0])
            T = np.eye(4)
            T[:3, 3] = [10 + index * 0.05, 20, 800 if index < 8 else 815]
            return T, 0.2

        with patch("calibrate_axis.detect_corners_with_contrast_fallback", side_effect=detect), \
             patch("calibrate_axis.estimate_pose", side_effect=pose):
            T, report = fit_stationary_board(images, object(), np.eye(3), np.zeros(5))
        self.assertEqual(report["inlier_count"], 8)
        self.assertEqual(report["observations"][-1]["status"], "pose_outlier")
        np.testing.assert_allclose(T[:3, 3], [10.175, 20, 800])

    def test_axis_is_transformed_into_board_coordinates(self):
        T_left_from_board = np.eye(4)
        T_left_from_board[:3, 3] = [10, 20, 0]
        point, direction = axis_in_board_frame([10, 15, 100], [0, 1, 0], T_left_from_board)
        np.testing.assert_allclose(point, [0, -5, 100])
        np.testing.assert_allclose(direction, [0, 1, 0])

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
