import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calibrate_axis import (ROOT, axis_in_board_frame, default_output_path,
                            fit_axis_from_poses, fit_stationary_board,
                            optional_turntable_top, turntable_top_from_target_poses)


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

    def test_target_bottom_gives_table_height_on_calibrated_axis(self):
        poses = [(angle, np.eye(3), np.array([20.0, 0.0, -180.0 + error]))
                 for angle, error in zip(range(0, 360, 30),
                                         [0, 0.2, -0.1, 0.1] * 3)]
        result = turntable_top_from_target_poses(
            poses, {"axis_point_mm": np.zeros(3),
                    "axis_direction": np.array([0.0, 0.0, 1.0])})
        self.assertAlmostEqual(result["turntable_top_axis_mm"], -180.0, delta=0.2)
        self.assertLess(result["turntable_top_height_spread_mm"], 1)

    def test_target_bottom_height_rejects_inconsistent_mount(self):
        poses = [(angle, np.eye(3), np.array([20.0, 0.0, -180.0 + angle / 10]))
                 for angle in range(0, 360, 30)]
        with self.assertRaisesRegex(ValueError, "bottom height"):
            turntable_top_from_target_poses(
                poses, {"axis_point_mm": np.zeros(3),
                        "axis_direction": np.array([0.0, 0.0, 1.0])})

    def test_reversed_axis_does_not_produce_unsafe_table_height(self):
        poses = [(angle, np.eye(3), np.array([20.0, 0.0, -180.0]))
                 for angle in range(0, 360, 30)]
        axis = {"axis_point_mm": np.zeros(3),
                "axis_direction": np.array([0.0, 0.0, -1.0])}
        with self.assertRaisesRegex(ValueError, "up direction"):
            turntable_top_from_target_poses(poses, axis)
        result = optional_turntable_top({"frame": "cuboid_center_at_bottom; z_up"}, poses, axis)
        self.assertIn("up direction", result["turntable_top_error"])

    def test_old_axis_target_remains_usable_with_manual_height(self):
        self.assertEqual(optional_turntable_top({"frame": "target_center"}, [], {}), {})


if __name__ == '__main__':
    unittest.main()
