import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generate_cube_target import build_config, draw_page
from calibrate_axis import marker_pose, stereo_marker_pose


class CubeTargetTests(unittest.TestCase):
    def test_eight_unique_markers_on_four_faces(self):
        config = build_config()
        self.assertEqual((config["width_mm"], config["depth_mm"], config["height_mm"]),
                         (45.0, 45.0, 100.0))
        self.assertEqual((config["marker_mm"], config["side_margin_mm"],
                          config["top_margin_mm"], config["marker_gap_mm"],
                          config["bottom_margin_mm"]), (21.0, 12.0, 26.0, 4.0, 28.0))
        self.assertEqual(config["markers"]["20"]["corners_mm"],
                         [[-10.5, -22.5, 74.0], [10.5, -22.5, 74.0],
                          [10.5, -22.5, 53.0], [-10.5, -22.5, 53.0]])
        self.assertEqual(config["markers"]["21"]["corners_mm"][2][2], 28.0)
        self.assertEqual(config["markers"]["22"]["corners_mm"][0][0], -22.5)
        self.assertEqual(set(map(int, config["markers"])), set(range(20, 28)))
        for marker in config["markers"].values():
            self.assertEqual(np.asarray(marker["corners_mm"]).shape, (4, 3))
        page = draw_page(config)
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        corners, ids, _ = cv2.aruco.ArucoDetector(dictionary).detectMarkers(np.asarray(page.convert("L")))
        self.assertEqual(set(ids.ravel().tolist()), set(range(20, 28)))

    def test_pose_uses_shared_three_dimensional_coordinates(self):
        config = build_config()
        K = np.array([[1800., 0, 1300.], [0, 1800., 970.], [0, 0, 1.]])
        D = np.zeros(5)
        rvec = np.array([0.1, -0.2, 0.05])
        tvec = np.array([10., 20., 700.])
        corners, ids = [], []
        for marker_id in (20, 21, 22):
            obj = np.asarray(config["markers"][str(marker_id)]["corners_mm"], np.float32)
            img, _ = cv2.projectPoints(obj, rvec, tvec, K, D)
            corners.append(img.reshape(1, 4, 2))
            ids.append([marker_id])
        R, t, count, rms = marker_pose(corners, np.asarray(ids), config, K, D)
        expected, _ = cv2.Rodrigues(rvec)
        self.assertEqual(count, 3)
        self.assertLess(np.linalg.norm(R - expected), 1e-3)
        self.assertLess(np.linalg.norm(t - tvec), 1e-2)
        self.assertLess(rms, 1e-2)

    def test_stereo_pose_recovers_nearly_front_facing_target(self):
        config = build_config()
        K1 = np.array([[3700., 0, 1268.], [0, 3700., 1004.], [0, 0, 1.]])
        K2 = np.array([[3686., 0, 1259.], [0, 3686., 988.], [0, 0, 1.]])
        D = np.zeros(5)
        Rlr = np.eye(3)
        tlr = np.array([-98.6, 0., 0.])
        rvec = np.array([0.08, -0.04, 0.03])
        R, _ = cv2.Rodrigues(rvec)
        tvec = np.array([14., 8., 690.])
        left_corners, right_corners, ids = [], [], []
        for marker_id in (20, 21, 26, 27):
            obj = np.asarray(config["markers"][str(marker_id)]["corners_mm"], np.float32)
            left, _ = cv2.projectPoints(obj, rvec, tvec, K1, D)
            right_rvec, _ = cv2.Rodrigues(Rlr @ R)
            right, _ = cv2.projectPoints(obj, right_rvec, Rlr @ tvec + tlr, K2, D)
            left_corners.append(left.reshape(1, 4, 2))
            right_corners.append(right.reshape(1, 4, 2))
            ids.append([marker_id])
        result = stereo_marker_pose(left_corners, np.asarray(ids),
                                    right_corners, np.asarray(ids), config,
                                    K1, D, K2, D, Rlr, tlr)
        self.assertEqual(result[2], 4)
        self.assertLess(np.linalg.norm(result[0] - R), 1e-3)
        self.assertLess(np.linalg.norm(result[1] - tvec), 1e-2)


if __name__ == "__main__":
    unittest.main()
