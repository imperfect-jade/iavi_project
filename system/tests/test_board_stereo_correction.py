import copy
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def synthetic_session(noise_px=0.1):
    rng = np.random.default_rng(11)
    size = (2592, 1944)
    K1 = np.array([[3900., 0, 1160.], [0, 3890., 1120.], [0, 0, 1.]])
    K2 = np.array([[3850., 0, 1220.], [0, 3870., 1050.], [0, 0, 1.]])
    D1 = np.zeros((1, 5))
    D2 = np.zeros((1, 5))
    original_R = cv2.Rodrigues(np.array([.018, .025, -.01]))[0]
    original_t = np.array([[-96.9], [1.7], [3.1]])
    true_R = cv2.Rodrigues(np.array([.012, 0, 0]))[0] @ original_R
    true_t = np.array([[-96.7], [-.5], [3.0]])
    world = np.array([[10*x, 10*y, 0] for y in range(4) for x in range(6)], np.float32)
    rows = []
    for i in range(24):
        pose = i // 12
        angle = i % 12
        left_R = cv2.Rodrigues(np.array([.06 + .006*np.sin(angle),
                                         -.12 + .005*np.cos(angle), .02]))[0]
        left_t = np.array([[-160. + 6*np.sin(angle)],
                           [(-25. if pose == 0 else 18.) + 5*np.cos(angle)],
                           [750. + 8*np.sin(angle)]])
        right_R = true_R @ left_R
        right_t = true_R @ left_t + true_t
        left, _ = cv2.projectPoints(world, cv2.Rodrigues(left_R)[0], left_t, K1, D1)
        right, _ = cv2.projectPoints(world, cv2.Rodrigues(right_R)[0], right_t, K2, D2)
        left = (left + rng.normal(0, noise_px, left.shape)).astype(np.float32)
        right = (right + rng.normal(0, noise_px, right.shape)).astype(np.float32)
        rows.append({"name": f"view_{i:03d}/repeat_001", "pose_id": pose,
                     "object": world.reshape(-1, 1, 3), "left": left, "right": right})
    original = {"image_size": np.array(size), "K_left": K1, "D_left": D1,
                "K_right": K2, "D_right": D2,
                "R_right_from_left": original_R,
                "t_right_from_left_mm": original_t}
    return rows, original, true_R, true_t


class BoardStereoCorrectionTests(unittest.TestCase):
    def test_fits_changed_geometry_and_preserves_original_baseline(self):
        from board_stereo_correction import estimate_session_geometry

        rows, original, true_R, _ = synthetic_session()
        corrected, report = estimate_session_geometry(rows, original)
        self.assertEqual(report["status"], "corrected")
        self.assertEqual(len(report["holdout_views"]), 6)
        self.assertLess(report["holdout_max_median_abs_px"], 1)
        self.assertLess(report["holdout_max_p95_abs_px"], 2)
        self.assertAlmostEqual(np.linalg.norm(corrected["t_right_from_left_mm"]),
                               np.linalg.norm(original["t_right_from_left_mm"]), places=6)
        rotation_error = np.rad2deg(np.linalg.norm(cv2.Rodrigues(
            corrected["R_right_from_left"] @ true_R.T)[0]))
        self.assertLess(rotation_error, .1)
        self.assertLess(max(x["median_abs_px"] for x in report["corrected_per_view"]), 1)

    def test_rejects_inconsistent_held_out_pairs(self):
        from board_stereo_correction import estimate_session_geometry

        rows, original, _, _ = synthetic_session()
        damaged = copy.deepcopy(rows)
        for i in (3, 7, 11, 15, 19, 23):
            damaged[i]["right"][:, :, 1] += 10
        with self.assertRaisesRegex(ValueError, "held-out"):
            estimate_session_geometry(damaged, original)

    def test_requires_enough_stereo_board_pairs(self):
        from board_stereo_correction import estimate_session_geometry

        rows, original, _, _ = synthetic_session()
        with self.assertRaisesRegex(ValueError, "at least 8"):
            estimate_session_geometry(rows[:7], original)

    def test_keeps_original_when_board_check_already_passes(self):
        from board_stereo_correction import estimate_session_geometry

        rows, original, true_R, true_t = synthetic_session()
        original["R_right_from_left"] = true_R
        original["t_right_from_left_mm"] = true_t
        result, report = estimate_session_geometry(rows, original)
        self.assertEqual(report["status"], "unchanged")
        self.assertIs(result, original)

    def test_small_scan_can_keep_original_when_it_passes(self):
        from board_stereo_correction import estimate_session_geometry

        rows, original, true_R, true_t = synthetic_session()
        original["R_right_from_left"] = true_R
        original["t_right_from_left_mm"] = true_t
        result, report = estimate_session_geometry(rows[:4], original)
        self.assertEqual(report["status"], "unchanged")
        self.assertIs(result, original)


if __name__ == "__main__":
    unittest.main()
