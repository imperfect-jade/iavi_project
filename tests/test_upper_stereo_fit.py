import sys
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))


class UpperPoseCalibrationTests(unittest.TestCase):
    def test_fits_upper_extrinsics_with_shared_intrinsics_and_baseline(self):
        from upper_stereo_fit import fit_upper_pose

        rng = np.random.default_rng(21)
        K = np.array([[3800., 0, 1296.], [0, 3800., 972.], [0, 0, 1.]])
        R_lower = np.eye(3)
        R_upper = cv2.Rodrigues(np.array([.007, 0., 0.]))[0]
        t_lower = np.array([[-100.], [0.], [0.]])
        t_upper = np.array([[-99.98], [.5], [1.]])
        obj = np.column_stack((np.mgrid[0:10, 0:6].T.reshape(-1, 2) * 12,
                               np.zeros(60))).astype(np.float32)
        lower = {"image_size": np.array([2592, 1944]), "K_left": K,
                 "D_left": np.zeros((1, 5)), "K_right": K.copy(),
                 "D_right": np.zeros((1, 5)),
                 "R_right_from_left": R_lower,
                 "t_right_from_left_mm": t_lower}
        records = []
        for i in range(29):
            board_R = cv2.Rodrigues(np.array([.08 + .2*np.sin(i),
                                              .12*np.cos(i), .05*np.sin(i/2)]))[0]
            board_t = np.array([[-120. + 60*np.sin(i)],
                                [-80. + 70*np.cos(i)], [760. + 80*np.sin(i/3)]])
            left, _ = cv2.projectPoints(obj, cv2.Rodrigues(board_R)[0],
                                         board_t, K, lower["D_left"])
            right, _ = cv2.projectPoints(obj,
                cv2.Rodrigues(R_upper @ board_R)[0],
                R_upper @ board_t + t_upper, K, lower["D_right"])
            left = (left + rng.normal(0, .1, left.shape)).astype(np.float32)
            right = (right + rng.normal(0, .1, right.shape)).astype(np.float32)
            records.append((f"calib_{i+1:03d}.png", left, right, {}))
        train, validation = records[:24], records[24:]
        upper, report = fit_upper_pose(train, lower, 12.0, validation_records=validation)
        self.assertEqual(report["heldout_pair_count"], 6)
        self.assertLess(report["holdout_max_p95_px"], 1.)
        self.assertEqual(report["validation_pair_count"], 5)
        self.assertLess(report["validation_max_p95_px"], 1.)
        np.testing.assert_array_equal(upper["K_left"], lower["K_left"])
        np.testing.assert_array_equal(upper["D_right"], lower["D_right"])
        self.assertAlmostEqual(np.linalg.norm(upper["t_right_from_left_mm"]),
                               100., places=5)
        error_deg = np.rad2deg(np.linalg.norm(cv2.Rodrigues(
            upper["R_right_from_left"] @ R_upper.T)[0]))
        self.assertLess(error_deg, .1)

        damaged = list(train)
        name, left, right, details = damaged[3]
        damaged[3] = (name, left, right + np.array([0., 10.], np.float32), details)
        with self.assertRaisesRegex(ValueError, "held-out"):
            fit_upper_pose(damaged, lower, 12.0, validation_records=validation)

        damaged_validation = list(validation)
        name, left, right, details = damaged_validation[0]
        damaged_validation[0] = (name, left, right + np.array([0., 10.], np.float32), details)
        with self.assertRaisesRegex(ValueError, "independent validation"):
            fit_upper_pose(train, lower, 12.0, validation_records=damaged_validation)

        from upper_stereo_fit import calibrate_upper
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            lower_path = root / "lower.npz"
            np.savez_compressed(lower_path, **lower)
            lower_path.with_suffix(".json").write_text(
                json.dumps({"left_serial": "left", "right_serial": "right"}), encoding="utf-8")
            output = root / "upper"
            output.mkdir()
            published = output / "stereo_calibration.npz"
            published.write_bytes(b"previous-good-model")
            args = SimpleNamespace(lower_calibration=lower_path, train=root / "train",
                validation=root / "validation", square_mm=12.0, square_y_mm=None,
                output=output)
            with patch("upper_stereo_fit.detect", side_effect=[
                (train, [], (2592, 1944)), (damaged_validation, [], (2592, 1944))]):
                with self.assertRaisesRegex(ValueError, "independent validation"):
                    calibrate_upper(args)
            self.assertEqual(published.read_bytes(), b"previous-good-model")


if __name__ == "__main__":
    unittest.main()
