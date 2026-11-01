import sys
import tempfile
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import calibrate_stereo
from calibrate_stereo import board_points, pairs, detect, check_validation_error, select_outliers, calibration_flags, find_corners
from calibrate_color import fit_color, chart_samples, apply_color_transform, color_pair_name


class CalibrationTests(unittest.TestCase):
    def test_stereo_calibration_requires_explicit_square_size(self):
        parser = calibrate_stereo.build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["--validation", "validation"])

    def test_stereo_calibration_uses_standard_five_coefficient_model_by_default(self):
        args = calibrate_stereo.build_parser().parse_args(
            ["--validation", "validation", "--square-mm", "12"])
        self.assertEqual(args.distortion_model, "standard5")
        self.assertEqual(args.square_mm, 12)

    def test_color_annotation_name_uses_prefix(self):
        self.assertEqual(color_pair_name("color_001"), "color_001")
        with self.assertRaises(ValueError): color_pair_name("001_color")

    def test_lab5_corner_detector_flags(self):
        with patch("calibrate_stereo.cv2.findChessboardCornersSB", return_value=(False, None)) as detector:
            self.assertIsNone(find_corners(np.zeros((32, 32, 3), np.uint8)))
        self.assertEqual(detector.call_args.kwargs["flags"],
                         cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)

    def test_lab5_board_geometry_and_distortion_models(self):
        obj = board_points(20, 25)
        self.assertTrue(np.array_equal(obj[10], [0, 25, 0]))
        self.assertEqual(calibration_flags("radial2"), cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3)
        self.assertEqual(calibration_flags("standard5"), 0)
        with self.assertRaises(ValueError): calibration_flags("other")

    def test_initial_fit_flags_only_clear_training_outlier(self):
        rows = [{"image": f"calib_{i:03d}.png", "epipolar_median_px": .20 + i*.01,
                 "epipolar_p95_px": .35 + i*.01,
                 "left_reprojection_rms_px": .20,
                 "right_reprojection_rms_px": .22} for i in range(1, 13)]
        rows[-1]["epipolar_median_px"] = 2.4
        rejected = select_outliers(rows)
        self.assertEqual(list(rejected), ["calib_012.png"])
        self.assertIn("epipolar", rejected["calib_012.png"])

    def test_auto_exclusion_removes_training_outlier_and_keeps_reason(self):
        records = [(f"calib_{i:03d}.png",) for i in range(1, 13)]
        rows = [{"image": record[0], "epipolar_median_px": .20 + i*.01,
                 "left_reprojection_rms_px": .20, "right_reprojection_rms_px": .22}
                for i, record in enumerate(records)]
        rows[-1]["epipolar_median_px"] = 2.4
        excluded, remaining, automatic = calibrate_stereo.choose_training_exclusions(
            records, select_outliers(rows), set())
        self.assertEqual([record[0] for record in excluded], ["calib_012.png"])
        self.assertEqual(len(remaining), 11)
        self.assertIn("epipolar", automatic["calib_012.png"])

    def test_auto_exclusion_stops_when_too_many_pairs_are_flagged(self):
        records = [(f"calib_{i:03d}.png",) for i in range(1, 11)]
        suggestions = {record[0]: "high error" for record in records[-3:]}
        with self.assertRaisesRegex(ValueError, "Too many automatic exclusions"):
            calibrate_stereo.choose_training_exclusions(records, suggestions, set())

    def test_manual_exclusion_can_be_used_without_automatic_exclusion(self):
        records = [(f"calib_{i:03d}.png",) for i in range(1, 13)]
        excluded, remaining, automatic = calibrate_stereo.choose_training_exclusions(
            records, {"calib_012.png": "high error"}, {3}, auto_enabled=False)
        self.assertEqual([record[0] for record in excluded], ["calib_003.png"])
        self.assertEqual(len(remaining), 11)
        self.assertEqual(automatic, {})

    def test_bad_independent_validation_is_rejected(self):
        rows = [{"image": "calib_001.png", "epipolar_median_px": 20.4,
                 "epipolar_p95_px": 22.5}]
        with self.assertRaisesRegex(ValueError, "Independent validation"):
            check_validation_error(rows, 2.0)
        check_validation_error([{"image": "calib_002.png", "epipolar_median_px": .4,
                                 "epipolar_p95_px": .8}], 2.0)

    def test_board_uses_millimetres(self):
        obj = board_points(20)
        self.assertEqual(obj.shape, (60, 3))
        self.assertTrue(np.array_equal(obj[1], [20, 0, 0]))

    def test_unpaired_images_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "left").mkdir()
            (root / "right").mkdir()
            (root / "left/calib_001.png").write_bytes(b"x")
            with self.assertRaises(ValueError):
                pairs(root)

    def test_old_calibration_filenames_are_not_discovered(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for side in ("left", "right"):
                folder = root / side
                folder.mkdir()
                (folder / "001_calib.png").touch()
            with self.assertRaises(ValueError):
                pairs(root)

    def test_stereo_detection_reports_unusable_pair(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "left").mkdir()
            (root / "right").mkdir()
            image = np.zeros((80, 80, 3), np.uint8)
            ok, encoded = cv2.imencode(".png", image)
            self.assertTrue(ok)
            for side in ("left", "right"):
                (root / side / "calib_001.png").write_bytes(encoded.tobytes())
            output = StringIO()
            with redirect_stdout(output):
                records, rejected, _ = detect(root)
            self.assertEqual(records, [])
            self.assertEqual(rejected[0]["image"], "calib_001.png")
            self.assertIn("UNUSABLE", output.getvalue())
            self.assertIn("calib_001.png", output.getvalue())

    def test_color_affine_recovery(self):
        right = np.array([[10, 20, 30], [80, 20, 30], [10, 80, 30],
                          [10, 20, 90], [100, 110, 120]], np.float64)
        expected = right * 0.8 + [2, 3, 4]
        matrix = fit_color(right, expected)
        actual = np.column_stack((right, np.ones(len(right)))) @ matrix.T
        self.assertTrue(np.allclose(actual, expected))

    def test_chart_samples_central_patch(self):
        image = np.full((100, 100, 3), (10, 20, 30), np.uint8)
        corners = [[0, 0], [99, 0], [99, 99], [0, 99]]
        self.assertTrue(np.allclose(chart_samples(image, corners), [10, 20, 30]))

    def test_color_transform_clips_and_preserves_image_shape(self):
        image = np.array([[[10, 20, 30], [250, 250, 250]]], np.uint8)
        matrix = np.array([[2, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 10]], np.float64)
        result = apply_color_transform(image, matrix)
        self.assertEqual(result.shape, image.shape)
        self.assertEqual(result.tolist(), [[[20, 20, 40], [255, 250, 255]]])


if __name__ == "__main__":
    unittest.main()
