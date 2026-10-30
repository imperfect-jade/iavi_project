import sys
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from reconstruct_view import load_run, q_for_scale, white_alignment_warnings, stereo_disparity, estimate_disparity_range, fuse_disparities, disparity_bounds_from_matches


class ReconstructionTests(unittest.TestCase):
    def test_sparse_dot_frames_can_estimate_one_pooled_range(self):
        frames = [[218.0, 219.0, 220.0] for _ in range(5)]
        with self.assertRaisesRegex(ValueError, "only 3 reliable matches"):
            disparity_bounds_from_matches(frames[0], 1296)
        far, near, count = disparity_bounds_from_matches(
            [d for frame in frames for d in frame], 1296)
        self.assertEqual(count, 15)
        self.assertLess(far, 218)
        self.assertGreater(near, 220)

    def test_gray_phase_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "manifest.json").write_text(json.dumps({
                "projector_size_px": [1920, 1080], "complete": True,
                "protocol": "gray_phase_c16_v1"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unsupported protocol"):
                load_run(tmp, {})

    def test_fusion_fills_holes_and_rejects_single_outlier(self):
        a = np.array([[10, np.nan, 10]], np.float32)
        b = np.array([[10.2, 12, 10.2]], np.float32)
        c = np.array([[50, 12.2, 10.1]], np.float32)
        fused, support = fuse_disparities([a, b, c])
        np.testing.assert_allclose(fused, [[10.1, 12.1, 10.1]], atol=.11)
        np.testing.assert_array_equal(support, [[2, 2, 3]])

    def test_auto_range_uses_matched_scene_disparities(self):
        rng = np.random.default_rng(5)
        left = rng.integers(0, 2, (96, 256), dtype=np.uint8) * 255
        right = np.zeros_like(left)
        right[:, :-16] = left[:, 16:]
        far, near, count = estimate_disparity_range(left, right)
        self.assertGreater(count, 20)
        self.assertLess(far, 16)
        self.assertGreater(near, 16)
        self.assertLess(near, 40)

    def test_auto_range_requires_scene_features(self):
        blank = np.zeros((96, 256), np.uint8)
        with self.assertRaisesRegex(ValueError, "Cannot estimate disparity"):
            estimate_disparity_range(blank, blank)

    def test_dense_dots_sgbm_recovers_known_shift(self):
        rng = np.random.default_rng(5)
        left = rng.integers(0, 2, (96, 256), dtype=np.uint8) * 255
        right = np.zeros_like(left)
        right[:, :-16] = left[:, 16:]
        disparity = stereo_disparity(left, right, 0, 64)
        center = disparity[20:76, 80:180]
        self.assertGreater(np.isfinite(center).mean(), .5)
        self.assertAlmostEqual(float(np.nanmedian(center)), 16, delta=1)

    def test_large_white_epipolar_error_is_warning(self):
        alignment = {"feature_matches": 22, "median_abs_epipolar_px": 16.67,
                     "p95_abs_epipolar_px": 18.27}
        warnings = white_alignment_warnings(alignment)
        self.assertEqual(len(warnings), 1)
        self.assertIn("16.67", warnings[0])
        self.assertEqual(white_alignment_warnings({**alignment,
            "median_abs_epipolar_px": 0.5}), [])
        self.assertEqual(white_alignment_warnings(None), [])

    def test_q_scale_preserves_3d_point(self):
        q = np.array([[1, 0, 0, -100], [0, 1, 0, -50],
                      [0, 0, 0, 500], [0, 0, .01, 0]], float)
        sample = np.array([200, 100, 20, 1])
        half = np.array([100, 50, 10, 1])
        actual = q_for_scale(q, .5) @ half
        expected = q @ sample
        self.assertTrue(np.allclose(actual/actual[3], expected/expected[3]))


if __name__ == "__main__":
    unittest.main()
