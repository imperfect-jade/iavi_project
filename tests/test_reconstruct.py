import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from reconstruct_view import decode_axis, match_rows, q_for_scale, white_alignment_warnings, rectify_decoded, stereo_disparity, estimate_disparity_range


class ReconstructionTests(unittest.TestCase):
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

    def test_rectification_keeps_valid_points_next_to_invalid_region(self):
        yy, xx = np.indices((5, 7), dtype=np.float32)
        xy = np.stack((100 + xx, 200 + yy), axis=-1)
        valid = np.ones((5, 7), bool)
        valid[:, 3] = False
        xy[~valid] = np.nan
        mapped, mask = rectify_decoded(xy, valid, (xx, yy))
        np.testing.assert_array_equal(mask, valid)
        np.testing.assert_allclose(mapped[valid], xy[valid])
        self.assertTrue(np.isnan(mapped[~valid]).all())

    def test_rectification_rejects_real_projector_coordinate_jump(self):
        yy, xx = np.indices((5, 7), dtype=np.float32)
        xy = np.stack((100 + xx, 200 + yy), axis=-1)
        xy[:, 3:, 0] += 16  # Gray cell address error between two valid regions.
        _, mask = rectify_decoded(xy, np.ones((5, 7), bool), (xx, yy))
        self.assertFalse(mask[:, 2:4].any())
        self.assertTrue(mask[:, :2].all())
        self.assertTrue(mask[:, 4:].all())

    def test_large_white_epipolar_error_is_warning(self):
        alignment = {"feature_matches": 22, "median_abs_epipolar_px": 16.67,
                     "p95_abs_epipolar_px": 18.27}
        warnings = white_alignment_warnings(alignment)
        self.assertEqual(len(warnings), 1)
        self.assertIn("16.67", warnings[0])
        self.assertEqual(white_alignment_warnings({**alignment,
            "median_abs_epipolar_px": 0.5}), [])
        self.assertEqual(white_alignment_warnings(None), [])

    def test_gray_phase_unwraps_both_cells_and_fraction(self):
        width, height, cell = 32, 8, 8
        u = np.broadcast_to(np.arange(width)[None, :], (height, width))
        g = (u // cell) ^ ((u // cell) >> 1)
        gray_pairs = []
        for bit in (1, 0):
            bright = np.where((g >> bit) & 1, 210, 30).astype(np.uint8)
            gray_pairs.append((bright, (240-bright).astype(np.uint8)))
        phase = [np.rint(120 + 90*np.cos(2*np.pi*(u % cell)/cell-k*np.pi/2)).astype(np.uint8)
                 for k in range(4)]
        coordinate, valid, _ = decode_axis(gray_pairs, phase, cell, width, 10, 12)
        self.assertTrue(valid.all())
        self.assertLess(np.max(np.abs(coordinate-u)), .05)
        self.assertGreater(coordinate[0, 8], coordinate[0, 7])

    def test_match_rows_checks_both_projector_axes(self):
        left = np.full((2, 12, 2), np.nan, np.float32)
        right = np.full_like(left, np.nan)
        for x in range(6, 10):
            left[0, x] = [x, 4]
            right[0, x-2] = [x, 4]
        right[0, 4, 1] = 20  # wrong y projector coordinate
        d = match_rows(left, right, 1, 4, .5)
        self.assertTrue(np.isnan(d[0, 6]))
        self.assertEqual(d[0, 7], 2)

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
