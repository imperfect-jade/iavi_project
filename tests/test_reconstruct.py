import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from reconstruct_view import decode_axis, match_rows, q_for_scale


class ReconstructionTests(unittest.TestCase):
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
