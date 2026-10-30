import sys
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from reconstruct_gray_neighborhood import (
    match_neighborhood, mutual_matches, triangulate_matches, load_decoded_cache, main,
)


class GrayNeighborhoodTests(unittest.TestCase):
    def test_recovers_neighbor_row_only_inside_window(self):
        left = np.full((5, 12, 2), np.nan, np.float32)
        right = np.full_like(left, np.nan)
        left[1, 8] = [100, 200]
        right[3, 5] = [100.1, 200.1]
        strict, _ = match_neighborhood(left, right, 2, 4, .5, 0)
        narrow, _ = match_neighborhood(left, right, 2, 4, .5, 1)
        wide, score = match_neighborhood(left, right, 2, 4, .5, 2)
        self.assertTrue(np.isnan(strict).all())
        self.assertTrue(np.isnan(narrow).all())
        np.testing.assert_array_equal(wide[1, 8], [5, 3])
        self.assertAlmostEqual(float(score[1, 8]), np.sqrt(.02), places=4)

    def test_checks_projector_y_and_disparity(self):
        left = np.full((3, 16, 2), np.nan, np.float32)
        right = np.full_like(left, np.nan)
        left[1, 10] = [100, 200]
        right[0, 2] = [100, 200]  # outside disparity range
        right[1, 7] = [100, 210]  # wrong projector y
        matches, _ = match_neighborhood(left, right, 2, 4, .5, 1)
        self.assertTrue(np.isnan(matches).all())

    def test_searches_all_repeated_projector_x_candidates(self):
        left = np.full((1, 30, 2), np.nan, np.float32)
        right = np.full_like(left, np.nan)
        left[0, 25] = [100, 200]
        right[0, :20] = [100, 230]
        right[0, 19] = [100, 200]
        matches, _ = match_neighborhood(left, right, 1, 26, .5, 0)
        np.testing.assert_array_equal(matches[0, 25], [19, 0])

    def test_rejects_ambiguous_codes(self):
        left = np.full((3, 12, 2), np.nan, np.float32)
        right = np.full_like(left, np.nan)
        left[1, 8] = [100, 200]
        right[0, 5] = [100, 200]
        right[2, 5] = [100, 200]
        matches, _ = match_neighborhood(left, right, 2, 4, .5, 1)
        self.assertTrue(np.isnan(matches).all())

    def test_outside_code_tolerance_does_not_make_valid_match_ambiguous(self):
        left = np.full((1, 12, 2), np.nan, np.float32)
        right = np.full_like(left, np.nan)
        left[0, 8] = [100, 200]
        right[0, 5] = [100.48, 200]
        right[0, 6] = [100, 200.52]
        matches, _ = match_neighborhood(left, right, 2, 4, .5, 0)
        np.testing.assert_array_equal(matches[0, 8], [5, 0])

    def test_target_uniqueness_is_global_across_rows(self):
        left = np.full((3, 12, 2), np.nan, np.float32)
        right = np.full_like(left, np.nan)
        left[0, 8] = [100, 200]
        left[2, 8] = [100.2, 200]
        right[1, 5] = [100, 200]
        matches, _ = match_neighborhood(left, right, 2, 4, .5, 2)
        np.testing.assert_array_equal(matches[0, 8], [5, 1])
        self.assertTrue(np.isnan(matches[2, 8]).all())

    def test_reverse_consistency_checks_both_pixel_axes(self):
        forward = np.full((5, 12, 2), np.nan, np.float32)
        reverse = np.full_like(forward, np.nan)
        forward[1, 8] = [5, 3]
        reverse[3, 5] = [8, 1]
        self.assertTrue(mutual_matches(forward, reverse)[1, 8])
        reverse[3, 5] = [8, 2]
        self.assertFalse(mutual_matches(forward, reverse)[1, 8])

    def test_empty_images_and_invalid_parameters(self):
        xy = np.full((3, 4, 2), np.nan, np.float32)
        matches, _ = match_neighborhood(xy, xy, 1, 3, .5, 8)
        self.assertTrue(np.isnan(matches).all())
        for radius in [-1, 1.5]:
            with self.assertRaises(ValueError):
                match_neighborhood(xy, xy, 1, 3, .5, radius)

    def test_triangulates_actual_y_with_vertical_camera_translation(self):
        p1 = np.array([[100, 0, 0, 0], [0, 100, 0, 0], [0, 0, 1, 0]], float)
        p2 = p1.copy()
        p2[:, 3] = [-2000, 1000, 0]
        matches = np.full((40, 50, 2), np.nan, np.float32)
        matches[20, 30] = [10, 30]
        calibration = {'P_left': p1, 'P_right': p2}
        xyz, residual = triangulate_matches(matches, calibration, 1)
        np.testing.assert_allclose(xyz[20, 30], [30, 20, 100], atol=1e-4)
        self.assertLess(float(residual[20, 30]), 1e-5)

    def test_inconsistent_y_has_nonzero_reprojection_error(self):
        p1 = np.array([[100, 0, 0, 0], [0, 100, 0, 0], [0, 0, 1, 0]], float)
        p2 = p1.copy()
        p2[0, 3] = -2000
        matches = np.full((40, 50, 2), np.nan, np.float32)
        matches[20, 30] = [10, 24]
        xyz, residual = triangulate_matches(matches, {'P_left': p1, 'P_right': p2}, 1)
        self.assertTrue(np.isfinite(xyz[20, 30]).all())
        self.assertGreater(float(residual[20, 30]), 1.9)

    def test_scaled_projection_preserves_3d_coordinates(self):
        p1 = np.array([[100, 0, 0, 0], [0, 100, 0, 0], [0, 0, 1, 0]], float)
        p2 = p1.copy()
        p2[:, 3] = [-2000, 1000, 0]
        matches = np.full((20, 25, 2), np.nan, np.float32)
        matches[10, 15] = [5, 15]
        xyz, _ = triangulate_matches(matches, {'P_left': p1, 'P_right': p2}, .5)
        np.testing.assert_allclose(xyz[10, 15], [30, 20, 100], atol=1e-4)

    def test_negative_camera_depth_is_rejected(self):
        p1 = np.array([[100, 0, 0, 0], [0, 100, 0, 0], [0, 0, 1, 0]], float)
        p2 = p1.copy()
        p2[0, 3] = -2000
        matches = np.full((40, 50, 2), np.nan, np.float32)
        matches[20, 10] = [30, 20]
        xyz, residual = triangulate_matches(matches, {'P_left': p1, 'P_right': p2}, 1)
        self.assertTrue(np.isnan(xyz).all())
        self.assertTrue(np.isnan(residual).all())

    def test_reverse_out_of_frame_target_is_rejected(self):
        forward = np.full((3, 4, 2), np.nan, np.float32)
        reverse = np.full_like(forward, np.nan)
        forward[0, 0] = [-1, 0]
        forward[2, 3] = [0, 3]
        self.assertFalse(mutual_matches(forward, reverse).any())

    def test_matches_small_random_case_against_exhaustive_oracle(self):
        rng = np.random.default_rng(12)
        left = rng.uniform(0, 8, (4, 9, 2)).astype(np.float32)
        right = rng.uniform(0, 8, (4, 9, 2)).astype(np.float32)
        left[rng.random((4, 9)) < .2] = np.nan
        right[rng.random((4, 9)) < .2] = np.nan
        expected = np.full_like(left, np.nan)
        selected = []
        for y in range(4):
            for x in range(9):
                candidates = []
                for yr in range(max(0, y-1), min(4, y+2)):
                    for xr in range(9):
                        distance = float(np.linalg.norm(left[y, x]-right[yr, xr]))
                        if 1 <= x-xr <= 4 and distance <= 2:
                            candidates.append((distance, xr, yr))
                candidates.sort()
                if candidates and (len(candidates) == 1 or candidates[1][0]-candidates[0][0] > .1):
                    score, xr, yr = candidates[0]
                    selected.append((score, y, x, xr, yr))
        used = set()
        for _, y, x, xr, yr in sorted(selected):
            if (xr, yr) not in used:
                used.add((xr, yr))
                expected[y, x] = [xr, yr]
        actual, _ = match_neighborhood(left, right, 1, 4, 2, 1)
        np.testing.assert_allclose(actual, expected, equal_nan=True)

    def test_cache_rejects_scale_and_dimension_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            calibration_path = root/'calibration.npz'
            np.savez(calibration_path, image_size=[24, 16])
            path = root/'decoded.npz'
            xy = np.full((8, 12, 2), np.nan, np.float32)
            np.savez(path, left=xy, right=xy, left_valid=np.zeros((8, 12), bool),
                     right_valid=np.zeros((8, 12), bool))
            (root/'qc.json').write_text(json.dumps({'scale': .5, 'calibration': str(calibration_path)}))
            calibration = {'image_size': np.array([24, 16])}
            with self.assertRaisesRegex(ValueError, 'scale'):
                load_decoded_cache(path, calibration, 1, calibration_path)
            with self.assertRaisesRegex(ValueError, 'shape|dimensions'):
                load_decoded_cache(path, {'image_size': np.array([30, 16])}, .5, calibration_path)
            alternate = root/'alternate.npz'
            alternate.write_bytes(calibration_path.read_bytes())
            with self.assertRaisesRegex(ValueError, 'calibration path'):
                load_decoded_cache(path, calibration, .5, alternate)
            (root/'qc.json').write_text(json.dumps({'scale': .5, 'calibration': str(calibration_path),
                                                  'calibration_sha256': '0'*64}))
            with self.assertRaisesRegex(ValueError, 'hash'):
                load_decoded_cache(path, calibration, .5, calibration_path)

    def test_cached_cli_writes_verified_coordinates_and_experimental_qc(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            p1 = np.array([[100, 0, 0, 0], [0, 100, 0, 0], [0, 0, 1, 0]], float)
            p2 = p1.copy()
            p2[:, 3] = [-600, 400, 0]
            calibration = root/'calibration.npz'
            np.savez(calibration, image_size=[24, 16], P_left=p1, P_right=p2, rectify_left=np.eye(3))
            left = np.full((8, 12, 2), np.nan, np.float32)
            right = np.full_like(left, np.nan)
            left[1, 8] = [100, 200]
            right[3, 5] = [100, 200]
            cache = root/'decoded.npz'
            np.savez(cache, left=left, right=right,
                     left_valid=np.isfinite(left).all(2), right_valid=np.isfinite(right).all(2))
            (root/'qc.json').write_text(json.dumps({'scale': .5, 'calibration': str(calibration)}))
            output = root/'output'
            result = main(['--decoded', str(cache), '--calibration', str(calibration),
                '--output', str(output), '--row-radius', '2', '--point-stride', '1',
                '--min-depth-mm', '50', '--max-depth-mm', '150'])
            self.assertEqual(result, 0)
            qc = json.loads((output/'qc.json').read_text())
            self.assertEqual(qc['point_count'], 1)
            self.assertEqual(qc['quality_status'], 'experimental_geometry_unverified')
            self.assertEqual(qc['color_source'], 'neutral_gray')
            self.assertAlmostEqual(float(np.load(output/'depth_mm.npy')[1, 8]), 100, places=3)
            np.testing.assert_array_equal(np.load(output/'right_xy_px.npy')[1, 8], [5, 3])
            with self.assertRaisesRegex(ValueError, 'exists'):
                main(['--decoded', str(cache), '--calibration', str(calibration), '--output', str(output)])
            # Actual cross-row matching survives, but inconsistent geometry must
            # be rejected by the explicit reprojection gate.
            p2[1, 3] = 0
            np.savez(calibration, image_size=[24, 16], P_left=p1, P_right=p2, rectify_left=np.eye(3))
            rejected_output = root/'geometry_rejected'
            result = main(['--decoded', str(cache), '--calibration', str(calibration),
                '--output', str(rejected_output), '--row-radius', '2', '--point-stride', '1',
                '--min-depth-mm', '50', '--max-depth-mm', '150', '--max-reprojection-error-px', '.5'])
            self.assertEqual(result, 2)
            rejected_qc = json.loads((rejected_output/'qc.json').read_text())
            self.assertEqual(rejected_qc['mutual_matches'], 1)
            self.assertEqual(rejected_qc['geometry_rejected_pixels'], 1)
            self.assertEqual(rejected_qc['point_count'], 0)

    def test_complete_raw_gray_phase_sequence_end_to_end(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            view = root/'view'
            for side in ('left', 'right'):
                (view/side).mkdir(parents=True)
            p1 = np.array([[100, 0, 0, 0], [0, 100, 0, 0], [0, 0, 1, 0]], float)
            p2 = p1.copy()
            p2[:, 3] = [-600, 400, 0]
            calibration = root/'calibration.npz'
            np.savez(calibration, image_size=[24, 16], P_left=p1, P_right=p2,
                     K_left=p1[:, :3], K_right=p1[:, :3], D_left=np.zeros(5),
                     D_right=np.zeros(5), rectify_left=np.eye(3), rectify_right=np.eye(3))
            calibration.with_suffix('.json').write_text(json.dumps({'left_serial': 'L', 'right_serial': 'R'}))
            names = [f'gray_{axis}_{bit:02d}{suffix}' for axis in ('x', 'y')
                     for bit in range(6, -1, -1) for suffix in ('', '_inv')]
            names += [f'phase_{axis}_{step:02d}' for axis in ('x', 'y') for step in range(4)]
            names += ['white', 'black']
            frames = []
            for index, name in enumerate(names):
                if name == 'white':
                    intensity = 220
                elif name == 'black':
                    intensity = 20
                else:
                    parts = name.split('_')
                    coordinate = 100 if parts[1] == 'x' else 200
                    if parts[0] == 'gray':
                        cell = coordinate//16
                        bit = ((cell ^ (cell >> 1)) >> int(parts[2])) & 1
                        if name.endswith('_inv'):
                            bit = 1-bit
                        intensity = 210 if bit else 30
                    else:
                        intensity = round(120+80*np.cos(2*np.pi*(coordinate % 16)/16-int(parts[2])*np.pi/2))
                row = {'index': index, 'pattern': name}
                for side, (x, y) in [('left', (16, 2)), ('right', (10, 6))]:
                    image = np.full((16, 24), 20, np.uint8)
                    image[y, x] = intensity
                    filename = f'{index:03d}_{name}.png'
                    cv2.imencode('.png', image)[1].tofile(view/side/filename)
                    row[side] = side+'/'+filename
                frames.append(row)
            manifest = {'complete': True, 'projector_size_px': [1920, 1080],
                'camera_size_px': [24, 16], 'left_serial': 'L', 'right_serial': 'R',
                'protocol': 'gray_phase_c16_v1', 'pattern_order': names,
                'frame_count': len(names), 'frames': frames}
            (view/'manifest.json').write_text(json.dumps(manifest))
            output = root/'output'
            result = main(['--view', str(view), '--calibration', str(calibration),
                '--output', str(output), '--row-radius', '2', '--point-stride', '1',
                '--min-depth-mm', '50', '--max-depth-mm', '150'])
            self.assertEqual(result, 0)
            qc = json.loads((output/'qc.json').read_text())
            self.assertEqual(qc['point_count'], 1)
            self.assertEqual(qc['color_source'], 'raw_left_white_rectified')
            self.assertAlmostEqual(float(np.load(output/'depth_mm.npy')[1, 8]), 100, places=3)
            # A manifest can lie about image dimensions even when all frames
            # agree with one another; check the actual decoder image shape.
            with np.load(calibration) as archive:
                changed = {key: archive[key].copy() for key in archive.files}
            changed['image_size'] = np.array([48, 32])
            np.savez(calibration, **changed)
            manifest['camera_size_px'] = [48, 32]
            (view/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'camera image dimensions'):
                main(['--view', str(view), '--calibration', str(calibration),
                    '--output', str(root/'wrong_dimensions'), '--row-radius', '2',
                    '--point-stride', '1', '--min-depth-mm', '50', '--max-depth-mm', '150'])


if __name__ == '__main__':
    unittest.main()
