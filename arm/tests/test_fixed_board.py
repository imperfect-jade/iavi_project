import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "arm"))

from fixed_board import (board_from_config, detect_corners,
                         detect_corners_with_contrast_fallback, estimate_pose,
                         relative_transform)
from calibrate_fixed_board import calibrate
from capture_fixed_board import annotate_corners


class FixedBoardTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "arm/targets/charuco_7x5_10mm_7mm_config.json").read_text())
        self.board = board_from_config(self.config)

    def test_project_board_detects_all_internal_corners(self):
        image = cv2.imread(str(ROOT / "arm/targets/charuco_7x5_10mm_7mm_85x65mm.png"))
        corners, ids = detect_corners(image, self.board)
        self.assertEqual(len(ids), 24)
        self.assertEqual(len(corners), 24)
        self.assertEqual(annotate_corners(image, corners, ids).shape, image.shape)

    def test_low_contrast_photo_uses_fallback(self):
        image = cv2.imread(str(ROOT / "arm/targets/charuco_7x5_10mm_7mm_85x65mm.png"), 0)
        noise = np.random.default_rng(10).normal(0, 2, image.shape)
        weak = np.clip(190 + 0.08 * (image.astype(float) - 127) + noise, 0, 255).astype(np.uint8)
        self.assertLess(len(detect_corners(weak, self.board)[1]), 12)
        corners, ids, method = detect_corners_with_contrast_fallback(weak, self.board, 12)
        self.assertGreaterEqual(len(ids), 12)
        self.assertEqual(len(corners), len(ids))
        self.assertEqual(method, "clahe")

    def test_synthetic_pose_and_relative_transform(self):
        K = np.array([[3700., 0., 1296.], [0., 3700., 972.], [0., 0., 1.]])
        D = np.zeros(5)
        xyz = self.board.getChessboardCorners().astype(np.float64)
        r1 = np.array([0.1, -0.2, 0.05])
        t1 = np.array([5., -10., 600.])
        r2 = np.array([0.15, 0.1, -0.1])
        t2 = np.array([-30., 20., 640.])
        measured = []
        for r, t in ((r1, t1), (r2, t2)):
            image, _ = cv2.projectPoints(xyz, r, t, K, D)
            T, rms = estimate_pose(self.board, image, np.arange(24).reshape(-1, 1), K, D)
            self.assertLess(rms, 1e-4)
            measured.append(T)
        T_1_2 = relative_transform(measured[0], measured[1])
        point_in_2 = measured[1] @ np.array([12., 24., 0., 1.])
        point_in_1 = measured[0] @ np.array([12., 24., 0., 1.])
        np.testing.assert_allclose(T_1_2 @ point_in_2, point_in_1, atol=1e-3)

    def test_offline_calibration_reads_capture_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            image = cv2.imread(str(ROOT / "arm/targets/charuco_7x5_10mm_7mm_85x65mm.png"))
            shifted = cv2.warpAffine(image, np.float32([[1, 0, 35], [0, 1, 10]]),
                                      (image.shape[1], image.shape[0]),
                                      borderValue=(255, 255, 255))
            rows = []
            for index, (pose, frame) in enumerate((('upper', image), ('lower', shifted), ('upper', image))):
                name = f'capture_{index:03d}_{pose}'
                folder = base / name
                folder.mkdir()
                (folder / 'left.png').write_bytes(cv2.imencode('.png', frame)[1].tobytes())
                rows.append({'name': name, 'pose': pose, 'left_image': f'{name}/left.png'})
            manifest = {'complete': True, 'board': self.config,
                        'camera_frame': 'original_left_camera',
                        'camera_size_px': [1700, 1300],
                        'left_serial': 'left', 'right_serial': 'right',
                        'observations': rows}
            (base / 'capture_manifest.json').write_text(json.dumps(manifest))
            stereo = {'K_left': [[1500, 0, 850], [0, 1500, 650], [0, 0, 1]],
                      'D_left': [0, 0, 0, 0, 0], 'image_size': [1700, 1300],
                      'left_serial': 'left', 'right_serial': 'right'}
            stereo_path = base / 'stereo.json'
            stereo_path.write_text(json.dumps(stereo))
            result = calibrate(base, stereo_path,
                               ROOT / 'arm/targets/charuco_7x5_10mm_7mm_config.json')
            self.assertEqual(result['summary']['upper']['count'], 2)
            self.assertAlmostEqual(result['summary']['upper']['max_return_translation_mm'], 0, places=2)
            self.assertGreater(result['summary']['lower']['max_reprojection_rms_px'], 0)
            self.assertEqual(len(result['observations']), 3)
            self.assertEqual(result['reference_capture'], 'capture_001_lower')


if __name__ == "__main__":
    unittest.main()
