import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fusion_pipeline


class FusionPipelineTests(unittest.TestCase):
    def test_missing_axis_capture_is_rejected_before_scanning(self):
        with tempfile.TemporaryDirectory() as tmp:
            images = Path(tmp)
            with self.assertRaisesRegex(ValueError, "complete axis capture"):
                fusion_pipeline.validate_axis_capture(images)
            (images / "capture_manifest.json").write_text(
                json.dumps({"complete": False}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "complete axis capture"):
                fusion_pipeline.validate_axis_capture(images)
            (images / "capture_manifest.json").write_text(
                json.dumps({"complete": True}), encoding="utf-8")
            fusion_pipeline.validate_axis_capture(images)

    def test_registers_axis_camera_through_fixed_board_in_scan_reference(self):
        axis_board = np.eye(4)
        axis_board[:3, 3] = [10, 20, 30]
        scan_board = np.eye(4)
        scan_board[:3, 3] = [4, 5, 6]
        reference_from_scan = np.eye(4)
        reference_from_scan[:3, 3] = [1, 2, 3]
        result = fusion_pipeline.axis_to_reference_transform(
            axis_board, reference_from_scan, scan_board)
        np.testing.assert_allclose(result[:3, 3], [-5, -13, -21])
        np.testing.assert_allclose(result[:3, :3], np.eye(3))

    def test_rejects_axis_images_without_usable_fixed_board(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "fixed board"):
                fusion_pipeline.find_axis_board_pose(Path(tmp), object(), np.eye(3),
                                                     np.zeros(5))

    def test_fusion_coordinator_uses_session_calibration_and_selected_repeats(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_root = root / "raw"
            session = raw_root / "object" / "session"
            selected = []
            for i in (0, 1, 11):
                source = session / f"view_{i:03d}" / "repeat_002"
                source.mkdir(parents=True)
                selected.append(source)
            calibration = root / "corrected.npz"
            calibration.write_bytes(b"npz")
            axis_images = root / "axis_images"
            axis_images.mkdir()
            (axis_images / "capture_manifest.json").write_text(
                json.dumps({"complete": True}), encoding="utf-8")
            board = root / "board.json"
            board.write_text("{}", encoding="utf-8")
            camera_poses = session / "camera_poses.json"
            camera_poses.write_text("{}", encoding="utf-8")
            calls = {}

            def fake_calibrate(argv):
                calls["axis"] = argv
                output = Path(argv[argv.index("--output") + 1])
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json.dumps({"point_rms_mm": 1.5,
                    "return_to_zero": {"left_corner_shift_rms_px": 0.5}}),
                    encoding="utf-8")
                return 0

            def fake_fuse(argv):
                calls["fuse"] = argv
                output = Path(argv[argv.index("--output") + 1])
                output.mkdir(parents=True)
                (output / "fused.ply").write_bytes(b"ply")
                (output / "fusion.json").write_text(json.dumps({"fused_points": 12,
                    "views": [1, 2]}), encoding="utf-8")
                return 0

            with patch.object(fusion_pipeline, "register_axis_camera",
                              return_value=root / "axis_to_reference.json"):
                output = fusion_pipeline.fuse_reconstructed_session(
                    session, selected, calibration, axis_images, board,
                    raw_root, root / "processed", exclude_view_ids={11},
                    calibrate=fake_calibrate, fuse=fake_fuse)
            self.assertEqual(Path(calls["axis"][calls["axis"].index("--stereo") + 1]),
                             calibration)
            chosen = calls["fuse"]
            repeats = chosen[chosen.index("--repeat-root") + 1:chosen.index("--raw-root")]
            self.assertEqual(repeats, [str(selected[0]), str(selected[1])])
            self.assertEqual(Path(chosen[chosen.index("--camera-poses") + 1]), camera_poses)
            self.assertTrue((output / "fused.ply").is_file())
            status = json.loads((output.parent / "fusion_status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["state"], "complete")
            self.assertEqual(status["excluded_view_ids"], [11])


if __name__ == "__main__":
    unittest.main()
