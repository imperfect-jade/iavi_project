import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fusion_pipeline
from fuse_views import load_axis_calibration


class FusionPipelineTests(unittest.TestCase):
    def test_fusion_reads_axis_in_stationary_board_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "axis.json"
            path.write_text(json.dumps({
                "coordinate_frame": "fixed_ChArUco_board", "unit": "mm",
                "axis_point_mm": [10, 20, 30], "axis_direction": [0, 1, 0],
                "angles": [{"command_deg": 30, "measured_deg": 29.5}],
                "fixed_board": {"board_type": "ChArUco"},
                "stereo_calibration": str(Path(tmp) / "stereo.npz")}), encoding="utf-8")
            result = load_axis_calibration(path)
            self.assertEqual(result["frame"], "fixed_ChArUco_board")
            self.assertEqual(result["angles_by_command"][30], 29.5)

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
            (session / "scan_plan.json").write_text(json.dumps({
                "board": {}, "reference_frame": "fixed_ChArUco_board"}), encoding="utf-8")
            camera_poses = session / "camera_poses.json"
            camera_poses.write_text("{}", encoding="utf-8")
            calls = {}

            def fake_calibrate(argv):
                calls["axis"] = argv
                output = Path(argv[argv.index("--output") + 1])
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json.dumps({"point_rms_mm": 1.5,
                    "coordinate_frame": "fixed_ChArUco_board", "fixed_board": {},
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
            self.assertIn("--axis-already-in-reference", chosen)
            self.assertNotIn("--axis-to-reference", chosen)
            self.assertEqual(Path(calls["axis"][calls["axis"].index("--fixed-board") + 1]), board)
            self.assertTrue((output / "fused.ply").is_file())
            status = json.loads((output.parent / "fusion_status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["state"], "complete")
            self.assertEqual(status["excluded_view_ids"], [11])


if __name__ == "__main__":
    unittest.main()
