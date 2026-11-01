import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scan_session
from scan_session import angles_for_range, pending_views, reconstruct_session, return_turntable_home


class SessionPlanTests(unittest.TestCase):
    def test_final_home_sends_zero_and_records_unverified_arrival(self):
        with tempfile.TemporaryDirectory() as td:
            class Controller:
                def __init__(self):
                    self.angles = []

                def move_to(self, angle):
                    self.angles.append(angle)
                    return b"#000P0500T2000!"

            controller = Controller()
            with patch("scan_session.time.sleep") as sleep:
                return_turntable_home(controller, Path(td), 12.0)
            self.assertEqual(controller.angles, [0])
            sleep.assert_called_once_with(12.0)
            record = json.loads((Path(td) / "turntable_final_home.json").read_text(encoding="utf-8"))
            self.assertEqual(record["command_deg"], 0)
            self.assertFalse(record["arrival_verified"])

    def test_range_and_view_ids(self):
        self.assertEqual(angles_for_range(60, 180, 60), [60, 120, 180])
        for values in [(0, 330, 45), (180, 0, 30), (0, 350, 30)]:
            with self.assertRaises(ValueError):
                angles_for_range(*values)

    def test_resume_only_skips_complete_matching_view(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            done = root / "view_014" / "repeat_001"
            done.mkdir(parents=True)
            (done / "left.png").write_bytes(b"left")
            (done / "right.png").write_bytes(b"right")
            (done / "localization").mkdir()
            (done / "localization" / "left.png").write_bytes(b"board")
            (done / "localization" / "right.png").write_bytes(b"board")
            (done / "localization.json").write_text("{}", encoding="utf-8")
            (done / "manifest.json").write_text(json.dumps({
                "complete": True, "view_id": 14, "arm_pose_id": 1,
                "turntable_deg_commanded": 60, "protocol": "dense_dots_v2",
                "frame_count": 1, "frames": [{"left": "left.png", "right": "right.png"}]}), encoding="utf-8")
            self.assertEqual(pending_views(root, [30, 60, 90], 1, 1, True), [30, 90])
            with self.assertRaises(FileExistsError):
                pending_views(root, [60], 1, 1, False)
            bad = root / "view_015" / "repeat_001"
            bad.mkdir(parents=True)
            with self.assertRaises(FileExistsError):
                pending_views(root, [90], 1, 1, True)

    def test_reconstructs_each_complete_view_and_resumes_failed_only(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            raw = root / "raw" / "object" / "session"
            processed = root / "processed" / "object" / "session"
            calibration = root / "stereo.npz"
            calibration.write_bytes(b"calibration")
            for view_id, pose_id in ((0, 0), (12, 1)):
                folder = raw / f"view_{view_id:03d}" / "repeat_001"
                folder.mkdir(parents=True)
                (folder / "left.png").write_bytes(b"image")
                (folder / "right.png").write_bytes(b"image")
                (folder / "localization").mkdir()
                (folder / "localization" / "left.png").write_bytes(b"board")
                (folder / "localization" / "right.png").write_bytes(b"board")
                (folder / "localization.json").write_text("{}", encoding="utf-8")
                (folder / "manifest.json").write_text(json.dumps({
                    "complete": True, "view_id": view_id, "arm_pose_id": pose_id,
                    "turntable_deg_commanded": 0, "protocol": "dense_dots_v2",
                    "frame_count": 1, "frames": [{"left": "left.png", "right": "right.png"}]}), encoding="utf-8")
            calls = []

            def fake_reconstruct(argv):
                view = Path(argv[argv.index("--view") + 1])
                output = Path(argv[argv.index("--output") + 1])
                calls.append(view.parent.name)
                if view.parent.name == "view_012" and calls.count("view_012") == 1:
                    raise ValueError("no reliable matches")
                output.mkdir(parents=True, exist_ok=True)
                (output / "cloud_left.ply").write_bytes(b"ply")
                (output / "qc.json").write_text(json.dumps({"point_count": 10, "warnings": []}), encoding="utf-8")
                return 0

            self.assertEqual(reconstruct_session(raw, processed, [0, 1], [0], calibration,
                                                 fake_reconstruct), 1)
            self.assertEqual(calls, ["view_000", "view_012"])
            self.assertEqual(reconstruct_session(raw, processed, [0, 1], [0], calibration,
                                                 fake_reconstruct), 0)
            self.assertEqual(calls, ["view_000", "view_012", "view_012"])
            summary = json.loads((processed / "reconstruction_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["failed"], 0)
            self.assertEqual(summary["complete"], 2)

            corrected = root / "corrected.npz"
            corrected.write_bytes(b"new calibration geometry")
            self.assertEqual(reconstruct_session(raw, processed, [0, 1], [0], corrected,
                                                 fake_reconstruct), 0)
            self.assertEqual(calls[-2:], ["view_000", "view_012"])

    def test_prepares_board_correction_from_same_latest_complete_repeats(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            raw = root / "raw" / "object" / "session"
            processed = root / "processed" / "object" / "session"
            base = root / "base.npz"
            board = root / "board.json"
            base.write_bytes(b"base")
            board.write_text("{}", encoding="utf-8")
            for repeat in (1, 2):
                folder = raw / "view_000" / f"repeat_{repeat:03d}"
                folder.mkdir(parents=True)
                (folder / "left.png").write_bytes(b"left")
                (folder / "right.png").write_bytes(b"right")
                (folder / "localization").mkdir()
                (folder / "localization" / "left.png").write_bytes(b"board")
                (folder / "localization" / "right.png").write_bytes(b"board")
                (folder / "localization.json").write_text("{}", encoding="utf-8")
                (folder / "manifest.json").write_text(json.dumps({
                    "complete": True, "view_id": 0, "arm_pose_id": 0,
                    "turntable_deg_commanded": 0, "protocol": "dense_dots_v2",
                    "frame_count": 1, "frames": [{"left": "left.png", "right": "right.png"}]}),
                    encoding="utf-8")
            chosen = root / "corrected.npz"
            with patch("board_stereo_correction.derive_session_calibration", return_value=chosen) as derive:
                result = scan_session.prepare_reconstruction_calibration(
                    raw, processed, [0], [0], base, board, enabled=True)
            self.assertEqual(result, chosen)
            self.assertEqual(derive.call_args.args[0], [raw / "view_000" / "repeat_002"])
            self.assertEqual(derive.call_args.args[3], processed / "board_stereo_correction")
            self.assertEqual(scan_session.prepare_reconstruction_calibration(
                raw, processed, [0], [0], base, board, enabled=False), base)


if __name__ == "__main__":
    unittest.main()
