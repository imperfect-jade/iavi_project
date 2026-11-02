import json
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scan_session
from scan_session import angles_for_range, pending_views, reconstruct_session, return_turntable_home


class SessionPlanTests(unittest.TestCase):
    def test_pose_calibrations_require_fresh_upper_fit_from_current_lower(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            models = [root / "upper.npz", root / "lower.npz"]
            for model in models:
                np.savez(model, image_size=np.array([2592, 1944]),
                         K_left=np.eye(3), D_left=np.zeros((1, 5)),
                         K_right=np.eye(3), D_right=np.zeros((1, 5)),
                         t_right_from_left_mm=np.array([[-100.], [0.], [0.]]))
                model.with_suffix(".json").write_text(json.dumps({
                    "left_serial": "L", "right_serial": "R",
                    "image_size": [2592, 1944]}), encoding="utf-8")
            upper_meta = json.loads(models[0].with_suffix(".json").read_text())
            upper_meta["lower_sha256"] = "stale"
            models[0].with_suffix(".json").write_text(json.dumps(upper_meta))
            with self.assertRaisesRegex(ValueError, "rerun upper"):
                scan_session.load_pose_calibrations(
                    models[0].with_suffix(".json"),
                    models[1].with_suffix(".json"), ("L", "R"))
            upper_meta["lower_sha256"] = hashlib.sha256(models[1].read_bytes()).hexdigest()
            models[0].with_suffix(".json").write_text(json.dumps(upper_meta))
            paths, _, _, size = scan_session.load_pose_calibrations(
                models[0].with_suffix(".json"),
                models[1].with_suffix(".json"), ("L", "R"))
            self.assertEqual(paths, {0: models[0], 1: models[1]})
            self.assertEqual(size, (2592, 1944))

    def test_each_scan_view_localizes_directly_in_stationary_board_frame(self):
        current = np.eye(4)
        current[:3, 3] = [12, -5, 800]
        with patch("scan_session.detect_corners_with_contrast_fallback",
                   return_value=(np.zeros((20, 1, 2)), np.arange(20), "raw")), \
             patch("scan_session.estimate_pose", return_value=(current, 0.2)):
            result = scan_session.localize(None, object(), np.eye(3), np.zeros(5))
        self.assertEqual(result["coordinate_frame"], "fixed_ChArUco_board")
        np.testing.assert_allclose(result["T_reference_from_left"], np.linalg.inv(current))
        self.assertEqual(result["corners"], 20)

    def test_post_reconstruction_fuses_only_after_all_views_succeed(self):
        args = SimpleNamespace(reconstruction_min_depth_mm=None,
            reconstruction_max_depth_mm=None, no_fusion=False,
            output=Path("raw"), processed_output=Path("processed"),
            fusion_axis_images=Path("axis"), board=Path("board.json"),
            fusion_radius_mm=65.0, fusion_height_mm=(-95.0, 110.0),
            fusion_voxel_mm=1.0, fusion_exclude_view=[11])
        session = Path("raw/object/session")
        sources = [session / "view_000/repeat_001", session / "view_012/repeat_001"]
        with patch("scan_session.reconstruct_session", return_value=0) as reconstruct, \
             patch("scan_session.selected_complete_sources", return_value=sources), \
             patch("fusion_pipeline.fuse_reconstructed_session") as fuse:
            calibrations = {0: Path("upper.npz"), 1: Path("lower.npz")}
            self.assertEqual(scan_session.finish_session(args, session,
                Path("processed/object/session"), [0, 1], [0], calibrations,
                lambda argv: 0), 0)
            self.assertEqual(fuse.call_args.args[:3],
                (session, sources, Path("lower.npz")))
            self.assertEqual(fuse.call_args.kwargs["exclude_view_ids"], {11})
            reconstruct.return_value = 1
            fuse.reset_mock()
            self.assertEqual(scan_session.finish_session(args, session,
                Path("processed/object/session"), [0, 1], [0], calibrations,
                lambda argv: 0), 1)
            fuse.assert_not_called()
            reconstruct.return_value = 0
            args.no_fusion = True
            self.assertEqual(scan_session.finish_session(args, session,
                Path("processed/object/session"), [0, 1], [0], calibrations,
                lambda argv: 0), 0)
            fuse.assert_not_called()

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
            upper = root / "upper.npz"
            lower = root / "lower.npz"
            upper.write_bytes(b"upper calibration")
            lower.write_bytes(b"lower calibration")
            calibrations = {0: upper, 1: lower}
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
                model = Path(argv[argv.index("--calibration") + 1])
                calls.append((view.parent.name, model.name))
                if view.parent.name == "view_012" and sum(
                        name == "view_012" for name, _ in calls) == 1:
                    raise ValueError("no reliable matches")
                output.mkdir(parents=True, exist_ok=True)
                (output / "cloud_left.ply").write_bytes(b"ply")
                (output / "qc.json").write_text(json.dumps({"point_count": 10, "warnings": []}), encoding="utf-8")
                return 0

            self.assertEqual(reconstruct_session(raw, processed, [0, 1], [0], calibrations,
                                                 fake_reconstruct), 1)
            self.assertEqual(calls, [("view_000", "upper.npz"), ("view_012", "lower.npz")])
            self.assertEqual(reconstruct_session(raw, processed, [0, 1], [0], calibrations,
                                                 fake_reconstruct), 0)
            self.assertEqual(calls, [("view_000", "upper.npz"),
                                     ("view_012", "lower.npz"),
                                     ("view_012", "lower.npz")])
            summary = json.loads((processed / "reconstruction_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["failed"], 0)
            self.assertEqual(summary["complete"], 2)

            upper.write_bytes(b"new upper geometry")
            self.assertEqual(reconstruct_session(raw, processed, [0, 1], [0], calibrations,
                                                 fake_reconstruct), 0)
            self.assertEqual(calls[-1], ("view_000", "upper.npz"))

    def test_refreshes_camera_poses_from_black_images_before_resume(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            raw = root / "raw" / "object" / "session"
            processed = root / "processed" / "object" / "session"
            sources = []
            ok, encoded = cv2.imencode(".png", np.zeros((4, 4, 3), np.uint8))
            self.assertTrue(ok)
            for view_id in (0, 12):
                folder = raw / f"view_{view_id:03d}" / "repeat_001"
                (folder / "left").mkdir(parents=True)
                (folder / "left" / "000_black.png").write_bytes(encoded.tobytes())
                sources.append(folder)
            raw.mkdir(parents=True, exist_ok=True)
            (raw / "camera_poses.json").write_text('{"stale": []}', encoding="utf-8")
            first = {"T_reference_from_left": np.eye(4).tolist(), "corners": 20,
                     "reprojection_rms_px": .2}
            second = {"T_reference_from_left": (2*np.eye(4)).tolist(), "corners": 22,
                      "reprojection_rms_px": .3}
            with patch("scan_session.localize", side_effect=[first, second]) as locate:
                scan_session.refresh_camera_poses(raw, processed, sources,
                    object(), np.eye(3), np.zeros(5), 12, 3.)
            self.assertEqual(locate.call_count, 2)
            poses = json.loads((raw / "camera_poses.json").read_text(encoding="utf-8"))
            self.assertEqual(set(poses), {"view_000/repeat_001", "view_012/repeat_001"})
            self.assertEqual(poses["view_012/repeat_001"], second["T_reference_from_left"])
            self.assertTrue((processed / "camera_pose_refresh.json").is_file())


if __name__ == "__main__":
    unittest.main()
