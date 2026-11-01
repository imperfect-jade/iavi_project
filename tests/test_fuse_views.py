import sys
import tempfile
import unittest
from pathlib import Path
import json

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from fuse_views import DEFAULT_AXIS_CALIBRATION, main, read_ply, rotate_about_axis, voxel_average, load_axis_calibration, crop_cylinder, transform_axis, validate_rigid_transform
from reconstruct_view import write_ply


class FuseViewsTests(unittest.TestCase):
    def test_explicit_repeats_use_custom_raw_root_and_ignore_old_repeat(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            raw = base / "capture" / "object" / "session"
            processed = base / "results"
            calibration = base / "stereo.npz"
            calibration.write_bytes(b"same calibration")
            axis = base / "axis.json"
            axis.write_text(json.dumps({
                "coordinate_frame": "unrectified_left_camera", "unit": "mm",
                "axis_point_mm": [0, 0, 0], "axis_direction": [0, 1, 0],
                "stereo_calibration": str(calibration),
                "angles": [{"command_deg": 30, "measured_deg": 30}]}), encoding="utf-8")
            transform = base / "axis_to_reference.json"
            transform.write_text(json.dumps({"status": "calibrated_fixed_board",
                                             "T_reference_from_axis_camera": np.eye(4).tolist()}),
                                 encoding="utf-8")
            poses = {}
            selected = []
            for view, angle, repeat in ((0, 0, 1), (0, 0, 2), (1, 30, 1)):
                folder = raw / f"view_{view:03d}" / f"repeat_{repeat:03d}"
                folder.mkdir(parents=True)
                name = f"{folder.parent.name}/{folder.name}"
                poses[name] = np.eye(4).tolist()
                (folder / "pose.json").write_text(json.dumps({"arm_pose_id": 0,
                    "turntable_deg_commanded": angle}), encoding="utf-8")
                (folder / "manifest.json").write_text(json.dumps({"complete": True,
                    "protocol": "dense_dots_v2", "arm_pose_id": 0,
                    "turntable_deg_commanded": angle}), encoding="utf-8")
                target = processed / "object" / "session" / folder.parent.name / folder.name
                target.mkdir(parents=True)
                (target / "qc.json").write_text(json.dumps({"point_count": 2,
                    "coordinate_frame": "rectified_left_camera",
                    "calibration": str(calibration)}), encoding="utf-8")
                write_ply(target / "cloud_left.ply",
                    np.array([[10, 0, 5], [11, 1, 5]], np.float32),
                    np.array([[4, 5, 6], [7, 8, 9]], np.uint8),
                    "original_left_camera")
                if view == 1 or repeat == 2:
                    selected.append(folder)
            poses_path = raw / "camera_poses.json"
            poses_path.write_text(json.dumps(poses), encoding="utf-8")
            output = base / "fusion"
            (output / "aligned").mkdir(parents=True)
            stale = output / "aligned" / "view_000_repeat_001.ply"
            stale.write_bytes(b"old fusion")
            self.assertEqual(main(["--repeat-root", *map(str, selected),
                "--raw-root", str(base / "capture"),
                "--processed-root", str(processed), "--output", str(output),
                "--camera-poses", str(poses_path), "--axis-calibration", str(axis),
                "--axis-to-reference", str(transform)]), 0)
            report = json.loads((output / "fusion.json").read_text(encoding="utf-8"))
            self.assertEqual([row["repeat"] for row in report["views"]],
                             ["view_000/repeat_002", "view_001/repeat_001"])
            self.assertTrue((output / "fused.ply").is_file())
            self.assertFalse(stale.exists())

    def test_axis_transforms_from_calibration_camera_to_reference(self):
        transform = np.eye(4)
        transform[:3, 3] = [10, 20, 30]
        point, direction = transform_axis([1, 2, 3], [0, 1, 0], transform)
        np.testing.assert_allclose(point, [11, 22, 33])
        np.testing.assert_allclose(direction, [0, 1, 0])

    def test_rejects_nonrigid_camera_transform(self):
        transform = np.eye(4)
        transform[0, 0] = 2
        with self.assertRaisesRegex(ValueError, "rigid"):
            validate_rigid_transform(transform, "test")

    def test_cylinder_crop_excludes_background_and_platform(self):
        points = np.array([[0, 0, 0], [40, 0, 0], [61, 0, 0], [0, -30, 0]])
        keep = crop_cylinder(points, np.zeros(3), np.array([0, 1, 0]), 60, (-20, 110))
        np.testing.assert_array_equal(keep, [True, True, False, False])

    def test_default_axis_path_is_the_calibration_directory(self):
        self.assertEqual(DEFAULT_AXIS_CALIBRATION,
            Path(__file__).resolve().parents[1] / "calibration/turntable_axis/axis_cube_01/axis_calibration.json")

    def test_axis_calibration_uses_measured_angles_and_original_left_frame(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "axis.json"
            path.write_text(json.dumps({"coordinate_frame": "unrectified_left_camera", "unit": "mm",
                "stereo_calibration": "C:/calibration.npz", "axis_point_mm": [1, 2, 3],
                "axis_direction": [0, -1, 0], "angles": [{"command_deg": 30, "measured_deg": 27}]}))
            axis = load_axis_calibration(path)
            self.assertEqual(axis["angles_by_command"][30.0], 27.0)
            np.testing.assert_array_equal(axis["point"], [1, 2, 3])
            self.assertEqual(axis["frame"], "original_left_camera")

    def test_axis_calibration_rejects_rectified_frame(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "axis.json"
            path.write_text(json.dumps({"coordinate_frame": "rectified_left_camera", "unit": "mm"}))
            with self.assertRaisesRegex(ValueError, "unrectified_left_camera"):
                load_axis_calibration(path)

    def test_inverse_turntable_rotation_restores_zero_angle_point(self):
        center = np.array([10., 0., 20.])
        original = np.array([[15., 3., 20.]], np.float32)
        moved = rotate_about_axis(original, center, [0, 1, 0], 30)
        recovered = rotate_about_axis(moved, center, [0, 1, 0], -30)
        np.testing.assert_allclose(recovered, original, atol=1e-5)

    def test_voxel_average_merges_overlapping_views_and_colors(self):
        xyz, bgr = voxel_average(np.array([[1.1, 2, 3], [1.2, 2, 3], [3, 4, 5]], np.float32),
                                 np.array([[10, 20, 30], [20, 30, 40], [50, 60, 70]], np.uint8), 1)
        self.assertEqual(len(xyz), 2)
        np.testing.assert_allclose(xyz[0], [1.15, 2, 3])
        np.testing.assert_array_equal(bgr[0], [15, 25, 35])

    def test_reads_single_view_ply_colors(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cloud.ply"
            write_ply(path, np.array([[1, 2, 3]], np.float32), np.array([[4, 5, 6]], np.uint8), "left")
            xyz, bgr = read_ply(path)
            np.testing.assert_array_equal(xyz, [[1, 2, 3]])
            np.testing.assert_array_equal(bgr, [[4, 5, 6]])
