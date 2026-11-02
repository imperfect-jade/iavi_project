import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from capture import capture_once, make_dense_dots, reserve_view, save_pair, finish_scan, select_projector_display, parser, save_manual_pair, next_manual_index, configure_camera, record_capture_session


class CaptureTests(unittest.TestCase):
    def test_dense_dots_protocol_has_black_board_frame_and_white_color_frame(self):
        a = make_dense_dots(80, 40, 8, 5, 3)
        b = make_dense_dots(80, 40, 8, 5, 3)
        self.assertEqual([p.name for p in a], ["black", "dense_dots_00", "dense_dots_01", "dense_dots_02", "white"])
        self.assertTrue(np.array_equal(a[0].image, b[0].image))
        self.assertEqual(np.count_nonzero(a[0].image), 0)
        self.assertFalse(np.array_equal(a[1].image, a[2].image))
        self.assertGreater(np.count_nonzero(a[1].image), 0)
        self.assertTrue(np.all(a[-1].image == 255))
        self.assertEqual(parser().parse_args(["scan", "--scan-protocol", "dense_dots"]).scan_protocol,
                         "dense_dots")
        self.assertEqual(parser().parse_args(["scan", "--scan-protocol", "dense_dots"]).dot_frames, 5)
        self.assertEqual(parser().parse_args(["scan"]).scan_protocol, "dense_dots")
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            parser().parse_args(["scan", "--scan-protocol", "gray_phase"])

    def test_black_capture_is_passed_to_board_localization_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "left").mkdir()
            (root / "right").mkdir()
            protocol = make_dense_dots(16, 16, 8, 5, 1)
            image = np.zeros((16, 16, 3), np.uint8)
            stamps = {"left": 1, "right": 2}
            board_rows = []
            def on_black(left, right, ticks, row):
                board_rows.append(row["pattern"])
                np.testing.assert_array_equal(left, image)
                self.assertEqual(ticks, stamps)
            with patch("capture.show_pattern"), patch("capture.grab_pair",
                 return_value=(image, image, stamps)), patch("capture.cv2.waitKey", return_value=-1), \
                 patch("capture.cv2.imshow"), patch("capture.preview_pair", return_value=image):
                capture_once({"destination": root, "metadata": {"protocol": "dense_dots_v3"},
                              "on_black": on_black},
                             (None, None, None, None, 1, 1000), protocol)
            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(board_rows, ["black"])
            self.assertEqual([row["pattern"] for row in manifest["frames"]],
                             ["black", "dense_dots_00", "white"])

    def test_session_rejects_changed_camera_settings(self):
        class Node:
            def __init__(self, value): self.Value = value
        class Camera:
            def __init__(self):
                for name, value in (("PixelFormat", "RGB8"), ("Gamma", 1.0),
                                    ("ExposureTime", 10000.0), ("Gain", 0.0)):
                    setattr(self, name, Node(value))
        with tempfile.TemporaryDirectory() as tmp:
            cameras = [Camera(), Camera()]
            args = parser().parse_args(["calibration", "--session-id", "test_01"])
            record_capture_session(Path(tmp), cameras, args, (2592, 1944))
            self.assertTrue((Path(tmp) / "capture_session.json").exists())
            cameras[1].Gamma.Value = 1.5
            with self.assertRaisesRegex(ValueError, "settings changed"):
                record_capture_session(Path(tmp), cameras, args, (2592, 1944))

    def test_lab5_camera_configuration_and_hardware_trigger(self):
        class Node:
            def __init__(self): self.value = None
            def SetValue(self, value): self.value = value
        class Camera:
            def __init__(self):
                for name in ("PixelFormat", "ExposureAuto", "GainAuto", "BalanceWhiteAuto",
                             "Gamma", "ExposureTime", "Gain", "TriggerSelector", "TriggerMode",
                             "TriggerSource", "TriggerActivation"):
                    setattr(self, name, Node())
        camera = Camera()
        configure_camera(camera, 10000, 0, "hardware")
        self.assertEqual(camera.PixelFormat.value, "RGB8")
        self.assertEqual(camera.Gamma.value, 1.0)
        self.assertEqual(camera.TriggerSource.value, "Line1")
        self.assertEqual(camera.TriggerActivation.value, "RisingEdge")
        self.assertEqual(parser().parse_args(["calibration"]).trigger, "software")

    def test_calibration_resume_uses_lab5_filenames_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            left = Path(tmp) / "left"
            left.mkdir()
            (left / "calib_001.png").touch()
            (left / "calib_012.png").touch()
            (left / "099_calib.png").touch()
            self.assertEqual(next_manual_index(tmp, "calib"), 13)
            right = Path(tmp) / "right"
            right.mkdir()
            (right / "calib_014.png").touch()
            self.assertEqual(next_manual_index(tmp, "calib"), 15)

    def test_color_capture_uses_prefix_and_resumes_numbering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for side in ("left", "right"):
                (root / side).mkdir()
            image = np.zeros((4, 5, 3), np.uint8)
            row = save_manual_pair(root, 1, "color", image, image,
                                   {"left": 1, "right": 2})
            self.assertEqual(row["left"], "left/color_001.png")
            (root / "right/color_012.png").touch()
            (root / "left/099_color.png").touch()
            self.assertEqual(next_manual_index(root, "color"), 13)

    def test_manual_capture_reports_only_after_both_images_are_saved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "left").mkdir()
            (root / "right").mkdir()
            image = np.zeros((4, 5, 3), np.uint8)
            output = StringIO()
            with redirect_stdout(output):
                row = save_manual_pair(root, 1, "calib", image, image,
                                       {"left": 1, "right": 2})
            self.assertTrue((root / row["left"]).exists())
            self.assertTrue((root / row["right"]).exists())
            self.assertIn("calib_001.png", output.getvalue())
            self.assertIn("Saved", output.getvalue())
            self.assertIn("calib_001.png", (root / "frames.jsonl").read_text())
            output = StringIO()
            with redirect_stdout(output), self.assertRaises(FileExistsError):
                save_manual_pair(root, 1, "calib", image, image,
                                 {"left": 3, "right": 4})
            self.assertEqual(output.getvalue(), "")

    def test_lab5_device_defaults_and_projector_position(self):
        args = parser().parse_args(["scan"])
        self.assertEqual((args.left_serial, args.right_serial), ("40031431", "40031435"))
        projector = {"name": "DISPLAY2", "x": 2560, "y": 0,
                     "width": 1920, "height": 1080}
        self.assertIs(select_projector_display([projector]), projector)
        with self.assertRaises(ValueError):
            select_projector_display([{**projector, "width": 1280}])

    def test_incomplete_scan_cannot_be_committed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = reserve_view(Path(tmp), "item", "s1", 0, 1)
            image = np.zeros((4, 5, 3), np.uint8)
            row = save_pair(root, 0, "white", image, image, {"left": 1, "right": 2})
            with self.assertRaises(ValueError):
                finish_scan(root, [row], ["white", "black"], {})
            self.assertFalse((root / "manifest.json").exists())
            self.assertTrue((root / "left" / "000_white.png").exists())

    def test_view_is_never_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            reserve_view(Path(tmp), "item", "s1", 0, 1)
            with self.assertRaises(FileExistsError):
                reserve_view(Path(tmp), "item", "s1", 0, 1)


if __name__ == "__main__":
    unittest.main()
