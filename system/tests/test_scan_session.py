import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scan_session import angles_for_range, pending_views


class SessionPlanTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
