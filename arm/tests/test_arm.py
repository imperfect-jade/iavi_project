import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
import types

ARM_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ARM_DIR))

from arm import ArmController, command_for_pose, main, open_controller, parse_position_reply


class FakeSerial:
    def __init__(self, partial=False, replies=()):
        self.writes = []
        self.partial = partial
        self.replies = list(replies)

    def write(self, data):
        self.writes.append(data)
        return len(data) - 1 if self.partial else len(data)

    def flush(self):
        pass

    def read_until(self, terminator):
        return self.replies.pop(0) if self.replies else b""


class ArmTests(unittest.TestCase):
    def test_pose_frames_use_all_six_tested_targets(self):
        self.assertEqual(command_for_pose("upper"),
                         b"{#000P1511T2000!#001P1900T2000!#002P2055T2000!#003P1040T2000!#004P1475T2000!#005P1569T2000!}")
        self.assertEqual(command_for_pose("lower"),
                         b"{#000P1511T2000!#001P2341T2000!#002P2403T2000!#003P0854T2000!#004P1469T2000!#005P1569T2000!}")

    def test_invalid_pose_cannot_write(self):
        fake = FakeSerial()
        with self.assertRaises(ValueError):
            ArmController(fake).move_to("middle")
        self.assertEqual(fake.writes, [])

    def test_short_write_is_an_error(self):
        with self.assertRaises(OSError):
            ArmController(FakeSerial(partial=True)).move_to("upper")

    def test_readback_rejects_wrong_id_and_invalid_reply(self):
        self.assertEqual(parse_position_reply(b"#002P2307!", 2), 2307)
        with self.assertRaises(ValueError):
            parse_position_reply(b"#003P2307!", 2)
        with self.assertRaises(ValueError):
            parse_position_reply(b"#002PRAD!", 2)

    def test_read_position_skips_echo_then_returns_matching_feedback(self):
        fake = FakeSerial(replies=[b"#002PRAD!", b"#002P2307!"])
        self.assertEqual(ArmController(fake).read_position(2), 2307)
        self.assertEqual(fake.writes, [b"#002PRAD!"])

    def test_read_position_reports_no_reply_without_claiming_arrival(self):
        fake = FakeSerial(replies=[b"#002PRAD!"])
        self.assertIsNone(ArmController(fake).read_position(2))

    def test_dry_run_needs_no_port(self):
        result = subprocess.run(
            [sys.executable, str(ARM_DIR / "arm.py"), "--pose", "upper", "--dry-run"],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("{#000P1511T2000!", result.stdout)

    def test_live_move_requires_port(self):
        result = subprocess.run(
            [sys.executable, str(ARM_DIR / "arm.py"), "--pose", "upper"],
            capture_output=True, text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--port", result.stderr)

    def test_open_port_keeps_dtr_and_rts_inactive(self):
        calls = []

        class Port:
            def __init__(self, **kwargs):
                calls.append(("construct", kwargs))
                self.dtr = True
                self.rts = True
                self.port = None

            def open(self):
                calls.append(("open", self.port, self.dtr, self.rts))

        fake_serial = types.SimpleNamespace(Serial=Port, SerialException=OSError)
        with patch.dict(sys.modules, {"serial": fake_serial}), patch("arm.time.sleep"):
            controller = open_controller("COM7")
        self.assertEqual(calls[0][1]["port"], None)
        self.assertEqual(calls[1], ("open", "COM7", False, False))
        self.assertIsInstance(controller, ArmController)

    def test_connect_only_sends_no_motion_command(self):
        fake = FakeSerial()
        fake.close = lambda: None
        with patch("arm.open_controller", return_value=ArmController(fake)):
            self.assertEqual(main(["--port", "COM7", "--connect-only"]), 0)
        self.assertEqual(fake.writes, [])


if __name__ == "__main__":
    unittest.main()
