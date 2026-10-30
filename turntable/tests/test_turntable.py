import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from turntable import ANGLE_TO_PWM, TurntableController, command_for_angle


class FakeSerial:
    def __init__(self):
        self.writes = []
        self.closed = False

    def write(self, data):
        self.writes.append(data)
        return len(data)

    def flush(self):
        pass

    def close(self):
        self.closed = True


class TurntableTests(unittest.TestCase):
    def test_all_verified_views_have_safe_commands(self):
        self.assertEqual(len(ANGLE_TO_PWM), 12)
        self.assertEqual(command_for_angle(0), b"#000P0500T2000!")
        self.assertEqual(command_for_angle(30), b"#000P0667T2000!")
        self.assertEqual(command_for_angle(330), b"#000P2333T2000!")
        with self.assertRaises(ValueError):
            command_for_angle(360)
        with self.assertRaises(ValueError):
            command_for_angle(15)

    def test_send_only_one_command_and_reuse_connection(self):
        fake = FakeSerial()
        controller = TurntableController(fake)
        controller.move_to(60)
        controller.move_to(90)
        self.assertEqual(fake.writes, [b"#000P0833T2000!", b"#000P1000T2000!"])
        self.assertFalse(fake.closed)


if __name__ == "__main__":
    unittest.main()
