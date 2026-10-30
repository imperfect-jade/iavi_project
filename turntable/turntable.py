"""Control the measured 12-view turntable positions through its Arduino USB serial port."""
from __future__ import annotations

import argparse
import sys
import time


# Measured and accepted on the experiment turntable; angles refer to the plate.
ANGLE_TO_PWM = {
    0: 500, 30: 667, 60: 833, 90: 1000,
    120: 1167, 150: 1333, 180: 1500, 210: 1667,
    240: 1833, 270: 2000, 300: 2167, 330: 2333,
}
BAUDRATE = 115200
MOVE_TIME_MS = 2000
STARTUP_WAIT_S = 2.5
SETTLE_WAIT_S = 2.5


def command_for_angle(angle: int) -> bytes:
    """Return only a previously tested target; never interpolate silently."""
    if angle not in ANGLE_TO_PWM:
        raise ValueError(f"Unsupported angle {angle}; choose 0, 30, ..., 330")
    return f"#000P{ANGLE_TO_PWM[angle]:04d}T{MOVE_TIME_MS:04d}!".encode("ascii")


class TurntableController:
    """Send commands over one persistent connection. No position feedback is assumed."""

    def __init__(self, connection):
        self.connection = connection

    def move_to(self, angle: int) -> bytes:
        command = command_for_angle(angle)
        written = self.connection.write(command)
        if written != len(command):
            raise OSError(f"Incomplete serial write: {written}/{len(command)} bytes")
        self.connection.flush()
        return command


def open_controller(port: str) -> TurntableController:
    try:
        import serial
    except ImportError as exc:
        raise RuntimeError("pyserial is missing; run: python -m pip install pyserial") from exc
    try:
        connection = serial.Serial(port=port, baudrate=BAUDRATE, timeout=0.5, write_timeout=2)
    except serial.SerialException as exc:
        raise RuntimeError(f"Cannot open {port}; close Zide or another serial monitor first: {exc}") from exc
    time.sleep(STARTUP_WAIT_S)  # Arduino may reset when USB serial opens.
    return TurntableController(connection)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Move the turntable to one of 12 tested plate angles")
    parser.add_argument("--port", default="COM6", help="Arduino USB serial port (default: COM6)")
    parser.add_argument("--angle", type=int, choices=ANGLE_TO_PWM, help="Send one target and exit")
    parser.add_argument("--list", action="store_true", help="Show the angle-to-command table")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without opening the port")
    args = parser.parse_args(argv)

    if args.list:
        for angle in ANGLE_TO_PWM:
            print(f"{angle:3d}°  {command_for_angle(angle).decode('ascii')}")
        return 0
    if args.dry_run:
        if args.angle is None:
            parser.error("--dry-run requires --angle")
        print(command_for_angle(args.angle).decode("ascii"))
        return 0

    controller = None
    try:
        controller = open_controller(args.port)
        print(f"Connected to {args.port} at {BAUDRATE} baud. Position feedback is not verified.")
        if args.angle is not None:
            targets = [args.angle]
        else:
            print("Enter a tested angle (0, 30, ..., 330), or q to quit.")
            targets = None
        while True:
            if targets is not None:
                if not targets:
                    break
                angle = targets.pop(0)
            else:
                raw = input("angle> ").strip().lower()
                if raw in {"q", "quit", "exit"}:
                    break
                try:
                    angle = int(raw)
                    command_for_angle(angle)
                except ValueError as exc:
                    print(exc)
                    continue
            command = controller.move_to(angle)
            print(f"Sent {command.decode('ascii')} for approximately {angle}°; actual arrival is unverified.")
            time.sleep(SETTLE_WAIT_S)
        return 0
    except (OSError, RuntimeError, KeyboardInterrupt) as exc:
        print(f"Turntable control stopped: {exc}", file=sys.stderr)
        return 1
    finally:
        if controller is not None:
            controller.connection.close()


if __name__ == "__main__":
    raise SystemExit(main())
