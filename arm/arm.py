"""Move the Jibot1 camera arm between two tested serial-command poses."""
from __future__ import annotations

import argparse
import re
import sys
import time


POSE_TARGETS = {
    "upper": (1511, 1900, 2055, 1040, 1475, 1569),
    "lower": (1511, 2341, 2403, 854, 1469, 1569),
}
BAUDRATE = 115200
MOVE_TIME_MS = 2000
STARTUP_WAIT_S = 2.5
SETTLE_WAIT_S = 2.5


def command_for_pose(pose: str) -> bytes:
    """Return one controller frame containing the six frozen joint targets."""
    if pose not in POSE_TARGETS:
        raise ValueError(f"Unsupported pose {pose!r}; choose upper or lower")
    parts = [f"#{index:03d}P{target:04d}T{MOVE_TIME_MS:04d}!"
             for index, target in enumerate(POSE_TARGETS[pose])]
    return ("{" + "".join(parts) + "}").encode("ascii")


def parse_position_reply(reply: bytes, servo_id: int) -> int:
    """Parse one matching `#IDPposition!` reply, not a command echo."""
    match = re.fullmatch(rb"#(\d{3})P(\d{4})!", reply.strip())
    if not match or int(match.group(1)) != servo_id:
        raise ValueError(f"Invalid position reply for ID {servo_id:03d}: {reply!r}")
    return int(match.group(2))


class ArmController:
    def __init__(self, connection):
        self.connection = connection

    def move_to(self, pose: str) -> bytes:
        command = command_for_pose(pose)
        written = self.connection.write(command)
        if written != len(command):
            raise OSError(f"Incomplete serial write: {written}/{len(command)} bytes")
        self.connection.flush()
        return command

    def read_position(self, servo_id: int) -> int | None:
        if not 0 <= servo_id <= 5:
            raise ValueError("servo_id must be 0 through 5")
        query = f"#{servo_id:03d}PRAD!".encode("ascii")
        if self.connection.write(query) != len(query):
            raise OSError(f"Incomplete serial write for ID {servo_id:03d} readback")
        self.connection.flush()
        for _ in range(5):
            reply = self.connection.read_until(b"!")
            if not reply:
                break
            try:
                return parse_position_reply(reply, servo_id)
            except ValueError:
                continue  # Ignore command echoes and unrelated responses.
        return None


def open_controller(port: str) -> ArmController:
    try:
        import serial
    except ImportError as exc:
        raise RuntimeError("pyserial is missing; run: python -m pip install pyserial") from exc
    try:
        connection = serial.Serial(port=port, baudrate=BAUDRATE, timeout=0.5,
                                   write_timeout=2)
    except serial.SerialException as exc:
        raise RuntimeError(f"Cannot open {port}; close Zide or another serial monitor: {exc}") from exc
    time.sleep(STARTUP_WAIT_S)  # Arduino may reset when USB serial opens.
    return ArmController(connection)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Move Jibot1 to a tested camera pose")
    parser.add_argument("--port", help="Arm Arduino USB serial port (required for live control)")
    parser.add_argument("--pose", choices=POSE_TARGETS, help="Move once and exit; omit for interactive mode")
    parser.add_argument("--list", action="store_true", help="List the two frozen commands")
    parser.add_argument("--dry-run", action="store_true", help="Print a command without opening a port")
    parser.add_argument("--readback", action="store_true", help="Read six servo positions after each move")
    args = parser.parse_args(argv)

    if args.list:
        for pose in POSE_TARGETS:
            print(f"{pose}: {command_for_pose(pose).decode('ascii')}")
        return 0
    if args.dry_run:
        if args.pose is None:
            parser.error("--dry-run requires --pose")
        print(command_for_pose(args.pose).decode("ascii"))
        return 0
    if not args.port:
        parser.error("--port is required for live control")

    controller = None
    try:
        controller = open_controller(args.port)
        print(f"Connected to {args.port} at {BAUDRATE} baud. Camera arrival is unverified.")
        if args.pose is None:
            print("Enter upper or lower; q to quit.")
        while True:
            if args.pose is not None:
                pose = args.pose
            else:
                raw = input("pose> ").strip().lower()
                if raw in {"q", "quit", "exit"}:
                    break
                if raw not in POSE_TARGETS:
                    print("Choose upper or lower")
                    continue
                pose = raw
            command = controller.move_to(pose)
            print(f"Sent {command.decode('ascii')}; waiting {SETTLE_WAIT_S:g} s. Physical arrival is unverified.")
            time.sleep(SETTLE_WAIT_S)
            if args.readback:
                positions = [controller.read_position(i) for i in range(6)]
                print("Readback: " + ", ".join(
                    f"{i:03d}={'no reply' if value is None else value:>4}"
                    for i, value in enumerate(positions)))
            if args.pose is not None:
                break
        return 0
    except (OSError, RuntimeError, KeyboardInterrupt, EOFError) as exc:
        print(f"Arm control stopped: {exc}", file=sys.stderr)
        return 1
    finally:
        if controller is not None:
            controller.connection.close()


if __name__ == "__main__":
    raise SystemExit(main())
