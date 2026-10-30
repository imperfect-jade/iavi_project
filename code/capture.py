"""Stereo structured-light acquisition for this project. Run --help first."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SAFE = re.compile(r"[A-Za-z0-9_-]+\Z")
PROJECTOR = "NBV projector"
PREVIEW = "NBV stereo preview"
DEFAULT_LEFT_SERIAL = "40031431"
DEFAULT_RIGHT_SERIAL = "40031435"
DEFAULT_PROJECTOR_XY = (2560, 0)


@dataclass(frozen=True)
class Pattern:
    name: str
    image: np.ndarray


def make_patterns(width=1920, height=1080, cell=16):
    """Gray cell addresses, negative four-step phase in x/y, white, black."""
    if min(width, height, cell) < 1:
        raise ValueError("Pattern dimensions and cell size must be positive")
    shape = (height, width)
    frames = []
    for axis, length, coord in (("x", width, np.arange(width)[None, :]),
                                ("y", height, np.arange(height)[:, None])):
        address = coord // cell
        gray = address ^ (address >> 1)
        bits = (math.ceil(length / cell) - 1).bit_length()
        for bit in range(bits - 1, -1, -1):
            image = np.broadcast_to(((gray >> bit) & 1) * 255, shape).astype(np.uint8)
            name = f"gray_{axis}_{bit:02d}"
            frames.extend((Pattern(name, image), Pattern(name + "_inv", 255 - image)))
    for axis, coord in (("x", np.arange(width)[None, :]),
                        ("y", np.arange(height)[:, None])):
        phase = 2 * np.pi * (coord % cell) / cell
        for step in range(4):
            image = np.broadcast_to(np.rint(127.5 + 127.5 * np.cos(
                phase - step * np.pi / 2)).clip(0, 255).astype(np.uint8), shape).copy()
            frames.append(Pattern(f"phase_{axis}_{step:02d}", image))
    frames.extend((Pattern("white", np.full(shape, 255, np.uint8)),
                   Pattern("black", np.zeros(shape, np.uint8))))
    return frames


def make_dense_dots(width=1920, height=1080, dot_cell=8, seed=5):
    """Lab5-style reproducible 20% random projector cells plus a color/QC frame."""
    if min(width, height, dot_cell) < 1:
        raise ValueError("Projector dimensions and dot cell size must be positive")
    rng = np.random.default_rng(seed)
    cells = rng.random((math.ceil(height / dot_cell),
                        math.ceil(width / dot_cell))) < .20
    dots = (np.repeat(np.repeat(cells, dot_cell, axis=0), dot_cell, axis=1)
            [:height, :width] * 255).astype(np.uint8)
    return [Pattern("dense_dots", dots),
            Pattern("white", np.full((height, width), 255, np.uint8))]


def reserve_view(root, object_id, session_id, view_id, repeat_id):
    for value in (object_id, session_id):
        if not SAFE.fullmatch(value):
            raise ValueError("object_id and session_id require letters, digits, _ or -")
    if view_id < 0 or repeat_id < 1:
        raise ValueError("view_id must be nonnegative and repeat_id positive")
    path = Path(root) / object_id / session_id / f"view_{view_id:03d}" / f"repeat_{repeat_id:03d}"
    path.mkdir(parents=True, exist_ok=False)
    (path / "left").mkdir()
    (path / "right").mkdir()
    return path


def _write_json(path, data):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temp, path)


def save_pair(directory, index, name, left, right, stamps):
    """Encode both images before publication; never replace a captured pair."""
    if not SAFE.fullmatch(name) or left.shape != right.shape or left.ndim != 3 or left.shape[2] != 3:
        raise ValueError("Invalid frame name or unequal/non-color camera images")
    if left.dtype != np.uint8 or right.dtype != np.uint8:
        raise ValueError("Expected uint8 BGR camera images")
    filename = f"{name}_{index:03d}.png" if name in ("calib", "color") else f"{index:03d}_{name}.png"
    paths = [Path(directory) / side / filename for side in ("left", "right")]
    if any(p.exists() for p in paths):
        raise FileExistsError(filename)
    encoded = []
    for image in (left, right):
        ok, data = cv2.imencode(".png", image)
        if not ok:
            raise RuntimeError("PNG encoding failed")
        encoded.append(data.tobytes())
    temps = [p.with_suffix(".png.tmp") for p in paths]
    try:
        for path, data in zip(temps, encoded):
            path.write_bytes(data)
        for temp, path in zip(temps, paths):
            os.replace(temp, path)
    except Exception:
        for path in (*temps, *paths):
            path.unlink(missing_ok=True)
        raise
    return {"index": index, "pattern": name, "left": f"left/{filename}",
            "right": f"right/{filename}", "left_device_ticks": stamps["left"],
            "right_device_ticks": stamps["right"], "monotonic_ns": time.monotonic_ns()}


def save_manual_pair(directory, index, name, left, right, stamps):
    """Save a calibration or color pair, then acknowledge its recorded files."""
    row = save_pair(directory, index, name, left, right, stamps)
    with (Path(directory) / "frames.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row) + "\n")
    print(f"Saved {Path(row['left']).name} (left + right) -> {directory}", flush=True)
    return row


def next_manual_index(directory, name):
    """Continue numbering from photos of the requested manual capture mode."""
    if name in ("calib", "color"):
        filenames = (p for side in ("left", "right")
                     for p in (Path(directory) / side).glob(f"{name}_*.png"))
        numbers = (p.stem.removeprefix(f"{name}_") for p in filenames)
    else:
        filenames = (Path(directory) / "left").glob(f"*_{name}.png")
        numbers = (p.stem.removesuffix(f"_{name}") for p in filenames)
    return 1 + max((int(number) for number in numbers if number.isdigit()), default=0)


def record_capture_session(directory, cameras, args, size):
    """Keep Lab5-style camera settings with each calibration/color session."""
    def value(camera, name):
        try:
            return getattr(camera, name).Value
        except Exception:
            return None

    settings = {"mode": args.mode, "trigger": args.trigger,
        "left_serial": args.left_serial, "right_serial": args.right_serial,
        "camera_size_px": list(size), "pattern_inner_corners": [10, 6],
        "cameras": [{"pixel_format": value(camera, "PixelFormat"),
                     "gamma": value(camera, "Gamma"),
                     "exposure_us": value(camera, "ExposureTime"),
                     "gain_db": value(camera, "Gain")}
                    for camera in cameras],
        "timestamp_note": "Raw camera ticks use independent clocks; do not subtract"}
    path = Path(directory) / "capture_session.json"
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if {key: previous.get(key) for key in settings} != settings:
            raise ValueError(f"Camera settings changed within {directory}; use a new session-id")
    else:
        _write_json(path, {"created_at": datetime.now().astimezone().isoformat(), **settings})


def finish_scan(directory, rows, order, metadata):
    if [row["pattern"] for row in rows] != list(order) or len(rows) != len(order):
        raise ValueError("Captured pattern order/count is incomplete")
    for row in rows:
        if not all((Path(directory) / row[side]).is_file() for side in ("left", "right")):
            raise ValueError("Missing image in captured pair")
    path = Path(directory) / "manifest.json"
    if path.exists():
        raise FileExistsError(path)
    _write_json(path, {**metadata, "complete": True, "frame_count": len(rows), "frames": rows})
    return path


def displays():
    """Physical monitor bounds via Windows API; empty on other systems."""
    if sys.platform != "win32":
        return []
    import ctypes
    from ctypes import wintypes
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except OSError:
        pass
    class RECT(ctypes.Structure):
        _fields_ = [(s, ctypes.c_long) for s in ("left", "top", "right", "bottom")]
    class INFO(ctypes.Structure):
        _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", RECT),
                    ("rcWork", RECT), ("dwFlags", wintypes.DWORD),
                    ("szDevice", wintypes.WCHAR * 32)]
    result = []
    callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HMONITOR,
                                       wintypes.HDC, ctypes.POINTER(RECT), wintypes.LPARAM)
    def collect(handle, _hdc, _rect, _data):
        info = INFO()
        info.cbSize = ctypes.sizeof(info)
        if not ctypes.windll.user32.GetMonitorInfoW(handle, ctypes.byref(info)):
            return False
        r = info.rcMonitor
        result.append({"name": info.szDevice, "x": r.left, "y": r.top,
                       "width": r.right-r.left, "height": r.bottom-r.top,
                       "primary": bool(info.dwFlags & 1)})
        return True
    callback = callback_type(collect)
    if not ctypes.windll.user32.EnumDisplayMonitors(None, None, callback, 0):
        raise OSError("Monitor enumeration failed")
    return result


def select_projector_display(available, name=None):
    """Use Lab5's recorded projector position, while checking today's layout."""
    candidates = [d for d in available if d["name"] == name] if name else [
        d for d in available if (d["x"], d["y"]) == DEFAULT_PROJECTOR_XY]
    if len(candidates) != 1:
        raise ValueError("Lab5 projector at (2560, 0) not found; run list-displays and pass --display")
    display = candidates[0]
    if (display["width"], display["height"]) != (1920, 1080):
        raise ValueError("Projector must be configured to native 1920x1080 desktop pixels")
    return display


def open_cameras(args):
    from pypylon import pylon
    factory = pylon.TlFactory.GetInstance()
    devices = {d.GetSerialNumber(): d for d in factory.EnumerateDevices()}
    if args.left_serial == args.right_serial or any(s not in devices for s in
          (args.left_serial, args.right_serial)):
        raise ValueError(f"Two distinct specified camera serials required; detected {list(devices)}")
    cameras = [pylon.InstantCamera(factory.CreateDevice(devices[s])) for s in
               (args.left_serial, args.right_serial)]
    try:
        for camera in cameras:
            camera.Open()
            configure_camera(camera, args.exposure_us, args.gain_db, args.trigger)
            camera.StartGrabbing(pylon.GrabStrategy_OneByOne)
        sizes = [(c.Width.Value, c.Height.Value) for c in cameras]
        if sizes[0] != sizes[1]:
            raise ValueError(f"Camera image sizes differ: {sizes}")
        converter = pylon.ImageFormatConverter()
        converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
        return cameras, converter, pylon, sizes[0]
    except Exception:
        close_cameras(cameras)
        raise


def close_cameras(cameras):
    for camera in cameras:
        if camera.IsGrabbing():
            camera.StopGrabbing()
        if camera.IsOpen():
            camera.Close()


def configure_camera(camera, exposure_us, gain_db, trigger_mode):
    """Match Lab5's fixed image settings and trigger source."""
    try:
        camera.PixelFormat.SetValue("RGB8")
    except Exception:
        print("RGB8 unavailable; keeping current camera pixel format", flush=True)
    camera.ExposureAuto.SetValue("Off")
    camera.GainAuto.SetValue("Off")
    for name, value in (("BalanceWhiteAuto", "Off"), ("Gamma", 1.0)):
        try:
            getattr(camera, name).SetValue(value)
        except Exception:
            print(f"{name} unavailable; keeping current value", flush=True)
    camera.ExposureTime.SetValue(exposure_us)
    camera.Gain.SetValue(gain_db)
    camera.TriggerSelector.SetValue("FrameStart")
    camera.TriggerMode.SetValue("On")
    camera.TriggerSource.SetValue("Software" if trigger_mode == "software" else "Line1")
    if trigger_mode == "hardware":
        try:
            camera.TriggerActivation.SetValue("RisingEdge")
        except Exception:
            print("TriggerActivation unavailable; verify Line1 rising-edge configuration", flush=True)


def grab_pair(cameras, converter, pylon, timeout_ms, trigger_mode="software"):
    if trigger_mode == "software":
        for camera in cameras:
            if camera.CanWaitForFrameTriggerReady():
                camera.WaitForFrameTriggerReady(timeout_ms, pylon.TimeoutHandling_ThrowException)
        for camera in cameras:
            camera.ExecuteSoftwareTrigger()
    images, ticks = [], []
    for camera in cameras:
        result = camera.RetrieveResult(timeout_ms, pylon.TimeoutHandling_ThrowException)
        try:
            if not result.GrabSucceeded():
                raise RuntimeError(f"Camera grab failed: {result.ErrorCode} {result.ErrorDescription}")
            images.append(converter.Convert(result).GetArray().copy())
            ticks.append(int(result.GetTimeStamp()))
        finally:
            result.Release()
    return images[0], images[1], {"left": ticks[0], "right": ticks[1]}


def show_pattern(pattern, display):
    cv2.imshow(PROJECTOR, pattern.image)
    cv2.moveWindow(PROJECTOR, display["x"], display["y"])
    cv2.setWindowProperty(PROJECTOR, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


def preview_pair(left, right, label):
    view = np.hstack((left, right))
    scale = min(1, 1600/view.shape[1], 800/view.shape[0])
    if scale < 1:
        view = cv2.resize(view, None, fx=scale, fy=scale)
    cv2.putText(view, label, (20, 35), cv2.FONT_HERSHEY_SIMPLEX,
                .65, (0, 255, 255), 2)
    return view


def capture_once(view_context, devices, protocol):
    """Capture one stationary view; caller is responsible for motion interlock.

    devices: (cameras, converter, pylon, display, settle_ms, timeout_ms).
    view_context: destination, metadata. Raises on any incomplete capture.
    """
    cameras, converter, pylon, display, settle_ms, timeout_ms = devices
    directory = Path(view_context["destination"])
    order = [p.name for p in protocol]
    rows = []
    try:
        for index, pattern in enumerate(protocol):
            show_pattern(pattern, display)
            if cv2.waitKey(settle_ms) & 255 in (27, ord("q")):
                raise KeyboardInterrupt("Scan cancelled")
            left, right, stamps = grab_pair(cameras, converter, pylon, timeout_ms)
            row = save_pair(directory, index, pattern.name, left, right, stamps)
            rows.append(row)
            with (directory / "frames.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")
            print(f"Saved {index+1}/{len(order)}: {Path(row['left']).name} (left + right)", flush=True)
            cv2.imshow(PREVIEW, preview_pair(left, right, f"{index+1}/{len(order)} {pattern.name}"))
            cv2.waitKey(1)
        return finish_scan(directory, rows, order, view_context["metadata"])
    finally:
        show_pattern(Pattern("black", np.zeros_like(protocol[0].image)), display)
        cv2.waitKey(settle_ms)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("list-cameras", "list-displays", "projector-test", "calibration", "color", "scan"))
    p.add_argument("--left-serial", default=DEFAULT_LEFT_SERIAL)
    p.add_argument("--right-serial", default=DEFAULT_RIGHT_SERIAL)
    p.add_argument("--display", help="override Lab5 projector position with exact monitor name from list-displays")
    p.add_argument("--output", type=Path, default=ROOT / "raw")
    p.add_argument("--object-id")
    p.add_argument("--session-id", default="session_01")
    p.add_argument("--view-id", type=int)
    p.add_argument("--repeat-id", type=int, default=1)
    p.add_argument("--arm-pose-id", type=int)
    p.add_argument("--turntable-deg", type=float)
    p.add_argument("--exposure-us", type=float, default=10000)
    p.add_argument("--gain-db", type=float, default=0)
    p.add_argument("--trigger", choices=("software", "hardware"), default="software",
                   help="Lab5 trigger mode; hardware uses a shared rising edge on Line1")
    p.add_argument("--settle-ms", type=int, default=250)
    p.add_argument("--timeout-ms", type=int, default=5000)
    p.add_argument("--cell-px", type=int, default=16)
    p.add_argument("--scan-protocol", choices=("gray_phase", "dense_dots"),
                   default="gray_phase")
    p.add_argument("--dot-cell-px", type=int, default=8)
    p.add_argument("--dot-seed", type=int, default=5)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if not SAFE.fullmatch(args.session_id):
        raise ValueError("Invalid session-id")
    if args.mode == "list-displays":
        for d in displays():
            print(d)
        return 0
    if args.mode == "list-cameras":
        from pypylon import pylon
        for d in pylon.TlFactory.GetInstance().EnumerateDevices():
            print(d.GetModelName(), d.GetSerialNumber())
        return 0
    ds = displays()
    display = select_projector_display(ds, args.display) if args.mode not in ("calibration", "color") else None
    if args.settle_ms < 1 or args.timeout_ms < 1 or args.exposure_us <= 0:
        raise ValueError("Timing and exposure must be positive")
    patterns = (make_dense_dots(dot_cell=args.dot_cell_px, seed=args.dot_seed)
                if args.mode in ("scan", "projector-test") and args.scan_protocol == "dense_dots" else
                make_patterns(cell=args.cell_px)
                if args.mode not in ("calibration", "color") else None)
    if args.mode == "projector-test":
        cv2.namedWindow(PROJECTOR, cv2.WINDOW_NORMAL)
        try:
            for pattern in patterns:
                show_pattern(pattern, display)
                print(pattern.name, "Space next / Q quit")
                while True:
                    key = cv2.waitKey(50) & 255
                    if key in (27, ord("q")):
                        return 0
                    if key == 32:
                        break
        finally:
            cv2.destroyAllWindows()
        return 0
    if args.mode == "scan" and (not args.object_id or args.view_id is None or
           args.arm_pose_id is None or args.turntable_deg is None):
        raise ValueError("Scan needs object-id, view-id, arm-pose-id, turntable-deg")
    if args.mode == "scan" and args.trigger != "software":
        raise ValueError("Scan sequences require software trigger after each projector pattern")
    cameras, converter, pylon, size = open_cameras(args)
    try:
        cv2.namedWindow(PREVIEW, cv2.WINDOW_NORMAL)
        if args.mode in ("calibration", "color"):
            target = args.output / args.mode / args.session_id
            target.mkdir(parents=True, exist_ok=True)
            (target / "left").mkdir(exist_ok=True)
            (target / "right").mkdir(exist_ok=True)
            record_capture_session(target, cameras, args, size)
            suffix = "calib" if args.mode == "calibration" else "color"
            index = next_manual_index(target, suffix)
            feedback = ""
            feedback_until = 0.0
            while True:
                left, right, stamps = grab_pair(cameras, converter, pylon, args.timeout_ms, args.trigger)
                found = [cv2.findChessboardCornersSB(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY),
                                                      (10, 6))[0]
                         for im in (left, right)] if args.mode == "calibration" else [True, True]
                label = (f"board L:{found[0]} R:{found[1]}" if args.mode == "calibration"
                         else "color chart: check both images")
                if time.monotonic() < feedback_until:
                    label = feedback
                cv2.imshow(PREVIEW, preview_pair(left, right, label + "  Space save / Q quit"))
                key = cv2.waitKey(1) & 255
                if key in (27, ord("q")):
                    break
                if key == 32 and not all(found):
                    feedback = "Not saved: board missing in one or both cameras"
                    feedback_until = time.monotonic() + 2.0
                    print(feedback, flush=True)
                if key == 32 and all(found):
                    row = save_manual_pair(target, index, suffix, left, right, stamps)
                    feedback = f"Saved {Path(row['left']).name} (left + right)"
                    feedback_until = time.monotonic() + 2.0
                    cv2.imshow(PREVIEW, preview_pair(left, right, feedback))
                    cv2.waitKey(1)
                    index += 1
            return 0
        cv2.namedWindow(PROJECTOR, cv2.WINDOW_NORMAL)
        black = Pattern("black", np.zeros_like(patterns[0].image))
        show_pattern(black, display)
        cv2.waitKey(args.settle_ms)
        print("Space starts one complete view; Q quits. Keep object and rig stationary.")
        while True:
            left, right, _ = grab_pair(cameras, converter, pylon, args.timeout_ms)
            cv2.imshow(PREVIEW, preview_pair(left, right, "Space scan / Q quit"))
            key = cv2.waitKey(1) & 255
            if key in (27, ord("q")):
                return 0
            if key != 32:
                continue
            directory = reserve_view(args.output, args.object_id, args.session_id,
                                     args.view_id, args.repeat_id)
            metadata = {"created_at": datetime.now().astimezone().isoformat(),
                "protocol": ("dense_dots_v1" if args.scan_protocol == "dense_dots" else
                             f"gray_phase_c{args.cell_px}_v1"),
                "dot_cell_px": args.dot_cell_px if args.scan_protocol == "dense_dots" else None,
                "dot_seed": args.dot_seed if args.scan_protocol == "dense_dots" else None,
                "pattern_order": [p.name for p in patterns],
                "pattern_sha256": {p.name: hashlib.sha256(p.image.tobytes()).hexdigest() for p in patterns},
                "projector_size_px": [1920, 1080], "display": display, "camera_size_px": list(size),
                "left_serial": args.left_serial, "right_serial": args.right_serial,
                "exposure_us": args.exposure_us, "gain_db": args.gain_db,
                "settle_ms": args.settle_ms, "view_id": args.view_id,
                "arm_pose_id": args.arm_pose_id, "turntable_deg_commanded": args.turntable_deg,
                "timestamp_note": "Camera ticks are from independent clocks; do not subtract"}
            _write_json(directory / "pose.json", {"arm_pose_id": args.arm_pose_id,
                "turntable_deg_commanded": args.turntable_deg,
                "verified_pose": False, "note": "Manual position only; no feedback or vision pose measured"})
            try:
                print("Complete:", capture_once({"destination": directory, "metadata": metadata},
                    (cameras, converter, pylon, display, args.settle_ms, args.timeout_ms), patterns))
            except BaseException as exc:
                _write_json(directory / "failure.json", {"error": str(exc),
                    "time": datetime.now().astimezone().isoformat()})
                raise
            return 0
    finally:
        close_cameras(cameras)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (Exception, KeyboardInterrupt) as exc:
        print(f"Capture stopped: {exc}", file=sys.stderr)
        sys.exit(1)
