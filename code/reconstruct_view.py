"""Reconstruct one 38-frame stereo Gray/phase view in millimetres.

Gray addresses unwrap both four-step phases. Projector coordinates establish
correspondence; metric geometry comes solely from calibrated stereo cameras.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import struct
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def pattern_names(width, height, cell):
    result = []
    for axis, length in (("x", width), ("y", height)):
        bits = (math.ceil(length/cell)-1).bit_length()
        for bit in range(bits-1, -1, -1):
            name = f"gray_{axis}_{bit:02d}"
            result.extend((name, name+"_inv"))
    result.extend(f"phase_{axis}_{step:02d}" for axis in ("x", "y") for step in range(4))
    return result + ["white", "black"]


def read_image(path, color=False):
    mode = cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE
    image = cv2.imdecode(np.frombuffer(Path(path).read_bytes(), np.uint8), mode)
    if image is None:
        raise ValueError(f"Cannot read {path}")
    return image


def decode_axis(gray_pairs, phase_frames, cell, projector_length, contrast_min=10,
                modulation_min=12):
    """Return absolute projector coordinate, mask and modulation.

    With I_k=a+b*cos(phi-k*pi/2), atan2(I1-I3,I0-I2) gives wrapped phi.
    Gray bits identify the cell; adding phi/(2*pi) unwraps it.
    """
    shape = phase_frames[0].shape
    code = np.zeros(shape, np.uint16)
    valid = np.ones(shape, bool)
    for positive, inverse in gray_pairs:
        if positive.shape != shape or inverse.shape != shape:
            raise ValueError("Gray frame shape mismatch")
        difference = positive.astype(np.int16)-inverse.astype(np.int16)
        valid &= np.abs(difference) > contrast_min
        code = (code << 1) | (difference > 0).astype(np.uint16)
    # Gray to binary via repeated XOR of shifted words.
    cell_index = code.copy()
    shift = 1
    while shift < 16:
        cell_index ^= cell_index >> shift
        shift *= 2
    c = phase_frames[0].astype(np.float32)-phase_frames[2].astype(np.float32)
    s = phase_frames[1].astype(np.float32)-phase_frames[3].astype(np.float32)
    modulation = .5*np.hypot(c, s)
    fraction = np.mod(np.arctan2(s, c), 2*np.pi)/(2*np.pi)
    coordinate = cell*(cell_index.astype(np.float32)+fraction.astype(np.float32))
    valid &= (cell_index < math.ceil(projector_length/cell))
    valid &= (modulation >= modulation_min) & (coordinate < projector_length)
    coordinate[~valid] = np.nan
    return coordinate, valid, modulation


def decode_side(directory, rows, projector_size, cell, contrast_min, illumination_min,
                modulation_min):
    names = [row["pattern"] for row in rows]
    def load(name):
        return read_image(directory / rows[names.index(name)]["file"])
    white, black = load("white"), load("black")
    illumination = white.astype(np.int16)-black.astype(np.int16)
    valid = illumination > illumination_min
    coords, mods = [], []
    for axis, length in (("x", projector_size[0]), ("y", projector_size[1])):
        bits = (math.ceil(length/cell)-1).bit_length()
        gray = [(load(f"gray_{axis}_{bit:02d}"),
                 load(f"gray_{axis}_{bit:02d}_inv")) for bit in range(bits-1, -1, -1)]
        phase = [load(f"phase_{axis}_{k:02d}") for k in range(4)]
        if any(im.shape != white.shape for pair in gray for im in pair) or any(
                im.shape != white.shape for im in phase):
            raise ValueError("Camera frame dimensions differ")
        coordinate, axis_valid, modulation = decode_axis(
            gray, phase, cell, length, contrast_min, modulation_min)
        coords.append(coordinate)
        mods.append(modulation)
        valid &= axis_valid
        print(f"  {axis}: {np.count_nonzero(axis_valid):,} phase-unwrapped pixels", flush=True)
    xy = np.stack(coords, axis=-1)
    xy[~valid] = np.nan
    return xy, valid, np.minimum(*mods), white


def rectification_maps(calibration, scale):
    size = tuple(int(x) for x in calibration["image_size"])
    target = tuple(int(round(x*scale)) for x in size)
    maps = []
    for side in ("left", "right"):
        projection = calibration[f"P_{side}"].copy()
        projection[:2] *= scale
        maps.append(cv2.initUndistortRectifyMap(
            calibration[f"K_{side}"], calibration[f"D_{side}"],
            calibration[f"rectify_{side}"], projection, target, cv2.CV_32FC1))
    return maps, target


def rectify_decoded(xy, valid, maps):
    """Interpolate continuous values only where the complete source footprint is valid."""
    mapped = cv2.remap(np.nan_to_num(xy, nan=0), *maps, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT)
    footprint = cv2.remap(valid.astype(np.float32), *maps, cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT)
    mask = footprint > .999
    # Reject interpolation over projector-coordinate jumps at Gray cell errors.
    jump_x = np.max(np.abs(np.diff(mapped, axis=1)), axis=2)
    jump_y = np.max(np.abs(np.diff(mapped, axis=0)), axis=2)
    mask[:, 1:] &= jump_x < 12
    mask[:, :-1] &= jump_x < 12
    mask[1:, :] &= jump_y < 12
    mask[:-1, :] &= jump_y < 12
    mapped[~mask] = np.nan
    return mapped, mask


def match_rows(left_xy, right_xy, min_disparity, max_disparity, tolerance):
    """Integer pixel baseline: nearest 2D projector coordinate on same epipolar row."""
    if left_xy.shape != right_xy.shape or left_xy.ndim != 3 or left_xy.shape[2] != 2:
        raise ValueError("Rectified coordinates must have matching HxWx2 shape")
    height, width = left_xy.shape[:2]
    result = np.full((height, width), np.nan, np.float32)
    offsets = np.arange(-5, 6)
    for y in range(height):
        lx = np.flatnonzero(np.isfinite(left_xy[y]).all(axis=1))
        rx = np.flatnonzero(np.isfinite(right_xy[y]).all(axis=1))
        if not len(lx) or not len(rx):
            continue
        order = np.argsort(right_xy[y, rx, 0], kind="stable")
        rx = rx[order]
        projected = right_xy[y, rx]
        center = np.searchsorted(projected[:, 0], left_xy[y, lx, 0])
        candidate = np.clip(center[:, None]+offsets, 0, len(rx)-1)
        candidate_x = rx[candidate]
        distance = np.linalg.norm(projected[candidate]-left_xy[y, lx, None], axis=2)
        disparity = lx[:, None]-candidate_x
        distance[(disparity < min_disparity) | (disparity > max_disparity)] = np.inf
        best = np.argmin(distance, axis=1)
        score = distance[np.arange(len(lx)), best]
        keep = score <= tolerance
        chosen_l, chosen_r, chosen_d, chosen_score = (lx[keep], candidate_x[np.arange(len(lx)), best][keep],
            disparity[np.arange(len(lx)), best][keep], score[keep])
        if not len(chosen_l):
            continue
        score_order = np.argsort(chosen_score, kind="stable")
        _, first = np.unique(chosen_r[score_order], return_index=True)
        unique = score_order[first]
        result[y, chosen_l[unique]] = chosen_d[unique]
    return result


def q_for_scale(q, scale):
    if scale <= 0:
        raise ValueError("Scale must be positive")
    return np.asarray(q, np.float64) @ np.diag([1/scale, 1/scale, 1/scale, 1])


def white_epipolar_diagnostic(left, right, far, near):
    """Independent feature check on white frames; returns None if texture is sparse."""
    gray = [cv2.cvtColor(view, cv2.COLOR_BGR2GRAY) if view.ndim == 3 else view
            for view in (left, right)]
    detector = cv2.SIFT_create(nfeatures=5000)
    k1, d1 = detector.detectAndCompute(gray[0], None)
    k2, d2 = detector.detectAndCompute(gray[1], None)
    if d1 is None or d2 is None:
        return None
    matches = cv2.BFMatcher().knnMatch(d1, d2, k=2)
    good = [a for a, b in matches if a.distance < .7*b.distance]
    residuals = [abs(k1[m.queryIdx].pt[1]-k2[m.trainIdx].pt[1]) for m in good
                 if far <= k1[m.queryIdx].pt[0]-k2[m.trainIdx].pt[0] <= near]
    if len(residuals) < 15:
        return None
    return {"feature_matches": len(residuals),
            "median_abs_epipolar_px": float(np.median(residuals)),
            "p95_abs_epipolar_px": float(np.percentile(residuals, 95))}


def write_ply(path, xyz, bgr, frame):
    points = np.asarray(xyz, np.float32).reshape(-1, 3)
    colors = np.asarray(bgr, np.uint8).reshape(-1, 3)
    if len(points) != len(colors):
        raise ValueError("Point/color count differs")
    vertex = np.empty(len(points), dtype=[("x", "<f4"), ("y", "<f4"),
        ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    for i, name in enumerate(("x", "y", "z")):
        vertex[name] = points[:, i]
    vertex["red"], vertex["green"], vertex["blue"] = colors[:, 2], colors[:, 1], colors[:, 0]
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"comment coordinates {frame}; unit mm\nelement vertex {len(points)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    with Path(path).open("wb") as stream:
        stream.write(header.encode("ascii"))
        vertex.tofile(stream)


def load_run(directory, calibration):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    projector = tuple(manifest["projector_size_px"])
    if projector != (1920, 1080) or not manifest.get("complete"):
        raise ValueError("Expected complete 1920x1080 scan")
    protocol = manifest["protocol"]
    if not protocol.startswith("gray_phase_c") or not protocol.endswith("_v1"):
        raise ValueError("Unsupported protocol")
    cell = int(protocol[len("gray_phase_c"):-len("_v1")])
    expected = pattern_names(*projector, cell)
    frames = manifest["frames"]
    if len(frames) != len(expected) or [r["pattern"] for r in frames] != expected or \
       manifest["pattern_order"] != expected or manifest["frame_count"] != len(expected):
        raise ValueError("Incomplete or misordered projection sequence")
    if tuple(manifest["camera_size_px"]) != tuple(calibration["image_size"]):
        raise ValueError("Camera image size differs from calibration")
    if manifest["left_serial"] != calibration["left_serial"] or \
       manifest["right_serial"] != calibration["right_serial"]:
        raise ValueError("Camera identities differ from calibration")
    for i, row in enumerate(frames):
        if row["index"] != i or row["pattern"] != expected[i]:
            raise ValueError("Frame index or name mismatch")
        for side in ("left", "right"):
            path = directory / row[side]
            if not path.is_file() or path.parent != directory / side:
                raise ValueError(f"Missing/unsafe camera frame: {path}")
    return manifest, frames, cell


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--view", type=Path, required=True, help="raw/.../view_NNN/repeat_NNN")
    p.add_argument("--calibration", type=Path, default=ROOT / "calibration/stereo/stereo_calibration.npz")
    p.add_argument("--output", type=Path, help="processed/.../view_NNN/repeat_NNN; defaults from --view")
    p.add_argument("--scale", type=float, choices=(.5, 1.0), default=.5)
    p.add_argument("--min-depth-mm", type=float, default=300)
    p.add_argument("--max-depth-mm", type=float, default=1500)
    p.add_argument("--gray-contrast-min", type=int, default=10)
    p.add_argument("--illumination-min", type=int, default=20)
    p.add_argument("--modulation-min", type=float, default=12)
    p.add_argument("--match-tolerance-px", type=float, default=1.5)
    p.add_argument("--point-stride", type=int, default=2)
    args = p.parse_args(argv)
    if args.min_depth_mm <= 0 or args.max_depth_mm <= args.min_depth_mm or args.point_stride < 1:
        p.error("Depth range and point stride invalid")
    if min(args.gray_contrast_min, args.illumination_min, args.modulation_min) < 0 or args.match_tolerance_px <= 0:
        p.error("Invalid decoder/matcher thresholds")
    with np.load(args.calibration) as archive:
        calibration = {key: archive[key].copy() for key in archive.files}
    metadata = json.loads(args.calibration.with_suffix(".json").read_text(encoding="utf-8"))
    calibration.update({"left_serial": metadata["left_serial"], "right_serial": metadata["right_serial"]})
    manifest, frames, cell = load_run(args.view, calibration)
    view = args.view.resolve()
    if args.output:
        output = args.output
    else:
        relative = view.relative_to((ROOT / "raw").resolve())
        output = ROOT / "processed" / relative
    output.mkdir(parents=True, exist_ok=True)
    decoded = {}
    for side in ("left", "right"):
        print(f"Decoding {side} with Gray-cell phase unwrapping...", flush=True)
        rows = [{"pattern": row["pattern"], "file": row[side]} for row in frames]
        xy, valid, modulation, white = decode_side(view, rows, (1920, 1080), cell,
            args.gray_contrast_min, args.illumination_min, args.modulation_min)
        decoded[side] = (xy, valid, modulation, white)
        print(f"  jointly valid: {valid.sum():,}/{valid.size:,}", flush=True)
    maps, target = rectification_maps(calibration, args.scale)
    rect = {}
    for side, mapping in zip(("left", "right"), maps):
        xy, valid, _, _ = decoded[side]
        rect[side] = rectify_decoded(xy, valid, mapping)
        print(f"Rectified {side}: {rect[side][1].sum():,} valid pixels", flush=True)
    q = q_for_scale(calibration["Q"], args.scale)
    def disparity_at(z):
        return (q[2, 3]/z-q[3, 3])/q[3, 2]
    far, near = disparity_at(args.max_depth_mm), disparity_at(args.min_depth_mm)
    if not 0 <= far < near < target[0]:
        raise ValueError(f"Invalid disparity range from calibration/depth: {far:.1f}..{near:.1f}")
    print(f"Expected disparity: {far:.1f}..{near:.1f}px", flush=True)
    white_views = [cv2.remap(decoded[side][3], *mapping, cv2.INTER_LINEAR)
                   for side, mapping in zip(("left", "right"), maps)]
    alignment = white_epipolar_diagnostic(white_views[0], white_views[1], far, near)
    if alignment:
        print(f"White-image epipolar check: {alignment['feature_matches']} matches, "
              f"median |dy|={alignment['median_abs_epipolar_px']:.2f}px", flush=True)
        if alignment["median_abs_epipolar_px"] > 3:
            raise ValueError("White-image epipolar error exceeds 3 processing pixels; "
                             "verify stereo rig has not moved since calibration and recapture calibration")
    disparity = match_rows(rect["left"][0], rect["right"][0], far, near,
                           args.match_tolerance_px)
    print(f"Forward stereo candidates: {np.isfinite(disparity).sum():,}", flush=True)
    # Reverse check keeps only mutually consistent integer correspondences.
    reverse = match_rows(rect["right"][0], rect["left"][0], -near, -far,
                         args.match_tolerance_px)
    yy, xx = np.nonzero(np.isfinite(disparity))
    xr = np.rint(xx-disparity[yy, xx]).astype(int)
    mutual = np.isfinite(reverse[yy, xr]) & (np.abs(reverse[yy, xr]+disparity[yy, xx]) <= 1)
    disparity[yy[~mutual], xx[~mutual]] = np.nan
    print(f"Stereo matched: {np.isfinite(disparity).sum():,} pixels", flush=True)
    xyz = cv2.reprojectImageTo3D(np.nan_to_num(disparity, nan=0), q)
    depth = xyz[:, :, 2].astype(np.float32)
    valid = np.isfinite(disparity) & np.isfinite(xyz).all(axis=2) & \
        (depth >= args.min_depth_mm) & (depth <= args.max_depth_mm)
    depth[~valid] = np.nan
    xyz[~valid] = np.nan
    white = cv2.remap(read_image(view / frames[-2]["left"], color=True),
                      *maps[0], cv2.INTER_LINEAR)
    sampled = valid.copy()
    sampled[::args.point_stride, ::args.point_stride] = False
    sampled = valid & ~sampled
    cloud = xyz[sampled]
    colors = white[sampled]
    if len(cloud) == 0:
        raise ValueError("No valid 3D points; inspect decoding and depth range")
    left_cloud = (calibration["rectify_left"].T @ cloud.T).T
    write_ply(output / "cloud_rectified_left.ply", cloud, colors, "rectified_left_camera")
    write_ply(output / "cloud_left.ply", left_cloud, colors, "original_left_camera")
    np.savez_compressed(output / "decoded_projector_xy.npz",
        left=rect["left"][0], right=rect["right"][0],
        left_valid=rect["left"][1], right_valid=rect["right"][1])
    np.save(output / "disparity_px.npy", disparity)
    np.save(output / "depth_mm.npy", depth)
    cv2.imencode(".png", (valid.astype(np.uint8)*255))[1].tofile(output / "valid_mask.png")
    preview = cv2.applyColorMap(np.uint8(np.clip((args.max_depth_mm-np.nan_to_num(
        depth, nan=args.max_depth_mm))/(args.max_depth_mm-args.min_depth_mm)*255, 0, 255)),
        cv2.COLORMAP_TURBO)
    preview[~valid] = 0
    cv2.imencode(".png", preview)[1].tofile(output / "depth_preview.png")
    qc = {"protocol": manifest["protocol"], "input": str(view),
          "calibration": str(args.calibration.resolve()), "scale": args.scale,
          "algorithm": "Gray cell plus negative four-step phase unwrap; integer stereo pixel match",
          "coordinate_frame": "rectified_left_camera", "unit": "mm",
          "decoded_left_pixels": int(decoded["left"][1].sum()),
          "decoded_right_pixels": int(decoded["right"][1].sum()),
          "rectified_left_pixels": int(rect["left"][1].sum()),
          "rectified_right_pixels": int(rect["right"][1].sum()),
          "matched_pixels": int(np.isfinite(disparity).sum()),
          "valid_depth_pixels": int(valid.sum()), "point_count": int(len(cloud)),
          "median_depth_mm": float(np.median(depth[valid])),
          "depth_range_mm": [args.min_depth_mm, args.max_depth_mm],
          "disparity_range_px": [far, near],
          "thresholds": {"gray_contrast_min": args.gray_contrast_min,
              "illumination_min": args.illumination_min,
              "modulation_min": args.modulation_min,
              "match_tolerance_px": args.match_tolerance_px}}
    (output / "qc.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
    print(f"Reconstructed {len(cloud):,} points; median depth {qc['median_depth_mm']:.1f} mm")
    print(f"Outputs: {output.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
