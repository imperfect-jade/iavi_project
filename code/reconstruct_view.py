"""Reconstruct a stereo dense-dots view in millimetres.

Metric geometry comes solely from calibrated stereo cameras.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import warnings

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read_image(path, color=False):
    mode = cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE
    image = cv2.imdecode(np.frombuffer(Path(path).read_bytes(), np.uint8), mode)
    if image is None:
        raise ValueError(f"Cannot read {path}")
    return image


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


def white_alignment_warnings(alignment, threshold_px=3.0):
    if alignment and alignment["median_abs_epipolar_px"] > threshold_px:
        return [f"White-image epipolar median |dy|={alignment['median_abs_epipolar_px']:.2f} "
                f"processing pixels exceeds {threshold_px:.2f}; reconstructed geometry may be inaccurate. "
                "Check stereo rig stability and calibration."]
    return []


def estimate_disparity_range(left, right):
    """Estimate a scene-specific disparity search interval from rectified features."""
    if left.shape != right.shape:
        raise ValueError("Rectified stereo images must have equal dimensions")
    gray = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) if im.ndim == 3 else im
            for im in (left, right)]
    detector = cv2.SIFT_create(nfeatures=5000)
    keypoints, descriptors = zip(*(detector.detectAndCompute(im, None) for im in gray))
    if any(desc is None for desc in descriptors):
        raise ValueError("Cannot estimate disparity: no image features; set both depth bounds manually")
    matches = cv2.BFMatcher().knnMatch(descriptors[0], descriptors[1], k=2)
    disparities = []
    for pair in matches:
        if len(pair) != 2 or pair[0].distance >= .7 * pair[1].distance:
            continue
        a, b = keypoints[0][pair[0].queryIdx].pt, keypoints[1][pair[0].trainIdx].pt
        disparity = a[0] - b[0]
        if abs(a[1]-b[1]) <= 3 and 0 < disparity < left.shape[1]:
            disparities.append(disparity)
    if len(disparities) < 15:
        raise ValueError(f"Cannot estimate disparity from only {len(disparities)} reliable matches; "
                         "set both --min-depth-mm and --max-depth-mm manually")
    low, high = np.percentile(disparities, (1, 99))
    padding = max(8, .20 * (high-low))
    far = max(1.0, float(np.floor(low-padding)))
    near = min(float(left.shape[1]-1), float(np.ceil(high+padding)))
    return far, near, len(disparities)


def stereo_disparity(left, right, min_disparity, num_disparities, block_size=5,
                     lr_tolerance_px=1.5):
    """Lab5-style SGBM with reciprocal left-right consistency."""
    if (left.shape != right.shape or left.ndim != 2 or left.dtype != np.uint8 or
            right.dtype != np.uint8):
        raise ValueError("SGBM expects equal-size uint8 grayscale images")
    if num_disparities <= 0 or num_disparities % 16 or block_size < 3 or block_size % 2 != 1:
        raise ValueError("Invalid SGBM disparity count or block size")
    def matcher(minimum):
        return cv2.StereoSGBM_create(
            minDisparity=minimum, numDisparities=num_disparities,
            blockSize=block_size, P1=8*block_size**2, P2=32*block_size**2,
            disp12MaxDiff=1, preFilterCap=31, uniquenessRatio=10,
            speckleWindowSize=100, speckleRange=2,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    disparity = matcher(min_disparity).compute(left, right).astype(np.float32)/16
    right_min = -(min_disparity+num_disparities)
    reverse = matcher(right_min).compute(right, left).astype(np.float32)/16
    height, width = left.shape
    x = np.arange(width, dtype=np.float32)[None, :]
    xr = np.rint(x-disparity).astype(np.int32)
    in_bounds = (xr >= 0) & (xr < width)
    sampled = reverse[np.arange(height)[:, None], np.clip(xr, 0, width-1)]
    valid = (disparity > min_disparity-.5) & in_bounds & (sampled > right_min-.5) & \
            (np.abs(disparity+sampled) <= lr_tolerance_px)
    disparity[~valid] = np.nan
    return disparity


def fuse_disparities(disparities, tolerance_px=1.5):
    """Keep per-pixel median of mutually agreeing estimates; retain lone valid hits."""
    if not disparities or any(d.shape != disparities[0].shape for d in disparities):
        raise ValueError("Disparity maps must be nonempty and equally sized")
    stack = np.stack(disparities).astype(np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        center = np.nanmedian(stack, axis=0)
        inliers = np.where(np.abs(stack-center) <= tolerance_px, stack, np.nan)
        fused = np.nanmedian(inliers, axis=0).astype(np.float32)
    support = np.isfinite(inliers).sum(axis=0).astype(np.uint8)
    fused[support == 0] = np.nan
    return fused, support


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
    if protocol == "dense_dots_v1":
        expected = ["dense_dots", "white"]
    elif protocol == "dense_dots_v2":
        count = manifest.get("dot_frames")
        if not isinstance(count, int) or not 1 <= count <= 32:
            raise ValueError("Invalid dense-dot frame count")
        expected = [f"dense_dots_{index:02d}" for index in range(count)] + ["white"]
    else:
        raise ValueError("Unsupported protocol")
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
    return manifest, frames


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--view", type=Path, required=True, help="raw/.../view_NNN/repeat_NNN")
    p.add_argument("--calibration", type=Path, default=ROOT / "calibration/stereo/stereo_calibration.npz")
    p.add_argument("--output", type=Path, help="processed/.../view_NNN/repeat_NNN; defaults from --view")
    p.add_argument("--scale", type=float, choices=(.5, 1.0), default=.5)
    p.add_argument("--min-depth-mm", type=float, help="manual lower depth bound; requires --max-depth-mm")
    p.add_argument("--max-depth-mm", type=float, help="manual upper depth bound; requires --min-depth-mm")
    p.add_argument("--point-stride", type=int, default=2)
    p.add_argument("--sgbm-block-size", type=int, default=5)
    args = p.parse_args(argv)
    if (args.min_depth_mm is None) != (args.max_depth_mm is None):
        p.error("Set both depth bounds, or omit both for automatic scene range")
    if (args.min_depth_mm is not None and
            (args.min_depth_mm <= 0 or args.max_depth_mm <= args.min_depth_mm)) or args.point_stride < 1:
        p.error("Depth range and point stride invalid")
    if args.sgbm_block_size < 3 or args.sgbm_block_size % 2 != 1:
        p.error("--sgbm-block-size must be odd and at least 3")
    with np.load(args.calibration) as archive:
        calibration = {key: archive[key].copy() for key in archive.files}
    metadata = json.loads(args.calibration.with_suffix(".json").read_text(encoding="utf-8"))
    calibration.update({"left_serial": metadata["left_serial"], "right_serial": metadata["right_serial"]})
    manifest, frames = load_run(args.view, calibration)
    view = args.view.resolve()
    if args.output:
        output = args.output
    else:
        relative = view.relative_to((ROOT / "raw").resolve())
        output = ROOT / "processed" / relative
    output.mkdir(parents=True, exist_ok=True)
    maps, target = rectification_maps(calibration, args.scale)
    q = q_for_scale(calibration["Q"], args.scale)
    def disparity_at(z):
        return (q[2, 3]/z-q[3, 3])/q[3, 2]
    dot_frames = frames[:-1]
    image_pair = None
    if args.min_depth_mm is None:
        estimated = []
        for index, source in enumerate(dot_frames):
            pair = [cv2.remap(read_image(view / source[side], color=True),
                              *mapping, cv2.INTER_LINEAR)
                    for side, mapping in zip(("left", "right"), maps)]
            if index == 0:
                image_pair = pair
            try:
                estimated.append(estimate_disparity_range(*pair))
            except ValueError as exc:
                print(f"[WARNING] {source['pattern']}: {exc}", flush=True)
        if not estimated:
            raise ValueError("No dot frame supplied enough matches to estimate depth; "
                             "set both depth bounds manually")
        far = min(item[0] for item in estimated)
        near = max(item[1] for item in estimated)
        auto_matches = sum(item[2] for item in estimated)
        def depth_at(disparity):
            return float(q[2, 3] / (q[3, 2] * disparity + q[3, 3]))
        min_depth_mm, max_depth_mm = depth_at(near), depth_at(far)
        if not np.isfinite([min_depth_mm, max_depth_mm]).all() or not 0 < min_depth_mm < max_depth_mm:
            raise ValueError("Automatic depth range is invalid for this calibration; "
                             "set both depth bounds manually")
        range_source = "auto_feature_matches"
        print(f"Auto disparity from {auto_matches} rectified matches: {far:.1f}..{near:.1f}px "
              f"(depth {min_depth_mm:.1f}..{max_depth_mm:.1f}mm)", flush=True)
    else:
        min_depth_mm, max_depth_mm = args.min_depth_mm, args.max_depth_mm
        far, near = disparity_at(max_depth_mm), disparity_at(min_depth_mm)
        auto_matches = None
        range_source = "manual_depth_bounds"
    if not 0 <= far < near < target[0]:
        raise ValueError(f"Invalid disparity range from calibration/depth: {far:.1f}..{near:.1f}")
    print(f"Expected disparity: {far:.1f}..{near:.1f}px", flush=True)
    per_pattern_pixels = []
    minimum = math.floor(far)
    count = math.ceil((near-minimum+1)/16)*16
    if count >= target[0]:
        raise ValueError("SGBM disparity range exceeds image width; narrow depth range or increase scale")
    estimates = []
    for index, frame in enumerate(dot_frames):
        pair = image_pair if index == 0 and image_pair is not None else [
            cv2.remap(read_image(view / frame[side], color=True),
                      *mapping, cv2.INTER_LINEAR)
            for side, mapping in zip(("left", "right"), maps)]
        one = stereo_disparity(
            cv2.cvtColor(pair[0], cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(pair[1], cv2.COLOR_BGR2GRAY),
            minimum, count, args.sgbm_block_size)
        one[(one < far) | (one > near)] = np.nan
        per_pattern_pixels.append(int(np.isfinite(one).sum()))
        estimates.append(one)
        print(f"  {frame['pattern']}: {per_pattern_pixels[-1]:,} matched pixels", flush=True)
    disparity, fusion_support = fuse_disparities(estimates)
    color_frame = frames[-1]
    algorithm = "dense_dots_sgbm_median_fusion" if len(dot_frames) > 1 else "dense_dots_sgbm_lr_consistent"
    white_views = [cv2.remap(read_image(view / color_frame[side]),
                             *mapping, cv2.INTER_LINEAR)
                   for side, mapping in zip(("left", "right"), maps)]
    alignment = white_epipolar_diagnostic(white_views[0], white_views[1], far, near)
    warnings = white_alignment_warnings(alignment)
    if alignment:
        print(f"White-image epipolar check: {alignment['feature_matches']} matches, "
              f"median |dy|={alignment['median_abs_epipolar_px']:.2f}px", flush=True)
    for warning in warnings:
        print(f"[WARNING] {warning}", flush=True)
    print(f"Stereo matched: {np.isfinite(disparity).sum():,} pixels", flush=True)
    xyz = cv2.reprojectImageTo3D(np.nan_to_num(disparity, nan=0), q)
    depth = xyz[:, :, 2].astype(np.float32)
    valid = np.isfinite(disparity) & np.isfinite(xyz).all(axis=2) & \
        (depth >= min_depth_mm) & (depth <= max_depth_mm)
    depth[~valid] = np.nan
    xyz[~valid] = np.nan
    white = cv2.remap(read_image(view / color_frame["left"], color=True),
                      *maps[0], cv2.INTER_LINEAR)
    sampled = valid.copy()
    sampled[::args.point_stride, ::args.point_stride] = False
    sampled = valid & ~sampled
    cloud = xyz[sampled]
    colors = white[sampled]
    if len(cloud) == 0:
        raise ValueError("No valid 3D points; inspect dot matching and depth range")
    left_cloud = (calibration["rectify_left"].T @ cloud.T).T
    write_ply(output / "cloud_rectified_left.ply", cloud, colors, "rectified_left_camera")
    write_ply(output / "cloud_left.ply", left_cloud, colors, "original_left_camera")
    np.save(output / "disparity_px.npy", disparity)
    np.save(output / "fusion_support.npy", fusion_support)
    np.save(output / "depth_mm.npy", depth)
    cv2.imencode(".png", (valid.astype(np.uint8)*255))[1].tofile(output / "valid_mask.png")
    preview = cv2.applyColorMap(np.uint8(np.clip((max_depth_mm-np.nan_to_num(
        depth, nan=max_depth_mm))/(max_depth_mm-min_depth_mm)*255, 0, 255)),
        cv2.COLORMAP_TURBO)
    preview[~valid] = 0
    cv2.imencode(".png", preview)[1].tofile(output / "depth_preview.png")
    qc = {"protocol": manifest["protocol"], "input": str(view),
          "calibration": str(args.calibration.resolve()), "scale": args.scale,
          "algorithm": algorithm,
          "coordinate_frame": "rectified_left_camera", "unit": "mm",
          "matched_pixels": int(np.isfinite(disparity).sum()),
          "dot_pattern_count": len(dot_frames),
          "per_pattern_matched_pixels": per_pattern_pixels,
          "fusion_support_at_least_two_pixels": int((fusion_support >= 2).sum()),
          "valid_depth_pixels": int(valid.sum()), "point_count": int(len(cloud)),
          "median_depth_mm": float(np.median(depth[valid])),
          "white_epipolar_diagnostic": alignment, "warnings": warnings,
          "depth_range_mm": [min_depth_mm, max_depth_mm],
          "depth_range_source": range_source,
          "auto_range_feature_matches": auto_matches,
          "disparity_range_px": [far, near],
          "thresholds": {"sgbm_block_size": args.sgbm_block_size}}
    (output / "qc.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
    print(f"Reconstructed {len(cloud):,} points; median depth {qc['median_depth_mm']:.1f} mm")
    print(f"Outputs: {output.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
