"""Experimental Gray/phase matching across nearby rectified camera rows.

The original reconstruction entry point is independent of this module.
Expanded correspondence search does not repair an inconsistent calibration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def match_neighborhood(left_xy, right_xy, min_disparity, max_disparity,
                       tolerance, row_radius=0, uniqueness_margin=.1):
    """Return right-camera (x,y) and projector distance at each left pixel.

    Candidate enumeration is exact within the code-distance/disparity/row
    bounds. The two best feasible candidates must differ by more than the
    uniqueness margin. A right pixel can be assigned to only one left pixel.
    """
    if left_xy.shape != right_xy.shape or left_xy.ndim != 3 or left_xy.shape[2] != 2:
        raise ValueError('Coordinates must have matching HxWx2 shape')
    if not isinstance(row_radius, (int, np.integer)) or row_radius < 0:
        raise ValueError('Row radius must be a nonnegative integer')
    if not np.isfinite([min_disparity, max_disparity, tolerance, uniqueness_margin]).all() or \
            min_disparity > max_disparity or tolerance <= 0 or uniqueness_margin < 0:
        raise ValueError('Invalid matching bounds or tolerance')
    height, width = left_xy.shape[:2]
    matches = np.full(left_xy.shape, np.nan, np.float32)
    scores = np.full((height, width), np.nan, np.float32)
    left_valid = np.isfinite(left_xy).all(axis=2)
    right_valid = np.isfinite(right_xy).all(axis=2)
    for y in range(height):
        lx = np.flatnonzero(left_valid[y])
        if not len(lx):
            continue
        y0, y1 = max(0, y-row_radius), min(height, y+row_radius+1)
        ry, rx = np.nonzero(right_valid[y0:y1])
        ry = ry+y0
        if not len(rx):
            continue
        projected = right_xy[ry, rx]
        order = np.argsort(projected[:, 0], kind='stable')
        rx, ry, projected = rx[order], ry[order], projected[order]
        query = left_xy[y, lx]
        low = np.searchsorted(projected[:, 0], query[:, 0]-tolerance, side='left')
        high = np.searchsorted(projected[:, 0], query[:, 0]+tolerance, side='right')
        counts = high-low
        active = np.flatnonzero(counts)
        if not len(active):
            continue
        max_count = int(counts[active].max())
        # Bound temporary array sizes even when projector x is highly repeated.
        batch_size = max(1, min(128, 65536//max_count))
        offsets = np.arange(max_count)
        for start in range(0, len(active), batch_size):
            ids = active[start:start+batch_size]
            indices = low[ids, None]+offsets
            available = indices < high[ids, None]
            indices = np.minimum(indices, len(rx)-1)
            disparity = lx[ids, None]-rx[indices]
            feasible = available & (disparity >= min_disparity) & (disparity <= max_disparity)
            distance = np.linalg.norm(projected[indices]-query[ids, None], axis=2)
            distance[~feasible | (distance > tolerance)] = np.inf
            best_index = np.argmin(distance, axis=1)
            best = distance[np.arange(len(ids)), best_index]
            second = (np.partition(distance, 1, axis=1)[:, 1] if max_count > 1
                      else np.full(len(ids), np.inf))
            gap = np.full(len(ids), np.nan)
            np.subtract(second, best, out=gap, where=np.isfinite(best))
            keep = (best <= tolerance) & (gap > uniqueness_margin)
            chosen = indices[np.arange(len(ids)), best_index][keep]
            x = lx[ids[keep]]
            matches[y, x, 0] = rx[chosen]
            matches[y, x, 1] = ry[chosen]
            scores[y, x] = best[keep]
    # This must be global: neighboring left rows may compete for one right pixel.
    yy, xx = np.nonzero(np.isfinite(scores))
    if len(xx):
        targets = matches[yy, xx, 1].astype(np.int64)*width + matches[yy, xx, 0].astype(np.int64)
        order = np.argsort(scores[yy, xx], kind='stable')
        _, first = np.unique(targets[order], return_index=True)
        keep = np.zeros(len(xx), bool)
        keep[order[first]] = True
        matches[yy[~keep], xx[~keep]] = np.nan
        scores[yy[~keep], xx[~keep]] = np.nan
    return matches, scores


def mutual_matches(forward, reverse, tolerance=0):
    """Check that right (x,y) maps back to the original left (x,y)."""
    if forward.shape != reverse.shape or forward.ndim != 3 or forward.shape[2] != 2:
        raise ValueError('Correspondences must have matching HxWx2 shape')
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError('Reverse tolerance must be finite and nonnegative')
    height, width = forward.shape[:2]
    keep = np.zeros((height, width), bool)
    yy, xx = np.nonzero(np.isfinite(forward).all(axis=2))
    xr, yr = np.rint(forward[yy, xx]).astype(np.int64).T
    inside = (xr >= 0) & (xr < width) & (yr >= 0) & (yr < height)
    xx, yy, xr, yr = xx[inside], yy[inside], xr[inside], yr[inside]
    back = reverse[yr, xr]
    distance = np.linalg.norm(back-np.column_stack((xx, yy)), axis=1)
    keep[yy, xx] = np.isfinite(back).all(axis=1) & (distance <= tolerance)
    return keep


def triangulate_matches(matches, calibration, scale):
    """Triangulate actual observations; return XYZ and max per-camera residual.

    Projection matrices from stereoRectify produce rectified-left coordinates.
    Negative camera depths and invalid homogeneous coordinates are rejected.
    Reprojection residuals are measured in processing-image pixels.
    """
    if matches.ndim != 3 or matches.shape[2] != 2:
        raise ValueError('Correspondences must have HxWx2 shape')
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Scale must be finite and positive')
    xyz_map = np.full((*matches.shape[:2], 3), np.nan, np.float32)
    residual_map = np.full(matches.shape[:2], np.nan, np.float32)
    yy, xx = np.nonzero(np.isfinite(matches).all(axis=2))
    if not len(xx):
        return xyz_map, residual_map
    projections = [np.asarray(calibration['P_'+side], np.float64).copy()
                   for side in ('left', 'right')]
    for projection in projections:
        if projection.shape != (3, 4) or not np.isfinite(projection).all():
            raise ValueError('Projection matrices must be finite 3x4 arrays')
        projection[:2] *= scale
    observations = [np.column_stack((xx, yy)).astype(np.float64),
                    matches[yy, xx].astype(np.float64)]
    homogeneous = cv2.triangulatePoints(*projections, observations[0].T, observations[1].T)
    with np.errstate(divide='ignore', invalid='ignore'):
        xyz = (homogeneous[:3]/homogeneous[3]).T
    valid = np.isfinite(xyz).all(axis=1) & (np.abs(homogeneous[3]) > 1e-12)
    residuals = []
    for projection, observed in zip(projections, observations):
        projected = np.column_stack((xyz, np.ones(len(xyz)))) @ projection.T
        valid &= projected[:, 2] > 0
        with np.errstate(divide='ignore', invalid='ignore'):
            predicted = projected[:, :2]/projected[:, 2, None]
        residuals.append(np.linalg.norm(predicted-observed, axis=1))
    residual = np.maximum(*residuals)
    valid &= np.isfinite(residual)
    xyz_map[yy[valid], xx[valid]] = xyz[valid]
    residual_map[yy[valid], xx[valid]] = residual[valid]
    return xyz_map, residual_map


def load_decoded_cache(path, calibration, scale, calibration_path):
    """Load rectified coordinates only with matching scale/calibration metadata."""
    path = Path(path)
    qc_path = path.parent/'qc.json'
    if not qc_path.is_file():
        raise ValueError('Decoder cache requires a sibling qc.json with scale and calibration')
    metadata = json.loads(qc_path.read_text(encoding='utf-8'))
    if metadata.get('scale') != scale:
        raise ValueError('Decoder cache scale differs from requested scale')
    declared = metadata.get('calibration')
    if not declared or Path(declared).resolve() != Path(calibration_path).resolve():
        raise ValueError('Decoder cache calibration path differs from requested calibration')
    current_hash = hashlib.sha256(Path(calibration_path).read_bytes()).hexdigest()
    expected_hash = metadata.get('calibration_sha256')
    if expected_hash and expected_hash != current_hash:
        raise ValueError('Decoder cache calibration hash differs from requested calibration')
    width, height = [int(round(float(x)*scale)) for x in calibration['image_size']]
    coordinates = []
    with np.load(path) as archive:
        for side in ('left', 'right'):
            xy = archive[side].astype(np.float32, copy=True)
            valid = archive[side+'_valid'].astype(bool)
            if xy.shape != (height, width, 2) or valid.shape != (height, width):
                raise ValueError('Decoder cache dimensions/shape differ from calibration and scale')
            xy[~valid] = np.nan
            coordinates.append(xy)
    metadata = {**metadata, 'calibration_hash_available': bool(expected_hash),
                'cache_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    return coordinates[0], coordinates[1], metadata


def _percentiles(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if not len(values):
        return None
    return {name: float(value) for name, value in zip(
        ('p05', 'median', 'p95'), np.percentile(values, [5, 50, 95]))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--view', type=Path, help='complete Gray/phase raw view')
    source.add_argument('--decoded', type=Path, help='rectified decoder NPZ plus sibling qc.json')
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='new output directory')
    parser.add_argument('--scale', type=float, choices=(.5, 1.), default=.5)
    parser.add_argument('--row-radius', type=int, default=0, help='vertical search radius in processing pixels')
    parser.add_argument('--match-tolerance-px', type=float, default=1.5, help='2D projector-coordinate distance')
    parser.add_argument('--uniqueness-margin-px', type=float, default=.1, help='minimum second-best code-distance gap')
    parser.add_argument('--lr-tolerance-px', type=float, default=0, help='reverse pixel-coordinate distance')
    parser.add_argument('--max-reprojection-error-px', type=float, default=1,
                        help='maximum error in either camera, in processing pixels')
    parser.add_argument('--min-depth-mm', type=float, default=300)
    parser.add_argument('--max-depth-mm', type=float, default=1500)
    parser.add_argument('--point-stride', type=int, default=2)
    parser.add_argument('--gray-contrast-min', type=int, default=10)
    parser.add_argument('--illumination-min', type=int, default=20)
    parser.add_argument('--modulation-min', type=float, default=12)
    parser.add_argument('--left-color', type=Path, help='optional already rectified left color image for cache mode')
    args = parser.parse_args(argv)
    numeric = [args.match_tolerance_px, args.uniqueness_margin_px, args.lr_tolerance_px,
               args.max_reprojection_error_px, args.min_depth_mm, args.max_depth_mm,
               args.gray_contrast_min, args.illumination_min, args.modulation_min]
    if not np.isfinite(numeric).all() or args.row_radius < 0 or args.point_stride < 1 or \
            args.match_tolerance_px <= 0 or args.uniqueness_margin_px < 0 or \
            args.lr_tolerance_px < 0 or args.max_reprojection_error_px <= 0 or \
            not 0 < args.min_depth_mm < args.max_depth_mm or \
            min(args.gray_contrast_min, args.illumination_min, args.modulation_min) < 0:
        parser.error('Invalid search, geometry or decoder parameters')
    if args.output.exists():
        raise ValueError(f'Output already exists: {args.output}; choose a new directory')
    if args.view and args.left_color:
        parser.error('--left-color applies only to cached decoder input')
    with np.load(args.calibration) as archive:
        calibration = {key: archive[key].copy() for key in archive.files}
    calibration_hash = hashlib.sha256(args.calibration.read_bytes()).hexdigest()
    # Import only the stable shared utilities; this file never edits that module.
    from reconstruct_view import (read_image, load_run, decode_side,
                                  rectification_maps, rectify_decoded, write_ply)
    warnings = ['Experimental reconstruction: wider search does not update or validate stereo calibration.']
    if args.decoded:
        left, right, metadata = load_decoded_cache(
            args.decoded, calibration, args.scale, args.calibration)
        protocol = metadata.get('protocol', 'cached_gray_phase')
        if not metadata['calibration_hash_available']:
            warnings.append('Legacy decoder cache has no calibration hash; association checked by path and scale.')
        white = np.full((*left.shape[:2], 3), 180, np.uint8)
        color_source = 'neutral_gray'
        if args.left_color:
            white = read_image(args.left_color, color=True)
            if white.shape != (*left.shape[:2], 3):
                raise ValueError('Rectified left color image dimensions differ from decoder cache')
            color_source = str(args.left_color.resolve())
        source_path = args.decoded.resolve()
        source_hash = metadata['cache_sha256']
        decoder_thresholds = metadata.get('thresholds')
        print('Using rectified decoder cache; original decoder thresholds are unchanged.', flush=True)
    else:
        manifest_path = args.view/'manifest.json'
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if not manifest.get('protocol', '').startswith('gray_phase_c'):
            raise ValueError('This entry point supports Gray/phase protocols only')
        camera_metadata = json.loads(args.calibration.with_suffix('.json').read_text(encoding='utf-8'))
        calibration.update({key: camera_metadata[key] for key in ('left_serial', 'right_serial')})
        manifest, frames, cell = load_run(args.view, calibration)
        maps, _ = rectification_maps(calibration, args.scale)
        coordinates = []
        for side, mapping in zip(('left', 'right'), maps):
            rows = [{'pattern': row['pattern'], 'file': row[side]} for row in frames]
            xy, valid, _, _ = decode_side(args.view, rows, tuple(manifest['projector_size_px']), cell,
                args.gray_contrast_min, args.illumination_min, args.modulation_min)
            if xy.shape[:2] != tuple(int(x) for x in calibration['image_size'][::-1]):
                raise ValueError('Raw camera image dimensions differ from calibration')
            coordinates.append(rectify_decoded(xy, valid, mapping)[0])
        left, right = coordinates
        white_frame = next(row['left'] for row in frames if row['pattern'] == 'white')
        white = cv2.remap(read_image(args.view/white_frame, color=True), *maps[0], cv2.INTER_LINEAR)
        color_source = 'raw_left_white_rectified'
        protocol = manifest['protocol']
        source_path = args.view.resolve()
        source_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        decoder_thresholds = {'gray_contrast_min': args.gray_contrast_min,
            'illumination_min': args.illumination_min, 'modulation_min': args.modulation_min}
    p1, p2 = calibration['P_left'], calibration['P_right']
    def disparity_at(z):
        return args.scale*((p1[0, 2]-p2[0, 2])+(p1[0, 3]-p2[0, 3])/z)
    far, near = disparity_at(args.max_depth_mm), disparity_at(args.min_depth_mm)
    if not 0 <= far < near < left.shape[1]:
        raise ValueError(f'Invalid horizontal disparity bounds: {far:.2f}..{near:.2f}')
    print(f'Matching rows +/-{args.row_radius}; disparity {far:.2f}..{near:.2f}', flush=True)
    forward, scores = match_neighborhood(left, right, far, near, args.match_tolerance_px,
                                         args.row_radius, args.uniqueness_margin_px)
    reverse, _ = match_neighborhood(right, left, -near, -far, args.match_tolerance_px,
                                    args.row_radius, args.uniqueness_margin_px)
    forward_count = int(np.isfinite(forward).all(axis=2).sum())
    mutual = mutual_matches(forward, reverse, args.lr_tolerance_px)
    forward[~mutual] = np.nan
    scores[~mutual] = np.nan
    xyz, residual = triangulate_matches(forward, calibration, args.scale)
    depth = xyz[:, :, 2].copy()
    valid = np.isfinite(xyz).all(axis=2) & (depth >= args.min_depth_mm) & \
        (depth <= args.max_depth_mm) & (residual <= args.max_reprojection_error_px)
    yy, xx = np.indices(valid.shape)
    vertical = forward[:, :, 1]-yy
    sampled = valid & (yy % args.point_stride == 0) & (xx % args.point_stride == 0)
    point_count = int(sampled.sum())
    depth[~valid] = np.nan
    xyz[~valid] = np.nan
    offsets, counts = np.unique(vertical[valid].astype(int), return_counts=True)
    qc = {'quality_status': 'experimental_geometry_unverified', 'protocol': protocol,
        'algorithm': 'Exact bounded 2D Gray-coordinate match; actual two-camera pixel triangulation',
        'source': str(source_path), 'source_sha256': source_hash,
        'calibration': str(args.calibration.resolve()), 'calibration_sha256': calibration_hash,
        'scale': args.scale, 'unit': 'mm', 'coordinate_frame': 'rectified_left_camera',
        'color_source': color_source, 'row_radius_processing_px': args.row_radius,
        'match_tolerance_projector_px': args.match_tolerance_px,
        'uniqueness_margin_projector_px': args.uniqueness_margin_px,
        'reverse_tolerance_processing_px': args.lr_tolerance_px,
        'max_reprojection_error_processing_px': args.max_reprojection_error_px,
        'depth_range_mm': [args.min_depth_mm, args.max_depth_mm],
        'disparity_range_processing_px': [float(far), float(near)],
        'decoder_thresholds': decoder_thresholds,
        'rectified_left_pixels': int(np.isfinite(left).all(axis=2).sum()),
        'rectified_right_pixels': int(np.isfinite(right).all(axis=2).sum()),
        'forward_matches': forward_count, 'mutual_matches': int(mutual.sum()),
        'valid_depth_pixels': int(valid.sum()), 'point_count': point_count,
        'geometry_rejected_pixels': int(mutual.sum()-valid.sum()),
        'accepted_right_minus_left_row_histogram': {str(int(k)): int(v) for k, v in zip(offsets, counts)},
        'reprojection_error_before_filter_processing_px': _percentiles(residual[mutual]),
        'reprojection_error_accepted_processing_px': _percentiles(residual[valid]),
        'reprojection_error_accepted_original_px': _percentiles(residual[valid]/args.scale),
        'depth_mm': _percentiles(depth[valid]), 'warnings': warnings}
    args.output.mkdir(parents=True, exist_ok=False)
    np.save(args.output/'right_xy_px.npy', forward)
    np.save(args.output/'matching_score_projector_px.npy', scores)
    np.save(args.output/'reprojection_error_px.npy', residual)
    np.save(args.output/'vertical_offset_px.npy', vertical.astype(np.float32))
    np.save(args.output/'depth_mm.npy', depth)
    cv2.imencode('.png', valid.astype(np.uint8)*255)[1].tofile(args.output/'valid_mask.png')
    preview = cv2.applyColorMap(np.uint8(np.clip((args.max_depth_mm-np.nan_to_num(
        depth, nan=args.max_depth_mm))/(args.max_depth_mm-args.min_depth_mm)*255, 0, 255)), cv2.COLORMAP_TURBO)
    preview[~valid] = 0
    cv2.imencode('.png', preview)[1].tofile(args.output/'depth_preview.png')
    if point_count:
        cloud = xyz[sampled]
        colors = white[sampled]
        write_ply(args.output/'cloud_rectified_left_EXPERIMENTAL.ply', cloud, colors,
                  'rectified_left_camera; EXPERIMENTAL geometry unverified')
        original_left = (calibration['rectify_left'].T @ cloud.T).T
        write_ply(args.output/'cloud_left_EXPERIMENTAL.ply', original_left, colors,
                  'original_left_camera; EXPERIMENTAL geometry unverified')
    (args.output/'qc.json').write_text(json.dumps(qc, indent=2, allow_nan=False), encoding='utf-8')
    print(f'Forward {forward_count:,}; mutual {int(mutual.sum()):,}; valid depth {int(valid.sum()):,}; '
          f'point cloud {point_count:,}', flush=True)
    print(f'Experimental geometry remains unverified. Outputs: {args.output.resolve()}', flush=True)
    return 0 if point_count else 2


if __name__ == '__main__':
    raise SystemExit(main())
