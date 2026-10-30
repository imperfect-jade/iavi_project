"""Fit a relative right-to-left BGR color transform from paired chart photos.

This is optional for point-cloud color; geometric dense-dot matching uses raw frames.
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
import sys
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    image = cv2.imdecode(np.frombuffer(Path(path).read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unreadable image: {path}")
    return image


def chart_samples(image, corners, rows=4, columns=6):
    """Sample central 40% of each patch after mapping chart to a unit square."""
    corners = np.asarray(corners, np.float32)
    if corners.shape != (4, 2) or rows < 1 or columns < 1:
        raise ValueError("Need four ordered outer chart corners and positive grid dimensions")
    h = cv2.getPerspectiveTransform(np.float32([[0, 0], [1, 0], [1, 1], [0, 1]]), corners)
    samples = []
    for r in range(rows):
        for c in range(columns):
            x, y = (c+.5)/columns, (r+.5)/rows
            dx, dy = .2/columns, .2/rows
            box = np.float32([[[x-dx, y-dy], [x+dx, y-dy],
                               [x+dx, y+dy], [x-dx, y+dy]]])
            polygon = cv2.perspectiveTransform(box, h)[0]
            mask = np.zeros(image.shape[:2], np.uint8)
            cv2.fillConvexPoly(mask, np.rint(polygon).astype(np.int32), 255)
            if not np.any(mask):
                raise ValueError("Chart patch lies outside image")
            samples.append(cv2.mean(image, mask=mask)[:3])
    return np.asarray(samples, np.float64)


def fit_color(right, left):
    right, left = np.asarray(right, np.float64), np.asarray(left, np.float64)
    if right.shape != left.shape or right.ndim != 2 or right.shape[1] != 3 or len(right) < 4:
        raise ValueError("Need at least four paired BGR samples")
    design = np.column_stack((right, np.ones(len(right))))
    if np.linalg.matrix_rank(design) < 4:
        raise ValueError("Chart colors do not span an affine color model")
    return np.linalg.lstsq(design, left, rcond=None)[0].T


def apply_color_transform(image, matrix):
    """Apply the fitted right-to-left BGR matrix to an 8-bit image."""
    image = np.asarray(image)
    matrix = np.asarray(matrix, np.float32)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8 or matrix.shape != (3, 4):
        raise ValueError("Expected uint8 BGR image and 3x4 matrix")
    corrected = cv2.transform(image.astype(np.float32), matrix)
    return np.clip(np.rint(corrected), 0, 255).astype(np.uint8)


def save_png(path, image):
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"Cannot encode {path}")
    Path(path).write_bytes(encoded.tobytes())


def color_pair_name(name):
    if not isinstance(name, str) or not re.fullmatch(r"color_\d+", name):
        raise ValueError(f"Unsafe color pair name: {name}")
    return name


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images", type=Path, default=ROOT / "raw/color/session_01")
    p.add_argument("--corners", type=Path, required=True,
                   help="JSON pairs: name (e.g. color_001), left/right ordered TL TR BR BL pixel corners")
    p.add_argument("--output", type=Path, default=ROOT / "calibration/color/color_calibration.json")
    p.add_argument("--rows", type=int, default=4)
    p.add_argument("--columns", type=int, default=6)
    args = p.parse_args(argv)
    annotations = json.loads(args.corners.read_text(encoding="utf-8"))["pairs"]
    if not annotations:
        raise ValueError("No annotated chart pairs")
    lefts, rights, usable, rejected = [], [], [], []
    annotated = {item["name"] for item in annotations}
    captured = {path.stem for path in (args.images / "left").glob("color_*.png")} | {
        path.stem for path in (args.images / "right").glob("color_*.png")}
    for name in sorted(captured - annotated):
        reason = "missing chart-corner annotation"
        rejected.append({"image": name, "reason": reason})
        print(f"[UNUSABLE] {name}: {reason}", flush=True)
    for item in annotations:
        name = item["name"]
        color_pair_name(name)
        try:
            left = read(args.images / "left" / f"{name}.png")
            right = read(args.images / "right" / f"{name}.png")
            left_sample = chart_samples(left, item["left"], args.rows, args.columns)
            right_sample = chart_samples(right, item["right"], args.rows, args.columns)
        except (OSError, ValueError, KeyError, cv2.error) as exc:
            reason = str(exc)
            rejected.append({"image": name, "reason": reason})
            print(f"[UNUSABLE] {name}: {reason}", flush=True)
            continue
        lefts.append(left_sample)
        rights.append(right_sample)
        usable.append(name)
        print(f"[USABLE] {name}: {len(left_sample)} paired color patches", flush=True)
    if not usable:
        raise ValueError("No usable color-chart pairs; inspect image paths and corner annotations")
    left, right = np.concatenate(lefts), np.concatenate(rights)
    matrix = fit_color(right, left)
    corrected = np.column_stack((right, np.ones(len(right)))) @ matrix.T
    per_pair = []
    offset = 0
    for name, l_samples, r_samples in zip(usable, lefts, rights):
        count = len(l_samples)
        corrected_pair = corrected[offset:offset+count]
        row = {"image": name,
               "mae_before": float(np.mean(np.abs(r_samples-l_samples))),
               "mae_after": float(np.mean(np.abs(corrected_pair-l_samples)))}
        per_pair.append(row)
        offset += count
        print(f"[COLOR] {name}: MAE before={row['mae_before']:.2f}, "
              f"after={row['mae_after']:.2f} BGR levels", flush=True)
    result = {"model": "right_to_left_bgr_affine", "matrix_3x4": matrix.tolist(),
        "rows": args.rows, "columns": args.columns, "pairs": usable,
        "rejected_pairs": rejected, "per_pair": per_pair,
        "mean_absolute_error_before": float(np.mean(np.abs(right-left))),
        "mean_absolute_error_after": float(np.mean(np.abs(corrected-left))),
        "note": "Relative camera match only; not absolute color accuracy. Apply to color output, not dense-dot matching."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    previews = []
    for name in usable:
        right_image = read(args.images / "right" / f"{name}.png")
        preview_path = args.output.parent / f"{name}_right_corrected.png"
        save_png(preview_path, apply_color_transform(right_image, matrix))
        previews.append(str(preview_path.resolve()))
    result["corrected_preview_images"] = previews
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"RESULT: usable={len(usable)}, unusable={len(rejected)}; "
          f"MAE before={result['mean_absolute_error_before']:.2f}, "
          f"after={result['mean_absolute_error_after']:.2f} BGR levels", flush=True)
    print(f"Saved color calibration: {args.output.resolve()}", flush=True)
    for path in previews:
        print(f"Saved corrected right-image preview: {path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
