"""Generate four printable ArUco faces and their shared 3D corner coordinates."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent / "targets"
PX_PER_MM = 20  # 508 dpi


def _point(face: int, u: float, v: float, width: float, depth: float, height: float):
    z = height - v
    if face == 0:
        return [-width / 2 + u, -depth / 2, z]
    if face == 1:
        return [width / 2, -depth / 2 + u, z]
    if face == 2:
        return [width / 2 - u, depth / 2, z]
    return [-width / 2, depth / 2 - u, z]


def build_config(width_mm=45.0, depth_mm=45.0, height_mm=100.0,
                 marker_mm=21.0, top_margin_mm=26.0, marker_gap_mm=4.0):
    if min(width_mm, depth_mm, height_mm, marker_mm) <= 0 or \
       marker_mm >= min(width_mm, depth_mm) or \
       top_margin_mm < 0 or marker_gap_mm < 0 or \
       top_margin_mm + marker_mm * 2 + marker_gap_mm > height_mm:
        raise ValueError("Markers do not fit the cuboid faces")
    markers = {}
    # The mounted target runs 0, 3, 2, 1 around the model's positive perimeter.
    # This mapping was verified from the captured adjacent-face observations.
    mounted_face_to_model_face = (0, 3, 2, 1)
    for face in range(4):
        face_width = width_mm if face % 2 == 0 else depth_mm
        x0 = (face_width - marker_mm) / 2
        for row in range(2):
            marker_id = 20 + face * 2 + row
            y0 = top_margin_mm + row * (marker_mm + marker_gap_mm)
            uv = [(x0, y0), (x0 + marker_mm, y0),
                  (x0 + marker_mm, y0 + marker_mm), (x0, y0 + marker_mm)]
            markers[str(marker_id)] = {
                "face": face, "row": row,
                "edge_mm": marker_mm,
                "corners_mm": [_point(mounted_face_to_model_face[face],
                                      u, v, width_mm, depth_mm, height_mm)
                               for u, v in uv],
            }
    return {"target_type": "four_face_aruco_cuboid",
            "dictionary": "DICT_4X4_50", "unit": "mm",
            "frame": "cuboid_center_at_bottom; z_up; model face 0=-y, face 1=+x, face 2=+y, face 3=-x",
            "mounted_face_to_model_face": list(mounted_face_to_model_face),
            "width_mm": width_mm, "depth_mm": depth_mm, "height_mm": height_mm,
            "marker_mm": marker_mm, "top_margin_mm": top_margin_mm,
            "bottom_margin_mm": height_mm - top_margin_mm - 2 * marker_mm - marker_gap_mm,
            "side_margin_mm": (width_mm - marker_mm) / 2,
            "marker_gap_mm": marker_gap_mm, "markers": markers,
            "measurement_note": "User-reported mounted dimensions: 45x45x100 mm cuboid, 21 mm black marker edge, 12 mm side margins, 26 mm top margin, 4 mm vertical gap, 28 mm bottom margin. Assumes all four faces share this placement and remain flat."}


def draw_page(config):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    page = Image.new("RGB", (210 * PX_PER_MM, 297 * PX_PER_MM), "white")
    draw = ImageDraw.Draw(page)
    try:
        font = ImageFont.truetype("arial.ttf", 40)
        small = ImageFont.truetype("arial.ttf", 30)
    except OSError:
        font = small = ImageFont.load_default()
    width = config["width_mm"]
    depth = config["depth_mm"]
    height = config["height_mm"]
    marker_mm = config["marker_mm"]
    margin = config["top_margin_mm"]
    gap = config["marker_gap_mm"]
    separation = 6.0
    total_width = width * 2 + depth * 2 + separation * 3
    x_mm = (210 - total_width) / 2
    y_mm = 105.0
    for face in range(4):
        face_width = width if face % 2 == 0 else depth
        x = round(x_mm * PX_PER_MM)
        y = round(y_mm * PX_PER_MM)
        w = round(face_width * PX_PER_MM)
        h = round(height * PX_PER_MM)
        draw.rectangle((x, y, x + w - 1, y + h - 1), outline=(150, 150, 150), width=2)
        draw.text((x, y - 55), f"FACE {face}  IDs {20+face*2},{21+face*2}", fill="black", font=small)
        for row in range(2):
            marker_id = 20 + face * 2 + row
            size_px = round(marker_mm * PX_PER_MM)
            marker = cv2.aruco.generateImageMarker(dictionary, marker_id, size_px)
            mx = x + round((face_width - marker_mm) / 2 * PX_PER_MM)
            my = y + round((margin + row * (marker_mm + gap)) * PX_PER_MM)
            page.paste(Image.fromarray(marker).convert("RGB"), (mx, my))
        x_mm += face_width + separation
    draw.text((250, 1700), f"4-face ArUco cuboid | DICT_4X4_50 | marker {marker_mm:g} mm", fill="black", font=font)
    draw.text((250, 1770), "Print A4 at 100% / actual size; measure marker edges.", fill="black", font=small)
    return page


def main():
    config = build_config()
    page = draw_page(config)
    OUT.mkdir(parents=True, exist_ok=True)
    page.save(OUT / "aruco_cube_4faces_A4.png", dpi=(508, 508))
    page.save(OUT / "aruco_cube_4faces_A4.pdf", "PDF", resolution=508)
    (OUT / "aruco_cube_4faces_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    _, ids, _ = cv2.aruco.ArucoDetector(dictionary).detectMarkers(np.asarray(page.convert("L")))
    if ids is None or set(ids.ravel().tolist()) != set(range(20, 28)):
        raise RuntimeError("Generated page did not decode all eight IDs")
    print(f"Generated 8 markers: {OUT / 'aruco_cube_4faces_A4.pdf'}")


if __name__ == "__main__":
    main()
