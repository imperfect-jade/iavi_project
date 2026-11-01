"""Generate the 7 x 5 ChArUco fixed arm-pose target at exact print dimensions."""
from pathlib import Path
import json

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


OUT = Path(__file__).resolve().parent / "targets"
PX_PER_MM = 20  # 508 dpi; exact integer pixels per millimetre
COLS, ROWS = 7, 5  # chessboard squares, not inner corners
SQUARE_MM, MARKER_MM = 10, 7
MARGIN_MM = 7.5
BOARD_W_MM, BOARD_H_MM = 85, 65
DICT_NAME = "DICT_4X4_50"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, DICT_NAME))
    board = cv2.aruco.CharucoBoard(
        (COLS, ROWS), float(SQUARE_MM), float(MARKER_MM), dictionary
    )
    board_px = board.generateImage((COLS * SQUARE_MM * PX_PER_MM,
                                    ROWS * SQUARE_MM * PX_PER_MM), marginSize=0)
    target = Image.new("L", (BOARD_W_MM * PX_PER_MM, BOARD_H_MM * PX_PER_MM), 255)
    target.paste(Image.fromarray(board_px), (int(MARGIN_MM * PX_PER_MM),
                                             int(MARGIN_MM * PX_PER_MM)))
    target.save(OUT / "charuco_7x5_10mm_7mm_85x65mm.png", dpi=(508, 508))

    # A4 at the same 20 px/mm. The physical target is centered without resampling.
    page = Image.new("RGB", (210 * PX_PER_MM, 297 * PX_PER_MM), "white")
    x = (page.width - target.width) // 2
    y = (page.height - target.height) // 2
    page.paste(target.convert("RGB"), (x, y))
    draw = ImageDraw.Draw(page)
    draw.rectangle((x, y, x + target.width - 1, y + target.height - 1),
                   outline=(180, 180, 180), width=1)
    font = ImageFont.truetype("arial.ttf", 45)
    draw.text((x, y - 130), "ChArUco 7 x 5 squares | 10 mm square | 7 mm marker",
              fill="black", font=font)
    draw.text((x, y + target.height + 55),
              "Print A4 at 100% / actual size. Measure 5 squares = 50 mm.",
              fill="black", font=font)
    page.save(OUT / "charuco_7x5_10mm_7mm_A4_print.pdf", "PDF", resolution=508)

    metadata = {
        "board_type": "ChArUco", "dictionary": DICT_NAME,
        "squares_x": COLS, "squares_y": ROWS,
        "square_length_mm": SQUARE_MM, "marker_length_mm": MARKER_MM,
        "pattern_width_mm": COLS * SQUARE_MM,
        "pattern_height_mm": ROWS * SQUARE_MM,
        "target_width_mm": BOARD_W_MM, "target_height_mm": BOARD_H_MM,
        "margin_mm_each_side": MARGIN_MM,
        "png_dpi": 508,
        "opencv_legacy_pattern": False,
        "note": "7 x 5 means squares; interior ChArUco corners are 6 x 4.",
    }
    (OUT / "charuco_7x5_10mm_7mm_config.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    # Confirm OpenCV recognizes the exact generated pattern before delivery.
    corners, ids, _ = cv2.aruco.ArucoDetector(dictionary).detectMarkers(board_px)
    assert ids is not None and len(ids) == len(board.getIds()), f"Detected {0 if ids is None else len(ids)} markers"
    charuco_corners, charuco_ids, _, _ = cv2.aruco.CharucoDetector(board).detectBoard(board_px)
    found = 0 if charuco_ids is None else len(charuco_ids)
    assert found == 24, f"Detected {found} ChArUco corners"
    print(f"Generated target: {target.size}, markers: {len(ids)}, ChArUco corners: {found}")


if __name__ == "__main__":
    main()
