"""Generate an A4, actual-size chessboard for close-range stereo calibration."""

from pathlib import Path
import json

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "calibration" / "stereo" / "targets"
PX_PER_MM = 20  # 508 dpi; integer pixels per millimetre
SQUARE_MM = 12
SQUARES_X, SQUARES_Y = 11, 7
PAGE_W_MM, PAGE_H_MM = 210, 297  # A4 portrait
WHITE_MARGIN_MM = 10


def font(size_px):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size_px)
        except OSError:
            pass
    return ImageFont.load_default()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    page = Image.new("RGB", (PAGE_W_MM * PX_PER_MM, PAGE_H_MM * PX_PER_MM), "white")
    draw = ImageDraw.Draw(page)
    square = SQUARE_MM * PX_PER_MM
    board_w, board_h = SQUARES_X * square, SQUARES_Y * square
    x0 = (page.width - board_w) // 2
    y0 = (page.height - board_h) // 2

    for row in range(SQUARES_Y):
        for col in range(SQUARES_X):
            if (row + col) % 2 == 0:
                x, y = x0 + col * square, y0 + row * square
                draw.rectangle((x, y, x + square - 1, y + square - 1), fill="black")

    margin = WHITE_MARGIN_MM * PX_PER_MM
    draw.rectangle((x0 - margin, y0 - margin,
                    x0 + board_w + margin - 1, y0 + board_h + margin - 1),
                   outline=(190, 190, 190), width=2)
    title = "STEREO CHESSBOARD  |  10 x 6 INNER CORNERS  |  12 mm SQUARE"
    draw.text((PAGE_W_MM * PX_PER_MM // 2, 44 * PX_PER_MM), title,
              font=font(54), fill="black", anchor="mm")
    draw.text((PAGE_W_MM * PX_PER_MM // 2, 222 * PX_PER_MM),
              "Print A4 at 100% / actual size. Disable Fit to Page.",
              font=font(48), fill="black", anchor="mm")

    # A separate dimension check makes accidental printer scaling obvious.
    ruler_x, ruler_y = 55 * PX_PER_MM, 242 * PX_PER_MM
    ruler_len = 100 * PX_PER_MM
    draw.line((ruler_x, ruler_y, ruler_x + ruler_len, ruler_y), fill="black", width=5)
    for x in (ruler_x, ruler_x + ruler_len):
        draw.line((x, ruler_y - 10 * PX_PER_MM // 2,
                   x, ruler_y + 10 * PX_PER_MM // 2), fill="black", width=5)
    draw.text((PAGE_W_MM * PX_PER_MM // 2, 253 * PX_PER_MM),
              "100 mm check line (also measure 5 squares = 60 mm)",
              font=font(46), fill="black", anchor="mm")

    pdf_path = OUT / "chessboard_10x6_12mm_A4_print.pdf"
    page.save(pdf_path, "PDF", resolution=PX_PER_MM * 25.4)
    metadata = {
        "pattern": "chessboard", "page_size_mm": [PAGE_W_MM, PAGE_H_MM],
        "squares": [SQUARES_X, SQUARES_Y], "inner_corners": [SQUARES_X - 1, SQUARES_Y - 1],
        "square_size_mm": SQUARE_MM,
        "pattern_size_mm": [SQUARES_X * SQUARE_MM, SQUARES_Y * SQUARE_MM],
        "white_margin_mm": WHITE_MARGIN_MM, "pdf": pdf_path.name,
        "print_setting": "A4, 100% / actual size; no fit-to-page scaling",
    }
    (OUT / "chessboard_10x6_12mm_info.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # The board region must still match the current stereo program's detector.
    board = np.asarray(page.crop((x0, y0, x0 + board_w, y0 + board_h)).convert("L"))
    found, corners = cv2.findChessboardCornersSB(board, (10, 6))
    if not found or len(corners) != 60:
        raise RuntimeError("Generated chessboard is not detected as 10 x 6 corners")
    print(f"Wrote {pdf_path}; detected {len(corners)} inner corners")


if __name__ == "__main__":
    main()
