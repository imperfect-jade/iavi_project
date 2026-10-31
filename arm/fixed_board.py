"""ChArUco detection and pose math for the fixed two-pose calibration board.

Coordinates and translations use millimetres. T_L_W maps board coordinates to
the original (unrectified) left-camera coordinates.
"""
from __future__ import annotations

import cv2
import numpy as np


def board_from_config(config):
    if config.get("board_type") != "ChArUco":
        raise ValueError("Expected a ChArUco board config")
    dictionary_name = config["dictionary"]
    if not hasattr(cv2.aruco, dictionary_name):
        raise ValueError(f"Unknown ArUco dictionary: {dictionary_name}")
    board = cv2.aruco.CharucoBoard(
        (int(config["squares_x"]), int(config["squares_y"])),
        float(config["square_length_mm"]), float(config["marker_length_mm"]),
        cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name)))
    if config.get("opencv_legacy_pattern", False):
        board.setLegacyPattern(True)
    return board


def detect_corners(image, board):
    if image is None or image.size == 0:
        raise ValueError("Empty image")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    corners, ids, _, _ = cv2.aruco.CharucoDetector(board).detectBoard(gray)
    if corners is None or ids is None:
        return np.empty((0, 1, 2), np.float32), np.empty((0, 1), np.int32)
    return corners, ids


def detect_corners_with_contrast_fallback(image, board, min_corners=12):
    """Retry low-contrast photos with CLAHE while keeping original pixel coordinates."""
    corners, ids = detect_corners(image, board)
    if len(ids) >= min_corners:
        return corners, ids, "raw"
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    enhanced_corners, enhanced_ids = detect_corners(enhanced, board)
    if len(enhanced_ids) > len(ids):
        return enhanced_corners, enhanced_ids, "clahe"
    return corners, ids, "raw"


def estimate_pose(board, corners, ids, K, D):
    if len(ids) < 4:
        raise ValueError("At least four ChArUco corners are needed for PnP")
    obj, img = board.matchImagePoints(np.asarray(corners, np.float32),
                                      np.asarray(ids, np.int32))
    obj = np.asarray(obj, np.float64).reshape(-1, 3)
    img = np.asarray(img, np.float64).reshape(-1, 2)
    if np.linalg.matrix_rank(obj[:, :2] - obj[:, :2].mean(axis=0)) < 2:
        raise ValueError("Detected corners are collinear")
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok or not np.isfinite(rvec).all() or not np.isfinite(tvec).all():
        raise ValueError("solvePnP failed")
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.ravel()
    if T[2, 3] <= 0:
        raise ValueError("Board is behind the camera")
    projected, _ = cv2.projectPoints(obj, rvec, tvec, K, D)
    rms = float(np.sqrt(np.mean(np.sum((projected.reshape(-1, 2) - img) ** 2, axis=1))))
    return T, rms


def relative_transform(T_reference_W, T_current_W):
    """Map a current left-camera point into the reference left-camera frame."""
    return np.asarray(T_reference_W) @ np.linalg.inv(np.asarray(T_current_W))


def transform_delta(T_a, T_b):
    delta = np.asarray(T_a) @ np.linalg.inv(np.asarray(T_b))
    rotation_deg = float(np.degrees(np.arccos(np.clip((np.trace(delta[:3, :3]) - 1) / 2, -1, 1))))
    translation_mm = float(np.linalg.norm(delta[:3, 3]))
    return translation_mm, rotation_deg
