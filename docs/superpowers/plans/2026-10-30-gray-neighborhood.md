# Gray Neighborhood Reconstruction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recover Gray-code correspondences across nearby rows while exposing geometric residuals.

**Architecture:** Add `code/reconstruct_gray_neighborhood.py`, independent of the concurrently edited reconstruction entry point. Reuse stable image decoding and PLY utilities without modifying their source; add exact search, reverse validation, triangulation, and an independent CLI.

**Tech Stack:** Python, NumPy, OpenCV, unittest; no new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-30-gray-neighborhood-design.md`

## Global Constraints

- Do not modify the other chat's files, camera settings, raw images or calibration.
- Work in separate new files in the current workspace, as accepted by the user.
- No commits or branch changes while the user is concurrently editing.
- Row and reprojection limits use processing-image pixels; code tolerance uses projector pixels.
- Cross-row geometry is experimental and must be labeled as unverified in QC.

## Review Focus

- Reject correspondences outside the row window, including image boundaries.
- Search all feasible projector-x candidates, including repeated x values.
- Enforce uniqueness globally across rows and validate reverse x and y.
- Triangulate actual two-dimensional observations; never feed cross-row disparity to Q.
- Cache dimensions, calibration association and absent color must be explicit.

### Task 1: Matching and geometry

**Files:** Create `code/reconstruct_gray_neighborhood.py`; create `tests/test_gray_neighborhood.py`.

- [x] Write failing behavior tests for neighboring-row matching and triangulation.
- [x] Run the new tests and observe failure before implementation.
- [x] Implement exact bounded matching, ambiguity rejection, target uniqueness, reverse checks and triangulation.
- [x] Run the new tests, including a brute-force oracle comparison.

### Task 2: CLI and scan comparison

**Files:** Extend only the new module and new tests; outputs under `processed/gray_neighborhood_20261030`.

- [x] Add raw-view and cached-decoder input modes, fresh output checks and traceability.
- [x] Add explicit row window and reprojection controls, QC and neutral-color fallback.
- [x] Compare radius 0 with radius 4 on the saved scan cache, without changing decoding thresholds.
- [x] Verify PLY counts, actual vertical offsets and reprojection residuals; inspect comparison image.
- [x] Run the project tests once concurrent edits are stable; report any pre-existing failures by name.
- [x] Write usage instructions and a bounded conclusion about coverage and accuracy.
