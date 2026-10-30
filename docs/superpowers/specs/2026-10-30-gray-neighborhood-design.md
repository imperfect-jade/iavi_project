# Gray coordinate neighborhood matching

Add an independent Gray/phase reconstruction entry point because another chat is editing `reconstruct_view.py`. The user requested a wider matching range to recover correspondences lost to vertical calibration residuals, and accepted keeping this work in separate files.

Use exact candidate search within a configurable vertical row window, bounded by horizontal disparity and the two-dimensional projector-coordinate distance. Reject ambiguous candidates, enforce global target uniqueness and reverse consistency, and triangulate the actual left/right pixel coordinates using scaled projection matrices. Preserve original same-row behavior in the existing entry point. Default the new row radius to zero. Expanded search is experimental and cannot establish metric accuracy.

Alternatives considered: increasing projector-coordinate tolerance would also accept wrong code values; a fixed row shift assumes one constant residual across the image; neighborhood matching explicitly records each correspondence's observed row. Use neighborhood matching with an explicit reprojection threshold and unverified-geometry status.

Support either a complete raw view or the saved rectified decoder NPZ. The raw view has been removed during concurrent work, so use the previous decoder cache for the first comparison. Cached reconstructions without an explicitly supplied rectified color image use neutral colors. Preserve calibration and source cache files, and write fresh outputs.

Tests must cover row-radius boundaries, both projector axes, disparity limits, repeated codes and ambiguity, global target collisions, reverse checks in two dimensions, image borders, empty inputs, and exact triangulation with vertical camera translation. Point count alone must not be reported as a calibration fix.
