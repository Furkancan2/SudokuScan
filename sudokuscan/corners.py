"""Stage 2 - Harris corner detection and automatic corner selection.

Overview of the selection rule
------------------------------
1. Harris response -> threshold relative to its maximum -> non-maximum
   suppression. This gives many candidate points.
2. Find the puzzle REGION using my_region.py.
3. Throw away every candidate outside the region (my_region.py).
4. Select the four extreme corners using my_corners.py.
5. Refine the four points with cv2.cornerSubPix.
"""

import cv2
import numpy as np

from .my_corners import my_pick_corners, my_is_good_quad
from .my_region import my_find_puzzle_region, my_filter_points


# Default Harris parameters
HARRIS_BLOCK_FRAC = 0.004
HARRIS_KSIZE = 3
HARRIS_K = 0.04
HARRIS_REL_THRESH = 0.01
NMS_RADIUS_FRAC = 0.01


def harris_response(norm, block_frac=HARRIS_BLOCK_FRAC, ksize=HARRIS_KSIZE,
                    k=HARRIS_K):
    """Run cv2.cornerHarris on the illumination-normalised image."""
    w = norm.shape[1]
    block_size = max(2, int(round(w * block_frac)))
    response = cv2.cornerHarris(np.float32(norm), block_size, ksize, k)
    return response


def harris_candidates(response, rel_thresh=HARRIS_REL_THRESH, radius=3):
    """Turn the response map into discrete points."""
    r = max(1, int(radius))
    kernel = np.ones((2 * r + 1, 2 * r + 1), np.uint8)
    local_max = cv2.dilate(response, kernel)

    is_peak = (response == local_max) & (
        response > rel_thresh * response.max()
    )

    ys, xs = np.nonzero(is_peak)
    points = np.stack([xs, ys], axis=1).astype(np.float32)
    scores = response[ys, xs]

    return points, scores


def refine_subpix(gray, corners):
    """Move each corner to sub-pixel accuracy."""
    w = gray.shape[1]
    win = max(2, int(round(w * 0.004)))

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.01
    )

    pts = corners.reshape(-1, 1, 2).copy()

    cv2.cornerSubPix(
        gray,
        pts,
        (win, win),
        (-1, -1),
        criteria
    )

    refined = pts.reshape(-1, 2)

    moved = np.linalg.norm(refined - corners, axis=1)

    refined[moved > 2 * win] = corners[moved > 2 * win]

    return refined


def detect_corners(gray, norm, binary, params=None):
    """Full Stage 2 corner detection."""
    params = params or {}

    response = harris_response(
        norm,
        params.get("block_frac", HARRIS_BLOCK_FRAC),
        params.get("ksize", HARRIS_KSIZE),
        params.get("k", HARRIS_K)
    )

    # Use my own region finder
    region = my_find_puzzle_region(binary)

    side = (
        np.sqrt(cv2.contourArea(region))
        if region is not None
        else binary.shape[1] / 3
    )

    radius = max(2, round(NMS_RADIUS_FRAC * side))

    points, scores = harris_candidates(
        response,
        params.get("rel_thresh", HARRIS_REL_THRESH),
        radius
    )

    result = {
        "response": response,
        "candidates": points,
        "kept": np.empty((0, 2), np.float32),
        "corners": None
    }

    if region is None or len(points) < 4:
        return result

    # Use my own point filter
    kept = my_filter_points(points, region)

    result["kept"] = kept
    result["region"] = region

    if len(kept) < 4:
        return result

    # Use my own corner selection function
    corners = my_pick_corners(kept)

    # Use my own quadrilateral validation function
    if not my_is_good_quad(corners, region):
        return result

    result["corners"] = refine_subpix(gray, corners)

    return result
